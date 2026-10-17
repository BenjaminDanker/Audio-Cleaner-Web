//! Who gets in, and where each session runs (a port of server/acw/stream/worker.py's admission and
//! server/acw/cpusplit.py's core layout, for one process that serves every live core).
//!
//! Capacity is in units (live denoise = 1, local captions = 0.7, echo cancellation = 0.2): live cores x units per
//! core, shrunk by other load measured on the live cores. Each session is placed on one physical core (its thread is
//! pinned to that core's hyper-threads), filling the lowest cores first; the web process tells file jobs to keep off
//! the cores in use (the session's core comes back in its `ready` event).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

pub const DENOISE_UNITS: f64 = 1.0;
// A caption stream costs about 0.17 cores (its share of a recognizer) next to 0.14 for noise removal in this engine
// (the Python worker's 0.7 was relative to its own, dearer, noise removal). Measured 2026-10-17: 12 noise+caption
// sessions on 5 cores kept captions within 0.3 s, 20 fell 1.8 s behind.
pub const CAPTIONS_LOCAL_UNITS: f64 = 1.2;
pub const ECHO_UNITS: f64 = 0.2;
const LIVE_MIN_MEMORY: u64 = 300 * 1024 * 1024;
const MAX_SESSIONS: usize = 500;
pub const TRIAL_CAPACITY_FRACTION: f64 = 0.7;

pub fn session_units(cfg: &Value) -> f64 {
    let s = |k: &str| cfg.get(k).and_then(|v| v.as_str()).unwrap_or("");
    (if s("denoise") == "light" { DENOISE_UNITS } else { 0.0 })
        + (if s("captions") == "local" { CAPTIONS_LOCAL_UNITS } else { 0.0 })
        + (if cfg.get("echo_reference").and_then(|v| v.as_bool()).unwrap_or(false) { ECHO_UNITS } else { 0.0 })
}

/// Logical CPUs grouped by physical core, in core order (from /sys, read before the sandbox closes it off).
pub fn physical_cores() -> Vec<Vec<usize>> {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut cores: std::collections::BTreeMap<(i64, i64), Vec<usize>> = Default::default();
    for cpu in 0..n.max(1) {
        let base = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
        let read = |f: &str| std::fs::read_to_string(format!("{base}/{f}")).ok().and_then(|s| s.trim().parse::<i64>().ok());
        let key = match (read("physical_package_id"), read("core_id")) {
            (Some(p), Some(c)) => (p, c),
            _ => (0, cpu as i64),
        };
        cores.entry(key).or_default().push(cpu);
    }
    cores.into_values().map(|mut v| {
        v.sort();
        v
    }).collect()
}

/// Physical cores file jobs always keep (ACW_MIN_JOB_CORES, default 1 per 8 cores), as in cpusplit.min_job_cores.
pub fn max_live_cores(cores: usize) -> usize {
    let default = ((cores as f64 / 8.0).round() as usize).max(1);
    let want = std::env::var("ACW_MIN_JOB_CORES").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(default);
    let keep = want.max(1).min(cores.saturating_sub(1).max(1));
    cores.saturating_sub(keep).max(1)
}

pub fn pin_current_thread(cpus: &[usize]) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &c in cpus {
            libc::CPU_SET(c, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

/// CPU use of other programs on the live cores and the memory left, sampled about once a second from files opened
/// before the sandbox (an open file stays readable).
pub struct HostLoad {
    stat: Mutex<Option<File>>,
    selfstat: Mutex<Option<File>>,
    meminfo: Mutex<Option<File>>,
    last: Mutex<Option<(Instant, Vec<(u64, u64)>, u64)>>,
    other_cores: Mutex<f64>, // smoothed
    available: Mutex<u64>,
}

impl HostLoad {
    pub fn open() -> Self {
        Self {
            stat: Mutex::new(File::open("/proc/stat").ok()),
            selfstat: Mutex::new(File::open("/proc/self/stat").ok()),
            meminfo: Mutex::new(File::open("/proc/meminfo").ok()),
            last: Mutex::new(None),
            other_cores: Mutex::new(0.0),
            available: Mutex::new(u64::MAX),
        }
    }

    fn read(f: &Mutex<Option<File>>) -> Option<String> {
        let mut g = f.lock().unwrap();
        let file = g.as_mut()?;
        file.seek(SeekFrom::Start(0)).ok()?;
        let mut s = String::new();
        file.read_to_string(&mut s).ok()?;
        Some(s)
    }

    /// Call about once a second (main thread).
    pub fn sample(&self, live_cpus: &[usize]) {
        let Some(stat) = Self::read(&self.stat) else { return };
        let mut per_cpu: Vec<(u64, u64)> = Vec::new(); // (busy, total) jiffies
        for line in stat.lines() {
            if let Some(rest) = line.strip_prefix("cpu") {
                if rest.starts_with(' ') {
                    continue; // the "cpu" total line
                }
                let mut it = rest.split_whitespace();
                let idx: usize = match it.next().and_then(|s| s.parse().ok()) {
                    Some(i) => i,
                    None => continue,
                };
                let v: Vec<u64> = it.filter_map(|s| s.parse().ok()).collect();
                if v.len() < 5 {
                    continue;
                }
                let idle = v[3] + v[4];
                let total: u64 = v.iter().take(8).sum();
                if per_cpu.len() <= idx {
                    per_cpu.resize(idx + 1, (0, 0));
                }
                per_cpu[idx] = (total - idle, total);
            }
        }
        let own = Self::read(&self.selfstat)
            .and_then(|s| {
                let after = s.rsplit_once(')')?.1.split_whitespace().collect::<Vec<_>>();
                Some(after.get(11)?.parse::<u64>().ok()? + after.get(12)?.parse::<u64>().ok()?)
            })
            .unwrap_or(0);
        let now = Instant::now();
        let mut last = self.last.lock().unwrap();
        if let Some((t0, prev, own0)) = last.as_ref() {
            let dt = now.duration_since(*t0).as_secs_f64().max(0.2);
            let mut busy = 0.0;
            for &c in live_cpus {
                if let (Some(a), Some(b)) = (prev.get(c), per_cpu.get(c)) {
                    let dtot = b.1.saturating_sub(a.1) as f64;
                    if dtot > 0.0 {
                        busy += b.0.saturating_sub(a.0) as f64 / dtot;
                    }
                }
            }
            let own_cores = own.saturating_sub(*own0) as f64 / 100.0 / dt; // jiffies at 100 Hz
            let other = (busy / 2.0 - own_cores).max(0.0); // two hyper-threads make one physical core
            let mut o = self.other_cores.lock().unwrap();
            *o = 0.7 * *o + 0.3 * other;
        }
        *last = Some((now, per_cpu, own));
        if let Some(m) = Self::read(&self.meminfo) {
            if let Some(kb) = m.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()) {
                *self.available.lock().unwrap() = kb * 1024;
            }
        }
    }

    pub fn other_cores(&self) -> f64 {
        if std::env::var("ACW_LIVE_IGNORE_HOST_LOAD").is_ok() {
            return 0.0; // tests: capacity must not depend on what else the machine is doing
        }
        *self.other_cores.lock().unwrap()
    }

    pub fn memory_available(&self) -> u64 {
        *self.available.lock().unwrap()
    }
}

pub struct Admission {
    pub cores: Vec<Vec<usize>>, // the live physical cores (logical CPUs of each)
    pub units_per_core: f64,
    pub caption_cap: usize,
    load: Vec<f64>, // units placed on each live core
    pub active: usize,
    pub units: f64,
    pub caption_streams: usize,
}

pub enum Decision {
    Admit(usize), // the live core index
    Refuse(Value),
}

impl Admission {
    pub fn new(cores: Vec<Vec<usize>>, units_per_core: f64, caption_cap: usize) -> Self {
        let n = cores.len();
        Self { cores, units_per_core, caption_cap, load: vec![0.0; n], active: 0, units: 0.0, caption_streams: 0 }
    }

    pub fn capacity(&self, host: &HostLoad) -> f64 {
        let n = self.cores.len() as f64;
        n * self.units_per_core * (1.0 - host.other_cores() / n).max(0.0)
    }

    pub fn decide(&mut self, cfg: &Value, host: &HostLoad) -> Decision {
        let units = session_units(cfg);
        let trial = cfg.get("trial").and_then(|v| v.as_bool()).unwrap_or(false);
        let cap = self.capacity(host);
        if trial && self.units >= TRIAL_CAPACITY_FRACTION * cap {
            return Decision::Refuse(json!({"type": "error", "code": "trial_busy"}));
        }
        if self.active >= MAX_SESSIONS || host.memory_available() < LIVE_MIN_MEMORY {
            return Decision::Refuse(json!({"type": "error", "code": "full", "available": []}));
        }
        let mut no_echo = cfg.clone();
        no_echo["echo_reference"] = json!(false);
        let local = cfg.get("captions").and_then(|v| v.as_str()) == Some("local");
        let mut without_captions = cfg.clone();
        without_captions["captions"] = Value::Null;
        if session_units(&no_echo) > 0.0 {
            if local && self.caption_streams >= self.caption_cap {
                let options = if self.units + session_units(&without_captions) <= cap {
                    json!([{"captions": "premium"}])
                } else {
                    json!([{"denoise": null, "captions": "premium"}])
                };
                return Decision::Refuse(json!({"type": "error", "code": "full", "available": options}));
            }
            if self.units + units > cap {
                let mut options = Vec::new();
                if local && self.units + session_units(&without_captions) <= cap {
                    options.push(json!({"captions": "premium"}));
                }
                if cfg.get("captions").and_then(|v| v.as_str()) != Some("premium") && options.is_empty() {
                    options.push(json!({"denoise": null, "captions": "premium"}));
                }
                return Decision::Refuse(json!({"type": "error", "code": "full", "available": options}));
            }
        }
        // Lowest core with room for it, else the least loaded: live stays packed on the first cores. A session the
        // broker says may not open a core (a free trial while paid jobs run) only joins cores live already uses.
        let may_open = cfg.get("may_open_core").and_then(|v| v.as_bool()).unwrap_or(true);
        let fits = |i: usize, load: &[f64]| load[i] + units <= self.units_per_core && (may_open || load[i] > 0.0);
        let core = match (0..self.load.len()).find(|&i| fits(i, &self.load)) {
            Some(c) => c,
            None if !may_open => return Decision::Refuse(json!({"type": "error", "code": "trial_busy"})),
            None => (0..self.load.len()).min_by(|&a, &b| self.load[a].partial_cmp(&self.load[b]).unwrap()).unwrap_or(0),
        };
        self.load[core] += units;
        self.active += 1;
        self.units += units;
        if local {
            self.caption_streams += 1;
        }
        Decision::Admit(core)
    }

    pub fn load_of(&self, core: usize) -> f64 {
        self.load.get(core).copied().unwrap_or(0.0)
    }

    pub fn release(&mut self, cfg: &Value, core: usize) {
        let units = session_units(cfg);
        if let Some(l) = self.load.get_mut(core) {
            *l = (*l - units).max(0.0);
        }
        self.active = self.active.saturating_sub(1);
        self.units = (self.units - units).max(0.0);
        if cfg.get("captions").and_then(|v| v.as_str()) == Some("local") {
            self.caption_streams = self.caption_streams.saturating_sub(1);
        }
    }
}

pub fn sleep_until_next_second(t: Instant) {
    let elapsed = t.elapsed();
    if elapsed < Duration::from_secs(1) {
        std::thread::sleep(Duration::from_secs(1) - elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host() -> HostLoad {
        HostLoad { stat: Mutex::new(None), selfstat: Mutex::new(None), meminfo: Mutex::new(None), last: Mutex::new(None), other_cores: Mutex::new(0.0), available: Mutex::new(u64::MAX) }
    }

    #[test]
    fn fills_the_lowest_cores_first_and_refuses_when_full() {
        let mut a = Admission::new(vec![vec![0, 8], vec![1, 9]], 2.0, 10);
        let h = host();
        let d = json!({"denoise": "light"});
        let cores: Vec<usize> = (0..4).map(|_| match a.decide(&d, &h) { Decision::Admit(c) => c, Decision::Refuse(_) => 99 }).collect();
        assert_eq!(cores, vec![0, 0, 1, 1]);
        assert!(matches!(a.decide(&d, &h), Decision::Refuse(_)));
        assert!(matches!(a.decide(&json!({"captions": "premium"}), &h), Decision::Admit(_))); // no units: never refused
        a.release(&d, 0);
        assert!(matches!(a.decide(&d, &h), Decision::Admit(0)));
    }

    #[test]
    fn a_session_that_may_not_open_a_core_only_joins_used_ones() {
        let mut a = Admission::new(vec![vec![0], vec![1]], 2.0, 10);
        let h = host();
        let trial = json!({"denoise": "light", "trial": true, "may_open_core": false});
        assert!(matches!(a.decide(&trial, &h), Decision::Refuse(_))); // nothing runs yet: no core to join
        assert!(matches!(a.decide(&json!({"denoise": "light"}), &h), Decision::Admit(0)));
        assert!(matches!(a.decide(&trial, &h), Decision::Admit(0))); // joins core 0
        assert!(matches!(a.decide(&trial, &h), Decision::Refuse(_))); // core 0 is full; core 1 would be new
    }

    #[test]
    fn trial_sessions_keep_off_the_last_places() {
        let mut a = Admission::new(vec![vec![0], vec![1]], 2.0, 10);
        let h = host();
        for _ in 0..3 {
            assert!(matches!(a.decide(&json!({"denoise": "light"}), &h), Decision::Admit(_)));
        }
        match a.decide(&json!({"denoise": "light", "trial": true}), &h) {
            Decision::Refuse(v) => assert_eq!(v["code"], "trial_busy"),
            _ => panic!(),
        }
    }
}
