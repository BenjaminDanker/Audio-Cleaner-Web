//! After binding its socket the engine drops filesystem write access and reading outside the libraries and models
//! (Landlock), like the file-job runners (server/acw/worker/sandbox.py). The systemd unit already denies it the
//! network (RestrictAddressFamilies=AF_UNIX) and privileged system calls.

use std::path::{Path, PathBuf};

use landlock::{Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr, RulesetStatus, ABI};

/// Where the sherpa-onnx shared library we linked against lives (the Python venv's copy): it loads its
/// onnxruntime from there at run time.
pub fn sherpa_lib_dir() -> Option<PathBuf> {
    // The build records it (native/live/build.sh sets SHERPA_ONNX_LIB_DIR, also used as the rpath: build.rs).
    option_env!("SHERPA_ONNX_LIB_DIR").map(PathBuf::from)
}

pub fn restrict(read_dirs: &[&Path], sherpa: Option<&Path>) -> Result<String, String> {
    let abi = ABI::V2;
    let read = AccessFs::from_read(abi);
    let mut paths: Vec<PathBuf> = ["/usr", "/lib", "/lib64", "/bin", "/etc/ld.so.cache", "/etc/localtime", "/sys/devices/system/cpu", "/proc/cpuinfo", "/dev/urandom"]
        .iter()
        .map(PathBuf::from)
        .collect();
    paths.extend(read_dirs.iter().map(|p| p.to_path_buf()));
    if let Some(s) = sherpa {
        paths.push(s.to_path_buf());
    }
    let mut rs = Ruleset::default()
        .handle_access(AccessFs::from_all(abi))
        .map_err(|e| e.to_string())?
        .create()
        .map_err(|e| e.to_string())?;
    for p in paths {
        if let Ok(fd) = PathFd::new(&p) {
            rs = rs.add_rule(PathBeneath::new(fd, read)).map_err(|e| e.to_string())?;
        }
    }
    if let Ok(fd) = PathFd::new("/dev/null") {
        rs = rs.add_rule(PathBeneath::new(fd, AccessFs::from_read(abi) | AccessFs::WriteFile)).map_err(|e| e.to_string())?;
    }
    let status = rs.restrict_self().map_err(|e| e.to_string())?;
    match status.ruleset {
        RulesetStatus::FullyEnforced => Ok("fully enforced".into()),
        RulesetStatus::PartiallyEnforced => Ok("partially enforced".into()),
        RulesetStatus::NotEnforced => Err("not supported by this kernel".into()),
    }
}
