"""Unprivileged process confinement for code that touches untrusted media.

Uploaded files go through ffmpeg/ffprobe and our decoders, which is the most
likely way in for an attacker. The job runner therefore confines itself
before reading anything:

  * Landlock (kernel LSM, no root needed): the filesystem becomes read-only
    except for the job directory, and only the paths the runner needs are
    visible at all (no home directory, no secrets, no database, no /proc).
    ABI >= 4 also forbids TCP bind/connect.
  * no_new_privs, resource limits (CPU seconds, address space, file size,
    no core dumps).
  * a seccomp filter (apply_seccomp) that refuses every way of opening or using a socket: Landlock does not
    govern Unix-domain sockets before ABI 9, so without it a compromised decoder could reach the user's D-Bus,
    the live stream worker or any abstract socket. The unit's own filter additionally allows only AF_UNIX.
    The GPU runner keeps the ability to create, bind and listen on Unix sockets (the CUDA driver serves one of
    its own) but cannot connect to any socket or send to an address. socketpair() is allowed for both: DeepFilterNet's runtime needs it.

Children (ffmpeg) inherit all of it.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import resource
import sys
from pathlib import Path

import acw_common

# x86_64 / aarch64 share these syscall numbers (asm-generic)
SYS_landlock_create_ruleset = 444
SYS_landlock_add_rule = 445
SYS_landlock_restrict_self = 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
PR_SET_NO_NEW_PRIVS = 38
PR_SET_DUMPABLE = 4

# Filesystem access rights
FS_EXECUTE = 1 << 0
FS_WRITE_FILE = 1 << 1
FS_READ_FILE = 1 << 2
FS_READ_DIR = 1 << 3
FS_REMOVE_DIR = 1 << 4
FS_REMOVE_FILE = 1 << 5
FS_MAKE_CHAR = 1 << 6
FS_MAKE_DIR = 1 << 7
FS_MAKE_REG = 1 << 8
FS_MAKE_SOCK = 1 << 9
FS_MAKE_FIFO = 1 << 10
FS_MAKE_BLOCK = 1 << 11
FS_MAKE_SYM = 1 << 12
FS_REFER = 1 << 13  # ABI 2
FS_TRUNCATE = 1 << 14  # ABI 3
FS_IOCTL_DEV = 1 << 15  # ABI 5

NET_BIND_TCP = 1 << 0  # ABI 4
NET_CONNECT_TCP = 1 << 1

READ = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR
# No links, pipes or sockets can be created in the job directory: the web and worker processes (not sandboxed)
# read what the runner leaves there, and none of it needs any of those.
WRITE = FS_WRITE_FILE | FS_REMOVE_DIR | FS_REMOVE_FILE | FS_MAKE_DIR | FS_MAKE_REG


class SandboxError(RuntimeError):
    pass


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
_libc.syscall.restype = ctypes.c_long


def landlock_abi() -> int:
    abi = _libc.syscall(SYS_landlock_create_ruleset, None, ctypes.c_size_t(0), ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION))
    return int(abi) if abi > 0 else 0


def _handled_fs(abi: int) -> int:
    mask = (1 << 13) - 1
    if abi >= 2:
        mask |= FS_REFER
    if abi >= 3:
        mask |= FS_TRUNCATE
    if abi >= 5:
        mask |= FS_IOCTL_DEV
    return mask


def apply_landlock(read_paths: list[str | Path], write_paths: list[str | Path], *, required: bool = True) -> int:
    """Restrict this process (and future children) to the given paths.
    Returns the Landlock ABI used, or 0 if unavailable and not required."""
    abi = landlock_abi()
    if abi < 1:
        if required:
            raise SandboxError("Landlock is not available on this kernel")
        return 0
    handled = _handled_fs(abi)
    attr = _RulesetAttr(handled, (NET_BIND_TCP | NET_CONNECT_TCP) if abi >= 4 else 0)
    size = ctypes.sizeof(_RulesetAttr) if abi >= 4 else ctypes.sizeof(ctypes.c_uint64)
    ruleset = _libc.syscall(SYS_landlock_create_ruleset, ctypes.byref(attr), ctypes.c_size_t(size), ctypes.c_uint32(0))
    if ruleset < 0:
        raise SandboxError(f"landlock_create_ruleset failed: errno {ctypes.get_errno()}")
    try:
        def add(path: str | Path, access: int) -> None:
            p = str(path)
            if not os.path.exists(p):
                return
            fd = os.open(p, os.O_PATH | os.O_CLOEXEC)
            try:
                if not os.path.isdir(p):
                    # Only file-level rights apply to non-directories.
                    access &= FS_EXECUTE | FS_READ_FILE | FS_WRITE_FILE | FS_TRUNCATE | FS_IOCTL_DEV
                rule = _PathBeneathAttr(access & handled, fd)
                rc = _libc.syscall(SYS_landlock_add_rule, ctypes.c_int(ruleset), ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH), ctypes.byref(rule), ctypes.c_uint32(0))
                if rc < 0:
                    raise SandboxError(f"landlock_add_rule({p}) failed: errno {ctypes.get_errno()}")
            finally:
                os.close(fd)

        for p in read_paths:
            add(p, READ)
        for p in write_paths:
            add(p, READ | WRITE | (FS_TRUNCATE if abi >= 3 else 0) | (FS_REFER if abi >= 2 else 0))
        if _libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise SandboxError("prctl(NO_NEW_PRIVS) failed")
        if _libc.syscall(SYS_landlock_restrict_self, ctypes.c_int(ruleset), ctypes.c_uint32(0)) < 0:
            raise SandboxError(f"landlock_restrict_self failed: errno {ctypes.get_errno()}")
    finally:
        os.close(ruleset)
    return abi


def apply_rlimits(cpu_seconds: int, address_space_bytes: int, max_file_bytes: int) -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 30))
    if address_space_bytes > 0:  # 0: no limit (CUDA reserves far more address space than it uses)
        resource.setrlimit(resource.RLIMIT_AS, (address_space_bytes, address_space_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))
    _libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)


# ---------------------------------------------------------------- seccomp

PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
AUDIT_ARCH_X86_64 = 0xC000003E
AUDIT_ARCH_AARCH64 = 0xC00000B7
AF_UNIX = 1
EPERM = 1
ENOENT = 2

# name -> syscall number, per architecture
SYSCALLS = {
    AUDIT_ARCH_X86_64: {
        "socket": 41, "connect": 42, "accept": 43, "sendto": 44, "sendmsg": 46, "bind": 49, "listen": 50,
        "socketpair": 53, "ptrace": 101, "accept4": 288, "sendmmsg": 307, "process_vm_readv": 310,
        "process_vm_writev": 311, "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
        "pidfd_getfd": 438, "bpf": 321, "unshare": 272, "setns": 308, "perf_event_open": 298, "userfaultfd": 323,
        "add_key": 248, "request_key": 249, "keyctl": 250,
    },
    AUDIT_ARCH_AARCH64: {
        "socket": 198, "socketpair": 199, "bind": 200, "listen": 201, "accept": 202, "connect": 203,
        "sendto": 206, "sendmsg": 211, "ptrace": 117, "accept4": 242, "sendmmsg": 269, "process_vm_readv": 270,
        "process_vm_writev": 271, "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
        "pidfd_getfd": 438, "bpf": 280, "unshare": 97, "setns": 268, "perf_event_open": 241, "userfaultfd": 282,
        "add_key": 217, "request_key": 218, "keyctl": 219,
    },
}
# Refused for every runner. io_uring can open and use sockets by itself, so it goes too; ptrace, process_vm_*
# and pidfd_getfd would let a compromised runner reach into other processes of the same user. bpf, perf_event_open
# and userfaultfd are kernel attack surface no decoder needs; unshare and setns would give it new namespaces (a user
# namespace brings mount and more); the keyring calls reach the user's kernel keys.
ALWAYS_DENIED = ("ptrace", "process_vm_readv", "process_vm_writev", "pidfd_getfd", "io_uring_setup", "io_uring_enter",
                 "io_uring_register", "bpf", "unshare", "setns", "perf_event_open", "userfaultfd", "add_key", "request_key",
                 "keyctl")
# CPU runner: no way to get a socket that could reach anything. socketpair() stays: the connected pair it makes
# leads nowhere (DeepFilterNet's Rust runtime builds one for its signal handling and aborts without it).
NO_SOCKETS = ("socket", "connect", "bind", "listen", "accept", "accept4", "sendto", "sendmmsg")
# GPU runner: the CUDA driver serves an abstract Unix socket of its own ("cuda-uvmfd-...": bind, listen) and
# does not start without it, so those stay allowed. Reaching anything else is not: connect() answers "no such
# file" (what the driver sees for the MPS control socket when MPS isn't running) and sending to an address fails.
NO_SOCKET_REACH = ("sendto", "sendmmsg")
CONNECT_ANSWERS_ENOENT = ("connect",)

# classic BPF
_BPF_LD_W_ABS = 0x20
_BPF_JEQ_K = 0x15
_BPF_RET_K = 0x06
_SECCOMP_DATA_NR = 0
_SECCOMP_DATA_ARCH = 4
_SECCOMP_DATA_ARG0 = 16  # low 32 bits (little-endian)


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8), ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(_SockFilter))]


def seccomp_program(arch: int, *, unix_sockets: bool = False) -> list[tuple[int, int, int, int]]:
    """The filter as (code, jt, jf, k) rows for `arch`: a syscall on the deny list fails with EPERM, anything else
    runs; a call from another architecture (the 32-bit compat entry) ends the process. With `unix_sockets`
    socket() is allowed for AF_UNIX only and the calls that reach another socket stay refused."""
    nums = SYSCALLS[arch]
    denied = list(ALWAYS_DENIED) + list(NO_SOCKET_REACH if unix_sockets else NO_SOCKETS)
    errno = SECCOMP_RET_ERRNO | EPERM
    prog: list[tuple[int, int, int, int]] = [
        (_BPF_LD_W_ABS, 0, 0, _SECCOMP_DATA_ARCH),
        (_BPF_JEQ_K, 1, 0, arch),
        (_BPF_RET_K, 0, 0, SECCOMP_RET_KILL_PROCESS),
        (_BPF_LD_W_ABS, 0, 0, _SECCOMP_DATA_NR),
    ]
    for name in denied:
        prog += [(_BPF_JEQ_K, 0, 1, nums[name]), (_BPF_RET_K, 0, 0, errno)]
    if unix_sockets:
        for name in CONNECT_ANSWERS_ENOENT:
            prog += [(_BPF_JEQ_K, 0, 1, nums[name]), (_BPF_RET_K, 0, 0, SECCOMP_RET_ERRNO | ENOENT)]
    if unix_sockets:  # socket(domain, ...): only AF_UNIX
        prog += [
            (_BPF_JEQ_K, 0, 4, nums["socket"]),
            (_BPF_LD_W_ABS, 0, 0, _SECCOMP_DATA_ARG0),
            (_BPF_JEQ_K, 0, 1, AF_UNIX),
            (_BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW),
            (_BPF_RET_K, 0, 0, errno),
        ]
    prog.append((_BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW))
    return prog


def apply_seccomp(*, unix_sockets: bool = False, required: bool = True) -> bool:
    """Install the filter on this process and its children (needs no_new_privs, which apply_landlock sets)."""
    import platform

    arch = {"x86_64": AUDIT_ARCH_X86_64, "aarch64": AUDIT_ARCH_AARCH64}.get(platform.machine())
    if arch is None:
        if required:
            raise SandboxError(f"no seccomp filter for {platform.machine()}")
        return False
    rows = seccomp_program(arch, unix_sockets=unix_sockets)
    filters = (_SockFilter * len(rows))(*[_SockFilter(*r) for r in rows])
    prog = _SockFprog(len(rows), filters)
    if _libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise SandboxError("prctl(NO_NEW_PRIVS) failed")
    if _libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(prog), 0, 0) != 0:
        if required:
            raise SandboxError(f"seccomp filter failed: errno {ctypes.get_errno()}")
        return False
    return True


def default_read_paths(extra: list[str | Path]) -> list[str | Path]:
    """Read-only view: system libraries, the Python runtime, our code, binaries
    and models. Deliberately excludes $HOME (except what is listed), /proc,
    /run, /var and most of /etc."""
    paths: list[str | Path] = [
        "/usr",
        "/lib",
        "/lib64",
        "/bin",
        "/etc/ld.so.cache",
        "/etc/localtime",
        "/etc/ssl/certs",
        "/sys/devices/system/cpu",
        "/proc/cpuinfo",  # ONNX Runtime / CTranslate2 pick their SIMD kernels from it
        "/dev/urandom",
        "/dev/null",
        sys.base_prefix,
        sys.prefix,
        str(Path(__file__).resolve().parents[2]),  # the directory holding the acw package
        str(Path(acw_common.__file__).resolve().parents[1]),  # ... and the shared package (same dir when deployed)
    ]
    paths.extend(extra)
    return paths


def confine(
    job_dir: Path,
    extra_read: list[str | Path],
    *,
    cpu_seconds: int,
    address_space_bytes: int,
    max_file_bytes: int,
    extra_write: list[str | Path] = (),
    unix_sockets: bool = False,
) -> int:
    """`extra_write`: e.g. the GPU device files for the GPU worker's runner; `unix_sockets`: the GPU runner may
    create Unix sockets and serve them (not connect to any or send to an address)."""
    abi = apply_landlock(default_read_paths(extra_read), [job_dir, "/dev/null", *extra_write])
    apply_rlimits(cpu_seconds, address_space_bytes, max_file_bytes)
    apply_seccomp(unix_sockets=unix_sockets)
    return abi
