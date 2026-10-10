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
  * The worker's systemd unit adds a seccomp filter that only allows AF_UNIX
    sockets (see deploy/systemd), which also covers UDP.

Children (ffmpeg) inherit all of it.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import resource
import sys
from pathlib import Path

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
WRITE = (
    FS_WRITE_FILE | FS_REMOVE_DIR | FS_REMOVE_FILE | FS_MAKE_DIR | FS_MAKE_REG
    | FS_MAKE_SYM | FS_MAKE_FIFO | FS_MAKE_SOCK
)


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
    resource.setrlimit(resource.RLIMIT_AS, (address_space_bytes, address_space_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))
    _libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)


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
        "/dev/urandom",
        "/dev/null",
        sys.base_prefix,
        sys.prefix,
        str(Path(__file__).resolve().parents[2]),  # the directory holding the acw package
    ]
    paths.extend(extra)
    return paths


def confine(job_dir: Path, extra_read: list[str | Path], *, cpu_seconds: int, address_space_bytes: int, max_file_bytes: int) -> int:
    abi = apply_landlock(default_read_paths(extra_read), [job_dir, "/dev/null"])
    apply_rlimits(cpu_seconds, address_space_bytes, max_file_bytes)
    return abi
