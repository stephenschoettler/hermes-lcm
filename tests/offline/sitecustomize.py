"""Opt-in fixture-only guard inherited by every Python test subprocess.

This is test infrastructure, never imported by the plugin. Run with a disposable
HOME/HERMES_HOME/TMPDIR and Python host source under LCM_TEST_FIXTURE_ROOT.
"""
import contextvars
import os
from pathlib import Path
import sys
from urllib.parse import unquote

_fixture_root = Path(os.environ['LCM_TEST_FIXTURE_ROOT']).resolve()
_guard_dir = str(Path(__file__).resolve().parent)
_open_target = contextvars.ContextVar('lcm_test_open_target', default=None)
_fd_paths = {}
_original_open = os.open
_original_close = os.close


def _path(raw, directory_fd=None):
    if isinstance(raw, int):
        if raw not in _fd_paths:
            raise RuntimeError('untracked descriptor in fixture mutation')
        return _fd_paths[raw]
    path = Path(os.fsdecode(raw))
    if not path.is_absolute() and directory_fd not in (None, -1):
        if directory_fd not in _fd_paths:
            raise RuntimeError('untracked directory descriptor in fixture mutation')
        path = _fd_paths[directory_fd] / path
    return path.resolve()


def _inside(path):
    return path.is_relative_to(_fixture_root)


def _open(raw, flags, mode=0o777, *, dir_fd=None):
    target = _path(raw, dir_fd)
    token = _open_target.set(target)
    try:
        fd = _original_open(raw, flags, mode, dir_fd=dir_fd)
    finally:
        _open_target.reset(token)
    _fd_paths[fd] = target
    return fd


def _close(fd):
    _fd_paths.pop(fd, None)
    return _original_close(fd)


os.open = _open
os.close = _close


def _audit(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto', 'socket.sendmsg'}:
        raise RuntimeError('network forbidden in disposable LCM tests')
    if event == 'sqlite3.connect':
        database = str(args[0])
        if database == ':memory:' or (database.startswith('file:') and 'mode=memory' in database):
            return
        if database.startswith('file:'):
            database = unquote(database[5:].split('?', 1)[0])
        if not _inside(_path(database)):
            raise RuntimeError('non-fixture SQLite connection forbidden')
    if event == 'open' and not isinstance(args[0], int):
        path = _open_target.get() or _path(args[0])
        if _inside(path) or str(path) == '/dev/null':
            return
        flags = args[2]
        if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            raise RuntimeError('non-fixture write forbidden: ' + str(path))
        sensitive = {'.env', '.netrc', 'auth.json', 'credentials', 'credentials.json',
                     'config.yaml', 'config.yml', 'config.json'}
        if (path.name in sensitive or path.suffix in {'.db', '.sqlite', '.sqlite3', '.tiktoken'}
                or any(part in {'.aws', '.ssh', '.gnupg'} for part in path.parts)):
            raise RuntimeError('non-fixture credential/config/data read forbidden: ' + str(path))
    if event in {'os.remove', 'os.rmdir'}:
        if not _inside(_path(args[0], args[1])):
            raise RuntimeError('non-fixture removal forbidden')
    if event == 'os.rename':
        if not all(_inside(_path(raw, fd)) for raw, fd in ((args[0], args[2]), (args[1], args[3]))):
            raise RuntimeError('non-fixture rename forbidden')
    if event in {'os.mkdir', 'os.chmod'}:
        if not _inside(_path(args[0], args[2])):
            raise RuntimeError('non-fixture mutation forbidden')
    if event == 'os.system':
        raise RuntimeError('unguarded shell test subprocess forbidden')
    if event in {'subprocess.Popen', 'os.exec', 'os.posix_spawn'}:
        if event == 'subprocess.Popen':
            executable, argv, cwd, supplied_env = args
        else:
            executable, argv, supplied_env = args
        child_env = os.environ if supplied_env is None else supplied_env
        if os.path.realpath(os.fsdecode(executable)) != os.path.realpath(sys.executable):
            # Native git/shell commands cannot inherit Python's audit guard.
            # Product git identity probes fail closed to unknown in this mode.
            raise RuntimeError('unguarded non-Python test subprocess forbidden')
        if any(flag in argv for flag in ('-I', '-E', '-S')):
            raise RuntimeError('test subprocess may not disable its guard')
        if (child_env.get('LCM_TEST_FIXTURE_ROOT') != str(_fixture_root)
                or _guard_dir not in child_env.get('PYTHONPATH', '').split(os.pathsep)):
            raise RuntimeError('test subprocess must inherit its fixture guard')
        for name in ('HOME', 'HERMES_HOME', 'TMPDIR'):
            if not _inside(_path(child_env[name])):
                raise RuntimeError('test subprocess home/temp is not isolated')


sys.addaudithook(_audit)

# Python 3.14 multiprocessing otherwise probes /tmp even with TMPDIR set.
import multiprocessing.util
multiprocessing.util._get_base_temp_dir = lambda tempfile: os.environ['TMPDIR']
