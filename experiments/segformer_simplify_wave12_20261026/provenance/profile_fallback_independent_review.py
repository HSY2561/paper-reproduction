"""CPU-only incremental review of protected-process creation-time fallback."""
from pathlib import Path
import contextlib
import ctypes
import hashlib
import importlib.util
import json
import os
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'profile_w12.py'
EXPECTED = 'd489652019f76fb04e3f1791a240aa2b44b4f3d4cdddb755d2afcb9831728102'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def patched(obj, name, value):
    original = getattr(obj, name)
    try:
        setattr(obj, name, value)
        yield
    finally:
        setattr(obj, name, original)


def main():
    assert sha(ENTRY) == EXPECTED
    old = json.loads((ROOT / 'provenance/profile_independent_review.json').read_text())
    assert old['passed'] and old['source_sha256'] == '17662665cdbea20241bb4053d9d1c7efd31e2773f404edcad00675569f8e1213'
    module_spec = importlib.util.spec_from_file_location('w12_profile_fallback_review', ENTRY)
    p = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(p)
    checks = {}
    def record(key, truth):
        assert truth, key
        checks[key] = True

    record('CPU_import_without_Paddle', 'paddle' not in sys.modules)
    own = p._open_process_identity(os.getpid())
    snapshot = p.process_birth_snapshot()
    record('real_snapshot_validates_own_exact_birth_and_name', snapshot['valid'] and
           snapshot['own_validation']['exact_equal'] and
           snapshot['entries'][os.getpid()]['creation_time_filetime'] == own['creation_time_filetime'] and
           Path(own['executable_path']).name.casefold() == snapshot['entries'][os.getpid()]['basename'].casefold())
    denied = lambda pid: dict(pid=pid, running=None, creation_time_filetime=None,
                             executable_path=None, winerror=5)
    with patched(p, '_open_process_identity', denied):
        fallback = p.process_identity(os.getpid(), snapshot)
        record('access_denied_uses_validated_snapshot_birth', fallback['identity_source'] == 'validated_NtQuerySystemInformation' and
               fallback['creation_time_filetime'] == own['creation_time_filetime'])
        record('birth_reuse_still_refused', not p.identity_matches(fallback, dict(fallback, creation_time_filetime=1)))
        record('no_bare_PID_exception', not p.identity_matches(fallback, dict(pid=os.getpid())))
        record('invalid_snapshot_fails_closed', p.process_identity(os.getpid(), dict(valid=False, entries=snapshot['entries']))['creation_time_filetime'] is None)
        record('missing_PID_fails_closed', p.process_identity(0xffffffff, snapshot)['creation_time_filetime'] is None)
    record('OpenProcess_success_not_replaced_by_snapshot', p.process_identity(os.getpid(), dict(valid=False))['identity_source'] == 'OpenProcess_GetProcessTimes')

    # Byte-for-byte synthetic x64 snapshots exercise bounds before dereference.
    encoded = Path(own['executable_path']).name.encode('utf-16-le')
    used = 96 + len(encoded)
    class Query:
        def __init__(self, change): self.change = change
        def __call__(self, kind, storage, capacity, needed):
            base = ctypes.addressof(storage)
            raw = bytearray(used)
            struct.pack_into('<q', raw, 32, own['creation_time_filetime'])
            struct.pack_into('<HH', raw, 56, len(encoded), len(encoded))
            struct.pack_into('<Q', raw, 64, base + 96)
            struct.pack_into('<QQ', raw, 80, os.getpid(), 0)
            raw[96:] = encoded
            if self.change == 'pointer_before': struct.pack_into('<Q', raw, 64, base - 2)
            if self.change == 'pointer_after': struct.pack_into('<Q', raw, 64, base + used)
            if self.change == 'odd_length': struct.pack_into('<H', raw, 56, 3)
            if self.change == 'length_gt_maximum': struct.pack_into('<H', raw, 58, 0)
            if self.change == 'short_step': struct.pack_into('<I', raw, 0, 8)
            if self.change == 'step_outside': struct.pack_into('<I', raw, 0, (used + 7) // 8 * 8)
            if self.change == 'negative_birth': struct.pack_into('<q', raw, 32, -1)
            if self.change == 'invalid_PID': struct.pack_into('<Q', raw, 80, 0x100000000)
            if self.change == 'wrong_own_birth': struct.pack_into('<q', raw, 32, 1)
            ctypes.memmove(base, bytes(raw), used)
            ctypes.cast(needed, ctypes.POINTER(ctypes.c_ulong)).contents.value = 95 if self.change == 'short_return' else used
            return 0
    class NT:
        def __init__(self, change): self.NtQuerySystemInformation = Query(change)
    for change in ('valid', 'pointer_before', 'pointer_after', 'odd_length', 'length_gt_maximum',
                   'short_step', 'step_outside', 'negative_birth', 'invalid_PID', 'wrong_own_birth', 'short_return'):
        with patched(p.ctypes, 'WinDLL', lambda *a, **k: NT(change)), patched(p, '_open_process_identity', lambda pid: dict(own)):
            result = p.process_birth_snapshot()
        record('synthetic_x64_' + change, result['valid'] is (change == 'valid') and
               (result['valid'] or result['entries'] == {}))
    record('final_source_SHA_unchanged', sha(ENTRY) == EXPECTED)
    record('Paddle_never_imported', 'paddle' not in sys.modules)
    result = dict(at=p.now(), passed=True, source_sha256=EXPECTED,
        prior_review=str(ROOT / 'provenance/profile_independent_review.json'),
        prior_review_sha256=sha(ROOT / 'provenance/profile_independent_review.json'),
        prior_35_contract_checks_retained=True,
        scope='Incremental independent review of protected-process birth fallback and telemetry reuse; no measurement protocol change',
        CPU_only=True, GPU_execution=False, dataset_read=False, apps_stopped=False,
        checks=checks,
        observed_real_snapshot=dict(record_count=snapshot['record_count'], structure_bounds_checked=snapshot['structure_bounds_checked'],
                                   unicode_pointer_bounds_checked=snapshot['unicode_pointer_bounds_checked'], own_exact_equal=True),
        limitations=['Actual GPU measurements and quiet-window success remain pending.',
                    'Process identity evidence is local-only; do not publish live process registries.',
                    'Snapshot fallback is x64 Windows only and fails closed if own exact GetProcessTimes/name cannot validate its layout.'])
    target = ROOT / 'provenance/profile_independent_review_d4896520.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(passed=True, checks=len(checks), source_sha256=EXPECTED, review=str(target))))


if __name__ == '__main__':
    main()
