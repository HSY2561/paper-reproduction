"""Independent CPU-only contract review of the W12 profiling entry.

All measurement-loop tests replace the model/Paddle/process launcher with
explicit in-memory doubles. No model, GPU, dataset or user process is touched.
"""
from pathlib import Path
import ast
import contextlib
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "profile_w12.py"
EXPECTED = "17662665cdbea20241bb4053d9d1c7efd31e2773f404edcad00675569f8e1213"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def patched(obj, **updates):
    old = {key: getattr(obj, key) for key in updates}
    try:
        for key, value in updates.items():
            setattr(obj, key, value)
        yield
    finally:
        for key, value in old.items():
            setattr(obj, key, value)


def fail(exc):
    raise exc


def main():
    assert sha(ENTRY) == EXPECTED, "Profile source changed before independent review"
    spec = importlib.util.spec_from_file_location("w12_profile_independent", ENTRY)
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    assert "paddle" not in sys.modules
    checks = {}
    def record(name, condition):
        assert condition, name
        checks[name] = True

    record("CPU_import_has_no_Paddle_or_GPU_context", "paddle" not in sys.modules)
    record("percentiles", p.percentile([1, 2, 3, 4], .5) == 2.5 and abs(p.percentile([1, 2, 3, 4], .95)-3.85) < 1e-12)
    record("sample_CV", abs(p.cv([1, 2, 3])-statistics.stdev([1, 2, 3])/2) < 1e-12)
    for n in (2, 3, 4, 5, 13, 14):
        rounds = p.balanced_rounds(n)
        positions = [[0] * n for _ in range(n)]
        for r in range(rounds):
            order = list(range(n))[r % n:] + list(range(n))[:r % n]
            for position, item in enumerate(order):
                positions[item][position] += 1
        record(f"complete_cyclic_position_balance_N{n}", rounds >= 5 and rounds % n == 0 and all(len(set(x)) == 1 for x in positions))
    record("exact_protocol_sizes", p.SIZES == (400, 480) and p.PROTOCOL["batch"] == 1 and p.PROTOCOL["precision"] == "fp32")

    directory = ROOT / "provenance" / ("profile_independent_cpu_assets_" + str(time.time_ns()))
    directory.mkdir(parents=True)
    exe = directory / "desktop.bin"; exe.write_bytes(b"Registered desktop identity CPU fixture")
    item = dict(pid=100001, type="C+G", process_name=str(exe), basename=exe.name,
                identity=dict(pid=100001, creation_time_filetime=11, executable_path=str(exe)))
    baseline = dict(processes=[], binary_identities=[dict(process_name=str(exe), basename=exe.name, sha256=sha(exe))])
    record("registered_exact_binary_new_PID_allowed", not p.classify_processes([item], baseline, -1)["blocked"])
    for name, changed in (
        ("pure_C_refused", dict(item, type="C")),
        ("unknown_path_refused", dict(item, process_name=str(directory / "other.bin"))),
        ("missing_path_refused", dict(item, process_name="")),
        ("Python_refused", dict(item, basename="python.exe"))):
        record(name, bool(p.classify_processes([changed], baseline, -1)["blocked"]))
    exe.write_bytes(b"changed hash")
    record("changed_binary_hash_refused", bool(p.classify_processes([item], baseline, -1)["blocked"]))
    record("bare_PID_without_birth_refused", not p.identity_matches(dict(pid=1), dict(pid=1)))
    record("PID_birth_reuse_refused", not p.identity_matches(dict(pid=1, creation_time_filetime=2), dict(pid=1, creation_time_filetime=3)))
    server = directory / "server.py"; server.write_text("CPU console identity fixture\n", encoding="utf-8")
    registration = dict(identity=dict(pid=7, creation_time_filetime=8, executable_path="python.exe"),
        command_line="python crack_console/server.py --port 8765", source_files={str(server): sha(server)})
    console = dict(pid=7, type="C+G", basename="python.exe", process_name="python.exe", identity=registration["identity"])
    registered = dict(processes=[], binary_identities=[], known_cpu_services=[registration])
    record("exact_CPU_console_CplusG_exception_allowed", not p.classify_processes([console], registered, -1, dict(cpu_verified=True))["blocked"])
    record("CPU_console_GPU_status_refused", bool(p.classify_processes([console], registered, -1, dict(cpu_verified=False))["blocked"]))
    record("CPU_console_pure_C_still_refused", bool(p.classify_processes([dict(console, type="C")], registered, -1, dict(cpu_verified=True))["blocked"]))
    server.write_text("changed server source\n", encoding="utf-8")
    record("CPU_console_changed_source_refused", bool(p.classify_processes([console], registered, -1, dict(cpu_verified=True))["blocked"]))
    record("status_endpoint_is_narrow", not p.console_status("http://127.0.0.1:9999/other")["cpu_verified"])

    fakeconfig, fakecheckpoint = directory / "cfg.yml", directory / "state.bin"
    fakeconfig.write_text("model: {}\n", encoding="utf-8"); fakecheckpoint.write_bytes(b"CPU metadata-only fixture")
    specpath = directory / "models.json"
    model = dict(id="CONTROL_Z", reference=True, config=str(fakeconfig), checkpoint=str(fakecheckpoint), checkpoint_sha256=sha(fakecheckpoint))
    p.write(specpath, dict(models=[dict(model, id="CANDIDATE_A", reference=False), model]))
    loaded = p.load_spec(specpath)
    record("arbitrary_reference_not_fixed_R0", loaded["reference_id"] == "CONTROL_Z")
    p.write(specpath, dict(models=[model, dict(model, id="second_reference")]))
    try:
        p.load_spec(specpath)
    except ValueError:
        record("duplicate_reference_refused", True)
    else:
        raise AssertionError("Two references accepted")

    class FakeCuda:
        def memory_allocated(self): return 10
        def memory_reserved(self): return 20
        def max_memory_allocated(self): return 30
        def max_memory_reserved(self): return 40
        def reset_max_memory_allocated(self): pass
        def reset_max_memory_reserved(self): pass
        def empty_cache(self): pass
        def get_device_name(self): return "CPU_REVIEW_DOUBLE_NOT_A_GPU"
    fakepaddle = SimpleNamespace(device=SimpleNamespace(cuda=FakeCuda()), seed=lambda *_: None,
        randn=lambda *a, **k: object(), no_grad=contextlib.nullcontext, __version__="CPU-review-double")
    count = {"calls": 0}
    def fake_model(_): count["calls"] += 1
    def build(item): return fakepaddle, fake_model, dict(id=item["id"])
    minimal = dict(models=[dict(id="A", reference=False), dict(id="Z", reference=True)],
                   reference_id="Z", sha256="0"*64, path=str(specpath), desktop_baseline_sha256="1"*64)
    counter = {"v": 0}
    def telemetry():
        counter["v"] += 1
        return dict(position=counter["v"])
    def background(_spec, snapshot):
        if snapshot["position"] == 2:
            raise p.ProfileDeferred("CPU fixture post-memory background invalid", snapshot)
    memorypath = directory / "memory_invalid.json"
    with patched(p, quiet_window=lambda *_: dict(passed=True), telemetry=telemetry,
                 assert_background=background, assert_inputs=lambda *_: None,
                 build_model=build, synchronize=lambda *_: None):
        try:
            p.memory_one(minimal, "A", 400, memorypath)
        except p.ProfileDeferred:
            invalid = p.read(memorypath)
            record("invalid_post_memory_measurement_preserved", invalid["measurement_valid"] is False and invalid["peak_allocated_bytes"] == 30 and count["calls"] == 110)
        else:
            raise AssertionError("Bad post-memory background accepted")
    latencydir = directory / "latency_invalid"; latencydir.mkdir()
    counter["v"] = 0; count["calls"] = 0
    def background_latency(_spec, snapshot):
        if snapshot["position"] == 3:
            raise p.ProfileDeferred("CPU fixture post-latency background invalid", snapshot)
    with patched(p, quiet_window=lambda *_: dict(passed=True), telemetry=telemetry,
                 assert_background=background_latency, assert_inputs=lambda *_: None,
                 build_model=build, synchronize=lambda *_: None):
        try:
            p.interleaved(minimal, latencydir)
        except p.ProfileDeferred:
            invalid = p.read(latencydir / "interleaved_partial.json")
            record("invalid_latency_round_all100_samples_preserved", invalid["status"] == "deferred_invalid_background" and len(invalid["rows"]) == 1 and len(invalid["rows"][0]["individual_ms"]) == 100 and invalid["rows"][0]["background_valid"] is False)
            record("invalid_latency_no_complete_result", not (latencydir / "interleaved_profile.json").exists())
        else:
            raise AssertionError("Bad post-round background accepted")

    class FakeChild:
        pid = 777777
        returncode = None
        def __init__(self, argv, **_):
            output = Path(argv[argv.index("--output") + 1])
            ident, size = argv[argv.index("--memory-one") + 1:argv.index("--memory-one") + 3]
            p.write(output, dict(id=ident, size=int(size), measurement_valid=True))
        def wait(self, timeout=None):
            raise p.subprocess.TimeoutExpired("CPU fixture", timeout)
        def poll(self): return None
    timeoutdir = directory / "timeout_race"; timeoutdir.mkdir()
    with patched(p.subprocess, Popen=FakeChild), patched(p, assert_inputs=lambda *_: None,
             process_identity=lambda pid: dict(pid=pid, creation_time_filetime=7),
             stop_owned_child=lambda *a: dict(stopped=True), finish_owned_stop=lambda *a: 0):
        try:
            p.memory_children(minimal, timeoutdir)
        except RuntimeError as error:
            receipt = p.read(timeoutdir / "memory_A_400_exit.json")
            record("timeout_even_later_exit0_never_adopted", receipt["exit_code"] == 0 and receipt["timed_out"] is True and "exceeded its budget" in str(error) and not (timeoutdir / "memory_summary.json").exists())
        else:
            raise AssertionError("Timed-out measurement with later0 adopted")
    child = FakeChild(["--output", str(directory / "stop_fixture.json"), "--memory-one", "A", "400"])
    with patched(p, process_identity=lambda pid: dict(pid=pid, creation_time_filetime=9)):
        stop = p.stop_owned_child(child, dict(pid=child.pid, creation_time_filetime=8))
        record("ownership_mismatch_no_process_stopped", stop["stopped"] is False and "Ownership" in stop["reason"])
    record_stop = dict(stopped=True)
    record("unknown_after_owned_stop_is_not_exit0", p.finish_owned_stop(child, record_stop) is None and record_stop["wait_timed_out"] is True and record_stop["final_exit_code"] is None)

    tree = ast.parse(ENTRY.read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    record("no_top_level_Paddle_import", not any((getattr(node, "module", "") or "").startswith("paddle") or any(alias.name.startswith("paddle") for alias in getattr(node, "names", [])) for node in imports))
    build_ast = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_model")
    record("builder_does_not_construct_dataset_or_loss", not any(isinstance(node, ast.Attribute) and node.attr in ("train_dataset", "val_dataset", "loss") for node in ast.walk(build_ast)))
    record("final_source_SHA_unchanged", sha(ENTRY) == EXPECTED)
    record("Paddle_never_imported_by_independent_review", "paddle" not in sys.modules)
    result = dict(at=p.now(), passed=True, reviewer="independent architecture agent; source reading plus CPU test doubles only",
        source=str(ENTRY), source_sha256=EXPECTED, CPU_only=True, GPU_execution=False,
        Paddle_imported=False, dataset_read=False, apps_stopped=False,
        checks=checks, fixture_directory=str(directory),
        protocol_findings=dict(reference="one explicit arbitrary ID", memory="2N actual child exits before parent GPU import", ordering="N*ceil(5/N) complete cyclic position balance, frozen before measurement", timing="100 warmup;100 synchronized forwards/round/model/size; pooled P50/P95 and paired-round ratios; sample CV<=.05", memory_windows="allocated/reserved plus baseline/increments and reset availability; whole-process peak labelled when no reset API", failure="Deferred exit3 never emits complete success; invalid post-measurement samples/counts retained; timeout or unknown code never adopted"),
        limitations=["CPU approval does not prove actual GPU speed or idle availability; real hardware measurements remain required.", "Baseline self-ratio CV0 is tautological, not a stability witness.", "Known CPU console exception is limited to registered PID birth/exe, command/source hashes and a read-only CPU-status GET; all other Python/pureC/unknown entries refuse.", "Source contains no hardcoded live PIDs; process identity registries are local artifacts and should not be published with scientific results."],
        source_modified=False)
    path = ROOT / "provenance/profile_independent_review.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(passed=True, checks=len(checks), source_sha256=EXPECTED,
                         review=str(path), review_sha256=sha(path), GPU_execution=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
