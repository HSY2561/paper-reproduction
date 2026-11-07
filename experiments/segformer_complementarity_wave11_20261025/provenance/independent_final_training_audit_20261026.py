"""Independent CPU-only file audit; never imports Paddle or changes frozen evidence."""
from pathlib import Path
import json, hashlib, pickle, csv, statistics, datetime

r = Path(__file__).resolve().parents[1]
issues = []

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()

def ck(ok, msg):
    if not ok:
        issues.append(msg)
    return bool(ok)

release = load(r/'provenance/release.json')
release_hash = sha(r/'provenance/release.json')
for p, s in release['files'].items():
    ck(Path(p).is_file() and sha(p) == s, 'release mismatch '+p)
summ = load(r/'results/four_cell_three_seed_summary.json')
summary_sha = sha(r/'results/four_cell_three_seed_summary.json')
rows = []
for row in summ['rows']:
    ident = row['id']
    checks = {}
    status = load(r/f'results/{ident}_status.json')
    sel = load(r/f'checkpoints/{ident}/selection.json')
    val = load(r/f'results/{ident}_val.json')
    checks['complete'] = ck(status['status'] == 'complete', ident+' status')
    for suffix in ['train', 'official_val', 'pixel_boundary']:
        x = load(r/f'results/{ident}_{suffix}_exit.json')
        checks[suffix+'_exit0'] = ck(x['exit_code'] == 0 and x.get('guard_reason') is None, ident+' '+suffix+' exit')
        ck(x['release_sha256'] == release_hash, ident+' '+suffix+' release')
    log = (r/f'logs/{ident}_train.stdout.log').read_text(encoding='utf-8', errors='replace')
    checks['full18000_log'] = ck('iter: 18000/18000' in log, ident+' no18k')
    checks['final18k_checkpoint'] = ck((r/f'training/{ident}/iter_18000/model.pdparams').is_file(), ident+' no18k cp')
    checks['frozen_sha'] = ck(sha(sel['checkpoint']) == sel['sha256'] == row['checkpoint_sha256'], ident+' frozen SHA')
    checks['config_sha'] = ck(sha(r/f'configs/{ident}.yml') == sel['config_sha256'], ident+' config SHA')
    checks['init_sha'] = ck(sha(r/f'weights/init/{row["init_id"]}.pdparams') == sel['initialization_sha256'], ident+' init SHA')
    states = []
    for c in sel['candidates']:
        preserved = r/f'checkpoints/{ident}'/('ordinary_model.pdstates' if c['variant'] == 'ordinary' else 'ema_ema_model.pdstates')
        ck(sha(preserved) == c['state_sha256'], ident+' '+c['variant']+' retained state SHA')
        ck(sha(c['state_path']) == c['state_sha256'], ident+' '+c['variant']+' original state SHA')
        ck(sha(c['weight']) == c['weight_sha256'], ident+' '+c['variant']+' original weight SHA')
        a = pickle.loads(preserved.read_bytes())
        ck(float(a['mIoU']) == c['mIoU'] and int(a['iter']) == c['iter'], ident+' '+c['variant']+' exact state')
        states.append(dict(variant=c['variant'], mIoU=float(a['mIoU']), iter=int(a['iter']), preserved_path=str(preserved), sha256=sha(preserved)))
    chosen = sorted(states, key=lambda a: (-a['mIoU'], a['iter'], a['variant'] != 'ordinary'))[0]
    checks['exact_normal_ema_selection'] = ck(chosen['variant'] == sel['selected']['variant'] and chosen['iter'] == sel['selected']['iter'] and chosen['mIoU'] == sel['selected']['mIoU'], ident+' selection')
    ck(val['n'] == 348 and abs(val['mIoU']-chosen['mIoU']) < 1e-12, ident+' independentval mismatch')
    with (r/f'results/{ident}_val_per_image.csv').open(encoding='utf-8-sig', newline='') as f:
        cs = list(csv.DictReader(f))
    sums = {k: sum(int(x[k]) for x in cs) for k in ['tn', 'fp', 'fn', 'tp', 'boundary_pred', 'boundary_gt', 'boundary_match_pred_1px', 'boundary_match_gt_1px', 'boundary_match_pred_2px', 'boundary_match_gt_2px', 'pred_skeleton_count', 'gt_skeleton_count', 'pred_skeleton_in_gt', 'gt_skeleton_in_pred']}
    tn, fp, fn, tp = [sums[k] for k in ['tn', 'fp', 'fn', 'tp']]
    vals = dict(precision=tp/(tp+fp), recall=tp/(tp+fn), f1=2*tp/(2*tp+fp+fn), crackIoU=tp/(tp+fp+fn), mIoU=(tp/(tp+fp+fn)+tn/(tn+fp+fn))/2)
    for radius in [1, 2]:
        p = sums[f'boundary_match_pred_{radius}px']/sums['boundary_pred']
        q = sums[f'boundary_match_gt_{radius}px']/sums['boundary_gt']
        vals[f'boundary_f1_{radius}px'] = 2*p*q/(p+q)
    p = sums['pred_skeleton_in_gt']/sums['pred_skeleton_count']
    q = sums['gt_skeleton_in_pred']/sums['gt_skeleton_count']
    vals['hard_cldice'] = 2*p*q/(p+q)
    checks['CSV_global_metrics'] = ck(len(cs) == 348 and all(abs(val[k]-v) < 1e-12 for k, v in vals.items()), ident+' CSV metrics')
    checks['CM'] = ck(val['confusion_matrix'] == [[tn, fp], [fn, tp]], ident+' CM')
    checks['summary_values'] = ck(all(abs(row[k]-val[k]) < 1e-12 for k in ['precision', 'recall', 'f1', 'crackIoU', 'mIoU', 'boundary_f1', 'hard_cldice']), ident+' summary')
    checks['unique348'] = ck(len({c['image'] for c in cs}) == 348, ident+' unique348')
    rng = load(r/f'provenance/rng/{ident}.json')
    checks['rng_full72000'] = ck(rng['status'] == 'complete' and rng['expected_training_samples'] == 72000 and rng['observed_samples'] >= 72000, ident+' rng samples')
    # Witness hashes the logical ASCII index stream with LF, while Windows text
    # output writes CRLF. Check logical normalized stream and budget prefix.
    indices = Path(rng['actual_indices_path']).read_text(encoding='ascii').splitlines()
    normalized = ''.join(x+'\n' for x in indices).encode('ascii')
    budget = ''.join(x+'\n' for x in indices[:72000]).encode('ascii')
    ck(hashlib.sha256(normalized).hexdigest() == rng['actual_indices_sha256'], ident+' actual logical indices SHA')
    ck(hashlib.sha256(budget).hexdigest() == rng['actual_training_indices_sha256'], ident+' actual budget indices SHA')
    rows.append(dict(id=ident, checks=checks, states=states, global_metrics_recomputed_from_CSV=vals, train_log_sha256=sha(r/f'logs/{ident}_train.stdout.log'), final18k_weight_sha256=sha(r/f'training/{ident}/iter_18000/model.pdparams'), selection_sha256=sha(r/f'checkpoints/{ident}/selection.json'), per_image_sha256=sha(r/f'results/{ident}_val_per_image.csv')))

rng_comparison = []
for seed in [42, 43, 44]:
    ids = [f'W11_{g}_s{seed}' for g in ['E0', 'E1', 'E2', 'E3', 'R0']]
    rs = [load(r/f'provenance/rng/{i}.json') for i in ids]
    fields = ['actual_training_indices_sha256', 'training_budget_numpy_state_sha256', 'training_budget_python_state_sha256', 'start_numpy_state_sha256', 'start_python_state_sha256']
    passed = all(len({x[k] for x in rs}) == 1 for k in fields)
    passed = passed and all(x['first_samples'][:8] == rs[0]['first_samples'][:8] for x in rs)
    ck(passed, f'seed{seed} shared RNG')
    for name in [f'paired_formal_s{seed}.json', f'paired_formal_R0_s{seed}.json']:
        ck(load(r/'provenance/rng'/name)['passed'], name+' failed')
    rng_comparison.append(dict(seed=seed, ids=ids, passed=passed, fields=fields))
grouped = {}
for g in ['R0', 'E0', 'E1', 'E2', 'E3']:
    a = [x for x in summ['rows'] if x['group'] == g]
    grouped[g] = {}
    for k in ['precision', 'recall', 'f1', 'crackIoU', 'mIoU', 'boundary_f1', 'hard_cldice']:
        grouped[g][k] = dict(mean=statistics.mean(x[k] for x in a), sample_sd=statistics.stdev(x[k] for x in a))
        ck(abs(grouped[g][k]['mean']-summ['groups'][g]['mean'][k]) < 1e-12, g+' mean '+k)
        ck(abs(grouped[g][k]['sample_sd']-summ['groups'][g]['sample_sd'][k]) < 1e-12, g+' sd '+k)
for p in ['results/workflow_exit.json', 'results/controller_20261025_185807_0583490_exit.json']:
    ck(load(r/p)['exit_code'] == 0, p+' exit')
ck(load(r/'results/queue_done.json')['complete_runs'] == 15, 'queue completeness')
data = load(r/'provenance/data_queue_after.json')
ck(data['checked_files'] == 4488 and not data['mismatches'], 'data hash')
independent_gates = {}
gate_fields = ['crackIoU', 'f1', 'mIoU', 'precision', 'recall', 'boundary_f1']
by_seed = {(x['group'], x['seed']): x for x in summ['rows']}
for left, right in [('E1', 'E0'), ('E2', 'E0'), ('E3', 'E0'), ('E3', 'E1'), ('E3', 'E2')]:
    deltas = [{k: by_seed[left, s][k]-by_seed[right, s][k] for k in gate_fields} for s in [42,43,44]]
    mean = {k: statistics.mean(d[k] for d in deltas) for k in gate_fields}
    limits = dict(crackIoU=.002, f1=.001, mIoU=0, precision=-.005, recall=-.005, boundary_f1=-.005)
    if left == 'E2':
        limits['boundary_f1'] = .01
    if left == 'E3' and right != 'E0':
        limits.update(crackIoU=.001, f1=0)
    mean_gate = all(mean[k] >= limits[k] for k in limits)
    keys = ['crackIoU'] if left == 'E3' and right != 'E0' else ['crackIoU', 'f1', 'boundary_f1'] if left == 'E2' else ['crackIoU', 'f1']
    direction = all(sum(d[k] > 0 for d in deltas) >= 2 for k in keys)
    old = summ['preregistered_comparisons'][left+'-'+right]
    ck(mean_gate == old['mean_effect_gate'] and direction == old['direction_gate'], left+'-'+right+' gates')
    independent_gates[left+'-'+right] = dict(mean_delta=mean, mean_gate=mean_gate, direction_gate=direction)
init = load(r/'provenance/initialization_manifest.json')['runs']
matched = load(r/'provenance/matched_R0_initialization.json')['runs']
independent_resource_gates = {}
for group in ['E0', 'E1', 'E2', 'E3']:
    deltas = [{k: by_seed[group,s][k]-by_seed['R0',s][k] for k in gate_fields} for s in [42,43,44]]
    mean = {k: statistics.mean(d[k] for d in deltas) for k in gate_fields}
    registered_ratio = init[group+'_s42']['registered_numel']/matched['R0_s42']['registered_numel']
    trainable_ratio = init[group+'_s42']['trainable_numel']/matched['R0_s42']['trainable_numel']
    passed = registered_ratio <= .9 and trainable_ratio <= .9
    passed = passed and all(mean[k] >= (-.002 if k in ['crackIoU','f1','mIoU'] else -.005) for k in gate_fields)
    within = sum(d['crackIoU'] >= -.002 for d in deltas)
    passed = passed and within >= 2
    ck(passed == summ['matched_R0_lightweight_comparisons'][group]['passed'],group+' resource gate')
    independent_resource_gates[group] = dict(mean_delta=mean, registered_ratio=registered_ratio, trainable_ratio=trainable_ratio, within_tolerance_seeds=within, passed=passed)
report = dict(at=datetime.datetime.now().astimezone().isoformat(), auditor='independent collaboration agent w11_final_training_audit; CPU-only file/evidence audit; not external cross-model skill verdict', scope='15 training runs and validation-only; no GPU, no test/custom data read; source/summary unchanged', release_sha256=release_hash, summary_sha256=summary_sha, release_files_checked=len(release['files']), complete_runs=len(rows), issues=issues, rows=rows, rng_comparison=rng_comparison, recomputed_grouped=grouped, conclusions=dict(all_main_module_gates_fail=all(not x['mean_effect_gate'] for x in summ['preregistered_comparisons'].values()), all_matched_R0_balanced_gates_fail=all(not x['passed'] for x in summ['matched_R0_lightweight_comparisons'].values()), interaction=summ['interaction_crackIoU'], supports_two_modules_plus_combination=summ['supports_two_modules_plus_combination']), notes=['GPU postprocessing and qualitative rendering not audited as complete. All original normal/EMA weights/states and preserved state copies are SHA-verified.'])
report.update(independently_recomputed_module_gates=independent_gates, independently_recomputed_resource_gates=independent_resource_gates)
p = r/'provenance/independent_final_training_audit_20261026.json'
p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(path=str(p), sha256=sha(p), issues=issues, complete_runs=len(rows), grouped=grouped, conclusions=report['conclusions']), ensure_ascii=False, indent=2))
