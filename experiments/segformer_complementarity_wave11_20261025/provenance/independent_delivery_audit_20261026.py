"""Read-only CPU verification of completed W11 quantitative package."""
from pathlib import Path
import json, hashlib, zipfile, csv, statistics, math, datetime

r = Path(__file__).resolve().parents[1]
issues = []
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(ok, message):
    if not ok: issues.append(message)
def close(a, b): return a is None and b is None or a is not None and b is not None and abs(float(a)-float(b)) < 1e-10
def pct(v, q):
    s=sorted(v); at=(len(s)-1)*q; low=math.floor(at); high=math.ceil(at)
    return s[low]+(s[high]-s[low])*(at-low)

done = load(r/'results/final_delivery_done.json')
ck(done['complete'] and done['actual_child_exit_code'] == 0, 'final delivery not complete/exit0')
post = done['postprocess']; d = Path(post['directory'])
ck(post['exit_code'] == 0 and post['stage'] == 'full_profile_and_report_complete', 'postprocess stage')
ck(load(r/'results/automatic_postprocess_child_exit.json')['exit_code'] == 0, 'automatic child exit')
ck(load(d/'postprocess_exit.json')['exit_code'] == 0, 'postprocess exit receipt')
summ = load(r/'results/four_cell_three_seed_summary.json')
ck(sha(r/'results/four_cell_three_seed_summary.json') == load(d/'postprocess_release.json')['inputs']['summary']['sha256'], 'summary changed')
profile = load(d/'interleaved_profile.json'); memory = load(d/'memory_summary.json')
ck(profile['status'] == 'complete' and len(profile['rows']) == 50 and len(profile['summary']) == 10, 'profile incomplete')
ck(not profile['protocol']['images_read'] and not profile['protocol']['test_read'], 'profile data access')
ck(profile['protocol']['precision'] == 'fp32' and profile['protocol']['batch'] == 1 and profile['protocol']['warmup_per_model_per_size'] == 100 and profile['protocol']['rounds'] == 5 and profile['protocol']['forwards_each_round'] == 100 and profile['protocol']['synchronize_each_forward'], 'profile fixed protocol')
ck(profile['quiet_background_window']['passed'] and not profile['absolute_idle_claim'], 'background disclosure')
init=load(r/'provenance/initialization_manifest.json')['runs']; init.update(load(r/'provenance/matched_R0_initialization.json')['runs'])
efficiency=[]
for item in profile['summary']:
    g=item['group']; size=item['size']; subset=[x for x in profile['rows'] if x['group']==g and x['size']==size]
    vals=[v for x in subset for v in x['individual_ms']]
    ck(len(subset)==5 and len(vals)==500 and all(v>0 and math.isfinite(v) for v in vals), f'{g} {size} samples')
    for x in subset:
        ck(close(pct(x['individual_ms'], .5),x['P50_ms']) and close(pct(x['individual_ms'],.95),x['P95_ms']),f'{g} {size} round percentiles')
        ck(x['order'] == ['R0','E0','E1','E2','E3'][x['round']:]+['R0','E0','E1','E2','E3'][:x['round']],f'{g} {size} rotation')
    ck(close(pct(vals,.5),item['P50_ms']) and close(pct(vals,.95),item['P95_ms']),f'{g} {size} percentiles')
    ratios=[x['P50_ms']/next(z['P50_ms'] for z in profile['rows'] if z['size']==size and z['round']==x['round'] and z['group']=='R0') for x in subset]
    cv=statistics.stdev(ratios)/statistics.mean(ratios)
    ck(close(statistics.median(ratios),item['paired_P50_ratio_R0_median']) and close(cv,item['paired_ratio_sample_CV']),f'{g} {size} ratio/CV')
    ck(item['stable_CV_le_5pct'] == (cv <= .05),f'{g} {size} stable flag')
    counts=init[g+'_s42']; ck(item['counts']['registered']==counts['registered_numel'] and item['counts']['trainable']==counts['trainable_numel'],g+' latency parameter counts')
    ck(sha(item['checkpoint'])==item['sha256'] and sha(item['config'])==item['config_sha256'],g+' latency frozen SHA')
    efficiency.append(dict(group=g,size=size,P50_ms=item['P50_ms'],P95_ms=item['P95_ms'],paired_ratio_R0=statistics.median(ratios),CV=cv,stable=cv<=.05,slower_than_R0=statistics.median(ratios)>1))
mem=[]
ck(len(memory['records'])==10 and len(memory['child_receipts'])==10, 'memory incomplete')
for item in memory['records']:
    g=item['group']; size=item['size']; receipt=load(d/f'memory_{g}_{size}_exit.json')
    ck(receipt['exit_code']==0 and not receipt['timed_out'],f'{g} {size} memory exit')
    ck(item['pid']>0 and item['counts']['registered']==init[g+'_s42']['registered_numel'] and item['counts']['trainable']==init[g+'_s42']['trainable_numel'],f'{g} {size} memory counts/PID')
    ck(item['peak_allocated_bytes']>=item['baseline_allocated_bytes'] and item['peak_reserved_bytes']>=item['baseline_reserved_bytes'] and item['increment_allocated_bytes']==item['peak_allocated_bytes']-item['baseline_allocated_bytes'],f'{g} {size} memory numbers')
    ck(sha(item['checkpoint'])==item['sha256'] and sha(item['config'])==item['config_sha256'],g+' memory frozen SHA')
    mem.append(dict(group=g,size=size,actual_pid=item['pid'],exit0=True,peak_allocated_MiB=item['peak_allocated_bytes']/2**20,peak_reserved_MiB=item['peak_reserved_bytes']/2**20,increment_allocated_MiB=item['increment_allocated_bytes']/2**20))
ck(len({x['actual_pid'] for x in mem})==10,'memory child PID uniqueness')
with Path(post['delivery']['csv']).open(encoding='utf-8-sig',newline='') as f: csv_rows=list(csv.DictReader(f))
ck(len(csv_rows)==5,'three seed CSV rows')
for row in csv_rows:
    g=row['group']
    for k,v in row.items():
        if k.endswith('_mean'):
            metric=k[:-5]; ck(close(float(v),summ['groups'][g]['mean'][metric]),g+' CSV '+k)
        elif k.endswith('_sample_SD'):
            metric=k[:-10]; ck(close(float(v),summ['groups'][g]['sample_sd'][metric]),g+' CSV '+k)
gates=load(d/'gates_and_interactions.json')
for k,v in gates.items(): ck(v==summ[k],'gates JSON '+k)
bundle=post['bundle']; ck(sha(bundle['path'])==bundle['sha256'] and sha(bundle['manifest'])==bundle['manifest_sha256'],'bundle/manifest SHA')
manifest=load(bundle['manifest'])
zip_verified=0
with zipfile.ZipFile(bundle['path']) as z:
    ck(z.testzip() is None,'ZIP CRC')
    ck(len(manifest['files'])==bundle['files'] and len(z.namelist())==bundle['files']+1,'ZIP count')
    ck(hashlib.sha256(z.read('postprocessing/bundle_manifest.json')).hexdigest()==bundle['manifest_sha256'],'ZIP internal manifest')
    for item in manifest['files']:
        raw=z.read(item['archive_name'])
        ck(len(raw)==item['size_bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],'ZIP evidence '+item['archive_name'])
        ck(not item['archive_name'].lower().endswith(('.pdparams','.pdopt','.pdstates','.jpg','.jpeg')), 'raw/binary inclusion '+item['archive_name'])
        zip_verified+=1
ck(not manifest['raw_data_included'] and not manifest['large_weights_included'],'bundle exclusion flags')
reports=[Path(post['delivery'][k]) for k in ['local_report','experiment_report','root_report']]
ck(len({sha(p) for p in reports})==1,'report versions mismatch')
text=reports[0].read_text(encoding='utf-8')
ck('False' in text and '不删除负结果' in text and '没有领域SOTA结论' in text and 'FLOPs：NA' in text,'report limitations')
for p in post['delivery']['figures']: ck(Path(p).is_file() and Path(p).stat().st_size>1000,'figure '+p)
result=dict(at=datetime.datetime.now().astimezone().isoformat(),auditor='independent collaboration agent; CPU-only completed-delivery file audit',issues=issues,passed=not issues,training_runs=15,summary_sha256=sha(r/'results/four_cell_three_seed_summary.json'),profile_sha256=sha(d/'interleaved_profile.json'),memory_receipts_verified=10,memory=mem,efficiency=efficiency,unstable_rows=[x for x in efficiency if not x['stable']],slower_rows=[x for x in efficiency if x['group']!='R0' and x['slower_than_R0']],zip_evidence_files_verified=zip_verified,bundle=bundle,report=str(reports[2]),postprocess_directory=str(d),scope='Statistics/efficiency/report/main ZIP only; qualitative supplement not yet audited; no GPU/test/custom data read',source_and_summary_modified=False)
out=r/'provenance/independent_delivery_audit_20261026.json'; out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(path=str(out),sha256=sha(out),issues=issues,efficiency=efficiency,memory=mem,bundle=bundle),ensure_ascii=False,indent=2))
