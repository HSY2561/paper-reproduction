"""Independent bounded CPU initialization audit; reads no dataset or GPU state."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import datetime as dt
import hashlib
import importlib.util
import json
import sys
import copy
import numpy as np
import paddle

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src/PaddleSeg-2.10.0'
sys.path.insert(0,str(SRC))
paddle.set_device('cpu')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arr(x): return np.asarray(x.numpy() if hasattr(x,'numpy') else x)
def equal(a,b): return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)
cache={}
def load(name):
    if name not in cache:
        cache[name]={k:arr(v) for k,v in paddle.load(manifest['runs'][name]['path']).items()}
    return cache[name]


manifest=read(ROOT/'provenance/initialization_manifest.json')
specs_doc=read(ROOT/'experiment_specs.json');specs={x['id']:x for x in specs_doc['experiments']}
protocol=read(ROOT/'protocol.json')
assert manifest['status']=='passed' and manifest['full_initial_states']==96
assert manifest['device']=='cpu' and not manifest['GPU_execution'] and not manifest['dataset_read']
assert manifest['input_hashes']['specs_sha256']==sha(ROOT/'experiment_specs.json')
assert manifest['source_script_sha256']==sha(ROOT/'prepare_initialization_w12.py')
assert manifest['input_hashes']['R0_manifest_sha256']==sha(Path(manifest['source_R0_manifest']))
assert manifest['input_hashes']['ImageNet_sha256']==sha(Path(manifest['original_imagenet']))
assert len(specs)==32 and set(manifest['runs'])=={f'{sid}_s{seed}' for sid in specs for seed in (42,43,44)}
assert specs_doc['maximum_full18k_new_runs']==18 and protocol['selection']['capacity_only_excluded']==['D03']
for p,h in manifest['model_source_hashes'].items():assert sha(p)==h,p

ledgers={};count_checks=[]
for name,entry in manifest['runs'].items():
    spec=specs[entry['spec_id']]
    assert sha(entry['path'])==entry['sha256'],name
    assert sha(entry['tensor_ledger'])==entry['tensor_ledger_sha256'],name
    ledger=read(entry['tensor_ledger']);ledgers[name]=ledger
    assert ledger['spec_id']==entry['spec_id'] and entry['seed'] in (42,43,44)
    assert entry['model']=={k:spec[k] for k in ('embedding_dim','decoder_kind','alignment','s4_hidden')}
    w=spec['embedding_dim'];h=spec['s4_hidden']
    backbone=3319392-(1024-h)*1046
    count=backbone+522*w+2
    if spec['decoder_kind']=='concat':count+=4*w*w
    if spec['alignment']!='none':
        f=w//2 if spec['alignment']=='full' else (16 if spec['alignment']=='flow16' else 32)
        count+=f*(2*w+(22 if spec['alignment']=='dwflow32' else 36))
    assert entry['registered_numel']==count and entry['trainable_numel']==count-2*w,name
    assert sum(x['numel'] for x in ledger['tensors'].values())==count,name
    count_checks.append(dict(id=name,registered=count,trainable=count-2*w))

source_r0=read(manifest['source_R0_manifest'])['runs'];r0={}
for seed in (42,43,44):
    p=source_r0[f'R0_s{seed}']['path'];assert sha(p)==source_r0[f'R0_s{seed}']['sha256']
    r0[seed]={k:arr(v) for k,v in paddle.load(p).items()}
    plain=load(f'D02_s{seed}')
    expected={k:v for k,v in r0[seed].items() if not k.startswith('balance_blocks.')}
    assert set(plain)==set(expected) and all(equal(v,plain[k]) for k,v in expected.items())
    for sid,f in (('D04',32),('D05',16)):
        state=load(f'{sid}_s{seed}')
        for k,v in plain.items():assert equal(v,state[k]),(sid,k)
        for side in ('down_h.weight','down_l.weight'):
            k='balance_blocks.aligned.'+side;assert equal(state[k],r0[seed][k][:f])
        k='balance_blocks.aligned.flow_make.weight';ix=list(range(f))+list(range(64,64+f))
        assert equal(state[k],np.take(r0[seed][k],ix,axis=1)),(sid,'flow_cat_block')
    assert ledgers[f'D04_s{seed}']['tensors'].keys()==ledgers[f'D07_s{seed}']['tensors'].keys()
    for key,info in ledgers[f'D04_s{seed}']['tensors'].items():
        other=ledgers[f'D07_s{seed}']['tensors'][key]
        assert all(info[x]==other[x] for x in ('shape','dtype','numel','sha256')),(seed,key)

indices=read(manifest['controlled_S4_indices'])
assert sha(manifest['controlled_S4_indices'])==manifest['controlled_S4_indices_sha256']
assert indices['nested_indices_verified'] and indices['W11_top768_reproduced']
for block in ('0','1'):
    for lo,hi in ((768,896),(896,960),(960,1024)):
        assert set(indices['chosen'][str(lo)][block]) <= set(indices['chosen'][str(hi)][block])
    score=np.array(indices['scores'][block]['score'],dtype=np.float64)
    order=np.lexsort((np.arange(1024),-score))
    for h in (768,896,960,1024):assert np.array_equal(np.sort(order[:h]),indices['chosen'][str(h)][block])

allowed_changes={f'backbone.block4.{b}.mlp.{suffix}' for b in (0,1) for suffix in (
    'fc1.weight','fc1.bias','dwconv.dwconv.weight','dwconv.dwconv.bias','fc2.weight')}
conditional=[]
for name,entry in manifest['runs'].items():
    if entry['stage']!='conditional':continue
    parent=entry['spec_id'].split('_H')[0]+f"_s{entry['seed']}"
    left,right=ledgers[parent]['tensors'],ledgers[name]['tensors']
    assert left.keys()==right.keys()
    changed={k for k in left if (left[k]['sha256'],left[k]['shape'])!=(right[k]['sha256'],right[k]['shape'])}
    assert changed==allowed_changes,(name,changed)
    conditional.append(dict(child=name,parent=parent,changed_tensors=len(changed)))
assert len(conditional)==60

tensor_samples=[]
for child,parent,h in (('C01_s42',None,960),('C02_s44',None,896),
                       ('D01_H960_s42','D01_s42',960),('D07_H896_s43','D07_s43',896),
                       ('D10_H960_s44','D10_s44',960)):
    actual=load(child);seed=manifest['runs'][child]['seed'];base=r0[seed] if parent is None else load(parent)
    assert actual.keys()==base.keys()
    for key,v in base.items():
        expected=v
        if key in allowed_changes:
            block=key.split('.')[2];suffix=key.split('.mlp.')[1]
            axis=1 if suffix=='fc1.weight' else 0
            expected=np.take(v,indices['chosen'][str(h)][block],axis=axis)
        assert equal(actual[key],expected),(child,key)
    tensor_samples.append(dict(child=child,parent='W11_R0' if parent is None else parent,
                               every_tensor_exact=True,changed_S4_only=True))

# Independently apply the algebra rather than calling the folding helper.
for seed in (42,44):
    plain=load(f'D02_s{seed}');native=load(f'D08_s{seed}')
    for i,stage in enumerate((4,3,2,1)):
        prefix=f'linear_c{stage}.proj.';b=plain['linear_fuse._conv.weight'][:,128*i:128*(i+1),0,0].astype('float64')
        for suffix in ('weight','bias'):
            expected=(plain[prefix+suffix].astype('float64')@b.T).astype('float32')
            assert equal(native[prefix+suffix],expected)
    for key,v in native.items():
        if key.startswith(('backbone.','linear_pred.')):assert equal(v,plain[key])
        if key.startswith('sum_norm.'):
            assert equal(v,plain[key.replace('sum_norm.','linear_fuse._batch_norm.')])

# Check width-prefix rules on both a narrowed and a widened factorized head.
for sid,seed,w in (('D01',42,96),('D03',44,160)):
    state=load(f'{sid}_s{seed}');base=r0[seed];n=min(w,128)
    for stage in (1,2,3,4):
        pre=f'linear_c{stage}.proj.'
        assert equal(state[pre+'weight'][:,:n],base[pre+'weight'][:,:n])
        assert equal(state[pre+'bias'][:n],base[pre+'bias'][:n])
    for b in range(4):
        assert equal(state['linear_fuse._conv.weight'][:n,b*w:b*w+n],
                     base['linear_fuse._conv.weight'][:n,b*128:b*128+n])
    assert equal(state['linear_pred.weight'][:,:n],base['linear_pred.weight'][:,:n])
    assert equal(state['linear_pred.bias'],base['linear_pred.bias'])

# A DW predictor's genuinely new kernels use independent named component seeds.
for label,kwargs in (('flow_make.0',dict(in_channels=64,out_channels=64,kernel_size=3,padding=1,groups=64,bias_attr=False)),
                     ('flow_make.1',dict(in_channels=64,out_channels=2,kernel_size=1,bias_attr=False))):
    material=('W12-explicit-initialization-v1|42|alignment.dwflow32.'+label).encode()
    named=int.from_bytes(hashlib.sha256(material).digest()[:4],'little')%2147483647
    paddle.seed(named);layer=paddle.nn.Conv2D(**kwargs)
    for suffix,value in layer.state_dict().items():
        assert equal(arr(value),load('D06_s42')['balance_blocks.aligned.'+label+'.'+suffix])

assert all(x['passed'] for x in manifest['masked_original_FFN_equivalence'])
assert all(x['passed'] for x in manifest['native128_plain128_initial_equivalence'])
receipt=read(ROOT/'provenance/initialization_actual_exec_receipt.json')
assert receipt['exit_code']==0 and receipt['manifest_sha256']==sha(ROOT/'provenance/initialization_manifest.json')
result=dict(at=dt.datetime.now().astimezone().isoformat(),passed=True,issues=[],CPU_only=True,
    GPU_started=False,datasets_read=False,initialization_generation_repeated=False,
    original_or_generated_initializations_modified=False,registered_definition_count=32,
    complete_init_and_ledger_SHA_checked=96,complete_count_formulas_checked=96,
    conditional_ledgers_only_ten_S4_tensors_changed=len(conditional),
    all_three_R0_plain128_exact=True,Flow32_Flow16_high_low_mapping_exact=True,
    Flow32_CoarseFlow32_same_initial_tensors_all_three_seeds=True,
    fixed_ImageNet_nested_indices_verified=True,conditional_capacity_D03_not_selection_eligible=True,
    finite_tensor_sample_files=len(cache),sampled_every_tensor_comparisons=tensor_samples,
    Native128_independent_algebra_seeds=[42,44],width_prefix_checked=[96,160],
    DW_named_seed_reproduction=True,source_review='prepare_initialization_w12.py and model helpers read; no model/GPU execution, only tiny CPU DW constructors',
    manifest_sha256=sha(ROOT/'provenance/initialization_manifest.json'),
    specs_sha256=sha(ROOT/'experiment_specs.json'),
    initialization_script_sha256=sha(ROOT/'prepare_initialization_w12.py'),
    model_source_hashes=manifest['model_source_hashes'],
    limitations='All96 file/ledger hashes and metadata reviewed; numeric tensor sampling is deliberately bounded, not a second full initialization generation. Shared data RNG still requires formal entry witness. Width-specific dropout masks are not asserted identical.')
out=ROOT/'provenance/initialization_independent_review.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(passed=True,sampled_files=len(cache),file=str(out),sha256=sha(out)),ensure_ascii=False))
