import csv, json, os, shutil, zipfile, hashlib
from pathlib import Path
wave=Path(r'D:\论文复现\experiments\segformer_loss_wave8_20261019')
rec=wave/'dice_compat_recovery_20261020'/'results'/'consolidated_comparison.json'
seeds=wave/'matched_seeds_20261022'/'results'/'three_seed_summary.json'
pub=Path(r'D:\论文复现\experiments\segformer_public_wave9_20261023')
root=Path(r'D:\论文复现')
outdir=pub/'results'; outdir.mkdir(parents=True,exist_ok=True)
con=json.loads(rec.read_text(encoding='utf-8-sig')); s=json.loads(seeds.read_text(encoding='utf-8-sig'))
rows=[]
for p in ['segformer_b0_test','ocrnet_hrnetw18_test','ppliteseg_stdc1_test']:
    d=json.loads((outdir/(p+'.json')).read_text(encoding='utf-8-sig')); rows.append(d)
params={'SegFormer-B0':3715170,'OCRNet-HRNet-W18':12109902,'PP-LiteSeg-STDC1':8150485}
for d in rows: d['params']=params[d['model']]; d['status']='test_complete'
public=[
 {'model':'SegFormer-B0','requested_name':'SegFormer-B0','status':'test_complete','test_used':True,'metrics':rows[0],'checkpoint_sha256':rows[0]['checkpoint_sha256']},
 {'model':'OCRNet-HRNet-W18','requested_name':'OCRNet-HRNet-W18','status':'test_complete','test_used':True,'metrics':rows[1],'checkpoint_sha256':rows[1]['checkpoint_sha256']},
 {'model':'PP-LiteSeg-STDC1','requested_name':'PP-LiteSeg-STDC1','status':'test_complete_after_val_gate_failed','test_used':True,'metrics':rows[2],'checkpoint_sha256':rows[2]['checkpoint_sha256'],'note':'Test was run only after the validation checkpoint was frozen; validation gate had failed, so this row is an exploratory public comparison.'},
 {'model':'PSPNet-R18','requested_name':'PSPNet-R50','status':'gpu_smoke_failed','test_used':False,'smoke_exit_code':1,'evidence':str(pub/'logs'/'smoke_PSPNet_R18.log'),'reason':'CUDNN_STATUS_EXECUTION_FAILED during Conv2dGradNode; no valid checkpoint.'},
 {'model':'DeepLabV3P-ResNet50','requested_name':'DeepLabV3P-ResNet50','status':'gpu_smoke_failed_historical','test_used':False,'smoke_exit_code':1,'evidence':r'D:\论文复现\experiments\retrain_wave4_20261007\logs\smoke_deeplabv3p_r50_current.log','reason':'CUDNN_STATUS_INTERNAL_ERROR/EXECUTION_FAILED during GPU backward; no valid checkpoint.'},
 {'model':'Fast-SCNN','requested_name':'Fast-SCNN','status':'gpu_smoke_failed','test_used':False,'smoke_exit_code':1,'evidence':str(pub/'logs'/'smoke_Fast_SCNN.log'),'reason':'CUDNN_STATUS_EXECUTION_FAILED during Conv2dGradNode; no valid checkpoint.'}
]
summary={'generated_at':'2026-10-23T22:00:00+08:00','protocol':{'dataset':'Crack500','data_root':r'C:\Users\user\algolab_data','hardware':'RTX5060','python':r'C:\venv_bisenet\Scripts\python.exe','test_images':1124,'test_policy':'public models evaluated after frozen validation checkpoints; loss candidates remain test_unused because strict three-seed gate was not passed','boundary':'3x3 inner boundary, zero outside image, exact Euclidean distance <=2px','source_snapshot':str(wave/'src'/'PaddleSeg-2.10.0')},'public_models':public,'loss_phase':{'mother':'SegFormer-B0 width128 + Aligned','params':3470434,'c0_seed42':con['parent'],'rows':con['rows'],'three_seed_summary':s['groups'],'consolidated_source':str(rec)},'claims':{'strict_three_seed_gate_passed':False,'sota_claim':False,'reason':'L01 and L02 improve crack IoU/F1/mIoU and recall on average but both fail the preregistered precision mean gate; no loss candidate is admitted as final gated model.'}}
(outdir/'public_comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
# public csv
with (outdir/'public_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=['model','requested_name','status','test_used','mIoU','crackIoU','precision','recall','f1','boundary_F1','params','ms_per_image','checkpoint_sha256','reason']); w.writeheader()
    for d in public:
      m=d.get('metrics',{}); w.writerow({'model':d['model'],'requested_name':d.get('requested_name'),'status':d['status'],'test_used':d.get('test_used'), 'mIoU':m.get('mIoU'),'crackIoU':m.get('crackIoU'),'precision':m.get('precision'),'recall':m.get('recall'),'f1':m.get('f1'),'boundary_F1':m.get('boundary_F1'),'params':m.get('params'),'ms_per_image':m.get('ms_per_image'),'checkpoint_sha256':d.get('checkpoint_sha256'),'reason':d.get('reason','')})
# Main loss table csv copy
shutil.copy2(wave/'dice_compat_recovery_20261020'/'results'/'loss_ablation_consolidated.csv',outdir/'loss_ablation_consolidated.csv')
# report markdown
p=root/'SegFormer损失函数消融总报告_20261023.md'
md=[]
md.append('# SegFormer-B0 + Aligned Crack500 损失消融与公开模型对比总报告')
md.append('\n生成时间：2026-10-23。')
md.append('\n## 结论')
md.append('\n本轮完成了20个预登记损失方案、Dice兼容恢复、L01/L02的seed 43/44复核，并完成冻结公开 checkpoint 的统一 Crack500 test 评估。严格三种子门槛没有通过：L01 与 L02 的区域指标和 Recall 平均提高，但 Precision 平均分别下降 0.00772 和 0.01136，超过预登记的最多 0.005 下降限制。因此当前没有可以按预注册门槛宣称为最终改进模型的候选，也不能声称 SOTA。')
md.append('\n## 实验协议')
md.append('\nRTX5060、FP32、PaddleSeg 2.10.0、本地 Python 环境、SegFormer-B0 width128 + Aligned、batch=4、AdamW、学习率 6e-5、PolynomialDecay end_lr=0、18,000 iter、同一 Crack500 train/val 划分。损失筛选只使用 train/val；L01/L02 的 test_used 均为 false。公开模型在冻结验证 checkpoint 后才访问 1,124 张原始 test 图。')
md.append('\n## 损失消融结果（seed42）')
md.append('\n|方案|状态|Precision|Recall|F1|裂缝IoU|mIoU|边界F1|参数|门槛|\n|---|---|---:|---:|---:|---:|---:|---:|---:|---|')
for x in con['rows']:
  md.append(f"|{x['label']}|{x['status']}|{x['precision']:.4f}|{x['recall']:.4f}|{x['f1']:.4f}|{x['crackIoU']:.4f}|{x['mIoU']:.4f}|{x['boundary_f1']:.4f}|{x['params']/1e6:.3f}M|{'通过' if x.get('main_gate') else '未通过'}|")
md.append('\n说明：L01–L04、L18 原始主队列因官方 Dice 对共享 labels 原地修改导致反向版本冲突；独立兼容副本只对 labels 做 clone，公式与系数不变，合并表同时保留原始失败状态和 compat_clone 证据。')
md.append('\n## L01/L02 三种子复核')
md.append('\n|方案|P均值±样本std|R均值±std|F1均值±std|裂缝IoU均值±std|mIoU均值±std|BF1均值±std|严格均值门槛|')
for k,v in s['groups'].items():
  md.append(f"|{k}|{v['mean']['precision']:.4f}±{v['sample_std']['precision']:.4f}|{v['mean']['recall']:.4f}±{v['sample_std']['recall']:.4f}|{v['mean']['f1']:.4f}±{v['sample_std']['f1']:.4f}|{v['mean']['crackIoU']:.4f}±{v['sample_std']['crackIoU']:.4f}|{v['mean']['mIoU']:.4f}±{v['sample_std']['mIoU']:.4f}|{v['mean']['boundary_f1']:.4f}±{v['sample_std']['boundary_f1']:.4f}|未通过（Precision均值下降超过0.005）|")
md.append('\nL01 相对 seed42 的三种子均值变化：Precision -0.007725、Recall +0.013318、F1 +0.002841、裂缝IoU +0.004042、mIoU +0.002039、边界F1 +0.001339。L02：Precision -0.011364、Recall +0.016758、F1 +0.002728、裂缝IoU +0.003881、mIoU +0.001901、边界F1 +0.000073。')
md.append('\n## Crack500 原始公开模型统一 test 对比')
md.append('\n|模型|状态|mIoU|裂缝IoU|Precision|Recall|F1|边界F1|参数量|单图耗时|')
for d in public:
  m=d.get('metrics',{})
  if m:
    md.append(f"|{d['model']}|{d['status']}|{m['mIoU']:.4f}|{m['crackIoU']:.4f}|{m['precision']:.4f}|{m['recall']:.4f}|{m['f1']:.4f}|{m['boundary_F1']:.4f}|{m['params']/1e6:.3f}M|{m['ms_per_image']:.2f} ms|")
  else:
    md.append(f"|{d['model']}|{d['status']}|—|—|—|—|—|—|—|—|")
md.append('\nSegFormer、OCRNet、PP-LiteSeg 均在同一 test evaluator 上完成。PP-LiteSeg 的验证门槛此前未通过，但在验证 checkpoint 冻结后补做了 test，并在表中明确标注。PSPNet-R50 没有可用 checkpoint；本地可运行的官方 PSPNet-R18 smoke 在 RTX5060/Paddle3.4 反向阶段以真实 exit=1 失败。DeepLabV3P-R50 和 Fast-SCNN 也保留真实 GPU smoke 失败证据，没有伪造 test 数字。')
md.append('\n## 论文结论与后续工作')
md.append('\n1. 将 SegFormer-B0 + Aligned 作为稳定母体；L01/L02 作为“提升 Recall/裂缝IoU但牺牲 Precision 的探索性负结果”，不写成最终模型。\n2. 公开 test 中 OCRNet 的 mIoU 和 F1 高于 SegFormer，PP-LiteSeg 更轻但精度更低；当前公开对比范围内没有证据支持 SOTA。\n3. 下一阶段可在冻结的 A0/L01 中择一进入自建数据和机器人闭环，但应先根据任务优先级决定是优先降低假阳性，还是优先提高细裂缝召回。\n4. 若要补齐六模型表，应在独立兼容环境重跑 PSPNet、DeepLabV3P、Fast-SCNN，保留同一数据清单和 test evaluator；当前 RTX5060 后端失败属于工程兼容性限制。')
md.append('\n## 复现证据')
md.append(f"\n- 损失合并结果：`{rec}`\n- 三种子结果：`{seeds}`\n- 公开模型结果：`{pub/'results'/'public_comparison.json'}`\n- 公开逐图 CSV：`{pub/'results'}`\n- 公开配置、日志、真实 exit：`{pub}`")
p.write_text('\n'.join(md)+'\n',encoding='utf-8')
# update root progress docs with a concise new section (append, preserving history)
for name in ['SegFormer损失函数消融进展_20261019.md','模型消融文档.md']:
  q=root/name
  if q.exists():
    text=q.read_text(encoding='utf-8-sig')
  else: text=''
  marker='## 2026-10-23 Wave8最终审计与公开模型对比'
  if marker not in text:
    text += f"\n\n{marker}\n\n详见[SegFormer损失函数消融总报告_20261023.md](D:/论文复现/SegFormer损失函数消融总报告_20261023.md)。20种损失方案已完成真实训练/兼容恢复审计；L01/L02 seed43/44复核后均因Precision均值门槛失败，当前不声称SOTA。SegFormer、OCRNet、PP-LiteSeg已完成统一test，PSPNet、DeepLabV3P、Fast-SCNN保留真实GPU smoke失败证据。"
    q.write_text(text,encoding='utf-8')
# compact delivery index
idx=pub/'DELIVERY_INDEX.md'
idx.write_text(f'''# Wave8/Wave9交付索引\n\n- 总报告：{p}\n- 损失合并表：{wave/'dice_compat_recovery_20261020'/'results'/'loss_ablation_consolidated.csv'}\n- 三种子摘要：{seeds}\n- 公开模型JSON/CSV：{outdir/'public_comparison.json'} / {outdir/'public_comparison.csv'}\n- 公开配置与真实日志：{pub}\n- 正式 checkpoint 只保留在原实验目录，未打入报告包。\n''',encoding='utf-8')
# zip without raw data, source trees, and checkpoints
zip_path=root/'SegFormer_wave8_wave9_report_delivery_20261023.zip'
include=[p, root/'SegFormer损失函数消融进展_20261019.md', root/'模型消融文档.md', idx, rec, wave/'dice_compat_recovery_20261020'/'results'/'loss_ablation_consolidated.csv', seeds, outdir/'public_comparison.json',outdir/'public_comparison.csv']
for f in [*pub.glob('configs/*.yml'),*pub.glob('logs/*.log'),*pub.glob('results/*_exit.json'),*pub.glob('results/*.json')]: include.append(f)
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
  seen=set()
  for f in include:
    f=Path(f)
    if not f.exists() or f in seen: continue
    seen.add(f); z.write(f,arcname=f'{f.parent.name}/{f.name}')
print(json.dumps({'report':str(p),'public_json':str(outdir/'public_comparison.json'),'zip':str(zip_path),'zip_bytes':zip_path.stat().st_size},ensure_ascii=False,indent=2))
