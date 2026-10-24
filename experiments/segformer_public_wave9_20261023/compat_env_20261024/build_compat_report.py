import json, csv, os
root = r'D:\论文复现\experiments\segformer_public_wave9_20261023\compat_env_20261024'
with open(os.path.join(root,'results','segformer_a0_aligned_test.json'),encoding='utf-8-sig') as f: a0=json.load(f)
with open(r'D:\论文复现\experiments\segformer_public_wave9_20261023\results\public_comparison.json',encoding='utf-8-sig') as f: old=json.load(f)
rows=[]
# Preserve existing valid public rows and exclude old failed placeholders.
for item in old['public_models']:
    if item['status'].startswith('test_complete'):
        m=item['metrics']; rows.append({
          'model': item['model'], 'requested_name': item['requested_name'], 'status': item['status'],
          'mIoU':m['mIoU'], 'crackIoU':m['crackIoU'], 'precision':m['precision'], 'recall':m['recall'], 'f1':m['f1'], 'boundary_F1':m['boundary_F1'], 'params':m.get('params'), 'ms_per_image':m.get('ms_per_image'), 'test_used':True, 'source':'wave9_existing'
    })
rows.insert(0, {'model':'SegFormer-B0+Aligned-A0','requested_name':'current_model_candidate','status':'test_complete_validation_frozen','mIoU':a0['mIoU'],'crackIoU':a0['crackIoU'],'precision':a0['precision'],'recall':a0['recall'],'f1':a0['f1'],'boundary_F1':a0['boundary_F1'],'params':3470434,'ms_per_image':a0['ms_per_image'],'test_used':True,'source':'compat_env_20261024'})
for model, req, status, p, flops in [
 ('PSPNet-ResNet50','PSPNet-ResNet50','gpu_smoke_failed_compat_env',67902052,162122215184),
 ('DeepLabV3P-ResNet50','DeepLabV3P-ResNet50','gpu_smoke_failed_compat_env',26789874,69671106848),
 ('Fast-SCNN','Fast-SCNN','gpu_smoke_failed_compat_env',1441332,629468624)]:
    rows.append({'model':model,'requested_name':req,'status':status,'mIoU':None,'crackIoU':None,'precision':None,'recall':None,'f1':None,'boundary_F1':None,'params':p,'ms_per_image':None,'test_used':False,'source':'compat_env_20261024','flops_smoke':flops})
out={'generated_at':'2026-10-24T11:10:00+08:00','protocol':{'dataset':'Crack500','data_root':r'C:\Users\user\algolab_data','test_images':1124,'current_model':'SegFormer-B0+Aligned-A0','compat_python':r'D:\论文复现\envs\paddle32\python.exe','compat_paddle':'3.2.2','gpu':'RTX5060','source':'D:\论文复现\external\PaddleSeg-2.10.0','test_policy':'test only after validation-frozen checkpoint; failed GPU smoke has no test metrics','boundary':'3x3 inner boundary, zero outside image, exact Euclidean <=2px'},'rows':rows,'compatibility':{'pspnet_gpu_exit':1,'deeplabv3p_gpu_exit':1,'fastscnn_gpu_exit':1,'cpu_smoke_all_exit':0,'failure':'CUDNN_STATUS_INTERNAL_ERROR or CUDNN_STATUS_EXECUTION_FAILED during backward in independent Paddle 3.2.2 GPU environment; no checkpoint created','cpu_estimate':'Fast-SCNN ~1.4h, DeepLabV3P ~30h, PSPNet ~50h for 18k at measured CPU smoke rates; not started as a substitute for RTX5060 protocol'}}
with open(os.path.join(root,'results','public_comparison_20261024.json'),'w',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2)
fields=['model','status','mIoU','crackIoU','precision','recall','f1','boundary_F1','params','ms_per_image','test_used','source']
with open(os.path.join(root,'results','public_comparison_20261024.csv'),'w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([{k:r.get(k) for k in fields} for r in rows])
md=[]
md.append('# Crack500 六模型与当前模型对照记录（2026-10-24）\n')
md.append('本记录基于 RTX5060 本地 Crack500。当前模型使用已完成验证集选模的 SegFormer-B0 + Aligned（A0）checkpoint；损失候选的严格三种子门槛尚未通过，因此未把 Dice/Lovasz 候选写成最终模型。\n')
md.append('| 模型 | 状态 | mIoU | 裂缝 IoU | Precision | Recall | F1 | 边界 F1 | 参数量 | 单图耗时 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for r in rows:
    def fmt(x): return '—' if x is None else (f'{x:.4f}' if isinstance(x,float) else str(x))
    p='—' if r.get('params') is None else f"{r['params']/1e6:.3f}M"
    ms=fmt(r.get('ms_per_image'))
    md.append(f"| {r['model']} | {r['status']} | {fmt(r.get('mIoU'))} | {fmt(r.get('crackIoU'))} | {fmt(r.get('precision'))} | {fmt(r.get('recall'))} | {fmt(r.get('f1'))} | {fmt(r.get('boundary_F1'))} | {p} | {ms} |")
md.extend(['','## 独立兼容环境证据','',f"- Python/Paddle：`{r'D:\论文复现\envs\paddle32\python.exe'}` / Paddle 3.2.2 GPU。",'- PSPNet-ResNet50、DeepLabV3P-ResNet50、Fast-SCNN 的 GPU smoke 均真实 exit=1，失败点为反向传播 cuDNN；对应日志和 exit JSON 均保存在 `compat_env_20261024/logs` 与 `compat_env_20261024/results`。','- 同环境 CPU 3 步 smoke 三个模型均 exit=0，只能证明模型和数据管线可运行，不能作为 RTX5060 正式训练结果。','- 未生成这三个模型的 checkpoint，因此没有访问 test，也没有填入伪造的 test 指标。','', '## A0 当前模型 test 结果','',f"- 1124 张 test 图，mIoU={a0['mIoU']:.6f}，裂缝 IoU={a0['crackIoU']:.6f}，Precision={a0['precision']:.6f}，Recall={a0['recall']:.6f}，F1={a0['f1']:.6f}，边界 F1={a0['boundary_F1']:.6f}。",f"- checkpoint SHA256：`{a0['checkpoint_sha256']}`。",'', '## 后续可执行项','','要补齐 PSPNet、DeepLabV3P、Fast-SCNN 的可比表格，需要在具有可用 Paddle GPU/cuDNN 后端的 Linux/服务器或另一台兼容 GPU 上按同一 18,000 iter、同一 train/val/test 清单训练、验证冻结后统一测试。当前 Windows RTX5060 的独立 Paddle 3.2.2 环境已证明仍无法完成这三种模型的 GPU 反向；CPU 训练预计从约 1.4 小时到约 50 小时，且会改变硬件协议，因此没有启动。'])
with open(r'D:\论文复现\实验结果汇总_20261024\六模型对照表_最新状态.md','w',encoding='utf-8') as f: f.write('\n'.join(md))
print(json.dumps({'json':os.path.join(root,'results','public_comparison_20261024.json'),'csv':os.path.join(root,'results','public_comparison_20261024.csv'),'report':r'D:\论文复现\实验结果汇总_20261024\六模型对照表_最新状态.md'},ensure_ascii=False))
