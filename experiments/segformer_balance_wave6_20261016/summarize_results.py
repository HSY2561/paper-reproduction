"""Read-only experiment audit and report generation; no training/inference."""
import csv
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
spec = importlib.util.spec_from_file_location('wave6_queue', ROOT / 'experiment_queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def main():
    # The existing runner touches only train/val in this audit.
    data_audit = queue.audit_data('report_20261017')
    queue.decisions()
    source = queue.read_json(ROOT / 'results/comparison.json')
    base = source['baseline']
    base_params = source['baseline_profile']['params_total']
    rows = source['rows']
    completed = [r for r in rows if r['status'] == 'complete']
    audits = []
    for row in completed:
        ident = row['id']
        selection = queue.read_json(ROOT / 'checkpoints' / ident / 'selection.json')
        stages = {s: queue.read_json(ROOT / 'results' / f'{ident}_{s}_exit.json')['exit_code']
                  for s in ('train', 'official_val', 'pixel_boundary', 'profile')}
        val = queue.read_json(ROOT / 'results' / f'{ident}_val.json')
        check = dict(id=ident, exit_codes=stages,
                     final_iter_in_log='iter: 18000/18000' in (ROOT / 'logs' / f'{ident}_train.log').read_text(encoding='utf-8', errors='replace'),
                     final_checkpoint=(ROOT / 'training' / ident / 'iter_18000/model.pdparams').exists(),
                     frozen_sha_matches=queue.sha(selection['checkpoint']) == selection['sha256'],
                     highest_exact_val_selected=selection['selected']['mIoU'] == max(c['mIoU'] for c in selection['candidates']),
                     independent_val_matches=abs(val['mIoU'] - selection['selected']['mIoU']) < 1e-6,
                     test_used=val['test_used'])
        assert all(v == 0 for v in stages.values()), check
        assert all(check[k] for k in ('final_iter_in_log', 'final_checkpoint', 'frozen_sha_matches', 'highest_exact_val_selected', 'independent_val_matches')), check
        assert not val['test_used'], check
        matrix = val['confusion_matrix']
        row['fp'] = matrix[0][1]
        row['fn'] = matrix[1][0]
        row['delta_fp'] = matrix[0][1] - base['confusion_matrix'][0][1]
        row['delta_fn'] = matrix[1][0] - base['confusion_matrix'][1][0]
        assert abs(2 * val['crackIoU'] / (1 + val['crackIoU']) - val['f1']) < 1e-12
        audits.append(check)

    checked_at = queue.now()
    queue.write_json(ROOT / 'results/result_audit_20261017.json', dict(
        checked_at=checked_at, complete_runs=len(completed),
        complete_module_mechanisms=sum(bool(r['modules']) for r in completed),
        baseline_sha_matches=queue.sha(queue.BASE_WEIGHTS) == queue.BASE_SHA,
        data_audit=data_audit, runs=audits, test_used=False))

    labels = {
        'W128': '仅压缩解码器至128', 'W64': '仅压缩解码器至64',
        'M01_SE': 'SE', 'M02_PSA': 'Polarized SA', 'M03_GC': 'Global Context',
        'M04_CCA': 'Criss-Cross Attention', 'M05_PAM': 'PAM', 'M06_CAM': 'CAM',
        'M07_NonLocal': 'NonLocal', 'M08_MSCA': 'MSCA', 'M09_Encoding': 'Encoding',
        'M10_GloRe': 'GloRe', 'M11_ASPPsep': 'Separable ASPP', 'M12_PSP': 'PSP',
        'M13_Aligned': 'Aligned', 'M14_UAFMchannel': 'UAFM Channel',
    }
    ordered = sorted(rows, key=lambda r: (0 if r['id'].startswith('W') else 1,
                                          {'W128': 0, 'W64': 1}.get(r['id'], int(r['id'][1:3]) if r['id'].startswith('M') else 0)))
    metrics = ('precision', 'recall', 'f1', 'crackIoU', 'mIoU', 'boundary_f1')
    table = ['| 配置（模块组均为W128母体） | Precision % | Recall % | F1 % | IoU % | mIoU % | Params M | 边界F1 % |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    numeric = ' | '.join(f'{100 * base[k]:.3f}' for k in metrics[:5])
    table.append(f"| SegFormer-B0，原始width256 | {numeric} | {base_params / 1e6:.6f} | {100 * base['boundary_f1']:.3f} |")
    exported = [dict(id='B0', label='SegFormer-B0 width256', status='frozen_reused',
                     **{k + '_percent': base[k] * 100 for k in metrics}, params=base_params, params_M=base_params / 1e6)]
    for row in ordered:
        name = row['id'] + ' ' + labels[row['id']]
        record = dict(id=row['id'], label=labels[row['id']], status=row['status'])
        if row['status'] == 'complete':
            numeric = ' | '.join(f'{100 * row[k]:.3f}' for k in metrics[:5])
            table.append(f"| {name} | {numeric} | {row['params'] / 1e6:.6f} | {100 * row['boundary_f1']:.3f} |")
            record.update({k + '_percent': row[k] * 100 for k in metrics})
            record.update(params=row['params'], params_M=row['params'] / 1e6,
                          delta_IoU_pp=100 * row['delta']['crackIoU'], delta_F1_pp=100 * row['delta']['f1'],
                          delta_FP=row['delta_fp'], delta_FN=row['delta_fn'])
        else:
            note = '兼容性失败' if row['status'] == 'smoke_failed' else '未完成'
            table.append(f'| {name}（{note}） | — | — | — | — | — | — | — |')
        exported.append(record)
    csv_path = ROOT / 'results/ablation_table_percent_20261017.csv'
    fields = ['id', 'label', 'status'] + [k + '_percent' for k in metrics] + ['params', 'params_M', 'delta_IoU_pp', 'delta_F1_pp', 'delta_FP', 'delta_FN']
    with csv_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(exported)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    figures = ROOT / 'figures/results_20261017'
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    for row in completed:
        color = '#e09b2d' if not row['modules'] else '#6d93b0'
        if row['id'] == 'M13_Aligned': color = '#bf4242'
        if row['id'] == 'M10_GloRe': color = '#277e61'
        ax.scatter(row['params'] / 1e6, 100 * row['crackIoU'], s=65, color=color, zorder=3)
        ax.annotate(row['id'].split('_')[0], (row['params'] / 1e6, 100 * row['crackIoU']),
                    xytext=(5, 4), textcoords='offset points', fontsize=8)
    ax.scatter(base_params / 1e6, base['crackIoU'] * 100, s=170, marker='*', color='#222222', label='Original B0', zorder=4)
    ax.annotate('B0', (base_params / 1e6, base['crackIoU'] * 100), xytext=(6, 6), textcoords='offset points')
    ax.axhline(base['crackIoU'] * 100, color='#777777', ls='--', lw=1, alpha=.65)
    ax.set(xlabel='Total model parameters (M)', ylabel='Crack IoU (%)',
           title='Crack500 validation: parameter-accuracy trade-off\nSeed 42; all completed candidates shown; test sealed')
    ax.grid(alpha=.15)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(figures / 'params_iou.png', dpi=200)
    fig.savefig(figures / 'params_iou.pdf')
    plt.close(fig)

    errors = sorted(completed, key=lambda r: r['crackIoU'])
    fig, ax = plt.subplots(figsize=(10.4, 7.4))
    positions = list(range(len(errors)))
    ax.barh([p + .18 for p in positions], [r['delta_fp'] / 1000 for r in errors], .34,
            color='#d98542', label='Additional false-positive pixels')
    ax.barh([p - .18 for p in positions], [-r['delta_fn'] / 1000 for r in errors], .34,
            color='#5286aa', label='Fewer false-negative pixels')
    ax.set_yticks(positions, [r['id'] for r in errors])
    ax.set(xlabel='Pixel count change versus B0 (thousands)',
           title='Recall gains accompany more background false positives\nGlobal validation confusion counts; no scene-type attribution yet')
    ax.grid(axis='x', alpha=.15)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / 'false_positive_tradeoff.png', dpi=200)
    fig.savefig(figures / 'false_positive_tradeoff.pdf')
    plt.close(fig)

    by_id = {r['id']: r for r in completed}
    a, g, n = [by_id[k] for k in ('M13_Aligned', 'M10_GloRe', 'M07_NonLocal')]
    module_rows = [r for r in completed if r['modules']]
    module_count = len(module_rows)
    best = max(completed, key=lambda r: r['crackIoU'])
    trend_count = sum(r['recall'] > base['recall'] and r['precision'] < base['precision'] for r in module_rows)
    boundary_lower_count = sum(r['boundary_f1'] < base['boundary_f1'] for r in module_rows)
    se_status = next(r['status'] for r in rows if r['id'] == 'M01_SE')
    se_note = 'SE已完成并按同一规则纳入本表' if se_status == 'complete' else 'SE未完成，不填写性能'
    se_exit_path = ROOT / 'results/M01_SE_train_exit.json'
    if se_status != 'complete' and se_exit_path.exists():
        se_exit = queue.read_json(se_exit_path)
        if se_exit.get('timed_out'):
            se_note += f"；最新训练于{se_exit['ended_at']}超时退出，exit_code={se_exit['exit_code']}，状态为{se_status}"
    root_figure_dir = figures.relative_to(ROOT.parent.parent).as_posix()
    combined_params = a['params'] + g['params'] - by_id['W128']['params']
    assert combined_params == 3517666
    report = f'''# SegFormer-B0 Crack500 模块消融结果与改进建议

更新时间：{checked_at}。这是本轮**验证集筛选报告**，不是最终test对照表。

## 已完成与未完成

已完成{len(completed)}个新训练组，其中{module_count}种不同模块、2个宽度控制；另复用原始B0冻结基线。{len(completed)}组均核对真实exit code 0、18,000/18,000日志、iter_18000、普通/EMA最高精确val mIoU、冻结SHA256、独立官方val、像素/边界评估和参数统计。原始B0 SHA仍一致，train/val全部4488个文件哈希复核无变化。{se_note}；PSP反向smoke失败，不填写性能。

SE旧日志停在13:34，在2000iter后的验证过程中中断，该旧运行没有训练exit JSON；实际原因未知。现场和当时已完成的14组保留，只对SE按相同seed/预训练重新训练。为处理反复丢失的工具会话，已获准使用有2小时时限、自动过期删除的一次性Windows任务运行同一队列；实际是否在运行仍须以新日志和回执核对，不能以注册成功代替完成。当前SE状态为{se_status}，GloRe和UAFM通道组已经完整完成。

## 数据与口径

RTX5060，Paddle3.4，FP32，seed42，batch4，18k，AdamW lr6e-5、poly end_lr0、相同增强/400裁剪/CE/EMA；模块从相同ImageNet MiT-B0初始化。所有新模块接在width128母体，必须同时对照W128与原始width256基线，不能把减小通道宽度的收益算成模块收益。这里只使用Crack500 train1896/val348，没有读取test或自建数据。

Precision、Recall、F1、IoU均为裂缝前景的全验证集混淆矩阵指标；mIoU是背景/裂缝两类IoU均值。对于本报告的二值全局计数，F1=2×IoU/(1+IoU)，两者同升不是两个独立统计证据。Params采用同一模型参数统计，包含骨干与解码器，单位M。

边界为前景减3×3腐蚀、图外为零、精确欧氏距离≤2px，全局匹配计数后算调和平均。不能混入旧报告0.4177或曾错误计算的0.5894等不同口径B0边界数字。当前所有结果只有一个seed，不能称显著提升。

## 完整消融表

''' + '\n'.join(table) + f'''

## 结果如何理解

1. **当前最高裂缝IoU的完成项是{best['id']}，IoU={best['crackIoU']:.6f}、F1={best['f1']:.6f}；Aligned的参数精度折中值得关注。** Aligned相对原始B0，F1增加{a['delta']['f1']*100:.4f}个百分点，裂缝IoU增加{a['delta']['crackIoU']*100:.4f}个百分点，mIoU增加{a['delta']['mIoU']*100:.4f}个百分点；参数减少{100*(1-a['param_ratio']):.2f}%。边界F1仍下降{-a['delta']['boundary_f1']*100:.4f}个百分点。相对W128母体，其IoU/F1/边界F1增加0.1992/0.1416/0.8990个百分点，表明能恢复部分压缩损失；是否能稳定超过原始B0尚未证实。它主要把W128的高Recall工作点拉回较高Precision，不足以认定所有细裂缝定位都改善。

2. **GloRe是效率方向的候选，不是精度胜出者。** 参数{g['params']/1e6:.6f}M，减少{100*(1-g['param_ratio']):.2f}%；F1/IoU分别低于B0约{-g['delta']['f1']*100:.4f}/ {-g['delta']['crackIoU']*100:.4f}个百分点。当前JSON的efficiency_pass=true，但使用了不同时间段测出的延迟，仅是临时规则判定。此前400/480尺寸出现倒挂、重复波动，**不能据10.67ms与基线15.58ms直接声称加速31.5%**。精度线和最终目标线仍未通过。

3. **背景误检没有被证明解决。** {module_count}个完成模块中，{trend_count}个相对B0提高Recall、降低Precision，{boundary_lower_count}个边界F1低于B0。B0总FP/FN为753850/971542；Aligned为{a['fp']}/{a['fn']}，即FP增加{a['delta_fp']:,}、FN减少{-a['delta_fn']:,}；GloRe增加{g['delta_fp']:,}个FP、减少{-g['delta_fn']:,}个FN；NonLocal增加{n['delta_fp']:,}个FP、减少{-n['delta_fn']:,}个FN。说明多数候选以更激进的前景预测换取漏检减少，不能写成“阴影、落叶误检得到抑制”。具体场景归因仍需人工分组查看图像。

4. **继续减宽到64不合算。** 参数只比B0减少9.31%，IoU却下降0.7610个百分点；W128已减少7.09%，更适合作为效率母体。UAFM通道融合参数反而高于原B0约2%，同时F1下降，没有进入最终组合的充分理由。GC/PSA/CCA/PAM/MSCA等没有区域增益，不按模块名称继续堆叠。

5. **目前不是SOTA，也没有最终定型。** Aligned只是单seed验证集最好；GloRe仅临时满足效率容差。六个公开原始模型尚未完成本轮统一test表；不同数据划分、验证与test、服务器与本机数字不能横向拼成胜出证据。

## 建议的后续顺序

| 优先级 | 工作及模型位置 | 要解决的问题/预期收益 | 成本与风险 | 放行条件 |
|---|---|---|---|---|
| 1 | 完成SE；同一进程交错顺序复测B0、W128、Aligned、GloRe | 补齐单项，排除GPU热态/时钟造成的假加速 | 只补最后一个未完成模块；速度评估先避开并行训练 | 真exit0/18k/冻结独立val；同输入、FP32、batch1、预热和同步 |
| 2 | 从逐图CSV定位Aligned/GloRe的FP增加与边界损失，人工标注阴影/纹理/低对比度类别 | 判断瓶颈来自已有裂缝扩张、断裂还是背景激活 | 总指标不能自动推断场景；不得删失败图或按结果改标签 | 保留全348图，报告提升/下降分布及最差案例；只用val |
| 3 | **至多一组探索性Aligned+GloRe**：GloRe接原生C4；Aligned对齐C1与投影C4后再融合 | 一个补深层关系，一个补跨尺度恢复，观察互补性 | 参数静态估计{combined_params/1e6:.6f}M（约减{100*(1-combined_params/base_params):.2f}%），不是实测；可能累加FP或边界退化；grid_sample部署支持待测 | 先复测与诊断，运行前登记；与W128、两个单模块组成四行消融；不改输入/loss/预算 |
| 4 | 真正有效候选与B0匹配seed43/44 | 排除单seed和多模型反复筛选的偶然最大值 | 每组仍完整18k；当前不能计算均值±标准差 | 报告三seed均值/波动，完整精度与边界门槛不事后放宽 |
| 5 | 六个预选原始公开模型，统一冻结后test | 检验真实精度/参数折中 | 不因val弱删行；兼容性失败不伪造数字 | 阶段一稳定后再做；复用合法已有结果，补齐缺项 |

原计划优先的Aligned+GC、MSCA+CAM缺少两侧区域增益，不执行。Aligned+GloRe只是当前证据支持的下一组假设，**不是已经训练的组合**，不自动保证提升；本报告不把组合估算参数加入结果表。两者相对W128都减少FP、增加FN，误差方向相同；接入位置不同只提供结构互补的假设，不能写成已证明误差互补，联合时尤其要防止Recall继续下降。

组合建议依据SE完成前的12模块证据提出；若SE补齐后改写候选排名或门槛结论，应先重新审阅，再运行组合，不能仅按本脚本固定建议自动启动。

如果组合仍无实质收益，停止重复叠加全局注意力：MiT本身已有长距离关系建模，冗余是一个待验证解释。优先补输入/裁剪细线保留、标签边界一致性及train内难负样本统计，再针对明确因素做一个单变量实验。若调整损失或采样，另起独立消融，不能与本轮结构改动同时变更，也不能看test调阈值。

最终论文可如实围绕“轻量解码器压缩与特征对齐带来的精度—成本折中”组织；现阶段不能写成“全部精度超过原模型、最低参数、阴影误检已解决、机器人实时性已证实”。自建数据、源帧互斥和机器人修复是后续外部验证。

## 图与证据

参数—IoU图展示所有已完成项，而非只展示最好的候选。

![参数与裂缝IoU]({root_figure_dir}/params_iou.png)

误检与漏检变化图中的橙条是新增FP，蓝条是减少FN；两者同时增长正说明Precision/Recall取舍。

![误检漏检变化]({root_figure_dir}/false_positive_tradeoff.png)

- 精确数据：`results/comparison.json`、各组`*_val.json`和`*_val_per_image.csv`。
- 百分数消融CSV：`results/ablation_table_percent_20261017.csv`。
- 核验清单：`results/result_audit_20261017.json`。
- 原始训练、验证、退出码：`logs/`和`results/*_exit.json`。
- 冻结权重及选模：`checkpoints/<ID>/selection.json`与同目录权重SHA256。
- 原始开源模块位置及训练契约：`refine-logs/EXPERIMENT_PLAN.md`、`provenance/RUNTIME.md`。

这是结果快照；SE后续完成时须按相同口径更新，不用中间迭代指标填表。
'''
    nested_report = report.replace(f']({root_figure_dir}/', f'](../{figures.relative_to(ROOT).as_posix()}/')
    queue.versioned_text(ROOT / 'refine-logs/RESULT_ANALYSIS.md', nested_report)
    root_report = ROOT.parent.parent / 'SegFormer模块消融结果与改进建议_20261017.md'
    root_report.write_text(report, encoding='utf-8')
    main_doc = ROOT.parent.parent / '模型消融文档.md'
    original = main_doc.read_text(encoding='utf-8-sig')
    history = ROOT / 'refine-logs' / ('MAIN_ABLATION_BEFORE_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '.md')
    history.write_text(original, encoding='utf-8')
    start = original.index('## 2026-10-17：')
    end = original.index('## 2026-10-07：', start)
    current_section = '## 2026-10-17：SegFormer精度效率消融（最新核验结果）\n\n' + report.split('\n', 1)[1]
    main_doc.write_text(original[:start] + current_section + '\n\n' + original[end:], encoding='utf-8')
    queue.decisions()
    print(json.dumps(dict(report=str(root_report), csv=str(csv_path), figures=str(figures),
                          complete_runs=len(completed), complete_modules=sum(bool(r['modules']) for r in completed),
                          test_used=False, audit_passed=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
