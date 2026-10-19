"""Audited report and scientific plots from existing train/val results."""
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent
import aligned_queue as a


def main(plots=False):
    a.report()
    c=a.read(ROOT/'results/comparison.json')
    audit=a.read(ROOT/'results/audit_complete_20261019.json')
    if audit['errors']:raise RuntimeError('Evidence audit has unresolved errors')
    base=c['baseline'];parent=c['parent'];complete=[r for r in c['rows'] if r['status']=='complete']
    records=[]
    for ident in ['B0','W128','A0']:
        m=a.read(ROOT/'results'/(ident+'_val.json'));p=a.read(ROOT/'results'/(ident+'_profile.json'))
        records.append(dict(id=ident,status='reused_frozen',**{k:m[k] for k in a.METRICS},params=p['params_total']))
    records+=c['rows']
    cols=['id','status',*[k+'_pct' for k in a.METRICS],'params_M','delta_iou_B0_pp','delta_f1_B0_pp','param_change_B0_pct','parent_accuracy_gate','B0_strict_metrics_params_gate','balanced_metrics_params_gate']
    bp=a.read(ROOT/'results/B0_profile.json')['params_total']
    raw=[]
    for r in records:
        row={k:r[k] for k in ('id','status')}
        if 'mIoU' in r:
            row.update({k+'_pct':r[k]*100 for k in a.METRICS})
            row.update(params_M=r['params']/1e6,delta_iou_B0_pp=(r['crackIoU']-base['crackIoU'])*100,
                       delta_f1_B0_pp=(r['f1']-base['f1'])*100,param_change_B0_pct=(r['params']/bp-1)*100)
            for k in cols:
                if k in r:row[k]=r[k]
        raw.append(row)
    with (ROOT/'results/ablation_percent_20261019.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(raw)
    lines=['# SegFormer-B0＋Aligned 第二模块筛选阶段报告','','更新时间：'+a.q.now(),'','本报告为阶段一完整筛选结果，三seed、稳定速度、六个公开模型与统一test仍未完成。','',
        '## 完成状态','','登记24个机制：22项完整18k、真实exit0、冻结SHA和独立验证通过；DAPPM反向smoke真实失败；PSP兼容性暂缓。审计错误为0。所有新结果均是Crack500验证集348图、seed42，未读取test或自建数据。','',
        '## 完整结果','','P/R/F1/IoU/mIoU/BF1均为百分数，Params为百万；待完成/兼容性失败不填性能。','',
        '| 模型 | 状态 | P | R | F1 | 裂缝IoU | mIoU | BF1 | Params M |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        values=' | '.join(f'{r[k]*100:.4f}' for k in a.METRICS)+f" | {r['params']/1e6:.6f}" if 'mIoU' in r else '— | — | — | — | — | — | —'
        lines.append(f"| {r['id']} | {r['status']} | {values} |")
    lines+=['','## 观察、解释和下一步','',
        '1. 区域IoU最高的是Aligned＋PAM（68.3507%），但Precision下降、边界F1也低，未通过预登记完整门槛。不能只按IoU最大值选最终模型。',
        '2. 同时通过A0贡献门槛的两项为A21 Polarized和A10 AFNB；已在新seed与test之前冻结为复核候选，不能写成最终模型。',
        '3. A21的IoU/F1/mIoU为68.0773/81.0071/82.9321%，较原B0分别增加0.2599/0.1843/0.1300个百分点，参数3.503330M，减少约5.70%。Precision比原B0下降约0.59个百分点、Recall提高约0.90个百分点；不能声称所有精度指标同时提升。',
        '4. A10的IoU/F1/mIoU为68.1660/81.0699/82.9836%，参数3.656994M，减少约1.57%。它是平衡备选，不满足减参5%的严格参数线。',
        '5. 更高IoU的APNB和Hamburger未通过A0的Precision下限；已保留其结果和预登记标志。不同数值门槛不能互相替代，不能事后调低Precision门槛。',
        '6. Aligned＋GloRe只有小幅区域收益，未通过A0贡献门槛；ContextEmbedding、PPContext等均保留负结果。单模块表现不保证联合表现。',
        '7. 以上是22路验证搜索后的单seed观察，含选择偏差。必须检验匹配seed43/44的平均变化和波动，速度稳定前不宣称加速；也不能据此声称SOTA。','',
        '解释假设：Polarized在融合特征上联合重标定空间/通道；AFNB在C3/C4之间补语义融合，与Aligned的C1/C4几何路径不同。当前数值支持进一步复核，并未证明具体阴影、落叶或长裂缝场景已解决。','',
        '## 下一阶段已登记','',
        '复核计划位于followup_20261019/refine-logs/PLAN.md。补W128＋AFNB seed42一项；B0、A0、A21、A10各seed43/44共8项。Polarized独立模块复用旧波M02_PSA。三seed齐备前不填写均值±标准差。','',
        '稳定配对速度先测四个冻结checkpoint；当P90/P10或配对比值IQR不稳定时保留测量并标为未确认。最终通过指标、参数、速度和多seed复核后，再补六个原始公开模型并冻结统一test。','',
        '## 证据与交付','',
        '- 全部原始小数和门槛：results/comparison.json。',
        '- 百分数完整CSV：results/ablation_percent_20261019.csv。',
        '- 22组退出/最终18k/SHA/选模/验证审计：results/audit_complete_20261019.json。',
        '- 数据前后4488文件哈希：provenance/data_before.json、data_after.json。',
        '- 冻结候选与新seed预算：followup_20261019/provenance/frozen_candidates.json。',
        '- 精确2px边界逐图CSV和全348空间归因：results、diagnostics。',
        '- 约1.77GiB已完成smoke的重复产物已按逐文件SHA清单清理；正式18k/普通EMA最佳/冻结权重与失败证据保留。清单见provenance/smoke_cleanup_manifest_20261019.json及receipt。','',
        '最终交付清单依旧是refine-logs/FINAL_DELIVERY.md。本报告不替代最终统一test和多seed总结。']
    if plots:
        os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/mplconfig')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        out=ROOT/'figures/screening_20261019';out.mkdir(parents=True,exist_ok=True)
        plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
        fig,ax=plt.subplots(figsize=(9,5.5))
        ax.scatter([r['params']/1e6 for r in complete],[r['crackIoU']*100 for r in complete],c='#9ca3af',s=35,label='Completed additions (seed42)')
        highlight=['B0','A0','A21_Polarized','A10_AFNB','A18_PAM','A22_Hamburger','A05_ContextEmbedding']
        colors=['black','#64748b','#dc2626','#2563eb','#d97706','#16a34a','#6b7280']
        for ident,color in zip(highlight,colors):
            row=next(r for r in records if r['id']==ident)
            x=row['params']/1e6;y=row['crackIoU']*100
            ax.scatter([x],[y],s=65,color=color,marker='D' if ident=='B0' else 'o')
            ax.annotate(ident,(x,y),xytext=(5,6),textcoords='offset points',fontsize=8,color=color)
        ax.axhline(base['crackIoU']*100,color='black',lw=.8,ls='--');ax.axvline(bp*.95/1e6,color='#64748b',lw=.8,ls=':')
        ax.set(xlabel='Parameters (million)',ylabel='Crack IoU on validation (%)',title='Accuracy / parameter trade-off (not a speed result)')
        ax.grid(alpha=.18);ax.legend(loc='lower left');fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/('params_iou.'+ext),dpi=180)
        plt.close(fig)
        selected=[next(r for r in records if r['id']==x) for x in ['A21_Polarized','A10_AFNB','A18_PAM']]
        fig,ax=plt.subplots(figsize=(8,4.8));metrics=['precision','recall','f1','crackIoU','boundary_f1']
        for i,row in enumerate(selected):ax.bar([j+(i-1)*.25 for j in range(len(metrics))],[(row[k]-base[k])*100 for k in metrics],width=.25,label=row['id'])
        ax.set_xticks(range(len(metrics)),['Precision','Recall','F1','Crack IoU','Boundary F1'])
        ax.axhline(0,color='black',lw=.7);ax.set(ylabel='Change vs original B0 (percentage points)',title='Gains and regressions must be reported together')
        ax.legend();ax.grid(axis='y',alpha=.18);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/('metric_tradeoffs.'+ext),dpi=180)
        plt.close(fig)
        lines+=['','## 结果图','', '![Parameter vs IoU](figures/screening_20261019/params_iou.png)','', '![Metric changes](figures/screening_20261019/metric_tradeoffs.png)']
    speed_file=ROOT/'followup_20261019/results/candidate_latency.json'
    if speed_file.exists():
        speed=a.read(speed_file)
        lines+=['','## 候选速度实测与限制','','四冻结模型、同进程平衡交错、GPU同步；本次所有稳定性门槛未通过。两个候选的实际中位数均更慢，不得省略变慢结果或宣称加速。','',
                '| 模型 | 400中位ms | 480中位ms | 480配对B0比值 | 稳定通过 |','|---|---:|---:|---:|---|']
        for ident in ['B0','A0','A21','A10']:
            s=next(x for x in speed['summary'] if x['id']==ident and x['size']==400)
            t=next(x for x in speed['summary'] if x['id']==ident and x['size']==480)
            lines.append(f"| {ident} | {s['median_ms']:.3f} | {t['median_ms']:.3f} | {t['paired_median_ratio_to_B0']:.3f} | {s['stability_pass'] and t['stability_pass']} |")
        lines+=['','多seed复核用于检验精度/参数收益，不表示完整效率目标已经通过。稳定速度前完整目标为空，参数更少不能替代延迟证据。']
    lines+=['','本轮筛选未访问test；原Crack500 test在项目历史阶段已有观察，因此不称整个项目从未观察的集合。最终保留source-frame-disjoint泛化复核。']
    content='\n'.join(lines)+'\n'
    a.q.versioned_text(ROOT/'refine-logs/STAGE1_ANALYSIS.md',content)
    target=a.q.PROJECT/'SegFormer_Aligned模块筛选阶段报告_20261019.md'
    # Root-level report uses real absolute figure links.
    content=content.replace('(figures/', '('+str(ROOT).replace('\\','/')+'/figures/')
    target.write_text(content,encoding='utf-8')
    print(json.dumps(dict(report=str(target),complete=len(complete),candidates=['A21_Polarized','A10_AFNB'],test_used=False),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plots',action='store_true');args=p.parse_args();main(args.plots)
