import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
GROUPS = [
    ('C0_B0', 'SegFormer-B0'),
    ('C1_Aligned', 'SegFormer-B0 + Aligned'),
    ('C2_WTConvWavelet', 'SegFormer-B0 + WTConv'),
    ('C3_Aligned_WTConvWavelet_bgce', 'SegFormer-B0 + Aligned + WTConv'),
]

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
    params = read_json(RESULTS / 'param_counts.json')
    rows = []
    for key, label in GROUPS:
        sel_path = RESULTS / f'{key}_s42_selection.json'
        test_path = RESULTS / f'{key}_s42_test_exact.json'
        if not sel_path.exists() or not test_path.exists():
            rows.append({'id': key, 'Variant': label, 'status': 'missing'})
            continue
        sel = read_json(sel_path)
        test = read_json(test_path)
        rows.append({
            'id': key, 'Variant': label, 'status': 'complete',
            'Precision (%)': test['precision'] * 100,
            'Recall (%)': test['recall'] * 100,
            'F1 (%)': test['f1'] * 100,
            'IoU (%)': test['crackIoU'] * 100,
            'mIoU (%)': test['mIoU'] * 100,
            'BF1@2 (%)': test.get('boundary_f1_2px', 0) * 100,
            'hard clDice (%)': test.get('hard_cldice', 0) * 100,
            'Params (M)': params.get(key, 0) / 1e6,
            'selected': sel.get('selected'),
            'val_mIoU (%)': sel.get('ordinary_val_mIoU' if sel.get('selected') == 'ordinary' else 'ema_val_mIoU', 0) * 100,
            'checkpoint_sha256': sel.get('selected_sha256'),
        })
    (RESULTS / 'deepcrack_ablation.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    fields = ['Variant', 'Precision (%)', 'Recall (%)', 'F1 (%)', 'IoU (%)', 'mIoU (%)', 'BF1@2 (%)', 'hard clDice (%)', 'Params (M)', 'selected', 'val_mIoU (%)', 'status']
    with (RESULTS / 'deepcrack_ablation.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fields})
    lines = [
        '# DeepCrack AW + 背景加权 CE 四模型消融报告（2026-11-12）', '',
        '## 数据与协议', '',
        '- 数据根目录：`C:/Users/user/Desktop/segformer_aw_cross_dataset_wave17_20261106/normalized/DeepCrack`。',
        '- 文件清单：train=255、val=45、test=237；本轮不修改清单。',
        '- seed=42、FP32、batch=4、18,000 iter、AdamW、初始学习率6e-5、PolynomialDecay到0、EMA。',
        '- 训练增强与 Crack500 AW 相同：ResizeStepScaling、400×400 RandomPaddingCrop、水平翻转、RandomDistort、Normalize。',
        '- 四组都使用相同背景加权 CE `[1.10, 1.0]`，只改变结构；普通/EMA按精确 val mIoU 选模，冻结后评估 test。', '',
        '## DeepCrack test 四组对照', '',
        '| Variant | Precision (%) | Recall (%) | F1 (%) | IoU (%) | mIoU (%) | BF1@2 (%) | hard clDice (%) | Params (M) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        if r.get('status') != 'complete':
            lines.append(f"| {r['Variant']} | missing | — | — | — | — | — | — | — |")
        else:
            lines.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.6f} |'.format(
                r['Variant'], r['Precision (%)'], r['Recall (%)'], r['F1 (%)'],
                r['IoU (%)'], r['mIoU (%)'], r['BF1@2 (%)'],
                r['hard clDice (%)'], r['Params (M)']))
    lines += ['', '## 模块贡献解释', '',
              '- C0 是同协议结构基线。',
              '- C1 相对 C0 的变化用于估计 Aligned 的空间对齐贡献。',
              '- C2 相对 C0 的变化用于估计 Haar-WTConv 的上下文与局部细节贡献。',
              '- C3 同时启用两者，用于判断联合效果是否超过单模块，并观察是否存在互补或指标折中。',
              '- 本表是单种子 DeepCrack 结果，不能据此宣称领域 SOTA；若要写稳定性结论，还需 seed43/44。', '',
              '## Crack500 六模型公开对照', '',
              '当前 AW + 背景加权 CE 与六个公开模型的统一历史 test 对照见：`../segformer_crack500_aw_wavelet_20261112/results/AW_CEbg110_vs_six_public_test.csv`。',
              '六个公开模型为 SegFormer-B0、OCRNet-HRNet-W18、PP-LiteSeg-STDC1、PSPNet-ResNet50、DeepLabV3P-ResNet50、Fast-SCNN。它们是历史冻结记录，训练后端、初始化、loss 和 EMA 不完全一致，因此只作公开参考。', '',
              '## 复现索引', '',
              '- 训练状态/exit：`results/*_s42_train_status.json`、`results/*_s42_train_exit.json`。',
              '- val 选择：`results/*_s42_val_ordinary_exact.json`、`results/*_s42_val_ema_exact.json`、`results/*_s42_selection.json`。',
              '- test 结果：`results/*_s42_test_exact.json`。',
              '- 参数：`results/param_counts.json`。',
              '- 配置与运行脚本：`configs/`、`run_deepcrack_ablation.ps1`。',
    ]
    (ROOT / 'DeepCrack四模型消融与Crack500公开对照报告_20261112.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
