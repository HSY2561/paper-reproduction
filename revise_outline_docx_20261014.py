from pathlib import Path
import shutil
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

SRC = Path(r"C:\Users\user\Desktop\大纲安排草稿.docx")
OUT = Path(r"D:\论文复现\大纲安排草稿_已修改_20261014.docx")
shutil.copy2(SRC, OUT)
doc = Document(OUT)

def clear_para(p):
    for r in list(p.runs):
        p._p.remove(r._r)

def set_font(run, name="Microsoft YaHei", size=10.5, bold=False, italic=False, color="000000"):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)

def set_para(p, text, align=None, font_name="Microsoft YaHei", size=10.5, bold=False, italic=False):
    clear_para(p)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    set_font(r, font_name, size, bold, italic)
    p.paragraph_format.line_spacing = 1.15
    return p

def insert_para_after(anchor, text, style=None, align=None, font_name="Microsoft YaHei", size=10.5, bold=False, italic=False):
    newp = OxmlElement("w:p")
    anchor._p.addnext(newp)
    p = anchor._parent.add_paragraph()
    p._p.getparent().remove(p._p)
    newp.addnext(p._p)
    if style is not None:
        p.style = style
    return set_para(p, text, align, font_name, size, bold, italic)

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)

def set_borders(table, color="D9D9D9"):
    tblPr = table._tbl.tblPr
    borders = tblPr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tblPr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn("w:" + edge)
        el = borders.find(tag)
        if el is None:
            el = OxmlElement("w:" + edge)
            borders.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)

def set_cell(cell, text, header=False):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(str(text))
    set_font(r, size=9, bold=header, color="FFFFFF" if header else "000000")
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if header:
        shade(cell, "1F4E79")

def insert_table_after(anchor, headers, rows, caption):
    cap = insert_para_after(anchor, caption, style=anchor.style, align=WD_ALIGN_PARAGRAPH.CENTER, size=10, bold=True)
    table = doc.add_table(rows=1, cols=len(headers))
    set_borders(table)
    for i, h in enumerate(headers):
        set_cell(table.rows[0].cells[i], h, True)
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, val in enumerate(row):
            set_cell(cells[i], val, False)
            if ridx % 2 == 1:
                shade(cells[i], "F3F6F9")
    cap._p.addnext(table._tbl)
    # Add a paragraph after the table so later text remains separated.
    after = doc.add_paragraph()
    table._tbl.addnext(after._p)
    return after

paras = doc.paragraphs

# SegFormer architecture: paragraphs 38, 40, 42, 44, 47 in the supplied outline.
set_para(paras[38],
    "本研究采用 SegFormer-B0 作为路面裂缝语义分割的基础模型。其编码器为四阶段 Mix Vision Transformer（MiT-B0），使用重叠式 patch embedding 和空间缩减注意力提取多尺度特征。MiT-B0 的四个阶段参数分别为：通道数 [32, 64, 160, 256]，attention heads [1, 2, 5, 8]，Transformer blocks [2, 2, 2, 2]，sr_ratio [8, 4, 2, 1]。各阶段的 MLP ratio 均为 4，qkv_bias 开启，drop path rate 为 0.1。对于 400 × 400 输入，四个阶段实际输出尺寸为 100 × 100 × 32、50 × 50 × 64、25 × 25 × 160 和 13 × 13 × 256，其中最后一个空间尺寸由 3 × 3、stride=2、padding=1 的重叠式 patch embedding 产生。"
)
set_para(paras[40],
    "第 i 个阶段的输出记为 F_i ∈ R^(C_i × H_i × W_i)，其中 (C_i, H_i, W_i) 依次为 (32,100,100)、(64,50,50)、(160,25,25) 和 (256,13,13)。四个阶段分别保留不同空间分辨率和语义层次，使网络同时利用长裂缝的区域上下文与细裂缝的局部纹理。"
)
set_para(paras[42],
    "对于每个阶段，PaddleSeg 实现首先将二维特征展平为 token 序列，再通过线性层投影到统一的 256 维嵌入空间： E_i = MLP_i(Flatten(F_i)) ∈ R^((H_iW_i) × 256)。随后将 token 重排为空间特征，并通过双线性插值将第 2 至第 4 阶段上采样到第一阶段的 100 × 100 空间尺寸： F̂_i = Up_{100×100}(Reshape(E_i))。"
)
set_para(paras[44],
    "四个尺度的特征按照 [F̂_4, F̂_3, F̂_2, F̂_1] 在通道维拼接，得到 1024 通道特征；随后使用 1 × 1 ConvBNReLU（1024 → 256）进行融合。融合结果经过 Dropout2D(p=0.1)，再由 1 × 1 分类卷积（256 → 2）产生背景/裂缝 logits，最后通过双线性插值恢复到输入图像的 H × W 尺寸： Z = Conv_{1×1}(Dropout(ConvBNReLU(Concat(F̂_4,F̂_3,F̂_2,F̂_1))))，P = Softmax(Up_{H×W}(Z))。"
)
set_para(paras[47],
    "该结构具有较低的解码开销，适合部署到搭载 RTX5060 训练、并在机器人端执行视觉巡检和局部修复的系统中。需要注意的是，13 × 13 的深层特征和 400 × 400 裁剪会压缩极细裂缝的空间信息，因此后续实验必须区分信息是在输入缩放、编码器特征提取、MLP 融合还是最终预测恢复阶段损失。UAFM 和 PPContextModule 仅作为受控消融模块，用于验证两个具体位置的结构假设。"
)

# Insert controlled decoder definition after the architecture description.
anchor = paras[47]
heading = insert_para_after(anchor, "Controlled decoder ablations", style=paras[37].style, bold=True, size=11)
uafm = insert_para_after(heading,
    "UAFM_SpAtten 接在基线的多尺度融合卷积之后、Dropout2D 和分类卷积之前。基线融合特征记为 C ∈ R^(B×256×100×100)，第一阶段细节特征记为 C1 ∈ R^(B×256×100×100)，UAFM 的输入为 (C1, C)，输出仍为 R^(B×256×100×100)，随后进入 Dropout2D 和 1 × 1 分类卷积。SegFormer-B0 的参数量为 3.715M，加入 UAFM 后为 4.897M，增加约 1.182M；RTX5060 上验证 batch 耗时由 0.0175 s 增至 0.0215 s。其设计目标是改善高低层特征融合、细裂缝端点和边界恢复，但当前 Crack500 验证结果未达到筛选门槛。"
)
ppc = insert_para_after(uafm,
    "PPContextModule 同样位于融合特征瓶颈，但其输入仅为 C ∈ R^(B×256×100×100)。按照当前 PaddleSeg 代码，模块参数为 PPContextModule(256, 64, 256, (1,3,6))，即以 64 通道中间表示和三个上下文尺度提取多尺度上下文，输出与 C 同形状的上下文残差；模型计算 C' = C + PPContextModule(C)，再执行 Dropout2D、1 × 1 分类卷积和最终双线性上采样。加入该模块后参数量为 3.914M，增加约 0.199M，验证 batch 耗时为 0.0183 s。其设计目标是提高长裂缝连续性以及阴影和低对比度区域的识别，但当前结果表现为 Recall 增加而 Precision、IoU 和边界 F1 下降。"
)

# Implementation details and formulas.
set_para(paras[48],
    "所有实验均基于 PaddleSeg-2.10.0 官方实现，在 RTX5060 GPU、FP32 精度下运行。随机种子为 42，batch size 为 4，总训练预算为 18,000 iterations。优化器采用 AdamW，β1=0.9、β2=0.999、weight decay=0.01，初始学习率 η0=6×10^−5。训练阶段使用 ResizeStepScaling（0.75–1.25，step=0.25）、400 × 400 RandomPaddingCrop、随机水平翻转、亮度/对比度/饱和度扰动和归一化；验证阶段只使用归一化。"
)
set_para(paras[50],
    "学习率按照 polynomial decay 下降至 0： η_t = η_0 (1 − t/T)^p，η_0=6×10^−5，T=18,000，p=1，t 为当前 iteration。"
    , align=WD_ALIGN_PARAGRAPH.CENTER, font_name="Cambria Math")
set_para(paras[52],
    "训练损失采用二分类交叉熵： L_CE = −(1/N) Σ_p Σ_{c=0}^{1} t_{p,c} log(p̂_{p,c})，其中 t_{p,c} 为像素 p 的 one-hot 真实标签，p̂_{p,c} 为模型预测概率，N 为参与计算的像素数。这里使用 t 和 p̂ 区分真实标签与预测概率，避免变量混用。"
)
set_para(paras[53],
    "每次训练同时保存普通模型和 EMA 模型 checkpoint。模型选择只使用验证集 mIoU： k* = argmax_{k∈{ordinary, EMA}} mIoU_val(k)。选中的 checkpoint 被复制到冻结目录并计算 SHA-256。测试集只在 checkpoint 冻结后访问一次，不能用 test 指标反向选择模型。"
)
set_para(paras[55],
    "该流程保证所有模型使用相同的数据清单、标签编码、数据增强、训练预算和评价脚本，也使验证集消融结果与最终测试集公开模型对照保持独立。"
)

# Crack500 crop/source-frame clarification.
set_para(paras[59],
    "Crack500 的 1,896、348 和 1,124 分别是官方 train、validation 和 test 清单中的裁剪图像—标签对数量，而不是 1,896、348 和 1,124 个独立原始采集 frame。每个 crop 应保留其 source_frame_id 和 crop_index；同一原始 frame 产生的相邻 crop 不能分别进入训练集和验证/测试集，否则会产生相邻裁剪泄漏。当前结果使用官方 crop 清单，已完成配对关系、文件存在性、标签值和 SHA-256 审计；source-frame-disjoint 复核还必须按照 source_frame_id 重新汇总，并检查 S_train∩S_val=∅、S_train∩S_test=∅、S_val∩S_test=∅。在该复核完成前，本文只能将这组结果称为 official crop-split results，不能称为 source-frame-disjoint results。"
)
source_audit = insert_para_after(paras[59],
    "Source-frame audit protocol. 先从文件名或原始元数据恢复 source_frame_id，再按 source_frame_id 分组统计 crop 数量、裂缝像素比例和尺寸。审计表应同时列出 crop 数、source frame 数、每个 source frame 的 crop 数分布、跨 split 重叠数和重复哈希数。该表应放在 Crack500 数据集描述之后，并与最终 test 结果一起保存为可复核的清单附件。"
)

# Metrics section before results.
metrics_heading = insert_para_after(source_audit, "Evaluation metrics and statistical protocol", style=paras[56].style, bold=True, size=11)
metrics1 = insert_para_after(metrics_heading,
    "本文同时记录两种统计口径。参考论文报告类别平均 Precision、Recall 和 F1，并使用 2 像素容差衡量边界匹配；本研究的主结果则由全验证集混淆矩阵聚合得到前景裂缝 IoU、Precision、Recall、F1 和 mIoU，边界 F1 也由全数据集边界匹配计数计算。两种口径不能直接混合比较。"
)
metrics2 = insert_para_after(metrics1,
    "对裂缝前景类别，IoU_crack = TP/(TP+FP+FN)，Precision = TP/(TP+FP)，Recall = TP/(TP+FN)，F1 = 2·Precision·Recall/(Precision+Recall)，mIoU = (IoU_bg + IoU_crack)/2。参考论文的类别平均指标应写为 mP=(P_bg+P_crack)/2、mR=(R_bg+R_crack)/2 和 mF1=(F1_bg+F1_crack)/2。"
)
metrics3 = insert_para_after(metrics2,
    "边界 F1 的计算先对二值掩膜 G 和 P 使用 3 × 3 腐蚀： B_G=G−Erode_3×3(G)，B_P=P−Erode_3×3(P)。若预测边界像素与真实边界像素的欧氏距离不超过 2 像素，则视为匹配，再由聚合后的边界 Precision 和 Recall 计算 Boundary F1。"
)

# Correct the results interpretation paragraph.
set_para(paras[64],
    "实验结果表明，SegFormer-B0 在当前 Crack500 official crop-split 协议下取得验证 mIoU 0.8280、裂缝 IoU 0.6782、Precision 0.8283、Recall 0.7891、F1 0.8082 和边界 F1 0.5894，因此保留为当前控制模型。UAFM 的验证结果为 mIoU 0.8263、裂缝 IoU 0.6752、Precision 0.8202、Recall 0.7924、F1 0.8059；其裂缝 IoU 和 F1 没有达到预设增益门槛，且 mIoU 下降，因此不能写成性能提升。PPContextModule 的验证结果为 mIoU 0.8249、裂缝 IoU 0.6728、Precision 0.8109、Recall 0.7979、F1 0.8044 和边界 F1 0.5776。该模块虽然提高 Recall，但同时降低 Precision、裂缝 IoU、F1 和边界 F1，说明在当前设置下扩大候选响应带来了更多阴影和纹理误检。两个模块均保留为受控消融负结果，不启动联合模型，也不根据这些失败模型访问 test 集。"
)
set_para(paras[66],
    "逐图分析应围绕输入缩放、编码器特征提取和预测恢复三个阶段区分错误来源。第一类样本为细裂缝和端点，用于检查 400 × 400 裁剪、13 × 13 深层特征和最终双线性上采样是否造成裂缝消失、变细或断裂；第二类为阴影和低对比度区域，用于区分漏检与误检；第三类为纹理密集路面，用于观察模型是否将沥青纹理误认为裂缝；第四类为宽裂缝，用于检查主体区域和边界是否同时恢复。每张图应记录 IoU、Boundary F1、FP/FN 像素数、裂缝连通分量数、是否零重叠以及 source frame 信息，不能只根据总体均值判断模块有效性。"
)

# Insert ablation results table after the corrected Crack500 results paragraph and before the computational-efficiency discussion.
table_anchor = paras[64]
ablation_after = insert_table_after(
    table_anchor,
    ["Model", "mIoU", "Crack IoU", "Precision", "Recall", "F1", "Boundary F1", "Decision"],
    [
        ["SegFormer-B0", "0.8280", "0.6782", "0.8283", "0.7891", "0.8082", "0.5894", "Control"],
        ["SegFormer-B0 + UAFM", "0.8263", "0.6752", "0.8202", "0.7924", "0.8059", "未计算", "Fail"],
        ["SegFormer-B0 + PPContext", "0.8249", "0.6728", "0.8109", "0.7979", "0.8044", "0.5776", "Fail"],
        ["PP-LiteSeg-STDC1", "0.8205", "0.6641", "0.8250", "0.7730", "0.7982", "0.5643", "Reference"],
    ],
    "Table 4. Controlled ablation results on the Crack500 validation set."
)

doc.save(OUT)
print(str(OUT))
