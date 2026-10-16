from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(r"D:\论文复现")
OUT = ROOT / "SegFormer_B0_消融实验对照表与图注_20261014.docx"
CHART = ROOT / "SegFormer_B0_消融指标对比图_20261014.png"

models = ["SegFormer-B0", "+ UAFM", "+ PPContext", "PP-LiteSeg"]
metrics = {
    "mIoU": [0.8280, 0.8263, 0.8249, 0.8205],
    "Crack IoU": [0.6782, 0.6752, 0.6728, 0.6641],
    "F1": [0.8082, 0.8059, 0.8044, 0.7982],
}

def font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def make_chart():
    W, H = 1800, 980
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    title_f = font(42, True)
    label_f = font(26)
    tick_f = font(22)
    d.text((70, 32), "Crack500 validation ablation comparison", fill=(0, 0, 0), font=title_f)
    colors = [(39, 90, 140), (84, 145, 108), (211, 142, 55), (140, 91, 150)]
    panel_y = 130
    panel_h = 700
    panel_w = 500
    gap = 65
    for j, (metric, vals) in enumerate(metrics.items()):
        x0 = 80 + j * (panel_w + gap)
        y0 = panel_y
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        d.text((x0, 88), metric, fill=(0, 0, 0), font=label_f)
        low = 0.60 if metric != "mIoU" else 0.78
        high = 0.85
        for k in range(6):
            v = low + (high-low)*k/5
            y = y1 - (v-low)/(high-low)*panel_h
            d.line((x0, y, x1, y), fill=(220, 220, 220), width=2)
            d.text((x0-58, y-12), f"{v:.2f}", fill=(75, 75, 75), font=tick_f)
        d.line((x0, y0, x0, y1), fill=(45, 45, 45), width=3)
        d.line((x0, y1, x1, y1), fill=(45, 45, 45), width=3)
        bw = 72
        for i, v in enumerate(vals):
            bx = x0 + 40 + i*110
            by = y1 - (v-low)/(high-low)*panel_h
            d.rounded_rectangle((bx, by, bx+bw, y1), radius=8, fill=colors[i])
            d.text((bx-2, by-34), f"{v:.3f}", fill=(0, 0, 0), font=tick_f)
            short = ["B0", "UAFM", "PC", "PL"][i]
            d.text((bx+12, y1+14), short, fill=(0, 0, 0), font=tick_f)
    legend_y = 885
    for i, name in enumerate(["B0", "UAFM", "PPContext", "PP-LiteSeg"]):
        x = 260 + i * 340
        d.rectangle((x, legend_y, x+26, legend_y+26), fill=colors[i])
        d.text((x+38, legend_y-2), name, fill=(0, 0, 0), font=tick_f)
    im.save(CHART)

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)

def borders(table, color="D9D9D9"):
    tbl = table._tbl
    tblPr = tbl.tblPr
    b = tblPr.first_child_found_in("w:tblBorders")
    if b is None:
        b = OxmlElement("w:tblBorders")
        tblPr.append(b)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        el = b.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            b.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)

def set_cell(cell, text, bold=False, color="000000", align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(str(text))
    r.bold = bold
    r.font.name = "Microsoft YaHei"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def add_table(doc, headers, rows, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(caption)
    r.bold = True
    r.font.name = "Microsoft YaHei"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    r.font.size = Pt(10.5)
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = True
    borders(table)
    for i, h in enumerate(headers):
        set_cell(table.rows[0].cells[i], h, bold=True, color="FFFFFF")
        shade(table.rows[0].cells[i], "1F4E79")
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, val in enumerate(row):
            set_cell(cells[i], val)
            if ridx % 2 == 1:
                shade(cells[i], "F3F6F9")
    doc.add_paragraph()
    return table

def add_body(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.name = "Microsoft YaHei"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    r.font.size = Pt(10.5)
    return p

def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.name = "Microsoft YaHei"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        r.font.color.rgb = RGBColor(0, 0, 0)
    return p

make_chart()
doc = Document()
sec = doc.sections[0]
sec.top_margin = Cm(2.2)
sec.bottom_margin = Cm(2.2)
sec.left_margin = Cm(2.4)
sec.right_margin = Cm(2.4)

styles = doc.styles
styles["Normal"].font.name = "Microsoft YaHei"
styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
styles["Normal"].font.size = Pt(10.5)

title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("SegFormer B0 消融实验对照表与结果分析")
r.font.name = "Microsoft YaHei"
r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
r.font.size = Pt(20)
r.font.color.rgb = RGBColor(0, 0, 0)

add_body(doc, "本节用于直接放入论文 Results and Discussion 部分，按照先给出统一实验协议、再给出定量表格、随后解释模块作用和筛选结论的顺序组织。所有数字均来自 Crack500 验证集的聚合混淆矩阵；测试集只在验证集冻结 checkpoint 后访问。")

add_heading(doc, "4.5.2 Ablation study on Crack500", level=1)
add_body(doc, "为分析不同结构模块对裂缝分割性能的影响，以 SegFormer-B0 为控制组，在相同数据划分、数据增强、训练预算和 checkpoint 选择规则下，分别引入 UAFM 和 PPContextModule。UAFM 用于加强多尺度特征融合，重点检验细裂缝和边界恢复能力；PPContextModule 用于增强融合特征中的上下文信息，重点检验长裂缝连续性、阴影和低对比度区域的识别能力。PP-LiteSeg-STDC1 作为保持原始结构的公开模型对照，不参与本文模块消融。")

add_table(doc,
          ["Model", "mIoU", "Crack IoU", "Precision", "Recall", "F1", "Boundary F1", "Decision"],
          [
              ["SegFormer-B0", "0.8280", "0.6782", "0.8283", "0.7891", "0.8082", "0.5894", "Control"],
              ["SegFormer-B0 + UAFM", "0.8263", "0.6752", "0.8202", "0.7924", "0.8059", "未计算", "Fail"],
              ["SegFormer-B0 + PPContext", "0.8249", "0.6728", "0.8109", "0.7979", "0.8044", "0.5776", "Fail"],
              ["PP-LiteSeg-STDC1", "0.8205", "0.6641", "0.8250", "0.7730", "0.7982", "0.5643", "Reference"],
          ],
          "Table 1. Quantitative comparison of the ablation models on the Crack500 validation set.")

add_body(doc, "从 Table 1 可以看出，SegFormer-B0 获得最高的 mIoU、裂缝 IoU、F1 和边界 F1，因此被保留为当前控制模型。UAFM 的 Recall 略有增加，说明其可能帮助网络保留部分难检裂缝像素；但是裂缝 IoU 仅增加约 0.0030，F1 增加约 0.0029，均未达到预设的 +0.005 和 +0.003 门槛，同时整体 mIoU 下降约 0.0017。因而 UAFM 没有通过单模块筛选。")
add_body(doc, "PPContextModule 的 Recall 提升约 0.0088，说明上下文信息有助于扩大裂缝候选区域，但 Precision 下降约 0.0173，裂缝 IoU 下降约 0.0054，边界 F1 下降约 0.0118。这表明该模块在当前训练协议下更容易产生纹理和阴影误检，扩大预测区域并不能转化为有效裂缝区域的准确恢复。PPContextModule 同样没有通过筛选，因此不启动 UAFM 与 PPContextModule 的联合实验。")

add_table(doc,
          ["Model", "ΔmIoU", "ΔCrack IoU", "ΔPrecision", "ΔRecall", "ΔF1", "ΔBoundary F1", "Gate"],
          [
              ["SegFormer-B0 + UAFM", "−0.0017", "+0.0030", "+0.0082", "−0.0037", "+0.0029", "未计算", "Not passed"],
              ["SegFormer-B0 + PPContext", "−0.0031", "−0.0054", "−0.0173", "+0.0088", "−0.0039", "−0.0118", "Not passed"],
          ],
          "Table 2. Metric changes relative to SegFormer-B0 and the module-selection decision.")

add_body(doc, "本文采用的完整通过条件为：裂缝 IoU 至少提升 0.005，F1 至少提升 0.003，Recall 下降不超过 0.005，mIoU 不下降，且边界 F1 不下降。由于 UAFM 的边界 F1 尚未形成完整的独立复核结果，UAFM 只能作为未通过的受控消融结果保存；PPContextModule 已有完整边界统计，但其边界 F1 低于控制组。该判定过程严格按照验证集结果完成，未使用测试集指标反向挑选模型。")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(CHART), width=Cm(16.0))
cap = doc.add_paragraph()
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = cap.add_run("Fig. 1. Crack500 validation comparison of mIoU, crack IoU and F1. B0 denotes SegFormer-B0, PC denotes PPContextModule, and PL denotes PP-LiteSeg-STDC1.")
r.italic = True
r.font.name = "Microsoft YaHei"
r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
r.font.size = Pt(9.5)

add_heading(doc, "Interpretation of the ablation results", level=2)
add_body(doc, "Figure 1 shows that the three principal foreground metrics decrease as the tested modules are added. This result does not mean that UAFM or PPContextModule is universally ineffective; it indicates that, under the present SegFormer-B0 feature resolution, 400 × 400 crop, cross-entropy objective and 18,000-iteration budget, neither module provides a sufficiently consistent gain on Crack500. The failure is therefore retained as an experimental finding rather than hidden by selecting a favorable test subset.")
add_body(doc, "对于论文叙述，建议将本组实验表述为“模块作用验证”，而不是把所有模块都描述成有效改进。UAFM 的结果支持这样的分析：多尺度融合可能改善少量细裂缝或端点，但收益不足以抵消部分区域误差；PPContextModule 的结果支持这样的分析：上下文建模增加了裂缝候选响应，却同时放大了阴影、纹理和低对比度路面中的误检。因此，后续工作应优先开展逐图错误归因、输入缩放与裁剪审计、标签连通性检查以及边界监督分析，而不是直接叠加更多注意力、Transformer 或 Mamba 模块。")

add_heading(doc, "Recommended qualitative figure and caption", level=2)
add_body(doc, "建议在本表之后插入一张四列或五列定性对照图。每一行使用同一张 Crack500 验证图，列依次为 Input、Ground truth、SegFormer-B0、SegFormer-B0 + UAFM、SegFormer-B0 + PPContext。至少选择细裂缝、阴影/低对比度、纹理密集和宽裂缝四类样本，并用红色标出 false positive、蓝色标出 false negative、黄色标出边界偏移。")
add_body(doc, "Fig. 2. Qualitative comparison of the baseline and ablation models on representative Crack500 validation images. The selected cases illustrate thin-crack discontinuity, low-contrast crack omission, shadow or texture false positives, and width or boundary recovery errors. The examples are used to explain the metric changes in Tables 1 and 2 and are not used for model selection.")

add_heading(doc, "Placement in the main manuscript", level=2)
add_body(doc, "在主论文中，Table 1 应放在 Crack500 验证结果的第一段之后，用于先呈现完整消融结果；Table 2 紧接其后，用于解释相对 SegFormer-B0 的增量和筛选门槛；Fig. 1 可放在 Table 2 之后，帮助读者快速比较三项核心指标；Fig. 2 放在定性分析段落之后，用于解释细裂缝、低对比度、阴影、纹理和宽度恢复等具体失败模式。随后再进入冻结测试集的公开模型对比表，避免将验证集消融和测试集对照混为一谈。")

doc.core_properties.title = "SegFormer B0 消融实验对照表与结果分析"
doc.core_properties.subject = "Crack500 validation ablation study"
doc.core_properties.author = ""
doc.save(OUT)
print(OUT)
