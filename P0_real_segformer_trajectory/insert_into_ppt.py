from pathlib import Path
import shutil
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
ppt=Path(r'C:\Users\user\Desktop\绘图.pptx')
backup=ppt.with_name('绘图_P0插图前备份.pptx')
if not backup.exists(): shutil.copy2(ppt,backup)
prs=Presentation(str(ppt))
slide=prs.slides[0]
# Clear existing shapes only if any; current deck is blank, but preserve any nonempty slide.
if len(slide.shapes)==0:
    pass
else:
    # do not delete user content; append a new slide instead
    slide=prs.slides.add_slide(prs.slide_layouts[6])
# dimensions
sw,sh=prs.slide_width,prs.slide_height
left=Inches(0.22); right=Inches(0.22); top=Inches(0.80); bottom=Inches(0.28)
gapx=Inches(0.12); gapy=Inches(0.14)
cellw=(sw-left-right-2*gapx)//3
cellh=(sh-top-bottom-gapy)//2
# slide title
box=slide.shapes.add_textbox(Inches(0.22),Inches(0.12),sw-Inches(0.44),Inches(0.42))
tf=box.text_frame; tf.clear(); p=tf.paragraphs[0]; p.text='P0  Real SegFormer Mask to Crack Skeleton and Repair Trajectory'; p.font.size=Pt(22); p.font.bold=True; p.font.color.rgb=RGBColor(31,55,79); p.alignment=PP_ALIGN.CENTER
panel_dir=Path(r'D:\论文复现\P0_real_segformer_trajectory\ppt_panels')
panels=[panel_dir/f'{i:02d}_{n}.png' for i,n in [(1,'original'),(2,'real_mask'),(3,'skeleton'),(4,'crack_map'),(5,'coordinates'),(6,'trajectory')]]
for i,panel in enumerate(panels):
    row,col=divmod(i,3)
    x=left+col*(cellw+gapx); y=top+row*(cellh+gapy)
    slide.shapes.add_picture(str(panel),x,y,width=cellw,height=cellh)
# footer note
foot=slide.shapes.add_textbox(Inches(0.18),sh-Inches(0.23),sw-Inches(0.36),Inches(0.18))
tf=foot.text_frame; tf.clear(); p=tf.paragraphs[0]; p.text='Crack500 valcrop | frozen SegFormer-B0 | real mask, skeleton and offline smoothed path | coordinates require measured camera-plane calibration'; p.font.size=Pt(8); p.font.color.rgb=RGBColor(90,90,90); p.alignment=PP_ALIGN.CENTER
out=ppt.with_name('绘图_P0六图.pptx')
prs.save(str(out))
print('saved',out,'slides',len(prs.slides),'shapes_first',len(prs.slides[0].shapes),'backup',backup)


