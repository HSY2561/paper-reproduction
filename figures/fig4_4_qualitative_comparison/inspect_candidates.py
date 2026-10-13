from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(r'C:\Users\user\algolab_data'); OUT=Path(__file__).resolve().parent
names=['20160407_165001_1_1081.jpg','20160307_162438_1921_1081.jpg','20160302_155857_641_721.jpg','20160328_153829_1_1.jpg','20160328_153829_1281_361.jpg','20160328_151057_1281_361.jpg','20160328_151244_1_721.jpg','20160328_152022_641_721.jpg','20160316_143527_1281_361.jpg','20160328_151244_1921_721.jpg','20160316_143445_1281_361.jpg','20160328_153629_1921_361.jpg']
tw,th=320,180; cols=4; rows=(len(names)+cols-1)//cols
sheet=Image.new('RGB',(cols*tw,rows*(th+30)),'white'); d=ImageDraw.Draw(sheet)
for i,n in enumerate(names):
    p=ROOT/'valcrop'/n; im=Image.open(p).convert('RGB'); im.thumbnail((tw,th)); x=i%cols*tw; y=i//cols*(th+30); sheet.paste(im,(x,y)); d.text((x+3,y+th+3),n[:-4],fill='black')
sheet.save(OUT/'candidate_inspect.jpg',quality=95)
