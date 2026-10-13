from pathlib import Path
from PIL import Image, ImageDraw
import random
ROOT=Path(r'C:\Users\user\algolab_data'); OUT=Path(__file__).resolve().parent
pairs=[]
for line in (ROOT/'val.txt').read_text().splitlines():
    a,b=line.split(); pairs.append((ROOT/a,ROOT/b))
r=random.Random(42); r.shuffle(pairs); pairs=pairs[:72]
tw,th=180,120; cols=8; rows=(len(pairs)+cols-1)//cols
sheet=Image.new('RGB',(cols*tw,rows*(th+20)),'white'); d=ImageDraw.Draw(sheet)
for i,(ip,gp) in enumerate(pairs):
    im=Image.open(ip).convert('RGB'); im.thumbnail((tw,th)); x=(i%cols)*tw; y=(i//cols)*(th+20); sheet.paste(im,(x,y)); d.text((x+2,y+th+2),ip.stem[:24],fill='black')
sheet.save(OUT/'candidate_originals.jpg',quality=92)
