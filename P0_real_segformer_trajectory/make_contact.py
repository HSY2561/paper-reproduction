from PIL import Image,ImageDraw
from pathlib import Path
root=Path(r"C:\Users\user\algolab_data")
lines=(root/'val.txt').read_text().splitlines()[:24]
thumbs=[]
for i,l in enumerate(lines):
 p=root/l.split()[0]; im=Image.open(p).convert('RGB'); im.thumbnail((240,180)); canvas=Image.new('RGB',(260,215),'white'); canvas.paste(im,((260-im.width)//2,5)); d=ImageDraw.Draw(canvas); d.text((5,190),f'{i}: {p.name}',fill='black'); thumbs.append(canvas)
out=Image.new('RGB',(260*4,215*6),(220,220,220))
for i,im in enumerate(thumbs): out.paste(im,((i%4)*260,(i//4)*215))
out.save(Path(r'D:\论文复现\P0_real_segformer_trajectory')/'val_samples_contact.png')
