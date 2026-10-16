from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
ROOT=Path(r'D:\论文复现\P0_real_segformer_trajectory')
OUT=ROOT/'ppt_panels'; OUT.mkdir(exist_ok=True)
img=np.asarray(Image.open(r'C:\Users\user\algolab_data\valcrop\20160222_080850_1281_361.jpg').convert('RGB'))
mask=np.asarray(Image.open(ROOT/'20160222_080850_1281_361_segformer_mask_raw.png'))>0
clean=np.asarray(Image.open(ROOT/'20160222_080850_1281_361_segformer_mask_clean.png'))>0
skel=np.asarray(Image.open(ROOT/'20160222_080850_1281_361_skeleton.png'))>0
width=np.load(ROOT/'20160222_080850_1281_361_width_px.npy')
traj=np.loadtxt(ROOT/'20160222_080850_1281_361_trajectory_pixels.csv',delimiter=',',skiprows=1)

def save(name,title,draw):
    fig,ax=plt.subplots(figsize=(8,4.5),dpi=200)
    draw(ax); ax.set_title(title,fontsize=17,fontweight='bold',pad=10); fig.tight_layout(pad=0.8); fig.savefig(OUT/name,bbox_inches='tight',facecolor='white'); plt.close(fig)

def base(ax): ax.imshow(img); ax.axis('off')
save('01_original.png','(a) Original image',lambda ax: (ax.imshow(img),ax.axis('off')))
def maskdraw(ax): ax.imshow(img); ax.imshow(mask,cmap='Reds',alpha=.58,vmin=0,vmax=1); ax.axis('off')
save('02_real_mask.png','(b) Real SegFormer-B0 mask',maskdraw)
def skdraw(ax): ax.imshow(img); y,x=np.where(skel); ax.scatter(x,y,s=1.2,c='#00e5ff',linewidths=0); ax.axis('off')
save('03_skeleton.png','(c) Crack skeleton',skdraw)
def crackdraw(ax):
    cmap=LinearSegmentedColormap.from_list('crackwidth',['black','#165dff','#00d4a8','#ffe600','#ff3b30'])
    ax.imshow(img,alpha=.28); show=np.ma.masked_where(~clean,width); im=ax.imshow(show,cmap=cmap,vmin=0,vmax=max(1,float(width.max()))); plt.colorbar(im,ax=ax,fraction=.046,pad=.03,label='Width / px'); ax.axis('off')
save('04_crack_map.png','(d) Crack map and width',crackdraw)
def coorddraw(ax):
    ax.plot(traj[:,0],traj[:,1],color='#1565c0',lw=2); ax.scatter(traj[::max(1,len(traj)//16),0],traj[::max(1,len(traj)//16),1],s=15,c='#e53935'); ax.invert_yaxis(); ax.set_aspect('equal','box'); ax.set_xlabel('X / mm (placeholder 1 px = 1 mm)'); ax.set_ylabel('Y / mm'); ax.grid(alpha=.25); ax.text(.02,.96,'Calibration placeholder: replace with measured mapping',transform=ax.transAxes,va='top',fontsize=9,color='#b71c1c')
save('05_coordinates.png','(e) Physical-coordinate path',coorddraw)
def trajdraw(ax):
    ax.imshow(img); ax.plot(traj[:,0],traj[:,1],color='#00ff75',lw=2.5); ax.scatter(traj[0,0],traj[0,1],s=48,c='#1976d2',label='start',zorder=4); ax.scatter(traj[-1,0],traj[-1,1],s=48,c='#d32f2f',label='end',zorder=4); ax.legend(loc='lower right',fontsize=9); ax.axis('off')
save('06_trajectory.png','(f) Smoothed skeleton trajectory',trajdraw)
print('\n'.join(str(p) for p in sorted(OUT.glob('*.png'))))
