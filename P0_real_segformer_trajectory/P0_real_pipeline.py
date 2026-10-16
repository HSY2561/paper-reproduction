import os, sys, json, hashlib, math
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(r"D:\论文复现")
OUT = ROOT / "P0_real_segformer_trajectory"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / "experiments" / "retrain_wave4_20261007" / "src" / "PaddleSeg-2.10.0"
CONFIG = ROOT / "experiments" / "retrain_wave4_20261007" / "configs" / "public_segformer_b0.yml"
CHECKPOINT = ROOT / "experiments" / "retrain_wave4_20261007" / "checkpoints" / "segformer_b0" / "model.pdparams"
DATA_ROOT = Path(r"C:\Users\user\algolab_data")
IMAGE_REL = "valcrop/20160222_080850_1281_361.jpg"
IMAGE_PATH = DATA_ROOT / IMAGE_REL
sys.path.insert(0, str(SOURCE))

import paddle
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.core import infer
from paddleseg import utils
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.interpolate import splprep, splev
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def infer_one():
    paddle.set_device('cpu')
    cfg = Config(str(CONFIG))
    builder = SegBuilder(cfg)
    model = builder.model
    utils.utils.load_entire_model(model, str(CHECKPOINT))
    model.eval()
    data={'img':str(IMAGE_PATH), 'trans_info':[]}
    data=builder.val_dataset.transforms(data)
    trans_info=data['trans_info']
    im=data['img'][np.newaxis,...]
    im=paddle.to_tensor(im)
    with paddle.no_grad():
        pred, logits = infer.inference(model, im, trans_info=trans_info)
    pred=np.squeeze(pred.numpy()).astype(np.uint8)
    # ensure original image geometry
    rgb=np.asarray(Image.open(IMAGE_PATH).convert('RGB'))
    if pred.shape != rgb.shape[:2]:
        from PIL import Image as PILImage
        pred=np.asarray(PILImage.fromarray(pred).resize((rgb.shape[1],rgb.shape[0]), PILImage.Resampling.NEAREST))
    return rgb, pred

def skeleton_path(skel):
    lab=label(skel, connectivity=2)
    if lab.max()==0: return np.empty((0,2),float), lab
    # largest skeleton component
    vals=np.bincount(lab.ravel()); vals[0]=0; comp=int(vals.argmax())
    yy,xx=np.where(lab==comp)
    nodes=list(zip(yy.tolist(),xx.tolist()))
    G=nx.Graph(); G.add_nodes_from(nodes)
    node_set=set(nodes)
    for y,x in nodes:
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if not (dy or dx): continue
                q=(y+dy,x+dx)
                if q in node_set:
                    G.add_edge((y,x),q,weight=math.sqrt(2) if dy and dx else 1.0)
    ends=[n for n in G.nodes if G.degree(n)<=1]
    if len(ends)>=2:
        best=None
        for a in ends:
            dist=nx.single_source_dijkstra_path_length(G,a,weight='weight')
            for b,d in dist.items():
                if b in ends and (best is None or d>best[0]): best=(d,a,b)
        if best:
            path=nx.shortest_path(G,best[1],best[2],weight='weight')
        else: path=nodes
    else:
        # principal axis order for loops/branch clusters
        pts=np.asarray(nodes,float)
        c=pts.mean(0); _,_,vh=np.linalg.svd(pts-c,full_matrices=False); axis=vh[0]
        path=[tuple(p) for p in pts[np.argsort((pts-c)@axis)]]
    arr=np.asarray([(x,y) for y,x in path],float)
    # remove duplicated consecutive points and smooth with spline when possible
    if len(arr)>=5:
        d=np.sqrt(((np.diff(arr,axis=0))**2).sum(1)); keep=np.r_[True,d>0]
        arr=arr[keep]
        if len(arr)>=5:
            s=max(1.0, 0.002*len(arr));
            try:
                tck,u=splprep([arr[:,0],arr[:,1]],s=s, k=min(3,len(arr)-1))
                uu=np.linspace(0,1,max(120,min(600,len(arr)*3)))
                arr=np.column_stack(splev(uu,tck))
            except Exception:
                arr=np.column_stack([gaussian_filter1d(arr[:,0],2),gaussian_filter1d(arr[:,1],2)])
    return arr, lab

def save_png(arr,path): Image.fromarray(arr).save(path)

def main():
    rgb,pred=infer_one()
    mask=(pred==1)
    # Restrict tiny isolated components from the displayed crack map only; preserve raw mask separately.
    lab=label(mask,connectivity=2); vals=np.bincount(lab.ravel()); clean=mask.copy()
    if len(vals)>1:
        clean=np.zeros_like(mask)
        for i,n in enumerate(vals[1:],1):
            if n>=8: clean[lab==i]=1
    skel=skeletonize(clean)
    traj,lab_s=skeleton_path(skel)
    dist=distance_transform_edt(clean)
    width_px=2*dist
    # We do not invent calibration. This explicit identity transform is a visualization placeholder.
    # The metadata and figure label state that measured camera calibration is still required.
    sx=sy=1.0; ox=oy=0.0
    if len(traj):
        xpix,ypix=traj[:,0],traj[:,1]
        xmm=(xpix-ox)*sx; ymm=(ypix-oy)*sy
    else:
        xmm=ymm=np.array([])
    stem='20160222_080850_1281_361'
    save_png((mask*255).astype(np.uint8),OUT/f'{stem}_segformer_mask_raw.png')
    save_png((clean*255).astype(np.uint8),OUT/f'{stem}_segformer_mask_clean.png')
    save_png((skel*255).astype(np.uint8),OUT/f'{stem}_skeleton.png')
    np.save(OUT/f'{stem}_width_px.npy',width_px.astype(np.float32))
    if len(traj):
        np.savetxt(OUT/f'{stem}_trajectory_pixels.csv',np.column_stack([xpix,ypix]),delimiter=',',header='x_px,y_px',comments='')
        np.savetxt(OUT/f'{stem}_trajectory_coordinates_provisional_mm.csv',np.column_stack([xmm,ymm]),delimiter=',',header='x_mm,y_mm',comments='')
    # Figure
    fig=plt.figure(figsize=(18,11),dpi=220,facecolor='white',constrained_layout=True)
    gs=fig.add_gridspec(2,3,wspace=0.18,hspace=0.34)
    ax=fig.add_subplot(gs[0,0]); ax.imshow(rgb); ax.set_title('(a) Original image',fontsize=14,fontweight='bold'); ax.axis('off')
    ax=fig.add_subplot(gs[0,1]); ax.imshow(rgb); ax.imshow(mask,cmap='Reds',alpha=.58,vmin=0,vmax=1); ax.set_title('(b) Real SegFormer-B0 mask',fontsize=14,fontweight='bold'); ax.axis('off')
    ax=fig.add_subplot(gs[0,2]); ax.imshow(rgb); yy,xx=np.where(skel); ax.scatter(xx,yy,s=1.1,c='#00e5ff',linewidths=0); ax.set_title('(c) Crack skeleton (skeletonize)',fontsize=14,fontweight='bold'); ax.axis('off')
    ax=fig.add_subplot(gs[1,0]);
    cmap=LinearSegmentedColormap.from_list('crackwidth',['black','#165dff','#00d4a8','#ffe600','#ff3b30'])
    show=np.ma.masked_where(~clean,width_px); ax.imshow(rgb,alpha=.3); im=ax.imshow(show,cmap=cmap,vmin=0,vmax=max(1,float(width_px.max()))); fig.colorbar(im,ax=ax,fraction=.035,pad=.02,shrink=.78,label='Local width / px'); ax.set_title('(d) Crack map (mask + width)',fontsize=14,fontweight='bold'); ax.axis('off')
    ax=fig.add_subplot(gs[1,1]);
    if len(traj):
        ax.plot(xmm,ymm,color='#1565c0',lw=1.8); ax.scatter(xmm[::max(1,len(xmm)//15)],ymm[::max(1,len(ymm)//15)],s=12,c='#e53935'); ax.invert_yaxis(); ax.set_aspect('equal','box')
        ax.set_xlabel('X / mm (placeholder scale: 1 px = 1 mm)'); ax.set_ylabel('Y / mm (placeholder scale: 1 px = 1 mm)')
        ax.text(.02,.97,'CALIBRATION PLACEHOLDER: replace with measured camera-plane mapping',transform=ax.transAxes,va='top',fontsize=9,color='#b71c1c')
    else: ax.text(.5,.5,'No usable skeleton',ha='center',va='center')
    ax.grid(alpha=.25); ax.set_title('(e) Physical path (calibration placeholder)',fontsize=12,fontweight='bold')
    ax=fig.add_subplot(gs[1,2]); ax.imshow(rgb); 
    if len(traj):
        ax.plot(traj[:,0],traj[:,1],color='#00ff75',lw=2.6); ax.scatter(traj[0,0],traj[0,1],s=42,c='#1976d2',label='start',zorder=4); ax.scatter(traj[-1,0],traj[-1,1],s=42,c='#d32f2f',label='end',zorder=4); ax.legend(loc='lower right',fontsize=9)
    ax.set_title('(f) Smoothed skeleton trajectory (offline)',fontsize=14,fontweight='bold'); ax.axis('off')
    fig.suptitle('P0: Real SegFormer mask to crack skeleton, coordinates, and repair path',fontsize=18,fontweight='bold',y=.985)
    fig.text(.5,.008,'Input: Crack500 valcrop | Model: frozen SegFormer-B0 | No simulated mask. Coordinate scale is a placeholder and must be replaced by measured camera-plane calibration before robot execution.',ha='center',fontsize=10,color='#555')
    fig.savefig(OUT/'P0_real_segformer_skeleton_trajectory.png',bbox_inches='tight')
    fig.savefig(OUT/'P0_real_segformer_skeleton_trajectory.pdf',bbox_inches='tight')
    plt.close(fig)
    meta={
      'status':'real_segformer_mask_p0_completed_with_calibration_placeholder',
      'simulated_mask':False,
      'dataset':'Crack500 validation', 'image':str(IMAGE_PATH), 'image_relative':IMAGE_REL,
      'config':str(CONFIG), 'checkpoint':str(CHECKPOINT), 'checkpoint_sha256':sha256(CHECKPOINT),
      'raw_mask_pixels':int(mask.sum()), 'clean_mask_pixels':int(clean.sum()), 'skeleton_pixels':int(skel.sum()),
      'trajectory_points':int(len(traj)), 'width_max_px':float(width_px.max()),
      'postprocess':{'skeleton':'skimage.morphology.skeletonize','trajectory':'largest_skeleton_component_endpoint_path_then_spline','crack_map':'binary_mask_and_distance_transform_width'},
      'coordinate_transform':{'type':'identity_placeholder','sx_mm_per_px':sx,'sy_mm_per_px':sy,'origin_px':[ox,oy],'status':'UNVERIFIED_REPLACE_WITH_MEASURED_CAMERA_PLANE_CALIBRATION'},
      'test_used':False,
      'outputs':[str(OUT/f'{stem}_segformer_mask_raw.png'),str(OUT/f'{stem}_skeleton.png'),str(OUT/'P0_real_segformer_skeleton_trajectory.png')]
    }
    (OUT/'P0_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=='__main__': main()



