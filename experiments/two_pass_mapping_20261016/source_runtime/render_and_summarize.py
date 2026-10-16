"""Render actual MATLAB output. No planning or segmentation algorithms here."""
from pathlib import Path
import json
import hashlib
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = scipy.io.loadmat(root/'scc_result.mat')
route = data['PathEdge']
nodes, edges = data['node'], data['edgeList'].astype(int)-1
np.savetxt(root/'scc_path_px.csv', route, delimiter=',', header='x_author_map_px,y_author_map_px', comments='')
np.savetxt(root/'scc_nodes_row_col.csv', nodes, delimiter=',', header='row_px,col_px', comments='')
np.savetxt(root/'scc_edges_1based.csv', edges+1, delimiter=',', fmt='%d', header='node_1,node_2', comments='')
img=plt.imread(root/'author_software'/'Crack Maps'/'Uniform'/'myCrack8_100_1.png')
fig,axes=plt.subplots(1,2,figsize=(13,6.3),layout='constrained')
for ax in axes:
    ax.imshow(img)
    ax.set(xlim=(0,img.shape[1]),ylim=(img.shape[0],0),xlabel='x (author map pixels)',ylabel='y (author map pixels)')
axes[0].set_title('Official artificial input map')
axes[1].plot(route[:,0],route[:,1],color='#de6719',lw=1.3,label='SCC base coverage route')
axes[1].scatter(route[0,0],route[0,1],color='#159f43',s=55,zorder=4,label='Start')
axes[1].scatter(route[-1,0],route[-1,1],color='#306bc0',marker='s',s=45,zorder=4,label='End')
axes[1].set_title('Official SCC output: 158 path points')
axes[1].legend(loc='lower right',fontsize=8)
fig.suptitle('Official SCC reproduction (MATLAB R2025b)\nSimulation on known map; not our robot or a stitched camera map',fontsize=13)
fig.savefig(root/'scc_official_simulation.png',dpi=180)
fig.savefig(root/'scc_official_simulation.svg')
plt.close(fig)

summary={
    'upstream':'https://github.com/Binghamton-ACSR-Lab/Crack-Filling-Robot',
    'upstream_commit':'d044d083fd9fb21296f9b8fb5d55c6b42145bf92',
    'source_match':{'verified_files':36,'matching_files':36,'note':'README and every Software blob match official main byte-for-byte before isolated visualization flag change'},
    'scc':{'exit_code':0,'stderr_empty':True,'points':len(route),'nodes':len(nodes),'edges':len(edges),'path_length_px':float(data['pathLength'].item()),'author_res':data['res'].tolist()[0],'author_res_labels':['areaCover1_percent','density','reported_compute_seconds','path_ft','coverage_area_sqft'],'author_scale_mm_per_pixel':2,'scale_is_user_calibration':False},
    'boustrophedon':json.loads((root/'boustrophedon_metrics.json').read_text(encoding='utf-8')),
    'segformer_interface':{'exit_code':0,'stderr_empty':True,'shape':[360,640],'foreground_pixels':13354,'raw_skeleton_pixels':526,'after_original_spur44_pixels':429,'removed_skeleton_pixels':97,'removed_fraction':97/526,'nodes':2,'edges':1,'source_prediction':'D:/论文复现/P0_real_segformer_trajectory/20160222_080850_1281_361_segformer_mask_raw.png','scope':'Existing real predicted mask reused; this run does not perform new neural-network inference or calibration'},
    'limits':['SCC is known-map simulation; oSCC full runtime not executed in this task','Boustrophedon is only a rectangle module and not a differential-drive controller','No pose estimation, mosaic fusion, live camera, hardware driver, repair execution, or new SegFormer inference was executed','The 44 px pruning and 2 mm/px author convention require replacing with calibrated metric parameters before deployment','No LICENSE was listed in upstream root; public source availability is not an unrestricted license grant']
}
(root/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
manifest=[]
for p in root.rglob('*'):
    if p.is_file() and p.name != 'artifact_sha256.json':
        manifest.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(root/'artifact_sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
