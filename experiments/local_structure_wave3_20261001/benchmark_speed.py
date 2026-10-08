import json, time, gc
import paddle
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.core import infer
from paddleseg.utils import utils

def sync():
    try: paddle.device.cuda.synchronize()
    except Exception: pass

models = [
 ('C0_BiSeNetV2', r'C:\local_wave3\configs\C4_test.yml', r'C:\local_c0clean\model.pdparams'),
 ('C4_BiSeNetV2_inputcrop', r'C:\local_wave3\configs\C4_test.yml', r'C:\local_wave3\checkpoints\C4\model.pdparams'),
 ('SegFormer_B0', r'C:\local_wave3\configs\public_segformer_b0_test.yml', r'C:\local_wave3\checkpoints\public_segformer_b0\model.pdparams'),
 ('OCRNet_HRNetW18', r'C:\local_wave3\configs\public_ocrnet_hrnetw18_test.yml', r'C:\local_wave3\checkpoints\public_ocrnet_hrnetw18\model.pdparams'),
]
out=[]
for name,cfg_path,ck in models:
    cfg=Config(cfg_path); builder=SegBuilder(cfg); model=builder.model; utils.load_entire_model(model,ck); model.eval(); ds=builder.val_dataset
    loader=paddle.io.DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,return_list=True)
    times=[]; n=0
    with paddle.no_grad():
        for data in loader:
            sync(); t0=time.perf_counter(); infer.inference(model,data['img'],trans_info=data['trans_info']); sync(); dt=time.perf_counter()-t0
            n+=1
            if n>20: times.append(dt)
            if n>=121: break
    avg=sum(times)/len(times); out.append({'model':name,'images_timed':len(times),'warmup':20,'avg_seconds_per_image':avg,'ms_per_image':avg*1000,'images_per_second':1/avg,'config':cfg_path,'checkpoint':ck})
    del model,loader,builder,ds; gc.collect();
    try: paddle.device.cuda.empty_cache()
    except Exception: pass
print(json.dumps(out,indent=2))
with open(r'C:\local_wave3\results\speed_benchmark_rtx5060.json','w') as f: json.dump(out,f,indent=2)
