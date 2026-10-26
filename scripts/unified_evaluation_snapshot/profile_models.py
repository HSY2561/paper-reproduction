"""Optional dataset-free, same-process interleaved profiles on the final backend."""
import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import time

from stage6_common import *


def telemetry():
    queries = {
        'gpu': ['nvidia-smi','--query-gpu=name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used','--format=csv,noheader'],
        'compute_processes': ['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],
    }
    result = {'at':now(), 'linux_pid':os.getpid()}
    for key,cmd in queries.items():
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        result[key] = {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    return result


def guard():
    if (ROOT/'results/STOP_REQUEST.json').exists() or shutil.disk_usage(ROOT).free<4*2**30:
        raise RuntimeError('Profile stopped by data-disk/STOP guard; original logs retained')


def load(item):
    paddle,np,model,builder,cfg = model_and_dataset(item,'val')
    return paddle,np,model


def interleaved():
    verify_stage_release()
    data = manifest()
    items = [x for x in data['items'] if x.get('include_in_profile')]
    models = {}; metadata = {}; snapshots = [telemetry()]
    for item in items:
        paddle,np,model = load(item);models[item['id']] = model
        metadata[item['id']] = {'params':sum(int(np.prod(p.shape)) for p in model.parameters()),
                                'checkpoint_sha256':item['checkpoint_sha256']}
    paddle.seed(20261025)
    rng = random.Random(20261025);rounds = []
    with paddle.no_grad():
        for size in (400,480):
            x = paddle.randn([1,3,size,size],dtype='float32')
            for model in models.values():
                for _ in range(50):model(x)
            paddle.device.cuda.synchronize()
            for rep in range(8):
                order = list(models);rng.shuffle(order)
                for ident in order:
                    guard()
                    before = telemetry()
                    paddle.device.cuda.synchronize();start = time.perf_counter()
                    for _ in range(100):models[ident](x)
                    paddle.device.cuda.synchronize();ms = (time.perf_counter()-start)*10
                    rounds.append({'size':size,'round':rep,'model':ident,'ms':ms,'order':order,
                                   'telemetry_before':before,'telemetry_after':telemetry()})
                    print(ident,size,rep,ms,flush=True)
    summaries = []
    for size in (400,480):
        for ident in models:
            vals = [r['ms'] for r in rounds if r['size']==size and r['model']==ident]
            ratios = [next(r['ms'] for r in rounds if r['size']==size and r['round']==i and r['model']==ident)/
                      next(r['ms'] for r in rounds if r['size']==size and r['round']==i and r['model']=='SegFormer_B0') for i in range(8)]
            cv = statistics.stdev(ratios)/statistics.mean(ratios)
            summaries.append({'model':ident,'size':size,'median_ms':statistics.median(vals),
                              'sample_sd_ms':statistics.stdev(vals),'paired_ratio_B0_median':statistics.median(ratios),
                              'paired_ratio_B0_cv':cv,'stable_pair_cv_below_10pct':cv<=.10,**metadata[ident]})
    snapshots.append(telemetry())
    write(ROOT/'results/interleaved_profile.json',{'at':now(),'summary':summaries,'rounds':rounds,
          'telemetry':snapshots,'paddle':paddle.__version__,'hardware':paddle.device.cuda.get_device_name(),
          'protocol':'All models resident in same process; FP32 batch1; input400/480;50warmup/model/size;8random interleaved rounds of100 synchronized forwards; no IO/postprocess',
          'speed_claim':'Descriptive only. Inspect telemetry for competing processes, thermal/clock/power changes and paired CV before any speed claim.'},exclusive=True)


def one(ident):
    verify_stage_release()
    item = item_by_id(ident);before = telemetry()
    paddle,np,model = load(item)
    records = []
    with paddle.no_grad():
        for size in (400,480):
            guard()
            x = paddle.randn([1,3,size,size],dtype='float32')
            for _ in range(10):model(x)
            paddle.device.cuda.synchronize()
            paddle.device.cuda.empty_cache()
            for method in ['reset_max_memory_allocated','reset_max_memory_reserved']:
                if hasattr(paddle.device.cuda,method):getattr(paddle.device.cuda,method)()
            allocated0 = int(paddle.device.cuda.memory_allocated())
            reserved0 = int(paddle.device.cuda.memory_reserved()) if hasattr(paddle.device.cuda,'memory_reserved') else None
            for _ in range(10):model(x)
            paddle.device.cuda.synchronize()
            allocated_peak = int(paddle.device.cuda.max_memory_allocated())
            reserved_peak = int(paddle.device.cuda.max_memory_reserved()) if hasattr(paddle.device.cuda,'max_memory_reserved') else None
            records.append({'size':size,'baseline_allocated_bytes':allocated0,
                            'peak_allocated_bytes':allocated_peak,'increment_allocated_bytes':allocated_peak-allocated0,
                            'baseline_reserved_bytes':reserved0,'peak_reserved_bytes':reserved_peak})
    aligned = any('Aligned' in type(layer).__name__ for layer in model.sublayers())
    write(ROOT/'results'/(ident+'_memory_profile.json'),{'at':now(),'model':ident,'records':records,
          'params':sum(int(np.prod(p.shape)) for p in model.parameters()),'checkpoint_sha256':item['checkpoint_sha256'],
          'hardware':paddle.device.cuda.get_device_name(),'paddle':paddle.__version__,'telemetry_before':before,'telemetry_after':telemetry(),
          'protocol':'Fresh process/model; eval FP32 batch1; peak Paddle allocated and reserved after warmup, not total display/context memory',
          'FLOPs':{'status':'not_measured','grid_sample_present':aligned,
                   'total_supported':False,'reason':'No verified complete counter for grid_sample/interpolation/pooling/custom fusion. Do not present partial convolution counts as total FLOPs.'}},exclusive=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--interleaved',action='store_true');parser.add_argument('--one');args=parser.parse_args()
    if args.interleaved:interleaved()
    elif args.one:one(args.one)
    else:parser.error('Choose --interleaved or --one MODEL')
