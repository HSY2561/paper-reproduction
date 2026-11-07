"""CPU-only W12 model checks; no dataset, GPU, training or metric selection."""
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src/PaddleSeg-2.10.0'
sys.path.insert(0, str(SRC))
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddleseg.models.segformer import SegFormer
from paddleseg.models.sfnet import AlignedModule
from paddleseg.models.segformer_simplify import (
    SegFormerSimplify, SimplifyAligned, fold_plain_state_to_native_sum)
from paddleseg.models.backbones.mix_transformer import MixVisionTransformer_B0
from paddleseg.models.backbones.mit_b0_s4ffn_controlled import MixVisionTransformer_B0_S4FFNControlled


def counts(model):
    registered = sum(math.prod(x.shape) for x in model.parameters())
    trainable = sum(math.prod(x.shape) for x in model.parameters() if not x.stop_gradient)
    return dict(registered=registered, trainable=trainable,
                nontrainable_statistics=registered - trainable)


def build(width=128, alignment='full', decoder_kind='concat', hidden=1024, layout='standard'):
    return SegFormerSimplify(2, MixVisionTransformer_B0_S4FFNControlled(hidden),
        embedding_dim=width, alignment=alignment, decoder_kind=decoder_kind,
        fusion_layout=layout)


def compare(a, b, shape, tolerance=5e-5):
    a.eval(); b.eval()
    inp = paddle.randn(shape)
    with paddle.no_grad():
        x, y = a(inp)[0], b(inp)[0]
    error = float(paddle.max(paddle.abs(x-y)))
    scale = max(float(paddle.max(paddle.abs(x))), 1.0)
    relative = error / scale
    if error > tolerance and relative > tolerance:
        raise AssertionError(dict(shape=shape, max_abs=error, relative=relative))
    return dict(shape=shape, max_abs=error, relative_to_max_abs_or_1=relative,
                atol=tolerance, rtol=tolerance, passed=True)


def main():
    started = time.time()
    paddle.set_device('cpu')
    paddle.seed(42012)
    np.random.seed(42012)
    matrix = [
        ('D01',96,'none','concat',1024), ('D02',128,'none','concat',1024),
        ('D03',160,'none','concat',1024), ('D04',128,'flow32','concat',1024),
        ('D05',128,'flow16','concat',1024), ('D06',128,'dwflow32','concat',1024),
        ('D07',128,'coarseflow32','concat',1024), ('D08',128,'none','native_sum',1024),
        ('D09',192,'none','native_sum',1024), ('D10',256,'none','native_sum',1024),
        ('C01',128,'full','concat',960), ('C02',128,'full','concat',896)]
    rows = []
    for ident,w,a,d,h in matrix:
        model = build(w,a,d,h)
        row = dict(id=ident,width=w,alignment=a,decoder_kind=d,hidden=h,counts=counts(model))
        model.eval()
        with paddle.no_grad():
            output = model(paddle.randn([1,3,400,400]))[0]
        assert list(output.shape)==[1,2,400,400] and bool(paddle.all(paddle.isfinite(output)))
        row['full400_eval_finite'] = True
        model.train()
        output = model(paddle.randn([2,3,65,63]))[0]
        loss = F.cross_entropy(output,paddle.randint(0,2,[2,65,63]),axis=1)
        loss.backward()
        missing,nonfinite=[],[]
        for name,value in model.named_parameters():
            if not value.stop_gradient:
                if value.grad is None: missing.append(name)
                elif not bool(paddle.all(paddle.isfinite(value.grad))): nonfinite.append(name)
        assert not missing and not nonfinite,dict(id=ident,missing=missing,nonfinite=nonfinite)
        row.update(finite_backward=True,unused_trainable_parameters=missing,
                   gradient_test_shape=[2,3,65,63])
        if d=='native_sum':
            assert not any(k.startswith(('linear_fuse.','balance_blocks.','fusion_post.')) for k in model.state_dict())
            assert row['counts']['registered']==3319392+522*w+2
        if d=='concat' and a=='none':
            assert row['counts']['registered']==3319392+4*w*w+522*w+2
        rows.append(row)
        print('CPU_MODEL',ident,row['counts'],flush=True)
        del model,output,loss

    original = SegFormer(2,MixVisionTransformer_B0(),128,balance_modules=['aligned'])
    standard = build()
    assert set(original.state_dict())==set(standard.state_dict())
    assert all(original.state_dict()[k].shape==standard.state_dict()[k].shape for k in original.state_dict())
    standard.set_state_dict(original.state_dict())
    base_exact = [compare(original,standard,s,0.0) for s in (
        [1,3,65,63],[1,3,400,400],[1,3,401,397],[1,3,481,479])]

    distributed = build(layout='distributed')
    distributed.set_state_dict(standard.state_dict())
    distributed_full = [compare(standard,distributed,s) for s in (
        [1,3,65,63],[1,3,400,400],[1,3,401,397],[1,3,481,479])]
    distributed.train()
    distributed(paddle.randn([2,3,65,63]))[0].square().mean().backward()
    assert distributed.linear_fuse._conv.weight.grad is not None
    assert bool(paddle.all(paddle.isfinite(distributed.linear_fuse._conv.weight.grad)))

    plain = build(alignment='none')
    plain_distributed = build(alignment='none',layout='distributed')
    plain_distributed.set_state_dict(plain.state_dict())
    distributed_none = [compare(plain,plain_distributed,s) for s in ([1,3,401,397],[1,3,481,479])]
    native = build(alignment='none',decoder_kind='native_sum')
    folded = fold_plain_state_to_native_sum(plain.state_dict(),128)
    assert set(folded)==set(native.state_dict())
    native.set_state_dict(folded)
    native_fold = [compare(plain,native,s) for s in (
        [1,3,65,63],[1,3,400,400],[1,3,401,397],[1,3,481,479])]

    backbone_shapes = {}
    for hidden in (1024,960,896,768):
        bb = MixVisionTransformer_B0_S4FFNControlled(hidden)
        assert bb.block4[0].mlp.fc1.weight.shape[1]==hidden
        assert bb.block4[1].mlp.fc2.weight.shape[0]==hidden
        backbone_shapes[str(hidden)] = dict(counts=counts(bb),output_channels=list(bb.feat_channels))
    inherited = AlignedModule(128,64)
    replacement = SimplifyAligned(128,'full')
    replacement.set_state_dict(inherited.state_dict())
    low,high=paddle.randn([2,128,17,19]),paddle.randn([2,128,3,4])
    with paddle.no_grad():
        exact = float(paddle.max(paddle.abs(inherited([low,high])-replacement([low,high]))))
    assert exact==0.0
    geometry=[]
    size=paddle.to_tensor([17,19],dtype='int64')
    for value in (0.0,0.5,-1.0):
        flow=paddle.full([2,2,17,19],value)
        with paddle.no_grad():
            original_warp=inherited.flow_warp(high,flow,size)
            replacement_warp=replacement.flow_warp(high,flow,size)
        assert bool(paddle.all(original_warp==replacement_warp))
        geometry.append(dict(constant_flow=value,original_coordinate_semantics_preserved=True))
    ones=paddle.ones([1,128,3,4]);zero=paddle.zeros([1,2,17,19])
    corner=float(replacement.flow_warp(ones,zero,size)[0,0,0,0])
    assert abs(corner-.25)<1e-6
    class ConstantPredictor(nn.Layer):
        def forward(self,x):return paddle.full([x.shape[0],2,x.shape[2],x.shape[3]],.75)
    coarse=SimplifyAligned(128,'coarseflow32');coarse.flow_make=ConstantPredictor()
    coarseflow=coarse.predict_flow(low,high,[8,10])
    assert list(coarseflow.shape)==[2,2,17,19]
    assert float(paddle.max(paddle.abs(coarseflow-.75)))<1e-6
    record=dict(at=dt.datetime.now().astimezone().isoformat(),passed=True,CPU_only=True,
        GPU_started=False,datasets_read=False,formal_training_started=False,
        source_root=str(SRC),paddle=paddle.__version__,elapsed_seconds=time.time()-started,
        candidates=rows,R0_state_keys_shapes_exact=True,R0_original_forward_bitwise=base_exact,
        distributed_full_checks=distributed_full,distributed_noalign_checks=distributed_none,
        distributed_fuse_gradient_used=True,native_fold_checks=native_fold,
        backbone_shapes=backbone_shapes,alignment_full_original_exact_error=exact,
        warp_coordinate_checks=geometry,zero_flow_ones_corner=corner,
        coarse_flow_constant_amplitude_preserved=True,
        limitations='CPU shape/gradient/numeric checks are not GPU throughput, full348 reproduction or scientific improvement evidence.')
    out=ROOT/'provenance/model_preflight_w12.json'
    out.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                         passed=True,elapsed_seconds=record['elapsed_seconds']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
