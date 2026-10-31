"""Ten one-at-a-time context/detail candidates on SegFormer-B0+Aligned.

The four projected features remain 256 channels and Aligned is retained. A
common 1024->128 bottleneck makes every candidate comparable; each candidate
then returns 128 channels before the common 128->256 expansion.
"""
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.models.segformer_simplify import SegFormerSimplify


class MSCA(nn.Layer):
    def __init__(self,c):
        super().__init__(); self.bs=nn.LayerList([nn.Sequential(nn.Conv2D(c,c,(1,k),padding=(0,k//2),groups=c,bias_attr=False),nn.Conv2D(c,c,(k,1),padding=(k//2,0),groups=c,bias_attr=False)) for k in (7,11,21)]); self.p=nn.Conv2D(c*3,c,1,bias_attr=False); self.n=nn.BatchNorm2D(c); self.a=nn.ReLU()
    def forward(self,x): return self.a(self.n(self.p(paddle.concat([b(x) for b in self.bs],1))))+x

class LSKA(nn.Layer):
    def __init__(self,c):
        super().__init__(); self.a=nn.Conv2D(c,c,(1,5),padding=(0,2),groups=c,bias_attr=False); self.b=nn.Conv2D(c,c,(5,1),padding=(2,0),groups=c,bias_attr=False); self.d=nn.Conv2D(c,c,7,padding=6,dilation=2,groups=c,bias_attr=False); self.p=nn.Conv2D(c,c,1,bias_attr=False); self.n=nn.BatchNorm2D(c); self.r=nn.ReLU()
    def forward(self,x): return x+self.r(self.n(self.p(self.a(self.b(x))+self.d(x))))

class DWR(nn.Layer):
    def __init__(self,c):
        super().__init__(); self.br=nn.LayerList([nn.Conv2D(c,c,3,padding=d,dilation=d,groups=c,bias_attr=False) for d in (1,2,3)]); self.p=nn.Conv2D(c*3,c,1,bias_attr=False); self.n=nn.BatchNorm2D(c); self.r=nn.ReLU()
    def forward(self,x): return x+self.r(self.n(self.p(paddle.concat([b(x) for b in self.br],1))))

class InceptionDWConv(nn.Layer):
    def __init__(self,c):
        super().__init__(); q=c//8; self.q=q; self.d3=nn.Conv2D(q,q,3,padding=1,groups=q,bias_attr=False); self.h=nn.Conv2D(q,q,(1,11),padding=(0,5),groups=q,bias_attr=False); self.v=nn.Conv2D(q,q,(11,1),padding=(5,0),groups=q,bias_attr=False); self.p=nn.Conv2D(c,c,1,bias_attr=False); self.n=nn.BatchNorm2D(c); self.r=nn.ReLU()
    def forward(self,x):
        q=self.q; y=paddle.concat([x[:,:-3*q],self.d3(x[:,-3*q:-2*q]),self.h(x[:,-2*q:-q]),self.v(x[:,-q:])],1); return x+self.r(self.n(self.p(y)))

class CG(nn.Layer):
    def __init__(self,c):
        super().__init__(); self.l=nn.Conv2D(c,c,3,padding=1,groups=c,bias_attr=False); self.d=nn.Conv2D(c,c,3,padding=2,dilation=2,groups=c,bias_attr=False); self.g=nn.Conv2D(c,c,1,bias_attr=False); self.n=nn.BatchNorm2D(c); self.r=nn.ReLU()
    def forward(self,x): return x+self.r(self.n(self.l(x)+self.d(x)+self.g(F.adaptive_avg_pool2d(x,1))))

class CAA(nn.Layer):
    def __init__(self,c):
        super().__init__(); h=max(8,c//4); self.pool=nn.AvgPool2D(7,stride=1,padding=3); self.f=nn.Conv2D(c,h,1,bias_attr=False); self.h=nn.Conv2D(h,h,(1,11),padding=(0,5),groups=h,bias_attr=False); self.v=nn.Conv2D(h,h,(11,1),padding=(5,0),groups=h,bias_attr=False); self.o=nn.Conv2D(h,c,1,bias_attr=False); self.s=nn.Sigmoid()
    def forward(self,x): return x*(1+self.s(self.o(self.v(self.h(self.f(self.pool(x)))))))

class SAFM(nn.Layer):
    def __init__(self,c):
        super().__init__(); self.c=c; self.dw=nn.LayerList([nn.Conv2D(c//4,c//4,3,padding=1,groups=c//4,bias_attr=False) for _ in range(3)]); self.mix=nn.Conv2D(c,c,1,bias_attr=False); self.n=nn.BatchNorm2D(c); self.r=nn.ReLU()
    def forward(self,x):
        q=c=self.c//4; parts=[x[:,:q]]
        for i,scale in enumerate((2,4,8),1):
            z=F.adaptive_avg_pool2d(x[:,i*q:(i+1)*q],[max(1,x.shape[2]//scale),max(1,x.shape[3]//scale)]); parts.append(F.interpolate(self.dw[i-1](z),size=x.shape[2:],mode='bilinear',align_corners=False))
        return x+self.r(self.n(self.mix(paddle.concat(parts,1))))

class StripPooling(nn.Layer):
    def __init__(self,c):
        super().__init__(); self.p=nn.Conv2D(c*2,c,1,bias_attr=False); self.n=nn.BatchNorm2D(c); self.r=nn.ReLU()
    def forward(self,x):
        h=F.adaptive_avg_pool2d(x,[x.shape[2],1]); v=F.adaptive_avg_pool2d(x,[1,x.shape[3]]); return x+self.r(self.n(self.p(paddle.concat([h.expand_as(x),v.expand_as(x)],1))))

class CoordAtt(nn.Layer):
    def __init__(self,c):
        super().__init__(); h=max(8,c//32); self.f=nn.Conv2D(c,h,1,bias_attr=False); self.h=nn.Conv2D(h,h,1,bias_attr=False); self.v=nn.Conv2D(h,h,1,bias_attr=False); self.oh=nn.Conv2D(h,c,1); self.ov=nn.Conv2D(h,c,1); self.s=nn.Sigmoid()
    def forward(self,x):
        ah=F.adaptive_avg_pool2d(x,[x.shape[2],1]); av=F.adaptive_avg_pool2d(x,[1,x.shape[3]]); z=self.f(paddle.concat([ah,av.transpose([0,1,3,2])],2)); h,w=x.shape[2],x.shape[3]; z1,z2=paddle.split(z,[h,w],2); return x*self.s(self.oh(self.h(z1))) * self.s(self.ov(self.v(z2.transpose([0,1,3,2]))))

class WTConv(nn.Layer):
    def __init__(self,c):
        super().__init__(); self.low=nn.AvgPool2D(2,2); self.up=nn.Conv2D(c,c,1,bias_attr=False); self.high=nn.Conv2D(c,c,5,padding=2,groups=c,bias_attr=False); self.p=nn.Conv2D(c,c,1,bias_attr=False); self.n=nn.BatchNorm2D(c); self.r=nn.ReLU()
    def forward(self,x): return x+self.r(self.n(self.p(self.high(x)+F.interpolate(self.up(self.low(x)),size=x.shape[2:],mode='bilinear',align_corners=False))))

MODULES={'msca':MSCA,'lska':LSKA,'dwr':DWR,'inception':InceptionDWConv,'cg':CG,'caa':CAA,'safm':SAFM,'strip':StripPooling,'coord':CoordAtt,'wtconv':WTConv}

@manager.MODELS.add_component
class SegFormerTen(SegFormerSimplify):
    def __init__(self,num_classes,backbone,embedding_dim=256,align_corners=False,pretrained=None,candidate='msca',alignment='full',decoder_kind='concat',fusion_layout='standard'):
        if candidate not in MODULES: raise ValueError('Unknown W14 candidate '+candidate)
        if decoder_kind != 'concat' or fusion_layout != 'standard':
            raise ValueError('W17 SegFormerTen uses concat/standard fusion')
        if alignment not in ('none', 'full'):
            raise ValueError('W17 supports alignment none or full')
        super().__init__(num_classes=num_classes,backbone=backbone,embedding_dim=embedding_dim,align_corners=align_corners,pretrained=None,decoder_kind='concat',alignment=alignment,fusion_layout='standard')
        self._sub_layers.pop('linear_fuse',None)
        self.reduce=nn.Conv2D(embedding_dim*4,128,1,bias_attr=False)
        self.candidate=candidate; self.alignment_kind=alignment; self.module=MODULES[candidate](128); self.expand=nn.Conv2D(128,embedding_dim,1,bias_attr=False)
        self.bn=nn.BatchNorm2D(embedding_dim); self.act=nn.ReLU()
        if pretrained is not None: raise ValueError('W14 uses backbone ImageNet initialization only')
    def forward(self,x):
        raw=self.backbone(x); p1,p2,p3,p4=self._native_projections(raw); high=(self.balance_blocks['aligned']([p1,p4]) if self.alignment_kind != 'none' else F.interpolate(p4,size=p1.shape[2:],mode='bilinear',align_corners=self.align_corners)); z=paddle.concat([high,F.interpolate(p3,size=p1.shape[2:],mode='bilinear',align_corners=self.align_corners),F.interpolate(p2,size=p1.shape[2:],mode='bilinear',align_corners=self.align_corners),p1],1); fused=self.act(self.bn(self.expand(self.module(self.reduce(z))))); return [F.interpolate(self.linear_pred(self.dropout(fused)),size=x.shape[2:],mode='bilinear',align_corners=self.align_corners)]


