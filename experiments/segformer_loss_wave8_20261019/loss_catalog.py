"""Configurations of official losses. No new loss formulas."""
def ce(**kwargs):return dict(type='CrossEntropyLoss',**kwargs)
def dice():return dict(type='DiceLoss')
def lovasz(**kwargs):return dict(type='LovaszSoftmaxLoss',**kwargs)
def ohem():return dict(type='OhemCrossEntropyLoss',thresh=.7,min_kept=40000)
def focal(gamma):return dict(type='MultiClassFocalLoss',alpha=1.,gamma=gamma)
def mix(left,right,weight):return dict(type='MixedLoss',losses=[left,right],coef=[1.,weight])


def catalog():
    specs=[]
    def add(ident,label,loss,family,purpose,risk,role='main'):
        specs.append(dict(id=ident,label=label,loss=loss,family=family,purpose=purpose,risk=risk,role=role))
    for n,w in enumerate([.1,.25,.5,1.],1):
        add(f'L{n:02d}_CE_Dice',f'CE + {w} Dice',mix(ce(),dice(),w),'CE_Dice','区域重叠与小前景召回；单独比较Dice系数','可能扩张前景、降低Precision；Dice必须位于MixedLoss最后')
    for n,w in enumerate([.1,.25,.5],5):
        add(f'L{n:02d}_CE_Lovasz',f'CE + {w} Lovasz',mix(ce(),lovasz(),w),'CE_Lovasz','像素分类与IoU代理目标结合','排序和更大训练开销；可能牺牲细小边界')
    for n,weights in enumerate([[1.5,1.],[2.,1.],[1.,2.]],8):
        add(f'L{n:02d}_WeightedCE',f'CE weights {weights}',ce(weight=weights),'weighted_CE','前两组增加背景误检代价；第三组为前景召回反向对照','背景加权可能漏检，前景加权可能假阳性；类别顺序0背景/1裂缝')
    for n,k in enumerate([.5,.25],11):
        add(f'L{n:02d}_TopKCE',f'CE top {k}',ce(top_k_percent_pixels=k),'topK_CE','降低易背景像素对梯度的支配','可能过拟合标注噪声；topK是CE内部机制变体')
    add('L13_OHEM','OHEM CE',ohem(),'OHEM','优先低真类概率的困难像素','过强难例挖掘可能放大噪声；min_kept针对整个batch')
    for n,g in enumerate([1.,2.],14):
        add(f'L{n:02d}_SoftmaxFocal',f'Softmax Focal gamma {g}',focal(g),'softmax_focal','按困难程度降低易样本贡献','不同于双sigmoid FocalLoss；标量alpha不是前景权重')
    add('L16_CE_Focal','CE + 0.5 Softmax Focal',mix(ce(),focal(2.),.5),'CE_focal','保留整体CE并强调困难像素','复合目标改变梯度幅值；固定学习率并记录')
    add('L17_OHEM_Dice','OHEM + 0.25 Dice',mix(ohem(),dice(),.25),'OHEM_Dice','难像素和区域约束是否互补','误检/漏检方向可能冲突；Dice置后')
    add('L18_WeightedCE_Dice','Weighted CE [1.5,1] + 0.25 Dice',mix(ce(weight=[1.5,1.]),dice(),.25),'weighted_CE_Dice','背景惩罚和前景重叠的取舍','两项目标强度可能抵消；不改变图像或标签文件')
    add('L19_CE_DualTask','CE + 0.01 DualTask',mix(ce(),dict(type='DualTaskLoss',tau=.5),.01),'CE_boundary_gradient','官方梯度边界一致性，沿用同一分割头','额外随机采样/卷积训练开销；必须先验证对logits梯度有效')
    add('L20_CE_RMI','CE + 0.01 RMI',mix(ce(),dict(type='RMILoss',num_classes=2,rmi_radius=3,rmi_pool_way=0,rmi_pool_size=3,rmi_pool_stride=3),.01),'CE_RMI','局部邻域互信息关系约束','内部float64/矩阵逆开销和数值稳定性；RMI值可为负，不伪称纯FP32损失计算')
    return specs
