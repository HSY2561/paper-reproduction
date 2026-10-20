"""Source-grounded mechanism descriptions and qualified follow-up directions."""

# These describe design properties and hypotheses, not measured causal effects.
# Observed deltas are injected from frozen result files by the report builder.
MODULES = {
 'aligned': ('SFNet AlignedModule', 'C1与C4投影特征之间，替换C4普通上采样',
  '用学习到的采样偏移对齐深浅层特征，针对跨尺度位置错位。',
  '新增参数少；在压缩解码器上恢复部分区域质量，适合作为已有研究母体。',
  'grid_sample与偏移预测有真实算子成本；极小裂缝在深层已丢失时，单靠对齐无法恢复。',
  '优先检查已有逐图漏检及恢复对齐，再比较更简洁的融合替代；若未来实验改变采样，必须单独登记并验证奇数尺寸，不把当前小涨点解释为所有边界已改善。'),
 'se': ('MobileNetV3 SE', '融合后128通道',
  '全局池化后产生通道权重，抑制无用通道。',
  '参数成本很低；可以利用全局通道统计，联合Aligned时存在一定单seed边界收益。',
  '全局统计可能把极少像素的细裂缝抑制掉；单模块未产生区域增益，池化也有后端兼容风险。',
  '当前保留低成本负结果；若有新诊断再考虑残差式使用或接入位置控制，不以改reduction系数凑独立模块。'),
 'polarized': ('Polarized Self Attention', '多尺度融合后',
  '分别保留空间或通道维信息进行重标定，服务于像素定位与通道判别。',
  '与Aligned联合后比独立加入更有价值，参数可维持原B0的95%以下；三个种子对原B0区域指标均有改善。',
  '第二模块对A0平均贡献不足原门槛；实际候选延迟更高，Precision均值有损失，不能当作最终通过模型。',
  '作为条件有效但门槛未过的消融保留；当前优先完成固定A0损失路线。以后如评估更轻的替换，应登记新假设，不能通过挑seed或降低0.2pp门槛放行。'),
 'gc': ('Global Context Block', '原生C4，MLP投影前',
  '用紧凑全局上下文向量调节深层语义。',
  '相对轻量；在W128上Precision与BF1有收益，说明可改变前景/背景取舍。',
  '区域IoU和F1下降；同一全局向量对不同位置作用有限，可能抑制小前景。',
  '不继续重复叠全局注意力；只有错误分组明确需要全局语义时再考虑，需保留Recall和小前景分组。'),
 'cca': ('Criss Cross Attention', '原生C4',
  '沿行列建立长距离位置关系，降低完整空间两两关系的开销。',
  'Aligned组合的Recall和区域指标明显上涨，可作为连通性方向的探索证据。',
  'Precision和BF1退化，可能把纹理也连成前景；当前没有骨架连续性实测证明长裂缝问题已解决。',
  '低优先级保留；未来如研究连通性，先用预测掩码测断点/骨架覆盖，而不是继续用IoU推断连通性。'),
 'pam': ('Position Attention Module', '原生C4',
  '计算位置相似度并聚合远处信息。',
  '当前Aligned组合seed42的裂缝IoU最高，是区域召回方向的强信号。',
  '明显Precision与BF1代价，未过完整门槛；位置相似并不保证是同一真实裂缝。',
  '不能按最高IoU替代A21；先诊断误检集中位置。只有新证据支持时才登记受约束的融合替代，不同时改loss和结构掩盖原因。'),
 'cam': ('Channel Attention Module', '原生C4',
  '聚合通道之间的关系，使用很少可学习参数。',
  '参数增量几乎可忽略；相对其他上下文模块有较好的参数成本。',
  '独立结果几乎无变化，Aligned组合边界与Precision也可能下降；少参数不代表矩阵乘无成本。',
  '保留为低参数控制，不以微小单seed上涨作为模块贡献；未来优先简化融合计算而非再叠同类通道块。'),
 'nonlocal': ('NonLocal2D', '原生C4',
  '利用全局成对相似度建模长距离上下文。',
  '独立时Recall提高，联合时区域也有部分收益。',
  'Precision/BF1取舍明显，位置关系与主干已有注意力可能重复；导出和内存访问也有开销。',
  '当前不作为优先方向；将其作为“强全局上下文不自动改善细裂缝”的负对照。'),
 'msca': ('MSCAN SpatialAttention', '融合后高分辨率128通道',
  '多尺度深度条带卷积提供不同方向与范围的局部上下文。',
  '与裂缝细长形状匹配；联合Aligned的BF1有正向变化。',
  '独立结果区域指标下降；融合后分辨率高，实际访存和延迟可能大于参数暗示。',
  '若后续修改，先定位细端点与宽度分组是否获益；可考虑低分辨率替换式接入，但那是新的单变量实验，不把未运行配置写成已有效。'),
 'encoding': ('EncModule', '原生C4，只取特征输出',
  '以可学习码本编码深层残差统计。',
  '可表达全局材质/纹理统计，给复杂背景提供另一种表示。',
  '参数和码本激活有成本；与Aligned联合时区域与边界收益不足，存在材质过拟合风险。',
  '当前保留负结果；如要证明背景抑制，应先建立盲法纹理场景标注并做域外验证。'),
 'glore': ('GloRe GruModule', '原生C4',
  '将特征映射到关系节点空间推理，再投回原空间。',
  '参数增量有限；Aligned组合有很小的区域正收益。',
  '单独和联合收益不稳定或过小；图推理可能把重复纹理聚合，不能凭机制名声称连续性改善。',
  '不再优先追加同质图上下文；保留其与Aligned样本级互补的观察，未来需具体错误类型证据。'),
 'aspp_sep': ('Separable ASPP', 'C4投影后、Aligned前',
  '用多个空洞率的可分离卷积提取多尺度上下文，无全局池化分支。',
  '相对W128和A0均恢复部分区域质量；深度可分离降低普通ASPP卷积成本。',
  'C4仅13×13时有效采样有限；增加参数后可能超过严格减参预算，BF1未必同步提高。',
  '作为有区域收益但成本受限的备选保留；如未来更改尺度或接入C3，必须登记为位置/尺度消融，不能计作新机制。'),
 'psp_pool': ('PSP PPModule', 'C4投影后',
  '多种自适应池化尺度聚合上下文。',
  '理论上补充全局与区域尺度信息；当前没有本机合法性能结果。',
  '当前Windows/RTX5060反向smoke失败或暂缓；不能讨论其性能涨跌。',
  '仅做等价后端兼容或在支持环境验证；修复前性能列保持空白，不为了凑模块数量改结构冒充同一实验。'),
 'external_attention': ('ExternalAttention', '原生C4，沿官方残差模式',
  '通过较紧凑的外部注意表示聚合信息，使用RTFormer中的一个子模块。',
  '参数增量较低，Aligned组合区域指标提高。',
  'Precision下降使完整门槛未过；这不是RTFormer完整EABlock或整网复现。',
  '保留区域/成本参考；后续先看误检而不是直接增加外部记忆维度。'),
 'context_embedding': ('ContextEmbeddingBlock', '原生C4',
  '全局池化语义与局部卷积结合。',
  '具备低空间分辨率全局上下文路径。',
  '3×3卷积参数大，当前总参数超过原B0，区域精度还下降。',
  '当前停止投入，作为高成本准确率控制；不能用“轻量上下文”标签掩盖实际参数超标。'),
 'dnl': ('Disentangled NonLocal', '原生C4',
  '分离成对关系和一元上下文，改造普通NonLocal的关系表达。',
  '比普通NonLocal更有结构化的关系分解；BF1有小幅恢复。',
  '区域收益不足，仍含成对关系计算，不能解决深层已丢失的小前景。',
  '保留为NonLocal家族对照，不优先重复同类搜索。'),
 'ppcontext': ('PPContextModule', '投影C4、Aligned前',
  '用池化尺度上下文融合低分辨率特征。',
  '参数相对较少；已有等价插值兼容路径。',
  '早期与Aligned联合均有区域下降，说明工程可运行不等于任务有效。',
  '当前不再继续堆叠；优先分析输入细节和损失目标，而非重复加池化尺度。'),
 'dappm': ('DAPPM', '原生C4',
  '逐级聚合多尺度池化上下文。',
  '可以递进结合不同尺度；没有当前合法性能结论。',
  '实际反向smoke失败；13×13特征上的较大池化尺度也可能退化，但这一性能解释尚未验证。',
  '仅保留兼容证据；等价修复成功后才可谈精度。'),
 'apnb': ('APNB', '原生C4',
  '池化后的非局部注意上下文，降低关系空间规模。',
  'Aligned组合区域IoU涨幅较大。',
  'Precision或BF1约束不满足，参数较高；不能用区域排序覆盖完整门槛。',
  '保留高召回/区域备选，不事后放宽Precision下限；未来需具体的假阳性抑制证据。'),
 'afnb': ('AFNB', 'C3到C4、MLP投影前',
  '用低层和高层语义之间的注意融合补充跨层信息。',
  '与Aligned的C1/C4几何对齐位置不同，单seed有互补性，且已补齐单独AFNB四行归因。',
  '三seed对A0平均IoU退化，第二模块不稳定；参数只比B0少约1.57%，速度代价也较大。',
  '结束这一组合的主线投入，保留完整失败归因；不要挑seed42成绩作最终表。'),
 'jpu': ('Joint Pyramid Upsampling', 'C2/C3/C4联合，重写C2路径',
  '多层投影与空洞卷积融合，尝试在中间尺度恢复语义。',
  '可同时利用中深层多尺度信息。',
  '不是简单残差插入，改变了C2路径；计算与访存增加，当前收益很小。',
  '若未来重做解码器，可作为整体替换参考；当前不宜与多个融合块继续叠加。'),
 'dcm': ('Dynamic Convolution Module', '原生C4，单一3×3过滤配置',
  '依据输入生成动态滤波，增加上下文适应能力。',
  'Aligned组合Precision较稳，有小幅区域收益。',
  '参数成本高、动态分组卷积导出风险；不是完整DMNet头。',
  '只有同机稳定延迟和实际部署支持后才值得继续；当前保留成本较高的对照。'),
 'emau': ('EM Attention Unit', '原生C4，重建特征与残差',
  '通过迭代基向量分解/重建聚合上下文。',
  '部分实现训练参数增量很少，可分离表示容量与参数量。',
  '有迭代计算和可变基状态，参数少不等于便宜；早期及Aligned组合区域增益不足。',
  '不作为优先候选；统计FLOPs时必须说明迭代算子的漏计，冻结时要保留基状态。'),
 'attention_refinement': ('AttentionRefinement', 'C4投影后',
  '局部卷积加通道注意细化特征。',
  '在低分辨率上细化语义较直接。',
  '有额外卷积参数，单seed区域贡献较小，未过完整门槛。',
  '作为简洁注意细化对照保留，不继续叠第三个同类块。'),
 'uafm_channel': ('UAFM Channel Attention', 'C1投影与融合后特征之间',
  '用通道权重融合两路特征。',
  'Aligned组合有较大的区域提升，说明浅深信息融合可改变工作点。',
  '高分辨率卷积导致参数总量超过B0；BF1/Precision有代价，与Aligned存在路径重叠。',
  '未来更值得研究“替换现有融合”而非追加UAFM；必须控制通道和高分辨率算子，单独验证收益。'),
 'hamburger': ('Hamburger', '融合后128通道',
  '用矩阵分解近似全局信息交互，区别于更早的完整LightHamHead替换。',
  '参数较低，Aligned组合区域和BF1均有正向变化。',
  'Precision下降超过原母体门槛；迭代训练/推理有额外代价，不能据参数判断速度。',
  '保留为有边界/区域信号的非通过备选；若进一步研究应固定因果对照，不与早期LightHam整头结果混为同一个实验。'),
}

LOSS_FAMILIES = [
 ('CE + Dice','区域重叠约束，可能帮助小前景召回','可能扩大前景导致误检；原MixedLoss有标签原地修改梯度冲突','等待原配置兼容恢复的GPU证据，不能把CPU预检当完成'),
 ('CE + Lovasz','直接提供IoU代理目标，当前区域涨点最明显','Precision下降超门槛，部分配方BF1也下降；系数改变总梯度幅值','当前只记录候选取舍；先完成所有预登记方案，不按已看成绩追加新权重'),
 ('Weighted CE','明确控制背景与裂缝像素的惩罚比例','背景权重大幅损失Recall，前景权重大幅损失Precision/BF1','用于解释工作点取舍；不能将P或R最高单独视为最终优胜'),
 ('TopK CE','减少容易像素对训练的支配','可能强调标注噪声，过强筛选使分布偏移','按既有0.5/0.25配置完整跑完；用FP/FN与极小前景统计解释'),
 ('OHEM CE','按预测置信度挖掘困难像素','困难像素可能是模糊标签；需注意全batch min_kept定义','按完整门槛比较，不凭loss数值小判断好坏'),
 ('Softmax Focal / CE + Focal','按难度连续降低易像素权重','gamma改变梯度分布；alpha=1是全局标量而非前景权重','保持softmax多类实现一致，先做既定超参数消融'),
 ('CE + DualTask','训练期梯度/边界一致性','额外随机采样、卷积与训练成本；不一定改善区域','先看预检可微性，再同时审查BF1/IoU，不只看边界涨点'),
 ('CE + RMI','邻域关系互信息目标','矩阵逆与内部float64开销，数值不稳定风险，loss可为负','仅按有限梯度和正式验证判断，记录训练时间，不以负loss断定失败'),
]

def append_analysis(add,table,rows6,rows7,parent6,a0,baseline):
    add('## 附录A 每个模块的优点 代价和提升方向')
    add('本附录把设计作用与实际观察分开。机制说明依据冻结的PaddleSeg官方类及其原始论文；参数/涨点取自本机实验。优势是设计特征或本次观察，风险与原因是待验证解释，不当作已证明因果。未来方向均为建议，本报告不新增训练或改变当前固定A0损失计划。')
    analysis_rows=[]
    for key,(name,pos,mechanism,advantage,risk,nextstep) in MODULES.items():
        samples=[]
        for phase,rows,ref in [('单独接W128',rows6,parent6),('在A0上联合',rows7,a0)]:
            for r in rows:
                if key in r.get('modules',[]) and not (key=='aligned' and phase=='在A0上联合'):
                    if key=='aligned' and r.get('modules')!=['aligned']:continue
                    if r['status']=='complete':
                        observation=f"{phase} {r['id']}：ΔIoU={(r['crackIoU']-ref['crackIoU'])*100:+.4f}pp，ΔP={(r['precision']-ref['precision'])*100:+.4f}pp，ΔR={(r['recall']-ref['recall'])*100:+.4f}pp，ΔBF1={(r['boundary_f1']-ref['boundary_f1'])*100:+.4f}pp，参数{r['params']/1e6:.6f}M"
                    else: observation=f"{phase} {r['id']}：{r['status']}，无合法性能数字"
                    samples.append(observation)
        if key=='gc' and not samples: samples=['仅Wave6独立结果；未登记在Aligned母体重复GC']
        add(f'### {name}')
        add(f'接入位置：{pos}。设计作用：{mechanism}')
        add('实测观察：'+'；'.join(samples)+'。')
        add(f'优点：{advantage}\n\n代价和风险：{risk}\n\n提升方向：{nextstep}')
        analysis_rows.append(dict(module=key,name=name,position=pos,mechanism=mechanism,observation='；'.join(samples),advantage=advantage,risk=risk,recommendation=nextstep))
    add('### 历史整头与空间融合补充')
    add('早期UAFM_SpAtten扩大高分辨率空间融合，参数约4.897M，超过原B0且Crack500验证区域下降；它与当前UAFM_Channel不是同一个变体。LightHamHead替换整个解码头，参数较少但原始区域信息恢复不足，正确EMA选模后仍明显弱于B0；这与当前融合后单独Hamburger块不同。DualAttention整块包含空间和通道支路，其很小的涨点已由PAM/CAM拆解对照进一步解释，不能把拆分变体计成更多独立新算法。')
    add('### 损失函数各家族的优劣')
    add(table(['家族','预期优点','主要代价或风险','当前提升方向'],LOSS_FAMILIES))
    add('### 按任务目标排列的建议')
    add('第一优先级是完成当前固定A0的损失实验并坚持Precision/Recall/BF1约束，解决已观察到的前景扩张与漏检取舍。第二优先级是用完整预测和场景标注检验细前景、低对比度与纹理误检，而非只按全局IoU排名。第三优先级才是登记轻量解码/融合“替换”实验，减少重复的上采样、拼接和高分辨率卷积；原MiT骨干先保留共同预训练。若要明显减参或提速，应先对各阶段算子计时再决定是否改变主干，下结论前需稳定的配对延迟。')
    add('当前不建议继续无目的叠多个全局注意模块：已有GC/NonLocal/DNL/GloRe/Encoding等结果显示大量作用重复或Precision/Recall代价。也不建议为了达到目标分数修改划分、测试阈值或反复查看test。模型之外的来源互斥、标注质量与实际机器人任务误差，可能比再增加一个模块更决定论文可信度。')
    add('### 原始参考入口')
    add('SegFormer的分层编码与轻量解码设计见[NeurIPS原论文](https://papers.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html)。其余模块具体接口以[官方PaddleSeg release/2.10源码](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/paddleseg/models)及本地冻结源码为准。Aligned参考[Semantic Flow原论文](https://arxiv.org/abs/2002.10120)，Polarized参考[Polarized Self-Attention原论文](https://arxiv.org/abs/2107.00782)，Lovasz参考[Lovász-Softmax原论文](https://arxiv.org/abs/1705.08790)。引用只支持设计动机，不提供本项目性能数字。')
    return analysis_rows
