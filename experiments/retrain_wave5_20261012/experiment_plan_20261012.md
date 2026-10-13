# Experiment Plan: SegFormer-B0 Crack500 Wave 5

## Problem and claims

- 长裂缝的连续性受到多尺度特征和长距离依赖不足的限制。
- 阴影、落叶、粗糙纹理和低对比度区域会产生假阳性。
- 机器人部署需要保持解码器的计算量和单图延迟可控。

主张只在统一 Crack500 协议下验证：官方 EMAU 是否改善上下文连续性、官方 DualAttentionModule 是否改善背景判别、官方 LightHamHead 是否降低解码成本并保持精度。

## Run matrix

| Run | Variant | Evidence | Gate |
|---|---|---|---|
| A | SegFormer-B0 + EMAU on C4 | 20-iter smoke, 18k train/val, global metrics and boundary F1 | pass precision/recall/F1/mIoU/boundary gate |
| B | SegFormer-B0 + DualAttention on C4 | same protocol | same gate |
| C | SegFormer-B0 + LightHamHead | same protocol, params/FLOPs/latency | same gate plus cost report |
| AB/AC/BC | only if two singles pass | one joint run | must exceed both singles |

## Fixed protocol

RTX5060, FP32, seed 42, batch 4, AdamW, lr 6e-5, PolynomialDecay end_lr=0, 18,000 iterations, 0.75–1.25 scaling, 400x400 crop, same train/val/test manifests as wave 4. Model selection uses validation mIoU only. Test is sealed until the validation gate passes.

## Risks

- Dual attention can increase memory; it is attached at 1/16 C4 resolution rather than the 1/4 feature map.
- LightHamHead discards the 1/4 decoder input by the official head API and may reduce fine crack recall; this is a deliberate lightweight trade-off to measure.
- A validation gain without test gain is recorded as a negative generalization result; no SOTA claim is made without a unified protocol.

## Completion decision (2026-10-13)

All three single-module runs completed 18,000 iterations with exit code 0. EMAU, DualAttentionModule and LightHamHead each failed the preregistered validation gate; the joint run was therefore not started and no candidate test evaluation was authorized. Continue with error analysis, input/label diagnostics and deployment measurements before considering another single-variable experiment.
