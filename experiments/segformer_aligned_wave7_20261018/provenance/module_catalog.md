# SegFormer-B0 + Aligned: official-module catalog

This catalog fixes `SegFormer-B0 + AlignedModule` as the parent. Every job adds exactly one named mechanism with `balance_modules: [aligned, <module>]`; decoder width, insertion position, and hyperparameter variants are not counted as additional mechanisms. The backbone, loss, data split, seed, augmentation, optimizer, schedule, and 18k-iteration protocol remain fixed.

The local source of record is:

`D:\璁烘枃澶嶇幇\experiments\segformer_aligned_wave7_20261018\src\PaddleSeg-2.10.0`

The source snapshot is based on the official PaddleSeg 2.10 release. Primary references are the [official PaddleSeg release/2.10 repository](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10), its [official model API](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.10/docs/apis/models/models.md), the [official model-zoo overview](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.10/docs/model_zoo_overview_cn.md), and the [official PP-LiteSeg description of UAFM/SPPM](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.10/configs/pp_liteseg/README.md). Exact executable interfaces below were checked against the local frozen source rather than inferred from paper names.

## Evidence from wave6

The Aligned parent has 3,470,434 parameters and validation mIoU/crack-IoU/F1/BF1 of 0.828184/0.678520/0.808474/0.449285. It remains the best single-module accuracy/parameter result, but its gains over B0 are below the preregistered final target.

Among additions previously tested without Aligned, GloRe gave the strongest efficiency evidence; separable ASPP recovered boundary quality; Encoding preserved boundary quality at a larger parameter cost. NonLocal nearly matched B0 region metrics but reduced BF1 by 0.017070. UAFM-Channel used more parameters than B0. SE completed a clean 18k run on 2026-10-18 with real exit0 and independent validation: crack-IoU 0.675527357, F1 0.806345959, mIoU 0.826359762. The prior 3200-iteration timeout remains archived as execution evidence. The completed SE is a negative single-module result; Aligned+SE is a separate conditional experiment. PPModule failed backward smoke testing with `CUDNN_STATUS_EXECUTION_FAILED` on the current Windows/RTX 5060 stack.

## Parameter and accuracy gates

With B0 at 3,715,170 parameters, the strict `<=0.95x B0` limit is 3,529,411 parameters. The Aligned parent therefore leaves only 58,977 parameters for an added module. A balanced secondary limit is `<=1.00x B0`, leaving 244,736 parameters. All 23 runnable entries are preregistered for attempts, including A05 as a high-cost accuracy control. Its 4,128,354 parameters cannot satisfy either smaller-parameter target and will be labelled explicitly. No control will be quietly omitted due to poor performance.

The original final target translated to the Aligned parent requires:

- crack-IoU at least 0.680173500, or at least +0.001653871 over Aligned;
- F1 at least 0.809228112, or at least +0.000754244 over Aligned;
- mIoU at least 0.828020627, allowing at most -0.000163644 from Aligned;
- recall at least 0.784134390, allowing at most -0.006839650 from Aligned;
- BF1 at least 0.446523666, allowing at most -0.002761158 from Aligned;
- total parameters at most 3,529,411;
- latency at most 1.10x B0 only after a stable, paired benchmark replaces the noisy wave6 measurements.

For seed-42 screening, advance a module only if it lies on the validation Pareto frontier and does not materially violate mIoU, recall, or BF1 floors. Freeze the top two or three candidates before running seeds 43/44. With only three seeds, report individual values and mean/range; do not claim statistical significance or SOTA from repeated use of the same validation set.

## Ordered 24-mechanism matrix

| Rank | Job / module | Official class and call | Fixed insertion | Cost and principal risk |
|---:|---|---|---|---|
| 1 | `A01_GloRe` / `glore` | `glore.GruModule(256,64,32,True)`, `x -> x` | raw C4 | +47,232 parameters in wave6; best efficiency evidence, but FP/FN shift may further lower recall with Aligned |
| 2 | `A02_ASPPsep` / `aspp_sep` | `layers.ASPPModule((1,2,4),128,128,...,use_sep_conv=True,image_pooling=False)` | projected C4 before Aligned | moderate parameters; strong wave6 BF1 recovery, possible thin-line smoothing |
| 3 | `A03_Encoding` / `encoding` | `encnet.EncModule(256,16)`, use second output | raw C4 | medium parameter cost; good BF1, codeword overfitting risk |
| 4 | `A04_ExternalAttention` / `external_attention` | `rtformer.ExternalAttention(256,256,64,8,False)` in the official residual pattern | raw C4 | low parameter/global mixing candidate; omits the rest of EABlock |
| 5 | `A05_ContextEmbedding` / `context_embedding` | `bisenet.ContextEmbeddingBlock(256,256)` | raw C4 | low spatial runtime but large 3x3 parameter cost |
| 6 | `A06_DNL` / `dnl` | `dnlnet.DisentangledNonLocal2D(0.05,...,reduction=4)` | raw C4 | centered pairwise plus unary context; still quadratic and related to failed NonLocal boundary behavior |
| 7 | `A07_PPContext` / `ppcontext` | `pp_liteseg.PPContextModule(128,32,128,(1,3,6),False)` | projected C4 before Aligned | likely strict-budget compatible; local snapshot has an RTX50/Windows interpolation compatibility path |
| 8 | `A08_DAPPM` / `dappm` | `ddrnet.DAPPM(256,32,256)` | raw C4 | roughly 0.18M parameters; pooled scales may collapse on a 13x13 map |
| 9 | `A09_APNB` / `apnb` | `ann.APNB(256,256,64,64,0.05,[1],(1,3,6,8))` | raw C4 | pooled global attention; medium-high parameter cost |
| 10 | `A10_AFNB` / `afnb` | `ann.AFNB(160,256,256,64,64,0.05,[1],...)` | C3-to-C4 before projection | fuses semantic levels distinct from Aligned's C1/C4 geometry; attribution is less isolated |
| 11 | `A11_JPU` / `jpu` | `layers.JPU([64,160,256],width=16)`, use fourth output as C2 | before standard projections | multi-level dilated fusion at 1/8; more runtime and changes C2 rather than using a residual |
| 12 | `A12_DCM` / `dcm` | `dmnet.DCM(3,True,256,256)` | raw C4 | dynamic grouped-convolution backend/export risk; one DCM filter size, not the full DMNet head |
| 13 | `A13_EMAU` / `emau` | `emanet.EMAU(256,32,3,0.1)` in `relu(x+reconstruction)` | raw C4 | few trainable parameters, iterative compute and mutable bases; earlier precision loss |
| 14 | `A14_NonLocal` / `nonlocal` | `layers.NonLocal2D(256,reduction=4)` | raw C4 | region metrics were strong but BF1 failure was large |
| 15 | `A15_AttentionRefinement` / `attention_refinement` | `bisenetv1.AttentionRefinement(128,128)` | projected C4 before Aligned | local 3x3 plus channel gate; medium parameter cost and partial overlap with SE |
| 16 | `A16_CAM` / `cam` | `layers.attention.CAM(256)` | raw C4 | negligible parameters; near-identity wave6 result |
| 17 | `A17_CCA` / `cca` | `ccnet.CrissCrossAttention(256)` | raw C4 | axial bias and weak wave6 result |
| 18 | `A18_PAM` / `pam` | `layers.attention.PAM(256)` | raw C4 | quadratic spatial affinity and no prior BF1 gain |
| 19 | `A19_MSCA` / `msca` | `mscan.SpatialAttention(128)` | fused 1/4 feature | high-resolution strip-convolution cost and prior accuracy loss |
| 20 | `A20_UAFMChannel` / `uafm_channel` | `layers.UAFM_ChAtten(128,128,128)` | C1 + fused 1/4 feature | +337,664 parameters in wave6; dominated by Aligned on primary metrics |
| 21 | `A21_Polarized` / `polarized` | `PolarizedSelfAttentionModule(128,128)` | fused 1/4 feature | high-resolution attention cost and poor wave6 result |
| 22 | `A22_Hamburger` / `hamburger` | `segnext.Hamburger(128,8,ham_kwargs)` with fixed NMF bases | projected C4 before Aligned | low parameters but iterative decomposition; prior full LightHamHead was worse |
| 23 | `A23_SE` / `se` | `mobilenetv3.SEModule(128,4)` | fused 1/4 feature | cheap; standalone SE now completed as a negative result; conditional Aligned+SE remains untested |
| 24 | `A24_PSP` / `psp_pool` | `layers.PPModule(128,128,(1,2,3,6),True,False)` | projected C4 before Aligned | currently incompatible with this GPU/software stack due reproduced cuDNN backward failure |

Ranks 1-23 are queued in the registered order; each adds one module to Aligned and starts cleanly from the common ImageNet state. Entries 14-23 are lower-priority conditional-effect controls, not omitted runs. Rank 24 must remain excluded until a source-neutral backend solution passes repeated forward/backward smoke tests.

`Aligned+GC` is deliberately absent from the 24 jobs, and no `MSCA+CAM` composite is scheduled, following the user's explicit exclusions. The catalog's MSCA and CAM rows are separate `Aligned+one-module` controls only; they must never be combined into one job.

## Official mechanisms deliberately not counted

- `ISANet.ISAHead` is a complete logits head with two attention passes, padding/reshape logic, and an auxiliary branch. Reusing only `SelfAttentionBlock` would not constitute the published interlaced mechanism; replacing the decoder would break the one-addition design.
- OCR's `SpatialGatherBlock`/`SpatialOCRModule` requires a region-probability proxy and an auxiliary classifier. It is a multi-component head and changes the loss/output contract.
- PointRend's `PointHead` changes training-time point sampling, inference subdivision, logits, and loss handling.
- GSCNN's `GatedSpatailConv2d` requires an independently generated one-channel edge gate; GSCNN ASPP also requires an edge input. Adding the edge branch would introduce more than one mechanism.
- BiSeNetV2 `BGA` assumes a fixed fourfold resolution relationship between its detail and semantic inputs. Adapting it would duplicate cross-scale fusion already supplied by Aligned and add a scale-specific intervention.
- `SqueezeBodyEdge` returns body and edge tensors and needs downstream dual heads/losses; it is not a feature-preserving drop-in block.
- FastSCNN `FeatureFusionModule` hard-codes `groups=128` in its depthwise convolution and is architecture-specific. It remains an implementation option, but it was excluded to avoid counting another closely related fusion variant.
- Decoder widths, alternate insertion stages, different dilation/bin sizes, and UAFM channel/spatial statistic variants are hyperparameter or placement variants and are not counted as separate mechanisms.

## Integration invariants

The only modified algorithm entry point is `paddleseg/models/segformer.py`. Official class implementations are unchanged. With `balance_modules: []` or `[aligned]`, no new candidate layer is instantiated, so the original and Aligned state keys and forward paths remain unchanged. Candidate jobs must use the ordered list `[aligned, module]`; the parent's `balance_blocks.aligned.*` keys therefore remain stable, and the candidate's state is appended under `balance_blocks.<module>.*`.

No test-set result should be read while screening. The 24-way validation search raises winner's-curse risk, so the final report must preserve all completed rows, freeze candidate selection, and evaluate every original public-model control on the same sealed test split regardless of validation rank.


Execution protocol clarification (2026-10-18): the authoritative preregistered parent/B0 gates are in protocol.json and EXPERIMENT_PLAN.md. The translated original-target values above are descriptive arithmetic only; they do not relax the new parent contribution gate. Speed remains provisional.
