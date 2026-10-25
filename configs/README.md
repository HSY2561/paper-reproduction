# Portable recorded recipes

`recipes.json` records each source config's SHA256, its relative experiment ID,
the template digest, the exact fields made portable, and command-level EMA and
backend information. Only dataset and initialization path values were replaced.
The templates remain self-contained PaddleSeg YAML after materialization.

| Template | Model | Initialization | Loss | EMA |
| --- | --- | --- | --- | --- |
| A0 | SegFormer MiT-B0, width 128, aligned | common MiT-B0 ImageNet | CE | yes |
| B0 | SegFormer MiT-B0, width 256, base | common MiT-B0 ImageNet | CE | yes |
| W96 | SegFormer MiT-B0, width 96, aligned | common MiT-B0 ImageNet | CE | yes |
| F08 | SegFormer MiT-B0, width 128, aligned + branch bottleneck rank 32 | common MiT-B0 ImageNet | CE | yes |
| public_segformer_b0 | SegFormer MiT-B0, width 256 | original MiT-B0 URL | CE | yes |
| public_ocrnet_hrnetw18 | OCRNet HRNet-W18 | random | CE weights 1, 0.4 | yes |
| public_ppliteseg_stdc1 | PP-LiteSeg STDC1 | original STDC1 URL | OHEM x3, min_kept 65000 | yes |
| public_pspnet_resnet50 | PSPNet ResNet50-vd | random | OHEM weights 1, 0.4 | no |
| public_deeplabv3p_resnet50 | DeepLabV3+ ResNet50-vd | random | OHEM | no |
| public_fastscnn | Fast-SCNN | random | OHEM weights 1, 0.4 | no |

All recorded recipes use 18,000 iterations, batch size 4, AdamW at 0.00006,
polynomial LR decay, 400x400 random crops, and validation with Normalize only.
The last three public recipes were trained in WSL2 Ubuntu 24.04 with Paddle GPU
3.3.0.dev20251209. The others were recorded under Windows with Paddle GPU 3.4.0.
These original public models are not a claim of a fully controlled common
initialization/loss/backend comparison.

`B0.yml` is the fresh control recipe with witnessed common initialization.
`public_segformer_b0.yml` preserves the historical public recipe separately;
its original URL alone does not establish its historical initialization binary
hash. Portable materialization points to the explicitly supplied local artifact.

The minimal runtime includes only the SegFormer overlay. Historical Windows
PP-LiteSeg training also used compatibility edits in `pp_liteseg.py` and
`pyramid_pool.py`. Those unrelated patches are not included, so the public
PP-LiteSeg template records its actual recipe without claiming byte-identical
reproduction of that historical training environment. The six public checkpoints
can be evaluated separately with their matching full model configurations.
