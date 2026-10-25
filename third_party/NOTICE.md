# Upstream sources and notices

The runtime is derived from [PaddlePaddle/PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg),
tag `v2.10.0`. The offline ZIP used by this package is
`https://github.com/PaddlePaddle/PaddleSeg/archive/refs/tags/v2.10.0.zip`, with
SHA256 `8ad8541e3ebb7fe9fc670f4026972be77385e73764f66e94390297e794c268d6`.
The bootstrap verifies that digest before extracting anything. It retains the
whole `paddleseg` core package because upstream model registration imports its
modules, plus only the official train/val entry points, requirements and LICENSE.
Documentation assets, examples, deployment stacks and unrelated application
repositories are not copied into this source package.

`PaddleSeg-LICENSE` is the unmodified upstream Apache License 2.0 text (SHA256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`).
The bootstrap also retains the upstream `LICENSE` inside the generated runtime.
Original file-level copyright and license notices remain intact.

One local overlay is applied: `overlays/paddleseg/models/segformer.py`, taken
without edits from the active wave10 source snapshot. SHA256:
`ad292bca4d0171028360ec00bbd1f7b87214c03a494ed2d13a02871e9c57d362`.
It implements the recorded decoder-width, alignment and fusion variants; no
other historical local model edits or compatibility patches are bundled.

The SegFormer source itself states that it was heavily based on
[NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) and points users to the
[NVlabs SegFormer license](https://github.com/NVlabs/SegFormer#license), with a
specific reminder about commercial adoption. That original notice is retained
in the overlay. The PaddleSeg top-level Apache license is not a replacement
for upstream component-specific notices or the licenses attached to externally
obtained pretrained weights and datasets.

`scripts/evaluate_exact.py` is the unchanged local validation evaluator from
the same experiment. Its source filename was `evaluate_val_boundary.py` and
its SHA256 is `97ac482974536c1cd9be268b126313741eedbb6697b5f9af65adc873be7f53a6`.
No dataset, trained checkpoint, or pretrained binary is redistributed here.
