# Generated local runtime

Run `python -B scripts/bootstrap_runtime.py --archive /path/to/PaddleSeg-v2.10.0.zip`
from the repository root. This verifies the pinned SHA256 and creates
`runtime/PaddleSeg-2.10.0` from the official archive, then applies the single
committed `segformer.py` overlay. It never starts training or downloads anything.

Generated runtime files, materialized configurations, and local outputs are ignored
here. Keep datasets and checkpoints outside the repository. See
`scripts/README.md` and `third_party/NOTICE.md` for the commands and provenance.
