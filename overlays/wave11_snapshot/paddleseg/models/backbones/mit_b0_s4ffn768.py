"""W11 isolated MiT-B0 adaptation; only S4 MixFFN hidden width changes."""
from functools import partial
from pathlib import Path

import paddle
from paddle import nn

from paddleseg.cvlibs import manager
from .mix_transformer import MixVisionTransformer


def load_exact_state(layer, path):
    """Load a complete local state; never silently skip shape/key mismatches."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    loaded = paddle.load(str(path))
    expected = layer.state_dict()
    if not isinstance(loaded, dict):
        raise TypeError('Expected a flat Paddle state_dict: ' + str(path))
    missing = sorted(set(expected) - set(loaded))
    extra = sorted(set(loaded) - set(expected))
    mismatch = {k: [list(loaded[k].shape), list(expected[k].shape)]
                for k in expected.keys() & loaded.keys()
                if list(loaded[k].shape) != list(expected[k].shape)}
    if missing or extra or mismatch:
        raise ValueError('Exact state load failed: missing=%r extra=%r shapes=%r'
                         % (missing, extra, mismatch))
    for key, value in loaded.items():
        if not bool(paddle.all(paddle.isfinite(value)).item()):
            raise ValueError('Non-finite initial tensor: ' + key)
    layer.set_state_dict(loaded)


@manager.BACKBONES.add_component
def MixVisionTransformer_B0_S4FFN768(pretrained=None):
    """Trainable compressed backbone, initialized only from an exact derived state.

    ``pretrained=None`` permits CPU construction for generation of the shared
    complete initial states. Formal training supplies the registered derived
    backbone state and/or the complete model initial state.
    """
    model = MixVisionTransformer(
        patch_size=4, embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 3], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0, drop_path_rate=0.1, pretrained=None)
    if pretrained is not None:
        load_exact_state(model, pretrained)
    return model
