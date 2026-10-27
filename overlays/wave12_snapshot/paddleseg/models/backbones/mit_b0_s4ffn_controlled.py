"""W12 controlled S4 Mix-FFN internal compression; other B0 structure unchanged."""
from functools import partial
from paddle import nn
from paddleseg.cvlibs import manager
from .mix_transformer import MixVisionTransformer
from .mit_b0_s4ffn768 import load_exact_state


@manager.BACKBONES.add_component
def MixVisionTransformer_B0_S4FFNControlled(hidden=1024, pretrained=None):
    if hidden not in (768, 896, 960, 1024):
        raise ValueError('W12 S4 hidden must be explicitly registered: 768/896/960/1024')
    model = MixVisionTransformer(
        patch_size=4, embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, hidden / 256],
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0, drop_path_rate=0.1, pretrained=None)
    if pretrained is not None:
        load_exact_state(model, pretrained)
    return model
