from pathlib import Path
p=Path('bisenet_remote.py')
s=p.read_text(encoding='utf-8')
s=s.replace("        if attention_type == 'se':\n            self.attention = SEModule(C5, reduction=8)\n        else:", "        if attention_type == 'se':\n            self.attention = SEModule(C5, reduction=8)\n        elif attention_type == 'dual':\n            self.attention = layers.DualAttentionModule(C5, C5)\n        else:")
p.write_text(s,encoding='utf-8')
Path('a8_dual.yml').write_text(Path('ohem_dice_remote.yml').read_text().replace('iters: 18000','iters: 3000').replace('  num_classes: 2\noptimizer:', '  num_classes: 2\n  context_type: none\n  attention_type: dual\noptimizer:'))
