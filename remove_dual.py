from pathlib import Path
p=Path('bisenet_remote.py')
s=p.read_text(encoding='utf-8')
s=s.replace("        elif attention_type == 'dual':\n            self.attention = layers.DualAttentionModule(C5, C5)\n", "")
p.write_text(s,encoding='utf-8')
