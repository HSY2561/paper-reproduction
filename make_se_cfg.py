from pathlib import Path
base=Path('ohem_dice_remote.yml').read_text()
for name,ctx,att in [('a6_se','none','se'),('a7_ppm_se','ppm','se')]:
 s=base.replace('iters: 18000','iters: 6000')
 s=s.replace('  num_classes: 2\noptimizer:', f'  num_classes: 2\n  context_type: {ctx}\n  attention_type: {att}\noptimizer:')
 Path(f'{name}.yml').write_text(s)
