from pathlib import Path
base=Path('ohem_dice_remote.yml').read_text()
for name,ctx in [('a2_ppm','ppm'),('a3_aspp','aspp')]:
 s=base.replace('iters: 18000','iters: 6000')
 s=s.replace('  num_classes: 2\noptimizer:', f'  num_classes: 2\n  context_type: {ctx}\noptimizer:')
 Path(f'{name}.yml').write_text(s)
 print(name)
