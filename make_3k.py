from pathlib import Path
for src,name in [('a6_se.yml','bisenetv2_a6_se_3k.yml'),('a7_ppm_se.yml','bisenetv2_a7_ppm_se_3k.yml')]:
 s=Path(src).read_text().replace('iters: 6000','iters: 3000')
 Path(name).write_text(s)
