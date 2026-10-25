"""Check the small publication tree, metric decisions and relative Markdown links."""
from pathlib import Path
import ast
import hashlib
import json
import re
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','__pycache__','.cache','.local_runs','.venv'}
TEXT={'.md','.py','.json','.csv','.yml','.yaml','.txt'}

def published_files():
    manifest=ROOT/'evidence/release_files.json'
    if manifest.exists():
        for rel in json.loads(manifest.read_text(encoding='utf-8'))['files']:
            p=(ROOT/rel).resolve()
            if not p.is_relative_to(ROOT.resolve()):raise ValueError('Manifest path escape')
            if not p.is_file():raise ValueError('Missing release file: '+rel)
            yield p
        return
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if not p.is_file() or any(x in SKIP for x in rel.parts):continue
        if rel.parts[:1]==('runtime',) and rel.as_posix() not in ('runtime/.gitignore','runtime/README.md'):continue
        yield p

def main():
    files=list(published_files());errors=[];links=0;code=0
    for p in files:
        rel=p.relative_to(ROOT).as_posix()
        if p.stat().st_size>2*1024**2:errors.append('oversized:'+rel)
        if p.suffix.lower() in {'.pdparams','.pdopt','.pt','.pth','.zip','.whl','.log'}:errors.append('forbidden binary:'+rel)
        if p.suffix.lower() not in TEXT:continue
        text=p.read_text(encoding='utf-8-sig')
        if re.search(r'[A-Za-z]:[\\/](?:Users[\\/]|论文复现|github[\\/])',text):errors.append('personal absolute path:'+rel)
        if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)',text):errors.append('credential pattern:'+rel)
        if p.suffix=='.py':ast.parse(text,filename=rel);code+=1
        if p.suffix=='.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
                if '://' in target or target.startswith('#'):continue
                target=unquote(target.split('#')[0].strip('<>'))
                if not target:continue
                links+=1
                if not (p.parent/target).exists():errors.append('missing link:'+rel+' -> '+target)
    fusion=json.loads((ROOT/'results/fusion20_seed42.json').read_text(encoding='utf-8'))
    assert len(fusion['rows'])==20
    multi=json.loads((ROOT/'results/fusion_multiseed.json').read_text(encoding='utf-8'))
    rows={r['id']:r for r in multi['rows']}
    assert rows['A0']['params']==3470434 and rows['F08_Branch32']['params']==3438178
    assert rows['W96']['accuracy_gate_mean'] is False
    assert rows['F08_Branch32']['accuracy_gate_mean'] is True
    assert rows['F08_Branch32']['parameter_reduction_vs_A0_pct']<1
    public=json.loads((ROOT/'results/public_models_historical_test.json').read_text(encoding='utf-8'))
    assert len(public['rows'])==7
    report={'files':len(files),'python_files_parsed':code,'relative_links_checked':links,'errors':errors,
            'checks':'AST, path/privacy patterns, size, links, current metric decisions; no training/inference',
            'weights_or_raw_data_published':False}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
