#!/usr/bin/env python3
"""Capture menu-entry QA without running providers, server, polling or features.

Static legacy candidates and intercepted dispatch are not feature readiness proof.
Refuse to overwrite an existing artifact directory. Raster conversion is optional;
SVG/HTML/text are produced directly by the actual Rich renderer.
"""
import argparse
import ast
from dataclasses import asdict
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def legacy_candidates(source):
    tree=ast.parse(source)
    candidates=[]
    for node in ast.walk(tree):
        if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) or not ('menu' in node.name):continue
        options=[]
        for child in ast.walk(node):
            if not isinstance(child,(ast.Tuple,ast.List)) or len(child.elts)<2:continue
            a,b=child.elts[:2]
            if (isinstance(a,ast.Constant) and isinstance(a.value,str) and
                re.fullmatch(r'\d+|[A-Za-z]',a.value) and isinstance(b,ast.Constant) and isinstance(b.value,str)):
                options.append({'choice':a.value,'label':b.value,'line':child.lineno})
        candidates.append({'method':node.name,'line':node.lineno,'literal_option_candidates':options,
                           'runtime_behavior_status':'not_audited','prerequisites_status':'not_audited'})
    return candidates


def capture(menu,width,path,method="_print_banner"):
    from rich.console import Console
    from rich._export_format import CONSOLE_SVG_FORMAT
    svg_format=re.sub(r'@font-face\s*\{\{.*?\}\}', '', CONSOLE_SVG_FORMAT, flags=re.S)
    if '@font-face' in svg_format:raise RuntimeError('Unexpected remote font export template')
    svg_format=svg_format.replace('font-family: Fira Code, monospace','font-family: monospace')
    output=io.StringIO();menu.console=Console(file=output,width=width,height=25,record=True,force_terminal=True,color_system='truecolor')
    if menu.console.width != width:raise RuntimeError('Requested rich capture width was not honored')
    getattr(menu,method)()
    menu.console.save_svg(str(path.with_suffix('.svg')),title='Actual Rich menu render',clear=False,code_format=svg_format)
    menu.console.save_html(str(path.with_suffix('.html')),clear=False)
    # Export plain text from a separate actual render, not ANSI regex removal.
    plain=io.StringIO();menu.console=Console(file=plain,width=width,height=25,color_system=None)
    if menu.console.width != width:raise RuntimeError('Requested plain capture width was not honored')
    getattr(menu,method)();path.with_suffix('.txt').write_text(plain.getvalue())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--baseline-ref',required=True,help='Explicit Git revision of preceding menu implementation')
    args=parser.parse_args()
    directory=Path(args.output);directory.mkdir(parents=True,exist_ok=False)
    from aneos_menu_v2 import ANEOSMenuV2
    from aneos_menu_catalog import MENU_OPTIONS
    from aneos_version import APPLICATION_VERSION
    revision=subprocess.run(['git','rev-parse',args.baseline_ref],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
    source=subprocess.run(['git','show',revision+':aneos_menu_v2.py'],cwd=ROOT,check=True,capture_output=True,text=True).stdout
    ns={'__name__':'qa_baseline_not_main','__file__':str(ROOT/'aneos_menu_v2.py')}
    exec(compile(source,'git:'+revision+':aneos_menu_v2.py','exec'),ns)
    for width in (80,120,160):
        capture(ns['ANEOSMenuV2'](),width,directory/f'before-{width}')
        capture(ANEOSMenuV2(),width,directory/f'after-{width}')
        capture(ANEOSMenuV2(),width,directory/f'help-{width}',method='_show_help')
    menu=ANEOSMenuV2();menu.console=None
    import contextlib
    text=io.StringIO()
    with contextlib.redirect_stdout(text):menu._print_banner()
    (directory/'after-plain.txt').write_text(text.getvalue())
    from aneos_menu import ANEOSMenu
    legacy=ANEOSMenu.__new__(ANEOSMenu);legacy.console=None;text=io.StringIO()
    with contextlib.redirect_stdout(text):legacy.display_header()
    (directory/'legacy-header-plain.txt').write_text(text.getvalue())
    files={p.name:sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
    register={
        'schema_version':'menu-entry-qa-v1','application_version':APPLICATION_VERSION,
        'baseline_commit':revision,'baseline_menu_sha256':sha256(source.encode()).hexdigest(),
        'current_menu_sha256':sha256((ROOT/'aneos_menu_v2.py').read_bytes()).hexdigest(),
        'source_files_sha256':{name:sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['aneos_version.py','aneos_menu_catalog.py','aneos_menu_v2.py','aneos_menu.py','scripts/qa_menu_entry.py']},
        'primary_options':[dict(asdict(item),label_render_status='captured_80_120_160',
            feature_prerequisite_qa='pending',feature_behavior_qa='pending',detailed_result_qa='pending') for item in MENU_OPTIONS],
        'primary_option_count':len(MENU_OPTIONS),'feature_count_excluding_exit':len(MENU_OPTIONS)-1,
        'legacy_static_candidates':legacy_candidates((ROOT/'aneos_menu.py').read_text()),
        'artifact_sha256':files,
        'coverage_limits':'Entry/labels and legacy header only. Static candidates are not a verified reachable feature denominator. No provider/feature execution, scientific readiness or full menu certification.'}
    (directory/'inventory.json').write_text(json.dumps(register,indent=2,ensure_ascii=False)+'\n')
    print(f'Captured {len(MENU_OPTIONS)} primary entries; feature QA remains pending. Artifacts: {directory}')


if __name__=='__main__':main()
