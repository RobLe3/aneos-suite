#!/usr/bin/env python3
"""Actual before/after option-3 renders with synthetic acquisition and network denied.

Does not poll providers or certify scientific validity. Refuses existing output.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    baseline='dd56b829fd61d6a4d47f9be61088c1975a2aff1e'
    source=subprocess.check_output(['git','show',baseline+':aneos_menu_v2.py'],cwd=ROOT,text=True)
    scope={'__name__':'frozen_batch_qa_not_main','__file__':str(ROOT/'aneos_menu_v2.py')}
    exec(compile(source,'git:'+baseline+':aneos_menu_v2.py','exec'),scope)
    from aneos_menu_v2 import ANEOSMenuV2
    from aneos_core.analysis.pipeline import AnalysisPipeline
    from aneos_core.data.models import NEOData,OrbitalElements,PhysicalProperties
    from aneos_core.utils.errors import DataSourceUnavailableError
    from rich.console import Console
    from rich._export_format import CONSOLE_SVG_FORMAT
    names=['candidate','missing','offline','candidate','wrong','long-label-[red]-'+'x'*45]
    path=args.output/'synthetic-requests.txt'
    path.write_text('# SYNTHETIC QA ONLY; not astronomical observations\n'+'\n'.join(names)+'\n')
    def neo(name):
        if name=='missing':return NEOData(name)
        return NEOData(name,orbital_elements=OrbitalElements(semi_major_axis=1.23,eccentricity=.2,
            inclination=180. if name=='candidate' else 1.49),
            physical_properties=PhysicalProperties(diameter_km=.011,albedo=.52,absolute_magnitude_h=25.9))
    def fetch(name,**kwargs):
        if name=='offline':raise DataSourceUnavailableError('SYNTHETIC source unavailable')
        return neo('other' if name=='wrong' else name)
    old_map={n:None if n=='offline' else neo('other' if n=='wrong' else n) for n in names}
    template=re.sub(r'@font-face\s*\{\{.*?\}\}','',CONSOLE_SVG_FORMAT,flags=re.S)
    template=template.replace('font-family: Fira Code, monospace','font-family: monospace')
    if '@font-face' in template:raise RuntimeError('Remote font reference remains')
    deny=Mock(side_effect=AssertionError('QA must not use network or DNS'))
    for label,cls in [('before',scope['ANEOSMenuV2']),('after',ANEOSMenuV2)]:
        for width in (80,120,160,None):
            menu=cls();menu.browse_files=Mock(return_value=str(path));menu._persist_detection_result_async=Mock()
            stream=io.StringIO()
            menu.console=Console(file=stream,width=width,record=True,force_terminal=True,
                color_system='truecolor',_environ={'TERM':'xterm-256color','COLUMNS':str(width),'LINES':'40'}) if width else None
            pipeline=AnalysisPipeline();pipeline.data_fetcher=SimpleNamespace(fetch_neo_data=fetch)
            with patch.object(socket.socket,'connect',deny),patch.object(socket,'getaddrinfo',deny), \
                 patch('aneos_core.data.fetcher.DataFetcher',return_value=SimpleNamespace(fetch_multiple=lambda d:old_map)), \
                 patch('aneos_core.analysis.pipeline.AnalysisPipeline',return_value=pipeline), \
                 contextlib.redirect_stdout(stream):
                if menu.console:
                    from rich.text import Text
                    menu.console.print(Text('SYNTHETIC QA CONTROLS — not astronomical observations'))
                else:
                    print('SYNTHETIC QA CONTROLS — not astronomical observations')
                menu._detect_batch()
            stem=args.output/f'{label}-{width or "plain"}'
            if width:
                if menu.console.width!=width:raise RuntimeError('Terminal width was not honored')
                menu.console.save_svg(str(stem.with_suffix('.svg')),clear=False,code_format=template)
                menu.console.save_html(str(stem.with_suffix('.html')),clear=False)
                text=menu.console.export_text(clear=False)
                if label=='after' and max(map(len,text.splitlines()),default=0)>width:
                    raise RuntimeError('New text exceeds requested width')
            else:text=stream.getvalue()
            stem.with_suffix('.txt').write_text(text)
            if label=='after' and width==80:
                with (args.output/'synthetic-batch-report.json').open('x') as target:
                    json.dump(menu._batch_reports[0],target,indent=2,allow_nan=False)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()}
    manifest={'terminal_capture_profile':'explicit xterm-256color; not a TERM=dumb qualification',
              'scope':'Actual option 3 with synthetic typed source controls; no scientific validation',
              'baseline_commit':baseline,'artifact_sha256':files,
              'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                  ('aneos_menu_v2.py','aneos_core/analysis/batch_reporting.py','aneos_core/analysis/reporting.py',
                   'aneos_core/reporting/exporters.py','scripts/qa_batch_menu.py')}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Captured actual synthetic-control before/after batch renders: {args.output}')


if __name__=='__main__':main()
