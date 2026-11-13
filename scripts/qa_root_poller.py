#!/usr/bin/env python3
"""Capture actual root result/browser/detail paths with synthetic offline controls."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def capture_all(directory):
    from rich.cells import cell_len
    from aneos_version import APPLICATION_VERSION
    from neo_poller import NEOPoller
    from aneos_core.data.models import NEOData, OrbitalElements
    from aneos_core.analysis.poller_presentation import dashboard_for
    from scripts.qa_menu_entry import capture

    directory.mkdir(parents=True, exist_ok=False)
    stamp = datetime(2026, 11, 13, 8, tzinfo=timezone.utc)
    neo = NEOData('SYNTHETIC_FOLLOW_UP', orbital_elements=OrbitalElements(
        semi_major_axis=2.5, eccentricity=.9, inclination=170),
        created_at=stamp, updated_at=stamp, fetched_at=stamp)
    data = [{'designation': neo.designation, 'neo_data': neo.to_dict()},
            {'designation': 'SYNTHETIC_LEGACY', 'artificial_probability': None}]
    poller = NEOPoller.__new__(NEOPoller)
    poller.console = None
    dashboard = dashboard_for(data, None)
    inputs = [c.to_payload() for c in dashboard.classifications]
    (directory/'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')

    def browser():
        with patch('rich.prompt.Prompt.ask', return_value='Q'):
            poller._interactive_object_explorer(dashboard)

    poller._qa_summary = lambda: poller.display_results(data)
    poller._qa_browser = browser
    poller._qa_typed_detail = lambda: poller._show_detailed_analysis(dashboard.classifications[0])
    poller._qa_legacy_detail = lambda: poller._show_detailed_analysis(dashboard.classifications[1])
    widths = {}
    for view in ('summary', 'browser', 'typed_detail', 'legacy_detail'):
        for width in (80, 120, 160):
            name = f'{view}-{width}'
            capture(poller, width, directory/name, method='_qa_'+view)
            lines = (directory/(name+'.txt')).read_text().splitlines()
            actual = max(map(cell_len, lines), default=0)
            assert actual <= width
            svg = (directory/(name+'.svg')).read_text()
            geometry = float(re.search(r'viewBox="0 0 ([\d.]+)', svg)[1])
            assert geometry == math.ceil(width*12.2+16)+2
            widths[name] = {'requested_columns': width, 'actual_text_max_cells': actual,
                            'svg_viewbox_width': geometry}
    sources = ['neo_poller.py', 'aneos_core/analysis/poller_presentation.py',
               'aneos_core/analysis/artificial_neo_dashboard.py',
               'aneos_core/analysis/dashboard_view.py', 'aneos_core/analysis/dashboard_browser.py',
               'scripts/qa_menu_entry.py', 'scripts/qa_root_poller.py']
    manifest = {'schema_version': 'root-poller-presentation-qa-v1',
                'application_version': APPLICATION_VERSION,
                'scope': 'Post-beta2 root presentation follow-up; not a change to the beta2 tag.',
                'widths': widths,
                'source_files_sha256': {p: sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                'artifact_sha256': {p.name: sha256(p.read_bytes()).hexdigest()
                                    for p in directory.iterdir() if p.is_file()},
                'limits': 'Actual rendering over synthetic computational/legacy controls, no providers. '
                          'Local screening receipt is not observation/discovery time. '
                          'Not all-menu QA, source authentication, calibration, physical fits or restoration of bulk polling.'}
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    capture_all(Path(parser.parse_args().output))
