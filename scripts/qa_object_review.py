#!/usr/bin/env python3
"""Capture actual object detail renders using explicitly synthetic controls."""
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.qa_menu_entry import capture
from aneos_core.analysis.object_review import build_object_review,render_object_review
from aneos_core.data.models import NEOData,OrbitalElements,PhysicalProperties

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    directory=Path(parser.parse_args().output);directory.mkdir(parents=True,exist_ok=False)
    cases={
        'small':NEOData(designation='SYNTHETIC small-body illustration',orbital_elements=OrbitalElements(
            semi_major_axis=1.23,eccentricity=.2,inclination=1.49,ra_of_ascending_node=0.,arg_of_periapsis=0.,mean_anomaly=0.),
            physical_properties=PhysicalProperties(diameter_km=.011,rotation_period_hours=.089193,
                albedo=.52,absolute_magnitude_h=25.9),sources_used=['synthetic-control']),
        'candidate':NEOData(designation='SYNTHETIC follow-up control',orbital_elements=OrbitalElements(
            semi_major_axis=1.,eccentricity=.2,inclination=180.),sources_used=['synthetic-control']),
        'missing':NEOData(designation='SYNTHETIC missing-data control')}
    class View:
        def __init__(self,review):self.review=review
        def _print_banner(self):render_object_review(self.review,console=self.console)
    inventory={}
    for name,neo in cases.items():
        review=build_object_review(neo)
        (directory/f'{name}.json').write_text(json.dumps(review,indent=2)+'\n')
        inventory[name]={'review_id':review['review_id'],'classification':review['assessment']['classification'],'widths':{}}
        for width in (80,120,160):
            path=directory/f'{name}-{width}';capture(View(review),width,path)
            lines=path.with_suffix('.txt').read_text().splitlines()
            inventory[name]['widths'][str(width)]={'lines':len(lines),'max_columns':max(map(len,lines))}
    files={p.name:sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
    (directory/'inventory.json').write_text(json.dumps({'schema_version':'object-detail-qa-v1',
        'cases':inventory,'artifact_sha256':files,'source_sha256':{p:sha256((ROOT/p).read_bytes()).hexdigest()
            for p in ['aneos_core/analysis/object_review.py','aneos_core/analysis/screening.py','scripts/qa_object_review.py']},
        'limits':'Synthetic illustrative controls only; no source acquisition, independent calibration, physical fits or all-menu certification.'},indent=2)+'\n')
    print(directory)
if __name__=='__main__':main()
