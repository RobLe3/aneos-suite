"""Pure batch input/report accounting. No acquisition, inference or network calls."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import base64

from aneos_core.data.enrichment import clone, fingerprint, identifier
from .reporting import verify_screening_report

SCHEMA = 'screening-batch-report-v1'
MAX_BYTES = 1024 * 1024
MAX_REQUESTS = 1000
POLICY = {'id':'batch-input-v1','max_bytes':MAX_BYTES,'max_requests':MAX_REQUESTS,
          'duplicate_policy':'reuse_exact_designation_only','csv':'one_column_optional_designation_header'}


def parse_input(raw, name):
    """Retain original bytes; reject bad encoding/shape/budgets before acquisition."""
    if not isinstance(raw, bytes) or len(raw)>MAX_BYTES:
        raise ValueError('Batch input must be at most 1 MiB')
    name=identifier(name,'input filename',limit=255)
    suffix=Path(name).suffix.lower()
    if suffix not in ('.txt','.csv'):
        raise ValueError('Choose a .txt or one-column .csv designation file')
    text=raw.decode('utf-8-sig')
    requests=[]
    if suffix=='.txt':
        rows=((i,[line]) for i,line in enumerate(text.splitlines(),1))
    else:
        reader=csv.reader(io.StringIO(text,newline=''),strict=True)
        def csv_rows():
            previous=0
            try:
                for row in reader:
                    start=previous+1;previous=reader.line_num
                    yield start,row
            except csv.Error as exc:
                raise ValueError(f'Invalid designation CSV: {exc}') from exc
        rows=csv_rows()
    first=True
    for line,row in rows:
        if not row or (len(row)==1 and (not row[0].strip() or row[0].lstrip().startswith('#'))):
            continue
        if len(row)!=1:
            raise ValueError(f'Line {line}: expected one designation column')
        value=row[0].strip()
        if suffix=='.csv' and first and value.casefold()=='designation':
            first=False;continue
        first=False
        designation=identifier(value,f'designation on line {line}')
        requests.append({'request_index':len(requests)+1,'source_line':line,'designation':designation})
        if len(requests)>MAX_REQUESTS:
            raise ValueError('Batch input exceeds 1,000 request rows; split it explicitly')
    if not requests:
        raise ValueError('File is empty or contains no designation rows')
    return {'name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
            'content_base64':base64.b64encode(raw).decode('ascii'),'requests':requests}


def read_input(path):
    path=Path(path)
    with path.open('rb') as stream:
        raw=stream.read(MAX_BYTES+1)
    return parse_input(raw,path.name)


def _summary(entries):
    unique={e['designation'] for e in entries}
    return {'request_count':len(entries),'distinct_exact_designations':len(unique),
            'reused_request_count':len(entries)-len(unique),
            'execution_status_counts':dict(sorted(Counter(e['report']['status'] for e in entries).items())),
            'classification_counts':dict(sorted(Counter(e['report']['classification'] for e in entries).items())),
            'semantics':'Request occurrences, not independent objects, observations or detection accuracy.'}


def build_batch_report(source_input, reports, created_at=None):
    original=clone(source_input)
    actual=parse_input(base64.b64decode(original['content_base64'],validate=True),original['name'])
    if actual!=original:raise ValueError('Batch input binding mismatch')
    expected=list(dict.fromkeys(r['designation'] for r in original['requests']))
    if set(reports)!=set(expected):raise ValueError('Each distinct requested designation needs one report')
    entries=[];seen=set()
    for request in original['requests']:
        name=request['designation'];report=clone(reports[name]);verify_screening_report(report)
        if report['designation']!=name:raise ValueError('Batch report identity mismatch')
        entries.append({**request,'reused_exact_request':name in seen,'report':report})
        seen.add(name)
    stamp=created_at or datetime.now(timezone.utc).isoformat()
    dt=datetime.fromisoformat(stamp)
    if dt.utcoffset() is None:raise ValueError('Batch creation timestamp needs an explicit UTC offset')
    body={'schema_version':SCHEMA,'input_policy':POLICY.copy(),'source_input':original,
          'created_at':stamp,'timestamp_role':'local_report_creation_not_observation',
          'requests':entries,'summary':_summary(entries),
          'scientific_status':'uncalibrated_screening_not_origin_inference'}
    body['batch_report_id']=fingerprint(body)
    return clone(body)


def _verify_batch_report(report):
    frozen=clone(report)
    if frozen.get('schema_version')!=SCHEMA:raise ValueError('Unsupported batch report')
    reports={}
    for entry in frozen['requests']:
        name=entry['designation']
        if name in reports and reports[name]!=entry['report']:
            raise ValueError('Exact duplicate request does not reuse its frozen report')
        reports[name]=entry['report']
    expected=build_batch_report(frozen['source_input'],reports,created_at=frozen['created_at'])
    if expected!=frozen:raise ValueError('Batch report does not reproduce')
    return frozen


def verify_batch_report(report):
    try:
        return _verify_batch_report(report)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise ValueError('Batch report is invalid or does not reproduce') from exc


def render_batch_report(report, console=None):
    """Pure readable request accounting; no side effects besides terminal output."""
    report=verify_batch_report(report)
    summary=report['summary']
    title=(f"Batch requests: {summary['request_count']} rows; "
           f"{summary['distinct_exact_designations']} exact designation labels; "
           f"reused rows: {summary['reused_request_count']}")
    columns=['Row / line','Designation','Execution','Screening / rank']
    rows=[]
    for entry in report['requests']:
        r=entry['report'];a=r.get('screening_assessment') or {};rank=a.get('screening_score')
        score='not available' if rank is None else format(rank,'.9g')
        reuse=' / reused' if entry['reused_exact_request'] else ''
        rows.append([f"{entry['request_index']} / {entry['source_line']}",entry['designation'],
                     r['status']+reuse,f"{r['classification']}\n{score}"])
    footer=('Rank: uncalibrated ranking units, not sigma. Origin probability and significance: unavailable. Missing is not zero. '
            'Repeated rows are not independent evidence. This view alone is not a storage acknowledgement; export portable copies explicitly.')
    if console is None:
        print(title);print(' | '.join(columns))
        from .object_review import literal
        for row in rows:print(' | '.join(literal(cell.replace('\n', '; rank ')) for cell in row))
        print(footer);print('Batch report ID: '+report['batch_report_id'])
    else:
        from rich.table import Table
        from rich.text import Text
        table=Table(title=Text(title),expand=True)
        for c in columns:table.add_column(c,overflow='fold')
        from .object_review import literal
        for row in rows:table.add_row(*(Text(literal(cell)) if i!=3 else Text(cell) for i,cell in enumerate(row)))
        console.print(table);console.print(Text(footer));console.print(Text('Batch report ID: '+report['batch_report_id']))


def render_batch_progress(job, console=None):
    """Display the host's verified execution read model; no invented pending reports."""
    from .object_review import literal
    title=f"Stored batch progress — {job['status']} (not a terminal batch report)"
    rows=[[str(item['request_index']),item['designation'],item['execution_status'],
           'ready checkpoint' if item['checkpoint_ready'] else 'not ready']
          for item in job['request_outcomes']]
    footer='Not assessed means no acknowledged report, not zero anomaly. No origin or significance conclusion.'
    if console is None:
        print(title)
        print('Row | Designation | Retained execution | Checkpoint')
        for row in rows:print(' | '.join(literal(cell) for cell in row))
        print(footer)
    else:
        from rich.table import Table
        from rich.text import Text
        table=Table(title=Text(title),expand=True)
        for name in ('Row','Designation','Retained execution','Checkpoint'):
            table.add_column(name,overflow='fold')
        for row in rows:table.add_row(*(Text(literal(cell)) for cell in row))
        console.print(table);console.print(Text(footer))
