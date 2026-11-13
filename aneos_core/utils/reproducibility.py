"""Read-only shared acquisition/report software identity; not an environment lock."""
from aneos_version import APPLICATION_VERSION, BUILD_CHANNEL
from hashlib import sha256


def software_metadata(source_files=None):
    """Read-only run metadata; file hashes are not a dependency lock or signature."""
    import sys
    import subprocess
    from importlib.metadata import version, PackageNotFoundError
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    commit, dirty = None, None
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
            capture_output=True, text=True, timeout=2, check=True)
        status = subprocess.run(['git', '--no-optional-locks', 'status', '--porcelain'], cwd=root,
            capture_output=True, text=True, timeout=2, check=True)
        commit, dirty = revision.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass  # A packaged/non-Git installation is explicitly unknown, never clean.
    packages = {}
    for name in ('numpy', 'scipy', 'pydantic', 'SQLAlchemy', 'requests', 'astroquery'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    modules = source_files if source_files is not None else ['aneos_version.py', 'aneos_core/analysis/pipeline.py', 'aneos_core/analysis/screening.py',
        'aneos_core/analysis/reporting.py', 'aneos_core/analysis/object_review.py',
        'aneos_core/analysis/artificial_neo_dashboard.py', 'aneos_core/analysis/dashboard_view.py', 'aneos_core/analysis/dashboard_browser.py',
        'aneos_core/data/physical_evidence.py', 'aneos_core/data/epoch_evidence.py', 'simple_neo_analyzer.py',
        'aneos_core/analysis/advanced_aneos_analyzer.py', 'aneos_core/analysis/impact_evidence.py',
        'aneos_core/interfaces/unified_analysis.py', 'aneos_core/data/models.py',
        'aneos_core/data/orbit_evidence.py', 'aneos_core/data/fetcher.py',
        'aneos_core/data/sources/sbdb.py', 'aneos_core/data/sources/horizons.py',
        'aneos_core/data/sources/mpc.py', 'aneos_core/data/sources/neodys.py',
        'aneos_core/data/acquisition.py', 'aneos_core/data/chronology.py', 'aneos_core/analysis/chronology_view.py', 'aneos_core/data/neodys_evidence.py', 'aneos_core/data/reconciliation.py',
        'aneos_core/data/enrichment.py', 'aneos_core/data/cache.py',
        'aneos_api/enrichment_store.py', 'aneos_api/acquisition_repository.py',
        'aneos_core/detection/validated_sigma5_artificial_neo_detector.py',
        'aneos_core/polling/root_candidate_controller.py', 'neo_poller.py',
        'aneos_core/utils/reproducibility.py']
    hashes = {name:sha256((root/name).read_bytes()).hexdigest() if (root/name).is_file() else None
        for name in modules}
    return {'application_version':APPLICATION_VERSION, 'build_channel':BUILD_CHANNEL,
        'repository_commit':commit, 'working_tree_dirty':dirty,
        'python_version':sys.version.split()[0], 'package_versions':packages,
        'float_mantissa_bits':sys.float_info.mant_dig, 'float_epsilon':sys.float_info.epsilon,
        'source_hashes_at_report_time':hashes,
        'limitations':'Source-file hashes describe files at report time, not loaded bytecode, authentication or a complete environment lock. Dirty code must be preserved separately to reproduce it.'}

