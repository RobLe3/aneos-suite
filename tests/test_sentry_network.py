"""Optional live query: never required by offline CI or scientific calibration."""
import pytest


@pytest.mark.network
def test_ky26_live_sentry_response_retains_scope_and_replays():
    from aneos_core.data.sources.sentry import SentryImpactService, verify_sentry_report
    report = SentryImpactService().assess('1998 KY26')
    assert verify_sentry_report(report)
    assert report['status'] in {'available', 'removed', 'not_listed'}
    if report['status'] == 'available':
        assert report['identity']['canonical_designation'] == '1998 KY26'
        assert report['source']['method'] in {'IOBS', 'LOV', 'MC'}
    else:
        assert report['collision_probability'] is None
    # A successful source transaction is not evidence of origin or local solver accuracy.
