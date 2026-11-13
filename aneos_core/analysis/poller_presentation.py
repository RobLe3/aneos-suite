"""Thin root-poller adapters over the shared screening read models.

Display and navigation are read-only. Export is an explicit action using the existing
exclusive report writer. Request context is not proof of source/date coverage.
"""
from .artificial_neo_dashboard import ArtificialNEODashboard
from .dashboard_view import render_detail, message


def dashboard_for(results, console):
    dashboard = ArtificialNEODashboard(console=console)
    dashboard.console = console  # None means the caller requested plain output.
    dashboard.classifications = dashboard.classify_records(results)
    return dashboard


def display(results, console):
    dashboard = dashboard_for(results, console)
    dashboard.display_dashboard(save_results=False)
    return dashboard


def export(results, console, context, output_dir='dashboard_results'):
    try:
        dashboard = dashboard_for(results, console)
        return dashboard._save_dashboard_results(output_dir, invocation_context=context)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        message('Export failed; no completed export has been verified. Existing reports were not overwritten.', console)
        return None


class _Browser:
    def __init__(self, console):
        self.console = console

    def _show_detailed_neo_view(self, record):
        return render_detail(record, self.console)

    def _export_current_view(self, records, suffix):
        return export(records, self.console, {
            'entrypoint': 'root-poller-browser', 'selected_page': suffix,
            'semantics': 'Requested display/export context, not verified source coverage',
        })

    def _has_impact_risk_indicators(self, record):
        from .impact_evidence import require_sentry_report
        try:
            report = require_sentry_report(record.impact_evidence, record.designation)
            return (report['status'] == 'available'
                    and report['collision_probability'] is not None
                    and report['collision_probability'] > 0)
        except (AttributeError, ValueError):
            return False


def explore(dashboard, console):
    from .dashboard_browser import browse
    return browse(_Browser(console), dashboard.classifications, 'Root poller screening records')
