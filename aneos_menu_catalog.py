"""Primary-menu navigation contract, not proof of backend/scientific readiness.

Keep labels, option identity and dispatch together. Per-feature prerequisites,
behavior, detailed results and the legacy menu remain subject to the QA matrix.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MenuOption:
    choice: str
    group: str
    label: str
    basis: str
    handler: str | None


MENU_OPTIONS = (
    MenuOption('1','Screening','Screen one object','Uncalibrated follow-up ranking','_detect_single'),
    MenuOption('2','Screening','Inspect screening evidence','Inputs, proxy contributions and limitations','_detect_multi_evidence'),
    MenuOption('3','Screening','Screen a batch','Designation file; per-object results','_detect_batch'),
    MenuOption('4','Screening','Inspect orbital history','Catalogue history, not manoeuvre proof','_orbital_history_analysis'),
    MenuOption('5','External evidence','Inspect impact evidence','Scoped JPL Sentry model results','_impact_assessment'),
    MenuOption('6','External evidence','Inspect encounter history','Computed encounters, not observations','_close_approach_history'),
    MenuOption('7','Acquisition & research','Run historical polling','Configured acquisition; verify coverage','_live_pipeline'),
    MenuOption('8','Acquisition & research','Inspect population patterns','Descriptive groups; significance unavailable','_population_pattern_analysis'),
    MenuOption('15','Acquisition & research','Screen orbital proximity','Pairwise catalogue prefilter, not rendezvous proof','_rendezvous_scan'),
    MenuOption('9','Results','Browse results','Session and persisted result views','_results_browser'),
    MenuOption('10','Results','Export results','Machine-readable JSON or CSV','_export_results'),
    MenuOption('11','System','Check components','Imports/reachability, not scientific certification','_system_health'),
    MenuOption('12','System','Start API server','Starts a local server process','_start_api_server'),
    MenuOption('13','System','Inspect session analytics','Session summary and export','_detection_analytics'),
    MenuOption('14','System','Read scientific help','Method definitions and limits','_show_help'),
    MenuOption('0','Navigation','Exit','Return without starting a feature',None),
)
