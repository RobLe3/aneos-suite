"""Polling screening dashboard over the shared scientific contracts.

The historical class/module names remain import-compatible. Origin categories,
probabilities and invented confidence have been deprecated, not recalibrated.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from aneos_core.data.models import NEOData
from aneos_core.data.enrichment import clone, fingerprint, identifier
from .screening import ScreeningAnalysisService, ScreeningPolicy
from .object_review import (
    build_object_review,
    build_assessment_review,
    verify_object_review,
)

CATEGORIES = {
    "FOLLOW_UP_CANDIDATE": "follow_up_candidate",
    "NOT_FLAGGED": "not_flagged",
    "INCONCLUSIVE": "inconclusive",
}
CATEGORY_LABELS = {
    "follow_up_candidate": "Follow-up candidate",
    "not_flagged": "Not flagged",
    "inconclusive": "Inconclusive",
}
LIMITS = (
    "Rank is an uncalibrated follow-up heuristic, not Gaussian significance or an origin probability. "
    "Not flagged does not mean natural. Impact risk is not assessed here. "
    "Legacy unverified rows are retained, not promoted to measured evidence."
)


def utc_now():
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class NEOClassification:
    """Compatibility view; every supported inference derives from a verified review.

    Old positional/keyword fields remain accepted. Their claims are archived and
    nullified; constructing a legacy object cannot confirm its asserted origin.
    """

    designation: str
    category: str = "inconclusive"
    confidence: Optional[float] = None
    artificial_probability: Optional[float] = None
    risk_factors: list = field(default_factory=list)
    analysis_details: dict = field(default_factory=dict)
    data_sources: list = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=utc_now)
    orbital_elements: dict = field(default_factory=dict)
    review: Optional[dict] = None
    input_record: object = None
    input_status: str = "legacy_unverified"
    input_id: Optional[str] = None

    def __post_init__(self):
        raw = self.input_record
        if raw is None:
            raw = {
                "designation": self.designation,
                "category": self.category,
                "artificial_probability": self.artificial_probability,
                "confidence": self.confidence,
                "orbital_elements": self.orbital_elements,
                "data_sources": self.data_sources,
                "analysis_details": self.analysis_details,
                "analysis_timestamp": self.analysis_timestamp.isoformat(),
            }
        raw = clone(raw)
        review = verify_object_review(self.review) if self.review is not None else None
        if review is not None and review["designation"] != self.designation:
            raise ValueError("Review/object identity mismatch")
        if review is not None:
            source = (review.get("inventory") or {}).get("source_snapshot") or {}
            evidence = review["assessment"]["evidence_snapshot"]
            contexts = (
                source.get("evidence_metadata") or {},
                evidence.get("provenance") or {},
                evidence.get("additional") or {},
            )
            if any(
                context.get(k) is True
                for context in contexts
                for k in ("synthetic", "is_synthetic", "mock", "demo")
            ):
                review = None
                object.__setattr__(
                    self, "input_status", "declared_synthetic_unverified"
                )
        assessment = review["assessment"] if review is not None else None
        snapshot = (
            review["inventory"]["source_snapshot"]
            if review and review["inventory"]
            else None
        )
        object.__setattr__(self, "input_record", raw)
        object.__setattr__(self, "input_id", fingerprint(raw))
        object.__setattr__(self, "review", review)
        object.__setattr__(
            self,
            "category",
            CATEGORIES[assessment["classification"]] if assessment else "inconclusive",
        )
        object.__setattr__(self, "confidence", None)
        object.__setattr__(self, "artificial_probability", None)
        object.__setattr__(
            self,
            "risk_factors",
            (
                [
                    e["type"] + " (uncalibrated proxy)"
                    for e in assessment["evidence_sources"]
                    if e["analyzed"]
                ]
                if assessment
                else []
            ),
        )
        object.__setattr__(
            self,
            "orbital_elements",
            clone(snapshot["orbital_elements"] or {}) if snapshot else {},
        )
        object.__setattr__(
            self, "data_sources", list(snapshot["sources_used"]) if snapshot else []
        )
        object.__setattr__(
            self,
            "analysis_details",
            {
                "assessment": clone(assessment),
                "data_completeness": None,
                "basic_field_presence": (
                    clone(review["inventory"]["coverage"]) if snapshot else None
                ),
                "interpretation": (
                    review["decision"]["text"]
                    if review
                    else "Screening unavailable: no replayable typed evidence or canonical assessment. Legacy input is retained for review."
                ),
                "limits": LIMITS,
            },
        )

    @property
    def screening_score(self):
        return self.review["assessment"]["screening_score"] if self.review else None

    @property
    def physical_properties(self):
        return (
            clone(
                self.review["inventory"]["source_snapshot"]["physical_properties"] or {}
            )
            if self.review and self.review["inventory"]
            else {}
        )

    @property
    def impact_evidence(self):
        raw = self.input_record if isinstance(self.input_record, dict) else {}
        return clone(raw.get("impact_evidence", raw.get("impact_assessment")))

    @property
    def basic_presence(self):
        return (
            self.review["inventory"]["coverage"]["basic_catalogue"]["fraction_present"]
            if self.review and self.review["inventory"]
            else None
        )

    def to_payload(self):
        # Do not allow a mutable nested dictionary to change a frozen report silently.
        if self.input_id != fingerprint(self.input_record):
            raise ValueError("Classification input changed")
        review = verify_object_review(self.review) if self.review else None
        if review and (
            review["designation"] != self.designation
            or CATEGORIES[review["assessment"]["classification"]] != self.category
        ):
            raise ValueError("Classification review changed")
        value = {
            "schema_version": "polling-screening-record-v1",
            "designation": self.designation,
            "category": self.category,
            "screening_score": self.screening_score,
            "confidence": None,
            "artificial_probability": None,
            "sigma_confidence": None,
            "combined_p_value": None,
            "false_discovery_rate": None,
            "origin_status": "inconclusive",
            "input_record": clone(self.input_record),
            "input_status": self.input_status,
            "input_id": self.input_id,
            "review": review,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "five_sigma_claim_permitted": False,
            "limits": LIMITS,
        }
        value["record_id"] = fingerprint(value)
        return value


def verify_record(value):
    value = clone(value)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "polling-screening-record-v1"
    ):
        raise ValueError("Unsupported screening record")
    stamp = datetime.fromisoformat(value["analysis_timestamp"])
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("Screening receipt requires explicit UTC")
    result = NEOClassification(
        value["designation"],
        review=value["review"],
        input_record=value["input_record"],
        input_status=value["input_status"],
        analysis_timestamp=stamp,
    )
    if result.to_payload() != value:
        raise ValueError("Screening record does not reproduce")
    return result


def classify_record(value, service=None):
    """No acquisition/I/O. Accept typed snapshots or replayable canonical evidence.

    Old score-only/unschematized packets retain all finite JSON values as unverified
    input. They cannot establish measured orbit/physical inputs by field-name alone.
    """
    service = service or ScreeningAnalysisService()
    is_typed = isinstance(value, NEOData)
    if is_typed:
        value = {"designation": value.designation, "neo_data": value.to_dict()}
    try:
        raw = clone(value)
    except (ValueError, TypeError, OverflowError):
        return NEOClassification(
            "Unidentified record",
            input_record={
                "retention_error": "NONFINITE_UNSERIALIZABLE_OR_OVERSIZE_INPUT"
            },
            input_status="invalid_input",
        )
    if not isinstance(raw, dict):
        return NEOClassification(
            "Unidentified record",
            input_record={"unverified_input": raw},
            input_status="invalid_input",
        )
    name = "Unidentified record"
    try:
        if not isinstance(raw, dict):
            raise ValueError("Record must be an object")
        name = identifier(
            raw.get("designation") or (raw.get("neo_data") or {}).get("designation"),
            "designation",
        )
        if raw.get("schema_version") == "polling-screening-record-v1":
            return verify_record(raw)
        review = None
        if raw.get("schema_version") == "object-evidence-review-v1":
            return NEOClassification(
                name,
                review=verify_object_review(raw),
                input_record=raw,
                input_status="replayed_review_not_authenticated",
            )
        if any(
            raw.get(k) is True for k in ("synthetic", "is_synthetic", "mock", "demo")
        ):
            return NEOClassification(
                name, input_record=raw, input_status="declared_synthetic_unverified"
            )
        if "neo_data" in raw and raw["neo_data"] is not None:
            n = NEOData.from_dict(raw["neo_data"])
            normalized = n.to_dict()
            supplied = raw["neo_data"]
            # The model recomputes its old orbital-only completeness cache. Do
            # not reject valid primary fields or use that cache as scientific quality.
            same_snapshot = set(normalized) == set(supplied) and all(
                normalized[k] == supplied[k] for k in normalized if k != "completeness"
            )
            if n.designation != name or not same_snapshot:
                raise ValueError("Snapshot identity/schema/default mismatch")
            if any(
                n.evidence_metadata.get(k) is True
                for k in ("synthetic", "is_synthetic", "mock", "demo")
            ):
                return NEOClassification(
                    name, input_record=raw, input_status="declared_synthetic_unverified"
                )
            if raw.get("screening_assessment") is not None:
                p = raw["screening_assessment"]
                actual = service.analyze_neo(
                    n, p["evidence_snapshot"]["additional"]
                ).to_payload()
                if actual != p:
                    raise ValueError("Source and supplied assessment do not match")
                review = build_assessment_review(actual, neo=n, policy=service.policy)
            else:
                review = build_object_review(n, policy=service.policy)
            status = "typed_snapshot_not_authenticated"
        elif (
            isinstance(raw.get("screening_assessment"), dict)
            or raw.get("schema_version") == "screening-assessment-v1"
        ):
            p = (
                raw
                if raw.get("schema_version") == "screening-assessment-v1"
                else raw["screening_assessment"]
            )
            assessment = service.replay(p["evidence_snapshot"], p["evidence_id"])
            if (
                p.get("policy_id") != assessment.policy_id
                or assessment.designation != name
            ):
                raise ValueError("Assessment binding mismatch")
            review = build_assessment_review(
                assessment.to_payload(), policy=service.policy
            )
            status = "replayed_assessment_not_authenticated"
        else:
            status = "legacy_unverified"
        return NEOClassification(
            name, review=review, input_record=raw, input_status=status
        )
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return NEOClassification(name, input_record=raw, input_status="invalid_input")


def safe_classification(value):
    if isinstance(value, NEOClassification):
        try:
            return verify_record(value.to_payload())
        except (ValueError, KeyError, TypeError, OverflowError):
            return NEOClassification(
                value.designation,
                input_record={"retention_error": "CLASSIFICATION_CHANGED"},
                input_status="invalid_input",
            )
    return classify_record(value)


def rank_key(value):
    c = safe_classification(value)
    score = c.screening_score
    return (
        score is None,
        -score if score is not None else 0,
        c.designation,
        c.input_id,
    )


@dataclass
class DashboardMetrics:
    total_objects: int
    artificial_confirmed: Optional[int] = None
    suspicious_objects: Optional[int] = None
    edge_cases: Optional[int] = None
    natural_objects: Optional[int] = None
    high_confidence_detections: Optional[int] = None
    data_quality_score: Optional[float] = None
    analysis_timestamp: datetime = field(default_factory=utc_now)
    total_records: int = 0
    follow_up_candidates: int = 0
    not_flagged: int = 0
    inconclusive: int = 0
    records_with_inventory: int = 0


class ArtificialNEODashboard:
    """Historical entry point, now an evidence-bound candidate-screening dashboard."""

    def __init__(self, console=None, service=None):
        if console is None:
            try:
                from rich.console import Console

                console = Console()
            except ImportError:
                pass
        self.console = console
        self.service = service or ScreeningAnalysisService()
        self.classifications = []
        self.metrics = None
        self.thresholds = {
            "deprecated_origin_thresholds": None,
            "policy": self.service.describe_policy(),
        }

    async def analyze_polling_results(self, polling_results):
        self.classifications = self.classify_records(polling_results)
        self.metrics = self._calculate_metrics(self.classifications)
        return self.classifications

    def classify_records(self, records):
        if not isinstance(records, list) or len(records) > 10000:
            raise ValueError("Dashboard requires at most 10000 explicit records")
        return [safe_classification(r) if isinstance(r, NEOClassification)
                else classify_record(r, self.service) for r in records]

    async def _classify_neo(self, result):
        return classify_record(result, self.service)

    def _detect_orbital_anomalies(self, orbital_elements):
        raise NotImplementedError(
            "Legacy orbit cutoffs/round-number origin indicators are deprecated; inspect the canonical ranking trace"
        )

    def _analyze_temporal_patterns(self, result):
        raise NotImplementedError(
            "Unverified discovery/observation defaults are not scientific temporal evidence; inspect source chronology"
        )

    def _analyze_physical_characteristics(self, result):
        raise NotImplementedError(
            "Unitless diameter/metallicity-origin heuristics are deprecated; inspect typed physical evidence"
        )

    def _calculate_metrics(self, classifications):
        items = [safe_classification(c) for c in classifications]
        return DashboardMetrics(
            total_objects=len({c.designation for c in items}),
            total_records=len(items),
            follow_up_candidates=sum(
                c.category == "follow_up_candidate" for c in items
            ),
            not_flagged=sum(c.category == "not_flagged" for c in items),
            inconclusive=sum(c.category == "inconclusive" for c in items),
            records_with_inventory=sum(c.basic_presence is not None for c in items),
        )

    def display_dashboard(self, save_results=True, output_dir="dashboard_results"):
        self.classifications = [safe_classification(c) for c in self.classifications]
        self.metrics = self._calculate_metrics(self.classifications)
        if self.console:
            self._display_rich_dashboard()
        else:
            self._display_text_dashboard()
        if save_results:
            return self._save_dashboard_results(output_dir)

    def _display_rich_dashboard(self):
        from .dashboard_view import render_dashboard

        render_dashboard(self.classifications, self.metrics, self.console)

    def _display_text_dashboard(self):
        from .dashboard_view import render_dashboard

        render_dashboard(self.classifications, self.metrics, None)

    def _display_category_details(self):
        from .dashboard_view import render_rows

        render_rows(self.classifications, self.console)

    def _display_risk_assessment(self):
        from .dashboard_view import message

        message(LIMITS, self.console)

    def _save_dashboard_results(self, output_dir, *, invocation_context=None):
        """Exclusive run directory, verified JSON and Markdown; never overwrite history."""
        from .dashboard_view import dashboard_markdown, message
        from .reporting import _software_metadata

        records = [safe_classification(c).to_payload() for c in self.classifications]
        report = {
            "schema_version": "polling-screening-dashboard-v1",
            "records": records,
            "software": _software_metadata(),
            "policies": record_policies(records),
            "limits": LIMITS,
        }
        if invocation_context is not None:
            if not isinstance(invocation_context, dict):
                raise ValueError("Invocation context must be a finite JSON object")
            report["invocation_context"] = clone(invocation_context)
        report["report_id"] = fingerprint(report)
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        run = path / (
            "screening_dashboard_"
            + utc_now().strftime("%Y%m%dT%H%M%S%fZ")
            + "_"
            + uuid4().hex[:8]
        )
        run.mkdir()
        (run / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        markdown = dashboard_markdown(self.classifications)
        if invocation_context is not None:
            context_json = json.dumps(report["invocation_context"], indent=2, ensure_ascii=True)
            context_json = context_json.replace("`", "\\u0060").replace("<", "\\u003c").replace(">", "\\u003e")
            markdown += "\n## Requested invocation context (not verified coverage)\n\n```json\n" + context_json + "\n```\n"
        (run / "report.md").write_text(markdown)
        from hashlib import sha256

        complete = {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in (run / "report.json", run / "report.md")
        }
        (run / "complete.json").write_text(json.dumps(complete, indent=2) + "\n")
        message("Screening report saved: " + str(run), self.console)
        return run


async def create_dashboard_from_polling(polling_results, display=True, save=True):
    dashboard = ArtificialNEODashboard()
    await dashboard.analyze_polling_results(polling_results)
    if display:
        dashboard.display_dashboard(save_results=save)
    return dashboard


def record_policies(records):
    """Frozen per-record policy descriptors, not an envelope-wide re-ranking."""
    result = {}
    for record in records:
        review = record["review"]
        if review is not None:
            description = ScreeningAnalysisService(
                ScreeningPolicy(**review["screening_policy"])
            ).describe_policy()
            if description["policy_id"] != review["assessment"]["policy_id"]:
                raise ValueError("Record policy cannot be reproduced")
            result[description["policy_id"]] = description
    return result


def records_from_file(data):
    """Current reports replay; old formats are re-screened/retained, never trusted."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise ValueError("Unsupported dashboard file")
    if data.get("schema_version") == "polling-screening-dashboard-v1":
        if data.get("report_id") != fingerprint(
            {k: v for k, v in data.items() if k != "report_id"}
        ):
            raise ValueError("Dashboard fingerprint mismatch")
        records = [verify_record(r).to_payload() for r in data["records"]]
        if data["policies"] != record_policies(records):
            raise ValueError("Dashboard record policies do not reproduce")
        return records
    for key in ("classifications", "results", "objects"):
        if key in data:
            if not isinstance(data[key], list):
                raise ValueError("Expected a record list")
            return data[key]
    return [data]
