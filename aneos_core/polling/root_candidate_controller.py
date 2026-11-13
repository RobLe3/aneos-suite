"""Root acquisition orchestration, not another detector or persistence engine."""
from ..data.candidate_discovery import CADCandidateService
from ..data.models import NEOData
from ..data.enrichment import clone
from ..analysis.screening import ScreeningAnalysisService
from ..utils.errors import EvidencePersistenceError


def validate_enrichment_budget(value):
    if type(value) is not int or not 0 <= value <= 20:
        raise ValueError('Explicit enrichment budget must be 0..20 object acquisitions')
    return value


class RootCandidateController:
    def __init__(self, service=None, fetcher_factory=None):
        self.service = service if service is not None else CADCandidateService()
        self.fetcher_factory = fetcher_factory

    def poll(self, start, end, limit, *, enrich_objects=0):
        validate_enrichment_budget(enrich_objects)
        saved = self.service.poll(start, end, limit=limit, object_policy='cad-physical-v1')
        try:
            return saved, self.consume(saved, enrich_objects=enrich_objects)
        except Exception as exc:
            exc.candidate_query = clone(saved)
            raise

    def replay(self, query_id):
        if self.service.repository is None:
            from aneos_api.candidate_repository import CandidateQueryRepository
            self.service.repository = CandidateQueryRepository()
        saved = self.service.repository.load(query_id)
        if saved is None:
            raise KeyError('Candidate query not found')
        return saved, self.consume(saved)  # Never enrich/poll on replay.

    def consume(self, saved, *, enrich_objects=0):
        validate_enrichment_budget(enrich_objects)
        if saved.get('commit_status') != 'committed':
            raise EvidencePersistenceError('Only committed candidate readback may be consumed')
        result = []
        fetcher = None
        for index, packet in enumerate(saved['records']):
            candidate = NEOData.from_dict(packet['neo_data'])
            neo = candidate
            local_projection = None
            enrichment = {'status': 'not_requested', 'object_invocation_budget': enrich_objects,
                          'provider_requests': 'separate installed source plan, caching and retry policy'}
            if index < enrich_objects:
                if fetcher is None:
                    if self.fetcher_factory is None:
                        from ..data.fetcher import DataFetcher
                        fetcher = DataFetcher()
                    else:
                        fetcher = self.fetcher_factory()
                from aneos_api.acquisition_repository import SQLEvidenceRepository
                repository = getattr(fetcher, 'evidence_repository', None) or SQLEvidenceRepository()
                if repository.engine.url != self.service.repository.engine.url:
                    raise EvidencePersistenceError('Discovery/enrichment must use the same evidence ledger')
                fetcher.evidence_repository = repository
                fetcher.candidate_query_id = saved['receipt']['query_id']
                enriched = fetcher.fetch_neo_data(candidate.designation)
                # DataFetcher binds a committed acquisition receipt before returning.
                if not isinstance(enriched, NEOData) or enriched.designation != candidate.designation:
                    raise ValueError('Enrichment object identity mismatch')
                receipt = enriched.evidence_metadata.get('acquisition', {}).get('storage_receipt')
                if not isinstance(receipt, dict) or receipt.get('commit_status') != 'committed':
                    raise EvidencePersistenceError('Enrichment has no committed acquisition receipt')
                from ..data.acquisition import result_from_bundle
                persisted = repository.load_acquisition(receipt['query_id'])
                if (persisted is None or persisted['receipt'] != receipt
                        or result_from_bundle(persisted['bundle'], persisted['receipt']).to_dict() != enriched.to_dict()):
                    raise EvidencePersistenceError('Enrichment commit readback mismatch')
                from aneos_api.enrichment_store import latest_projection
                with repository.sessions() as db:
                    local_projection = latest_projection(db, candidate.designation)
                if local_projection is None:
                    raise EvidencePersistenceError('Combined persisted object projection unavailable')
                neo = NEOData.from_dict(local_projection['projection']['neo_data'])
                new_sources = [o['source'] for o in persisted['bundle']['outcomes']
                               if o['source'] != 'CAD' and o['status'] == 'available']
                enrichment = {**enrichment, 'status': 'committed' if new_sources else 'no_new_source_data',
                              'new_source_contributions': new_sources, 'coverage': clone(persisted['bundle']['coverage']),
                              'cad_acquisition': 'reused_committed_collection_no_new_cad_request', 'receipt': clone(receipt)}
                # Persistence or validation failure propagates; no successful scientific fallback.
            elif enrich_objects:
                enrichment['status'] = 'outside_object_budget'
            assessment = ScreeningAnalysisService().analyze_neo(neo).to_payload()
            result.append({**clone(packet), 'neo_data': neo.to_dict(),
                'candidate_neo_data': candidate.to_dict(),
                'candidate_query_id': saved['receipt']['query_id'],
                'candidate_receipt_id': saved['receipt']['receipt_id'],
                'enrichment': enrichment, 'local_projection': clone(local_projection), 'screening_assessment': assessment,
                'limitations': ['Encounter-selected candidates are not independently observed anomalies.',
                    'Original candidate snapshot/all encounter rows remain in candidate_neo_data and the ledger.',
                    'Enrichment and collection are separate source assertions, not independent votes.',
                    'Scientific completeness and origin remain unestablished.']})
        return result
