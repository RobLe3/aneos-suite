"""
SBDB (Small-Body Database) data source implementation.

This module provides integration with NASA's Small-Body Database API
for fetching Near Earth Object orbital elements and physical parameters.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime
try:
    from datetime import UTC  # Python 3.11+
except ImportError:
    from datetime import timezone as _tz
    UTC = _tz.utc

from .base import HTTPDataSource, FetchResult, DataSourceException
from ...config.settings import APIConfig
from ..cache import CacheManager
from ..models import NonGravitationalParameters
from ..orbit_evidence import OrbitModelParameter, OrbitalCovariance, finite_number
import copy
import hashlib
import json

logger = logging.getLogger(__name__)

class SBDBSource(HTTPDataSource):
    """NASA Small-Body Database data source."""
    
    def __init__(self, config: APIConfig, cache_manager: Optional[CacheManager] = None):
        super().__init__(
            name="SBDB",
            base_url=config.sbdb_url,
            config=config,
            cache_manager=cache_manager
        )
    
    async def fetch_orbital_elements(self, designation: str) -> FetchResult:
        """Fetch orbital elements from SBDB API."""
        try:
            params = {
                "sstr": designation,
                "phys-par": "true",
                "full-prec": "true",
                "cov": "mat",
            }

            response_data = await self._http_get("", params)

            # A1/A2/A3 and force-model constants are in orbit.model_pars.
            # Do not issue the undocumented nongrav=1 probe.

            if "orbit" not in response_data:
                return FetchResult(
                    success=False,
                    error_message=f"No orbital data found for {designation}",
                    source=self.name
                )
            
            # Parse SBDB response format
            orbital_data = self._parse_sbdb_response(response_data, designation)
            
            return FetchResult(
                success=True,
                data=orbital_data,
                source=self.name
            )
            
        except Exception as e:
            error_msg = f"SBDB API error for {designation}: {str(e)}"
            logger.error(error_msg)
            
            return FetchResult(
                success=False,
                error_message=error_msg,
                source=self.name
            )
    
    def _parse_sbdb_response(self, data: Dict[str, Any], designation: str) -> Dict[str, Any]:
        """Parse SBDB API response into standardized format."""
        if not isinstance(data, dict):
            raise ValueError('SBDB payload must be an object')
        signature = data.get('signature', {})
        if not isinstance(signature, dict):
            raise ValueError('SBDB signature must be an object')
        if 'orbit' in data and not isinstance(data['orbit'], dict):
            raise ValueError('SBDB orbit section must be an object')
        if signature and signature.get('version') != '1.3':
            raise ValueError(f"Unsupported SBDB schema version: {signature.get('version')}")
        orbital_data = {}

        # Extract orbital elements (list of {name, value, ...} dicts)
        if "orbit" in data:
            elements = data["orbit"].get("elements", [])

            # Maps the SBDB 'name' field to OrbitalElements field names
            element_mapping = {
                "e":  "eccentricity",
                "i":  "inclination",
                "a":  "semi_major_axis",
                "om": "ra_of_ascending_node",
                "w":  "arg_of_periapsis",
                "ma": "mean_anomaly",
            }

            for element in elements:
                sbdb_name = element.get("name")
                value = element.get("value")
                if sbdb_name in element_mapping and value is not None:
                    try:
                        orbital_data[element_mapping[sbdb_name]] = float(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing {sbdb_name}='{value}' for {designation}: {e}")

        # Native physical assertions are normalized only after response context exists.
        phys_list = data.get('phys_par', [])

        orbit_section = data.get('orbit', {})
        if not isinstance(orbit_section, dict):
            raise ValueError('SBDB orbit section must be an object')
        parameters = {}
        model_pars = orbit_section.get('model_pars', [])
        if not isinstance(model_pars, list):
            raise ValueError('SBDB model_pars must be an array')
        for item in model_pars:
            if not isinstance(item, dict) or 'name' not in item:
                raise ValueError('SBDB model parameter must be a named object')
            record = OrbitModelParameter.from_provider(item)
            if record.name in parameters:
                raise ValueError(f'Duplicate model parameter: {record.name}')
            parameters[record.name] = record
        native_cov = orbit_section.get('covariance')
        if native_cov is not None and (not isinstance(native_cov, dict) or
                                      not {'labels', 'data', 'epoch'}.issubset(native_cov)):
            raise ValueError('Incomplete SBDB covariance object')
        covariance = OrbitalCovariance.from_provider(native_cov) if native_cov is not None else None
        if covariance is not None:
            base = {'e', 'q', 'tp', 'node', 'peri', 'i'}
            actual = set(covariance.labels)
            if not base.issubset(actual):
                raise ValueError('Covariance missing required orbital-coordinate labels')
            expected = {key for key, parameter in parameters.items() if parameter.kind in {'EST', 'CON'} and parameter.n}
            if not expected.issubset(actual):
                raise ValueError('Covariance missing estimated/considered model parameters')
            if actual - base - set(parameters):
                raise ValueError('Unknown covariance labels without model parameter definitions')
            if any(parameters[key].kind == 'SET' for key in actual - base):
                raise ValueError('Fixed model parameter cannot be an estimated covariance coordinate')
        solution = {
            'source': 'JPL SBDB', 'schema_version': signature.get('version'),
            'retrieved_at': datetime.now(UTC).isoformat(),
            'canonical_response_sha256': hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            'orbit_id': orbit_section.get('orbit_id'), 'epoch_jd_tdb': orbit_section.get('epoch'),
            'solution_date': orbit_section.get('soln_date'), 'time_scale': 'TDB',
            'equinox': orbit_section.get('equinox'),
            'fit_metadata': {key: orbit_section.get(key) for key in
                             ('rms', 'data_arc', 'n_obs_used', 'n_del_obs_used', 'n_dop_obs_used', 'producer')},
            'object': copy.deepcopy(data.get('object', {})),
            'elements': copy.deepcopy(orbit_section.get('elements', [])),
            'first_obs': orbit_section.get('first_obs'), 'last_obs': orbit_section.get('last_obs'),
        }
        if solution['epoch_jd_tdb'] is not None:
            finite_number(solution['epoch_jd_tdb'], 'orbit epoch')
            from ..epoch_evidence import NativeEpoch
            orbital_data['native_epoch'] = NativeEpoch(str(solution['epoch_jd_tdb']), 'JD', 'TDB', 'SBDB').to_dict()
        if any(key in parameters for key in ('A1', 'A2', 'A3')):
            ng = NonGravitationalParameters(parameters=parameters, covariance=covariance, solution=solution)
            orbital_data['_nongrav'] = ng.to_dict()
        else:
            orbital_data['_nongrav'] = None
        from ..physical_evidence import build_sbdb_evidence, physical_projection
        physical_evidence=build_sbdb_evidence(phys_list,{
            'object':copy.deepcopy(data.get('object',{})),
            'retrieved_at':solution['retrieved_at'],
            'response_sha256':solution['canonical_response_sha256'],
            'container_orbit_id':solution['orbit_id'],
            'container_orbit_epoch_jd_tdb':solution['epoch_jd_tdb'],
            'schema_version':signature.get('version'),
        })
        projection=physical_projection(physical_evidence)
        compatibility={'diameter_km':'diameter','rotation_period_hours':'rot_per'}
        orbital_data['_physical']={compatibility.get(k,k):v for k,v in projection.items()}
        orbital_data['_evidence_metadata'] = {
            'schema_status': 'supported' if signature else 'unverified_missing_signature',
            'signature': copy.deepcopy(signature), 'solution': solution,
            'nongrav_status': 'available' if orbital_data['_nongrav'] else 'not_available',
            'covariance': covariance.to_dict() if covariance else None,
            'model_pars': [parameter.to_dict() for parameter in parameters.values()],
            'physical_measurements': copy.deepcopy(phys_list),
            'physical_evidence':physical_evidence,
        }

        # Observation window (first_obs / last_obs from orbit section)
        orbit_section = data.get("orbit", {})
        for sbdb_key, out_key in (("first_obs", "first_observation_date"),
                                   ("last_obs",  "last_observation_date")):
            raw_date = orbit_section.get(sbdb_key)
            if raw_date:
                orbital_data[out_key] = str(raw_date)

        # Source metadata (filtered out before OrbitalElements construction)
        orbital_data["_source"] = self.name
        orbital_data["_designation"] = designation
        orbital_data["_fetched_at"] = datetime.now(UTC).isoformat()

        return orbital_data
    
    def _parse_epoch(self, epoch_str: str) -> Optional[str]:
        """Parse SBDB epoch format."""
        try:
            # SBDB typically uses format like "2023-01-01"
            parsed_date = datetime.strptime(epoch_str, "%Y-%m-%d")
            return parsed_date.isoformat()
        except ValueError:
            try:
                # Try alternative format
                parsed_date = datetime.strptime(epoch_str, "%Y-%b-%d")
                return parsed_date.isoformat()
            except ValueError:
                logger.warning(f"Could not parse SBDB epoch format: {epoch_str}")
                return None
    
    async def health_check(self) -> bool:
        """Perform health check by testing a known object (Apophis / 99942)."""
        try:
            result = await self.fetch_orbital_elements("99942")
            return result.success

        except Exception as e:
            logger.error(f"SBDB health check failed: {e}")
            return False