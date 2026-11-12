# aNEOS REST API Reference

Complete reference for the aNEOS REST API endpoints, authentication, and usage examples.

## 🌐 API Overview

The aNEOS REST API provides programmatic access to all platform functionality including NEO analysis, machine learning predictions, monitoring, and system administration.

**Base URL**: `http://localhost:8000/api/v1`  
**Documentation**: `http://localhost:8000/docs` (Interactive Swagger UI)  
**OpenAPI Spec**: `http://localhost:8000/openapi.json`

## Screening contract correction — 2026-11-12

Existing GET/POST `/analysis/detect` and `/analysis/detect/batch` return an
uncalibrated `screening_score`, follow-up `candidate`, evidence/policy content IDs,
ranking-input snapshots and explicit limitations. Sigma, p-value, FDR,
artificial-origin probability, binary origin and confidence are nullable and are
**unavailable** in the installed model. `NOT_FLAGGED` is not a natural-origin
verdict. A designation-only spacecraft lookup cannot create significance.

A1/A2/A3 fitted evidence, covariance and provenance remain descriptive; they do
not imply propulsion. This change does not validate legacy `/analyze`, network,
ATLAS or historical front ends. See
[the screening contract](../engineering/screening-result-contract.md) for scope,
controlled deprecations and outstanding acceptance requirements. Earlier examples
that suggest calibrated inference are superseded for these detection routes.

## 🔐 Authentication

### API Key Authentication

```bash
# Include API key in header (find your keys in .env file)
curl -H "X-API-Key: <YOUR_API_KEY>" \
     http://localhost:8000/api/v1/analysis/analyze
```

### Bearer Token Authentication

```bash
# Include bearer token in Authorization header
curl -H "Authorization: Bearer your-jwt-token" \
     http://localhost:8000/api/v1/analysis/analyze
```

### Default API Keys (Development)

```bash
# Admin user
X-API-Key: <YOUR_ADMIN_API_KEY>

# Analyst user  
X-API-Key: <YOUR_ANALYST_API_KEY>

# Viewer user
X-API-Key: <YOUR_VIEWER_API_KEY>
```

## 📊 Analysis Endpoints

### Analyze Single NEO

Perform comprehensive analysis of a single Near Earth Object.

**Endpoint**: `POST /api/v1/analysis/analyze`

**Request**:
```json
{
  "designation": "2024 AB123",
  "force_refresh": false,
  "include_raw_data": false,
  "include_indicators": true
}
```

**Response**:
```json
{
  "success": true,
  "timestamp": "2025-08-04T10:30:00Z",
  "designation": "2024 AB123",
  "anomaly_score": {
    "overall_score": 0.234,
    "confidence": 0.891,
    "classification": "natural",
    "risk_factors": [],
    "indicator_scores": {
      "orbital_anomaly": {
        "raw_score": 0.15,
        "weighted_score": 0.12,
        "confidence": 0.85
      }
    }
  },
  "processing_time": 1.45,
  "data_quality": {
    "completeness": 0.95,
    "accuracy": 0.88,
    "recency": 0.92
  }
}
```

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/analysis/analyze" \
     -H "X-API-Key: <YOUR_ANALYST_API_KEY>" \
     -H "Content-Type: application/json" \
     -d '{
       "designation": "2024 AB123",
       "include_indicators": true
     }'
```

**Python Example**:
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/analysis/analyze",
    headers={"X-API-Key": "<YOUR_ANALYST_API_KEY>"},
    json={
        "designation": "2024 AB123",
        "include_indicators": True
    }
)

result = response.json()
print(f"Overall Score: {result['anomaly_score']['overall_score']}")
print(f"Classification: {result['anomaly_score']['classification']}")
```

### Batch Analysis

Analyze multiple NEOs in a single request.

**Endpoint**: `POST /api/v1/analysis/analyze/batch`

**Request**:
```json
{
  "designations": ["2024 AB123", "2024 BX1", "2024 CY2"],
  "force_refresh": false,
  "include_raw_data": false,
  "progress_webhook": "https://your-app.com/progress"
}
```

**Response**:
```json
{
  "batch_id": "batch_20250804_103000",
  "status": "processing",
  "total_neos": 3,
  "estimated_completion": "5-15 minutes",
  "progress_url": "/api/v1/analysis/batch/batch_20250804_103000/status"
}
```

### Get Analysis Results

Retrieve cached analysis results.

**Endpoint**: `GET /api/v1/analysis/results/{designation}`

**Response**: Same as single analysis response.

### Search Analysis Results

Search and filter analysis results with pagination.

**Endpoint**: `GET /api/v1/analysis/search`

**Query Parameters**:
- `query`: Search query string
- `classification`: Filter by classification (natural, suspicious, etc.)
- `min_score`: Minimum anomaly score
- `max_score`: Maximum anomaly score
- `page`: Page number (default: 1)
- `page_size`: Results per page (default: 50)

**Response**:
```json
{
  "items": [
    {
      "designation": "2024 AB123",
      "anomaly_score": 0.234,
      "classification": "natural",
      "analysis_date": "2025-08-04T10:30:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 50,
  "total_pages": 1
}
```

## 🤖 Machine Learning Endpoints

### ML Prediction

Get machine learning-based anomaly predictions.

**Endpoint**: `POST /api/v1/prediction/predict`

**Request**:
```json
{
  "designation": "2024 AB123",
  "use_cache": true,
  "model_id": "ensemble_v1"
}
```

**Response**:
```json
{
  "success": true,
  "designation": "2024 AB123",
  "anomaly_score": 0.756,
  "anomaly_probability": 0.834,
  "is_anomaly": true,
  "confidence": 0.923,
  "model_id": "ensemble_v1",
  "feature_contributions": [
    {
      "feature_name": "orbital_eccentricity",
      "contribution": 0.15,
      "feature_value": 0.82
    }
  ],
  "model_predictions": {
    "isolation_forest": 0.78,
    "one_class_svm": 0.73,
    "autoencoder": 0.76
  }
}
```

### Model Management

**List Available Models**: `GET /api/v1/prediction/models`
```json
{
  "available_models": [
    {
      "model_id": "isolation_forest_v1",
      "type": "isolation_forest",
      "trained_date": "2025-08-01T12:00:00Z",
      "performance_score": 0.89
    }
  ],
  "active_model": "ensemble_v1",
  "ensemble_enabled": true
}
```

**Activate Model**: `POST /api/v1/prediction/models/{model_id}/activate`

### Feature Analysis

**Get NEO Features**: `GET /api/v1/prediction/features/{designation}`
```json
{
  "designation": "2024 AB123",
  "features": [0.15, 0.82, 0.34, ...],
  "feature_names": ["orbital_eccentricity", "inclination", ...],
  "feature_quality": 0.91,
  "total_features": 87
}
```

## 📊 Monitoring Endpoints

### System Health

**Health Check**: `GET /health`
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "version": "2.0.0",
  "services": {
    "analysis_pipeline": true,
    "ml_predictor": true,
    "database": true
  }
}
```

### Metrics

**Current Metrics**: `GET /api/v1/monitoring/metrics`
```json
{
  "success": true,
  "system_metrics": {
    "timestamp": "2025-08-04T10:30:00Z",
    "cpu_percent": 25.5,
    "memory_percent": 45.2,
    "disk_usage_percent": 12.8
  },
  "analysis_metrics": {
    "total_analyses": 1523,
    "successful_analyses": 1487,
    "cache_hit_rate": 0.85,
    "average_processing_time": 2.3
  },
  "ml_metrics": {
    "model_predictions": 892,
    "prediction_latency": 0.15,
    "ensemble_agreement": 0.91
  }
}
```

**Metrics History**: `GET /api/v1/monitoring/metrics/history?hours=24`

### Alerts

**List Alerts**: `GET /api/v1/monitoring/alerts`
```json
[
  {
    "alert_id": "alert_001",
    "alert_type": "system_performance",
    "alert_level": "medium",
    "title": "High CPU Usage",
    "message": "CPU usage above 80% for 10 minutes",
    "timestamp": "2025-08-04T10:25:00Z",
    "acknowledged": false,
    "resolved": false
  }
]
```

**Acknowledge Alert**: `POST /api/v1/monitoring/alerts/{alert_id}/acknowledge`

## ⚙️ Administration Endpoints

### System Status

**System Status**: `GET /api/v1/admin/status`
```json
{
  "status": "healthy",
  "uptime_seconds": 7200,
  "services": {
    "analysis_pipeline": true,
    "ml_predictor": true,
    "database": true
  },
  "version": "2.0.0"
}
```

### Configuration

**Get Configuration**: `GET /api/v1/admin/config`
```json
{
  "api_config": {
    "max_concurrent_requests": 100,
    "request_timeout": 300
  },
  "analysis_config": {
    "max_batch_size": 100,
    "cache_ttl": 3600
  },
  "ml_config": {
    "model_cache_size": 10,
    "ensemble_threshold": 0.7
  }
}
```

### Training Management

**Start Training**: `POST /api/v1/admin/training/start`
```json
{
  "designations": ["2024 AB123", "2024 BX1", ...],
  "model_types": ["isolation_forest", "one_class_svm"],
  "use_ensemble": true,
  "hyperparameter_optimization": true
}
```

**Training Status**: `GET /api/v1/admin/training/{session_id}/status`

## 🌊 Streaming Endpoints

### WebSocket Connection

**Connect**: `WebSocket /api/v1/stream/ws/{session_id}`

**Message Format**:
```json
{
  "type": "subscribe",
  "event_types": ["analysis_complete", "alert_generated"]
}
```

**Event Format**:
```json
{
  "event_type": "analysis_complete",
  "timestamp": "2025-08-04T10:30:00Z",
  "data": {
    "designation": "2024 AB123",
    "anomaly_score": 0.234
  }
}
```

### Server-Sent Events

**Live Metrics**: `GET /api/v1/stream/metrics/live`
```
event: metrics_update
data: {"cpu_percent": 25.5, "memory_percent": 45.2}

event: metrics_update  
data: {"cpu_percent": 26.1, "memory_percent": 45.8}
```

**Live Alerts**: `GET /api/v1/stream/alerts/live`

## 📝 Error Handling

### Error Response Format

```json
{
  "error": "Validation error",
  "status_code": 400,
  "timestamp": "2025-08-04T10:30:00Z",
  "details": {
    "field": "designation",
    "message": "Invalid NEO designation format"
  }
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created |
| 400 | Bad Request | Invalid request data |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

## 🔄 Rate Limiting

**Default Limits**:
- 100 requests per minute per API key
- Burst allowance: 20 requests
- Headers returned: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

**Rate Limit Response**:
```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60,
  "timestamp": "2025-08-04T10:30:00Z"
}
```

## 📚 SDK Examples

### Python SDK Example

```python
import requests
import json
from datetime import datetime

class ANEOSClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
    
    def analyze_neo(self, designation, include_indicators=True):
        """Analyze a single NEO."""
        response = self.session.post(
            f"{self.base_url}/api/v1/analysis/analyze",
            json={
                "designation": designation,
                "include_indicators": include_indicators
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_system_health(self):
        """Get system health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict_anomaly(self, designation):
        """Get ML prediction for NEO."""
        response = self.session.post(
            f"{self.base_url}/api/v1/prediction/predict",
            json={"designation": designation}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = ANEOSClient(api_key="<YOUR_ANALYST_API_KEY>")

# Analyze NEO
result = client.analyze_neo("2024 AB123")
print(f"Anomaly Score: {result['anomaly_score']['overall_score']}")

# Get ML prediction
prediction = client.predict_anomaly("2024 AB123")
print(f"ML Prediction: {prediction['anomaly_probability']}")

# Check system health
health = client.get_system_health()
print(f"System Status: {health['status']}")
```

### JavaScript SDK Example

```javascript
class ANEOSClient {
    constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
        this.baseUrl = baseUrl;
        this.apiKey = apiKey;
    }
    
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...(this.apiKey && { 'X-API-Key': this.apiKey }),
                ...options.headers
            },
            ...options
        };
        
        const response = await fetch(url, config);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    }
    
    async analyzeNEO(designation, includeIndicators = true) {
        return this.request('/api/v1/analysis/analyze', {
            method: 'POST',
            body: JSON.stringify({
                designation,
                include_indicators: includeIndicators
            })
        });
    }
    
    async getSystemHealth() {
        return this.request('/health');
    }
    
    async predictAnomaly(designation) {
        return this.request('/api/v1/prediction/predict', {
            method: 'POST',
            body: JSON.stringify({ designation })
        });
    }
}

// Usage
const client = new ANEOSClient('http://localhost:8000', '<YOUR_ANALYST_API_KEY>');

// Analyze NEO
client.analyzeNEO('2024 AB123')
    .then(result => {
        console.log('Anomaly Score:', result.anomaly_score.overall_score);
    })
    .catch(error => {
        console.error('Analysis failed:', error);
    });
```

## 🔧 Testing the API

### Using cURL

```bash
# Test health endpoint
curl http://localhost:8000/health

# Analyze NEO with authentication
curl -X POST \
  -H "X-API-Key: <YOUR_ANALYST_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"designation": "2024 AB123"}' \
  http://localhost:8000/api/v1/analysis/analyze

# Get system metrics
curl -H "X-API-Key: <YOUR_ADMIN_API_KEY>" \
  http://localhost:8000/api/v1/monitoring/metrics
```

### Using httpie

```bash
# Install httpie
pip install httpie

# Test endpoints
http GET localhost:8000/health
http POST localhost:8000/api/v1/analysis/analyze \
  X-API-Key:<YOUR_ANALYST_API_KEY> \
  designation="2024 AB123"
```

### Interactive Testing

Access the interactive API documentation at:
**http://localhost:8000/docs**

Features:
- Try all endpoints directly in the browser
- Automatic request/response examples
- Built-in authentication testing
- Schema validation
- Response format documentation

---

**The aNEOS REST API provides comprehensive programmatic access to all platform capabilities. Use the interactive documentation for real-time testing and exploration!** 🚀
### Non-gravitational evidence sidecar (issue #5)

`GET /api/v1/analysis/detect` and existing batch detection results include
`non_gravitational_evidence` when catalogue coefficients are supplied. Missing
coefficients remain `not_available`/null, not zero. This field is descriptive
orbit-fit evidence, not a new origin classifier.

The record retains A1/A2/A3 native values, uncertainty, units, SET/EST/CON status,
raw numerical literals, force-model constants, solution identity, provider schema,
retrieval time and canonical-response hash. Supported SI conversions label both
value and sigma as **coefficients**, not instantaneous accelerations. Unknown
units are retained natively and do not receive an invented conversion.

Covariance carries its native labels, order, matrix, raw matrix, TDB epoch and
solution-epoch elements. `covariance_applies_to_display_epoch=false` means that
it must not be applied to displayed elements without explicit propagation.
Off-diagonal correlations are preserved. The API does not infer propulsion from
A2, nor omit A3 in favor of a scalar velocity/acceleration proxy.

This does not fix the legacy score's calibration or establish artificial origin;
those remain separate work in issues #6 and #8–#11.


### CAD evidence correction (2026-11-12 review branch)

Encounter evidence in screening snapshots preserves native TDB, body, provenance
and query coverage; it is not raw astrometry or manoeuvre evidence. Supplemental
history queries can be bounded/partial. Source failures are explicit and must not
be read as empty successful surveys. See
[CAD integrity scope](../engineering/cad-evidence-integrity.md). No new REST route
or calibrated physical-origin capability is introduced.


### Legacy validation consumers (2026-11-12 review branch)

Required validation and dashboard responses expose nullable confidence/FP
probability and explicit validation status. Workflow outcomes are not classifier
accuracy. Scatter coordinates use a supplied uncalibrated screening rank, never
`1 - false_positive_probability`. Unknown coordinates are not plotted as zero or
perfect confidence. Standalone core legacy `/analyze` response migration remains
unfinished; do not infer that its older examples establish scientific validation.
See [the contract correction](../engineering/legacy-validation-integrity.md).

### Core `/analyze` correction (ADR-069, 2026-11-12 review branch)

`POST /api/v1/analysis/analyze` now acquires typed evidence and returns the same
`screening_assessment` as shared screening for identical input/policy. The
compatibility `anomaly_score.overall_score` is **uncalibrated ranking units**;
confidence is null. `status` distinguishes completed ranking from inconclusive
input; source failure is HTTP 503 and invalid measurements HTTP 422. Neither
condition means natural origin. Results include evidence/policy IDs and replay
inputs; the full observational fit/calibration capability is still unavailable.

`force_refresh` reaches DataFetcher. `include_indicators=false` omits descriptive
proxy diagnostics from the compatibility score view, not the canonical assessment.
`include_raw_data` controls the typed source-data sidecar. The existing batch
route now forwards refresh/raw-data flags; its jobs are still in-memory.
No designation-only result cache is read by `/analyze`.

Analysis JSON export retains a structured assessment and reproducibility metadata;
CSV identifies raw ranking units, classification and evidence/policy IDs. These
are in-memory convenience exports, not durable append-only case history. Other
advertised analysis export formats are explicitly unsupported rather than
silently returning JSON. ML prediction with the new core contract returns HTTP
503 until a compatible independently calibrated model is integrated; old cached
predictions cannot bypass that guard. Standalone research models are retained.

The separately mocked enhanced REST endpoint is **not** repaired by these core
changes. Do not use it as a scientific assessment; follow #6/#12 for its migration.

### Enhanced REST correction (ADR-070, 2026-11-12 review branch)

This supersedes the enhanced-REST migration warning in the ADR-069 section above.
The existing declared routes are now registered in the main application under
its existing authentication dependency:

- `POST /api/v1/analysis/enhanced`: existing request object; same canonical
  screening assessment as `/analyze` for identical evidence/policy. Source failure
  is 503, invalid source evidence 422, processing failure 500. Missing orbit can
  return an inconclusive assessment with null rank, never a natural/zero result.
- `POST /api/v1/analysis/enhanced/batch`: existing bare JSON list body; existing
  `force_refresh` and `enable_validation` query flags. One to 100 nonblank
  designations; duplicates and all per-object failures retain their input order.
  Failed items have null rank/assessment and explicit status/errors, not removal.
- `GET /api/v1/analysis/enhanced/summary?days=7`: descriptive stored screening
  rows by database assessment date, not observation date or independent-object
  performance. Replayable current-policy payloads are summarized; legacy,
  modified or unsupported records are counted as excluded, not rewritten.
  Scalar legacy score/classification columns do not supply validation evidence.

Enhanced results expose `screening_assessment`, provenance/replay IDs, optional
raw-data and proxy sidecars, required-review outcomes and execution options.
`overall_score` is uncalibrated ranking units; confidence, FP probability,
uncertainty intervals and uninstrumented diagnostic timings are null. Top-level
elapsed time is measured. Optional-review disablement is `not_requested`, never
an approval. Method/sampling preferences are recorded but explicitly not
performed without an installed scientific model; changing `validation_method`
cannot bypass prerequisites. A workflow review pass is not an origin finding.

Summary probability/risk estimates remain null (`not_reliably_estimated`),
recommendations remain unknown without recorded valid review evidence, and empty
rank statistics are null rather than zero. Counts are operational descriptions,
not FDR, classifier accuracy or independent validation. New enhanced reports are
not yet persisted by these routes; summaries concern existing stored screening
records, not a complete enhanced-review history. Jobs/case history, provider-wide
rate control and crash/restart recovery remain #15/#13/#16.

The enhanced review adapter also rejects incomplete, truthy, errored or stale
workflow passes using explicit stage outcomes and ranking/full-review IDs. A valid
candidate remains visible when review transport fails. Unsupported legacy review
probability/confidence scalars remain null until a calibrated model contract is
installed; native measured parameter evidence is preserved separately.
