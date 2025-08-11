# aNEOS REST API Reference

Complete reference for the aNEOS REST API endpoints, authentication, and usage examples.

## 🌐 API Overview

The aNEOS REST API provides programmatic access to all platform functionality including NEO analysis, machine learning predictions, monitoring, and system administration.

**Base URL**: `http://localhost:8000/api/v1`  
**Documentation**: `http://localhost:8000/docs` (Interactive Swagger UI)  
**OpenAPI Spec**: `http://localhost:8000/openapi.json`

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