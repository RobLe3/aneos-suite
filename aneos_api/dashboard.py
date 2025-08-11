"""
Web Dashboard for aNEOS API.

Provides a web-based interface for monitoring system status, viewing analysis results,
managing ML models, and system administration.
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path

try:
    from fastapi import APIRouter, Request, Depends, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    HAS_FASTAPI = True
    HAS_JINJA2 = True
except ImportError:
    HAS_FASTAPI = False
    HAS_JINJA2 = False
    logging.warning("FastAPI not available, dashboard disabled")

# Try to import Jinja2 separately for better error handling
try:
    import jinja2
    if HAS_FASTAPI and not HAS_JINJA2:
        HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    if HAS_FASTAPI:
        logging.warning("Jinja2 not available, templates disabled")

# Import moved to avoid circular imports
# from .app import get_aneos_app
from .auth import get_current_user

logger = logging.getLogger(__name__)

if HAS_FASTAPI:
    router = APIRouter()
    
    if HAS_JINJA2:
        # Setup templates and static files
        current_dir = Path(__file__).parent
        templates_dir = current_dir / "templates"
        static_dir = current_dir / "static"
        
        # Create directories if they don't exist
        templates_dir.mkdir(exist_ok=True)
        static_dir.mkdir(exist_ok=True)
        
        try:
            templates = Jinja2Templates(directory=str(templates_dir))
            # Static files
            router.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        except Exception as e:
            logger.warning(f"Failed to initialize templates: {e}")
            templates = None
    else:
        templates = None
else:
    # Fallback router for when FastAPI is not available
    class MockRouter:
        def get(self, *args, **kwargs): return lambda f: f
        def mount(self, *args, **kwargs): pass
    router = MockRouter()
    templates = None

@router.get("/", response_class=HTMLResponse)
async def dashboard_home(
    request: Request,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Main dashboard page."""
    if not HAS_FASTAPI:
        return HTMLResponse("<h1>Dashboard not available - FastAPI required</h1>")
    
    if not templates:
        return HTMLResponse("<h1>Dashboard not available - Templates not configured</h1>")
    
    try:
        # Get system status
        from .app import get_aneos_app  # Import here to avoid circular imports
        aneos_app = get_aneos_app()
        system_status = aneos_app.get_health_status()
        
        # Get basic metrics with safe fallbacks
        metrics_data = {
            'cpu_percent': 0,
            'memory_percent': 0,
            'total_analyses': 0,
            'model_predictions': 0,
            'status': 'limited_functionality'
        }
        
        if aneos_app.metrics_collector:
            try:
                system_metrics = aneos_app.metrics_collector.get_system_metrics()
                analysis_metrics = aneos_app.metrics_collector.get_analysis_metrics()
                ml_metrics = aneos_app.metrics_collector.get_ml_metrics()
                
                metrics_data.update({
                    'cpu_percent': system_metrics.cpu_percent if system_metrics else 0,
                    'memory_percent': system_metrics.memory_percent if system_metrics else 0,
                    'total_analyses': analysis_metrics.total_analyses if analysis_metrics else 0,
                    'model_predictions': ml_metrics.model_predictions if ml_metrics else 0,
                    'status': 'active'
                })
            except Exception as e:
                logger.error(f"Error getting dashboard metrics: {e}")
                metrics_data['error'] = str(e)
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "user": current_user,
            "system_status": system_status,
            "metrics": metrics_data,
            "page_title": "aNEOS Dashboard"
        })
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return HTMLResponse(f"<h1>Dashboard Error</h1><p>{str(e)}</p>", status_code=500)

@router.get("/monitoring", response_class=HTMLResponse)
async def monitoring_dashboard(
    request: Request,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """System monitoring dashboard."""
    if not HAS_FASTAPI:
        return HTMLResponse("<h1>Monitoring not available - FastAPI required</h1>")
    
    if not templates:
        return HTMLResponse("<h1>Monitoring not available - Templates not configured</h1>")
    
    return templates.TemplateResponse("monitoring.html", {
        "request": request,
        "user": current_user,
        "page_title": "System Monitoring"
    })

@router.get("/analysis", response_class=HTMLResponse)
async def analysis_dashboard(
    request: Request,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """NEO analysis dashboard."""
    if not HAS_FASTAPI:
        return HTMLResponse("<h1>Analysis not available - FastAPI required</h1>")
    
    if not templates:
        return HTMLResponse("<h1>Analysis not available - Templates not configured</h1>")
    
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "user": current_user,
        "page_title": "NEO Analysis"
    })

@router.get("/ml", response_class=HTMLResponse)
async def ml_dashboard(
    request: Request,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Machine Learning models dashboard."""
    if not HAS_FASTAPI:
        return HTMLResponse("<h1>ML Dashboard not available - FastAPI required</h1>")
    
    if not templates:
        return HTMLResponse("<h1>ML Dashboard not available - Templates not configured</h1>")
    
    return templates.TemplateResponse("ml.html", {
        "request": request,
        "user": current_user,
        "page_title": "ML Models"
    })

@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Administration dashboard."""
    if not HAS_FASTAPI:
        return HTMLResponse("<h1>Admin not available - FastAPI required</h1>")
    
    if not templates:
        return HTMLResponse("<h1>Admin not available - Templates not configured</h1>")
    
    # Check admin permissions
    if not current_user or current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "user": current_user,
        "page_title": "Administration"
    })

# Utility function to create dashboard templates
def create_dashboard_templates():
    """Create HTML templates for the dashboard."""
    if not HAS_FASTAPI or not HAS_JINJA2:
        return
    
    templates_dir = Path(__file__).parent / "templates"
    
    # Base template
    base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ page_title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 1rem; }
        .nav { background: #34495e; padding: 0.5rem; }
        .nav a { color: white; text-decoration: none; margin-right: 1rem; padding: 0.5rem; }
        .nav a:hover { background: #2c3e50; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .card { background: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; }
        .metric { text-align: center; }
        .metric-value { font-size: 2rem; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; }
        .status-healthy { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-critical { color: #e74c3c; }
        .btn { background: #3498db; color: white; padding: 0.5rem 1rem; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #2980b9; }
    </style>
</head>
<body>
    <header class="header">
        <h1>aNEOS - Advanced Near Earth Object detection System</h1>
        {% if user %}
            <p>Welcome, {{ user.username }} ({{ user.role }})</p>
        {% endif %}
    </header>
    
    <nav class="nav">
        <a href="/dashboard/">Dashboard</a>
        <a href="/dashboard/monitoring">Monitoring</a>
        <a href="/dashboard/analysis">Analysis</a>
        <a href="/dashboard/ml">ML Models</a>
        {% if user and user.role == 'admin' %}
            <a href="/dashboard/admin">Admin</a>
        {% endif %}
        <a href="/docs">API Docs</a>
    </nav>
    
    <div class="container">
        {% block content %}{% endblock %}
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>'''
    
    # Dashboard template
    dashboard_template = '''{% extends "base.html" %}

{% block content %}
<div class="card">
    <h2>System Status</h2>
    <p class="status-{{ 'healthy' if system_status.status == 'healthy' else 'warning' }}">
        Status: {{ system_status.status|title }}
    </p>
    <p>Version: {{ system_status.version }}</p>
    <p>Services Online: {{ system_status.services|length }}</p>
</div>

<div class="card">
    <h2>Key Metrics</h2>
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{{ "%.1f"|format(metrics.cpu_percent) }}%</div>
            <div class="metric-label">CPU Usage</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ "%.1f"|format(metrics.memory_percent) }}%</div>
            <div class="metric-label">Memory Usage</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ metrics.total_analyses }}</div>
            <div class="metric-label">Total Analyses</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ metrics.model_predictions }}</div>
            <div class="metric-label">ML Predictions</div>
        </div>
    </div>
</div>

<div class="card">
    <h2>Quick Actions</h2>
    <button class="btn" onclick="analyzeNEO()">Analyze NEO</button>
    <button class="btn" onclick="viewMetrics()">View Metrics</button>
    <button class="btn" onclick="checkAlerts()">Check Alerts</button>
</div>

<script>
    function analyzeNEO() {
        const designation = prompt("Enter NEO designation:");
        if (designation) {
            window.open(`/docs#/Analysis/analyze_neo_api_v1_analysis_analyze_post`, '_blank');
        }
    }
    
    function viewMetrics() {
        window.open('/dashboard/monitoring', '_blank');
    }
    
    function checkAlerts() {
        fetch('/api/v1/monitoring/alerts')
            .then(response => response.json())
            .then(data => alert(`Found ${data.length} alerts`))
            .catch(error => alert('Error fetching alerts'));
    }
</script>
{% endblock %}'''
    
    # Write templates
    try:
        (templates_dir / "base.html").write_text(base_template)
        (templates_dir / "dashboard.html").write_text(dashboard_template)
        
        # Create other templates
        for template_name in ["monitoring.html", "analysis.html", "ml.html", "admin.html"]:
            template_content = f'''{{%extends "base.html" %}}

{{%block content %}}
<div class="card">
    <h2>{template_name.replace('.html', '').title()}</h2>
    <p>This is the {template_name.replace('.html', '')} dashboard.</p>
    <p>Integration with API endpoints and real-time updates coming soon.</p>
</div>
{{%endblock %}}'''
            
            (templates_dir / template_name).write_text(template_content)
        
        logger.info("Dashboard templates created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create dashboard templates: {e}")

# Initialize templates on import
if HAS_FASTAPI and HAS_JINJA2:
    create_dashboard_templates()