/**
 * aNEOS Real-Time Validation Dashboard
 * 
 * Comprehensive JavaScript application for real-time monitoring of the
 * validation pipeline with interactive charts, WebSocket updates, and
 * responsive visualization components.
 */

class ValidationDashboard {
    constructor() {
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.charts = {};
        this.currentView = 'overview';
        this.updateIntervals = {};
        
        // Configuration
        this.config = {
            websocketUrl: this.getWebSocketUrl(),
            updateInterval: 1000,
            chartColors: {
                primary: '#3498db',
                success: '#27ae60',
                warning: '#f39c12',
                danger: '#e74c3c',
                info: '#17a2b8',
                secondary: '#6c757d'
            }
        };
        
        // Initialize dashboard
        this.init();
    }
    
    /**
     * Initialize the dashboard application
     */
    async init() {
        try {
            console.log('Initializing aNEOS Validation Dashboard...');
            
            // Setup DOM elements
            this.setupDOM();
            
            // Initialize charts
            this.initializeCharts();
            
            // Connect WebSocket for real-time updates
            this.connectWebSocket();
            
            // Setup event handlers
            this.setupEventHandlers();
            
            // Load initial data
            await this.loadInitialData();
            
            // Start update loops
            this.startUpdateLoops();
            
            console.log('Dashboard initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize dashboard:', error);
            this.showError('Failed to initialize dashboard: ' + error.message);
        }
    }
    
    /**
     * Setup DOM elements and structure
     */
    setupDOM() {
        // Create main dashboard structure if not exists
        if (!document.getElementById('dashboard-container')) {
            this.createDashboardStructure();
        }
        
        // Setup navigation
        this.setupNavigation();
        
        // Setup connection status indicator
        this.setupConnectionStatus();
    }
    
    /**
     * Create the main dashboard HTML structure
     */
    createDashboardStructure() {
        const body = document.body;
        
        const dashboardHTML = `
            <!-- Dashboard Header -->
            <header class="dashboard-header">
                <h1>aNEOS Real-Time Validation Dashboard</h1>
                <div class="status-indicator">
                    <span class="status-dot" id="system-status-dot"></span>
                    <span id="system-status-text">System Status: Loading...</span>
                </div>
            </header>
            
            <!-- Navigation -->
            <nav class="dashboard-nav">
                <button class="nav-button active" data-view="overview">Overview</button>
                <button class="nav-button" data-view="validation">Validation Pipeline</button>
                <button class="nav-button" data-view="detection">Detection Statistics</button>
                <button class="nav-button" data-view="alerts">Alerts & Monitoring</button>
                <button class="nav-button" data-view="analytics">Analytics</button>
            </nav>
            
            <!-- Main Dashboard Container -->
            <main class="dashboard-container" id="dashboard-container">
                <!-- System Overview Cards -->
                <div class="dashboard-card" id="system-overview-card">
                    <div class="card-header">
                        <h3 class="card-title">System Overview</h3>
                    </div>
                    <div class="metrics-grid" id="system-metrics">
                        <!-- Dynamic content -->
                    </div>
                </div>
                
                <!-- Validation Pipeline Performance -->
                <div class="dashboard-card" id="validation-performance-card">
                    <div class="card-header">
                        <h3 class="card-title">Validation Pipeline Performance</h3>
                    </div>
                    <div class="chart-container">
                        <canvas id="stage-performance-chart"></canvas>
                    </div>
                </div>
                
                <!-- Real-Time Validation Results -->
                <div class="dashboard-card" id="validation-results-card">
                    <div class="card-header">
                        <h3 class="card-title">Real-Time Validation Results</h3>
                    </div>
                    <div class="chart-container">
                        <canvas id="validation-scatter-chart"></canvas>
                    </div>
                </div>
                
                <!-- Detection Statistics -->
                <div class="dashboard-card" id="detection-stats-card">
                    <div class="card-header">
                        <h3 class="card-title">Detection Statistics</h3>
                    </div>
                    <div class="metrics-grid" id="detection-metrics">
                        <!-- Dynamic content -->
                    </div>
                    <div class="chart-container small">
                        <canvas id="recommendation-distribution-chart"></canvas>
                    </div>
                </div>
                
                <!-- Processing Trends -->
                <div class="dashboard-card" id="processing-trends-card">
                    <div class="card-header">
                        <h3 class="card-title">Processing Trends</h3>
                    </div>
                    <div class="chart-container">
                        <canvas id="processing-trends-chart"></canvas>
                    </div>
                </div>
                
                <!-- Artificial Object Alerts -->
                <div class="dashboard-card" id="alerts-card">
                    <div class="card-header">
                        <h3 class="card-title">Artificial Object Alerts</h3>
                        <span class="alert-count" id="alert-count">0</span>
                    </div>
                    <div class="alert-list" id="alert-list">
                        <!-- Dynamic content -->
                    </div>
                </div>
                
                <!-- Stage Performance Details -->
                <div class="dashboard-card" id="stage-details-card">
                    <div class="card-header">
                        <h3 class="card-title">Stage Performance Details</h3>
                    </div>
                    <div class="stage-list" id="stage-list">
                        <!-- Dynamic content -->
                    </div>
                </div>
                
                <!-- Module Availability -->
                <div class="dashboard-card" id="module-availability-card">
                    <div class="card-header">
                        <h3 class="card-title">Validation Module Availability</h3>
                    </div>
                    <div class="chart-container small">
                        <canvas id="module-availability-chart"></canvas>
                    </div>
                </div>
                
                <!-- System Health -->
                <div class="dashboard-card" id="system-health-card">
                    <div class="card-header">
                        <h3 class="card-title">System Health</h3>
                    </div>
                    <div class="metrics-grid" id="health-metrics">
                        <!-- Dynamic content -->
                    </div>
                </div>
            </main>
            
            <!-- Connection Status -->
            <div class="connection-status connecting" id="connection-status">
                Connecting...
            </div>
        `;
        
        body.innerHTML = dashboardHTML;
    }
    
    /**
     * Setup navigation functionality
     */
    setupNavigation() {
        const navButtons = document.querySelectorAll('.nav-button');
        
        navButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const view = e.target.getAttribute('data-view');
                this.switchView(view);
                
                // Update active button
                navButtons.forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');
            });
        });
    }
    
    /**
     * Setup connection status indicator
     */
    setupConnectionStatus() {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.addEventListener('click', () => {
                if (statusElement.classList.contains('disconnected')) {
                    this.connectWebSocket();
                }
            });
        }
    }
    
    /**
     * Initialize all charts
     */
    initializeCharts() {
        // Initialize Chart.js with global settings
        if (typeof Chart !== 'undefined') {
            Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
            Chart.defaults.color = '#6c757d';
        }
        
        // Initialize individual charts
        this.initializeStagePerformanceChart();
        this.initializeValidationScatterChart();
        this.initializeRecommendationDistributionChart();
        this.initializeProcessingTrendsChart();
        this.initializeModuleAvailabilityChart();
    }
    
    /**
     * Initialize stage performance chart
     */
    initializeStagePerformanceChart() {
        const ctx = document.getElementById('stage-performance-chart');
        if (!ctx) return;
        
        this.charts.stagePerformance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Stage 1: Data Quality', 'Stage 2: Cross-Match', 'Stage 3: Physical', 'Stage 4: Statistical', 'Stage 5: Expert Review'],
                datasets: [{
                    label: 'Average Processing Time (ms)',
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: this.config.chartColors.primary,
                    borderColor: this.config.chartColors.primary,
                    borderWidth: 1
                }, {
                    label: 'Pass Rate (%)',
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: this.config.chartColors.success,
                    borderColor: this.config.chartColors.success,
                    borderWidth: 1,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Processing Time (ms)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Pass Rate (%)'
                        },
                        max: 100,
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    legend: {
                        display: true
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                }
            }
        });
    }
    
    /**
     * Initialize validation scatter chart
     */
    initializeValidationScatterChart() {
        const ctx = document.getElementById('validation-scatter-chart');
        if (!ctx) return;
        
        this.charts.validationScatter = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Accepted',
                    data: [],
                    backgroundColor: this.config.chartColors.success,
                    borderColor: this.config.chartColors.success
                }, {
                    label: 'Rejected',
                    data: [],
                    backgroundColor: this.config.chartColors.danger,
                    borderColor: this.config.chartColors.danger
                }, {
                    label: 'Expert Review',
                    data: [],
                    backgroundColor: this.config.chartColors.warning,
                    borderColor: this.config.chartColors.warning
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Validation Score (1 - False Positive Probability)'
                        },
                        min: 0,
                        max: 1
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Confidence'
                        },
                        min: 0,
                        max: 1
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            title: function(context) {
                                const point = context[0].raw;
                                return point.object_designation || 'Unknown Object';
                            },
                            label: function(context) {
                                const point = context.raw;
                                return [
                                    `Score: ${point.x.toFixed(3)}`,
                                    `Confidence: ${point.y.toFixed(3)}`,
                                    `Processing: ${point.processing_time_ms}ms`
                                ];
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Initialize recommendation distribution chart
     */
    initializeRecommendationDistributionChart() {
        const ctx = document.getElementById('recommendation-distribution-chart');
        if (!ctx) return;
        
        this.charts.recommendationDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Accepted', 'Rejected', 'Expert Review'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: [
                        this.config.chartColors.success,
                        this.config.chartColors.danger,
                        this.config.chartColors.warning
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    /**
     * Initialize processing trends chart
     */
    initializeProcessingTrendsChart() {
        const ctx = document.getElementById('processing-trends-chart');
        if (!ctx) return;
        
        this.charts.processingTrends = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Validations per Hour',
                    data: [],
                    borderColor: this.config.chartColors.primary,
                    backgroundColor: this.config.chartColors.primary + '20',
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Average Processing Time (ms)',
                    data: [],
                    borderColor: this.config.chartColors.warning,
                    backgroundColor: this.config.chartColors.warning + '20',
                    fill: false,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Validations per Hour'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Processing Time (ms)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                }
            }
        });
    }
    
    /**
     * Initialize module availability chart
     */
    initializeModuleAvailabilityChart() {
        const ctx = document.getElementById('module-availability-chart');
        if (!ctx) return;
        
        this.charts.moduleAvailability = new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: ['ΔBIC Analysis', 'Spectral Analysis', 'Radar Polarization', 'Thermal-IR', 'Gaia Astrometry'],
                datasets: [{
                    label: 'Availability (%)',
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: this.config.chartColors.info,
                    borderColor: this.config.chartColors.info,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Availability (%)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    /**
     * Connect to WebSocket for real-time updates
     */
    connectWebSocket() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            return;
        }
        
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = 'Connecting...';
            statusElement.className = 'connection-status connecting';
        }
        
        try {
            this.websocket = new WebSocket(this.config.websocketUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                
                if (statusElement) {
                    statusElement.textContent = 'Connected';
                    statusElement.className = 'connection-status connected';
                }
                
                // Subscribe to all message types
                this.websocket.send(JSON.stringify({
                    type: 'subscribe',
                    subscriptions: {
                        validation_metrics: true,
                        system_health: true,
                        alerts: true,
                        stage_performance: true,
                        detection_statistics: true,
                        processing_trends: true
                    }
                }));
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleWebSocketMessage(message);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                
                if (statusElement) {
                    statusElement.textContent = 'Disconnected - Click to reconnect';
                    statusElement.className = 'connection-status disconnected';
                }
                
                this.scheduleReconnect();
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.scheduleReconnect();
        }
    }
    
    /**
     * Schedule WebSocket reconnection with exponential backoff
     */
    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
            
            setTimeout(() => {
                console.log(`Reconnecting WebSocket (attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);
                this.reconnectAttempts++;
                this.connectWebSocket();
            }, delay);
        }
    }
    
    /**
     * Handle incoming WebSocket messages
     */
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'initial_data':
                this.updateDashboard(message.data);
                break;
                
            case 'validation_metrics':
                this.updateValidationMetrics(message.data);
                break;
                
            case 'system_health':
                this.updateSystemHealth(message.data);
                break;
                
            case 'alerts':
                this.updateAlerts(message.data);
                break;
                
            case 'detection_statistics':
                this.updateDetectionStatistics(message.data);
                break;
                
            case 'validation_result':
                this.addValidationResult(message.data);
                break;
                
            case 'artificial_object_alert':
                this.showArtificialObjectAlert(message.data);
                break;
                
            default:
                console.log('Unknown WebSocket message type:', message.type);
        }
    }
    
    /**
     * Get WebSocket URL based on current location
     */
    getWebSocketUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/dashboard/ws/validation`;
    }
    
    /**
     * Setup event handlers
     */
    setupEventHandlers() {
        // Window resize handler for chart responsiveness
        window.addEventListener('resize', () => {
            Object.values(this.charts).forEach(chart => {
                if (chart && typeof chart.resize === 'function') {
                    chart.resize();
                }
            });
        });
        
        // Visibility change handler for performance optimization
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseUpdates();
            } else {
                this.resumeUpdates();
            }
        });
    }
    
    /**
     * Load initial dashboard data via REST API
     */
    async loadInitialData() {
        try {
            const response = await fetch('/dashboard/api/dashboard/data');
            const data = await response.json();
            
            this.updateDashboard(data);
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showError('Failed to load dashboard data');
        }
    }
    
    /**
     * Update entire dashboard with new data
     */
    updateDashboard(data) {
        try {
            this.updateSystemOverview(data.system_overview || {});
            this.updateValidationPipeline(data.validation_pipeline || {});
            this.updateDetectionStatistics(data.detection_statistics || {});
            this.updateAlerts(data.alerts_and_notifications || {});
            this.updateSystemHealth(data.system_health || {});
            
        } catch (error) {
            console.error('Error updating dashboard:', error);
        }
    }
    
    /**
     * Update system overview metrics
     */
    updateSystemOverview(data) {
        const container = document.getElementById('system-metrics');
        if (!container) return;
        
        const metrics = [
            { 
                label: 'Validations Today', 
                value: data.total_validations_today || 0,
                change: this.calculateChange(data.total_validations_today, data.total_validations_yesterday)
            },
            { 
                label: 'Current Throughput/Hr', 
                value: Math.round(data.current_throughput_per_hour || 0),
                change: null
            },
            { 
                label: 'Active Alerts', 
                value: data.active_alerts || 0,
                change: null,
                color: data.active_alerts > 0 ? 'danger' : 'success'
            },
            { 
                label: 'FP Prevention Rate', 
                value: ((data.false_positive_prevention_rate || 0) * 100).toFixed(1) + '%',
                change: null
            },
            { 
                label: 'Avg Confidence', 
                value: ((data.average_confidence || 0) * 100).toFixed(1) + '%',
                change: null
            },
            { 
                label: 'Expert Review Queue', 
                value: data.expert_review_queue || 0,
                change: null
            }
        ];
        
        container.innerHTML = metrics.map(metric => `
            <div class="metric-item">
                <span class="metric-value ${metric.color || ''}">${metric.value}</span>
                <div class="metric-label">${metric.label}</div>
                ${metric.change !== null ? `<div class="metric-change ${metric.change.type}">${metric.change.text}</div>` : ''}
            </div>
        `).join('');
        
        // Update system status
        this.updateSystemStatus(data);
    }
    
    /**
     * Update system status indicator
     */
    updateSystemStatus(data) {
        const statusDot = document.getElementById('system-status-dot');
        const statusText = document.getElementById('system-status-text');
        
        if (!statusDot || !statusText) return;
        
        let status = 'healthy';
        let statusMessage = 'System Operational';
        
        if (data.active_alerts > 5) {
            status = 'critical';
            statusMessage = 'Multiple Alerts Active';
        } else if (data.active_alerts > 0) {
            status = 'warning';
            statusMessage = 'Alerts Present';
        }
        
        statusDot.className = `status-dot ${status}`;
        statusText.textContent = `System Status: ${statusMessage}`;
    }
    
    /**
     * Update validation pipeline performance
     */
    updateValidationPipeline(data) {
        const stagePerformance = data.stage_performance || {};
        const stageList = document.getElementById('stage-list');
        
        if (stageList) {
            const stages = [
                { number: 1, name: 'Data Quality Filter', key: 'stage_1' },
                { number: 2, name: 'Known Object Cross-Match', key: 'stage_2' },
                { number: 3, name: 'Physical Plausibility', key: 'stage_3' },
                { number: 4, name: 'Statistical Significance', key: 'stage_4' },
                { number: 5, name: 'Expert Review Threshold', key: 'stage_5' }
            ];
            
            stageList.innerHTML = stages.map(stage => {
                const stageData = stagePerformance[stage.key] || {};
                const processingTime = stageData.avg_processing_time_ms || 0;
                const passRate = (stageData.pass_rate || 0) * 100;
                const avgScore = (stageData.avg_score || 0) * 100;
                
                let statusClass = 'healthy';
                if (processingTime > 1000) statusClass = 'critical';
                else if (processingTime > 500) statusClass = 'warning';
                
                return `
                    <div class="stage-item">
                        <div class="stage-header">
                            <span class="stage-name">Stage ${stage.number}: ${stage.name}</span>
                            <span class="stage-status ${statusClass}">${statusClass}</span>
                        </div>
                        <div class="stage-metrics">
                            <div class="stage-metric">
                                <div class="stage-metric-value">${processingTime.toFixed(0)}ms</div>
                                <div class="stage-metric-label">Avg Time</div>
                            </div>
                            <div class="stage-metric">
                                <div class="stage-metric-value">${passRate.toFixed(1)}%</div>
                                <div class="stage-metric-label">Pass Rate</div>
                            </div>
                            <div class="stage-metric">
                                <div class="stage-metric-value">${avgScore.toFixed(1)}%</div>
                                <div class="stage-metric-label">Avg Score</div>
                            </div>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill ${statusClass}" style="width: ${Math.min(passRate, 100)}%"></div>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        // Update stage performance chart
        if (this.charts.stagePerformance) {
            const processingTimes = [];
            const passRates = [];
            
            for (let i = 1; i <= 5; i++) {
                const stageData = stagePerformance[`stage_${i}`] || {};
                processingTimes.push(stageData.avg_processing_time_ms || 0);
                passRates.push((stageData.pass_rate || 0) * 100);
            }
            
            this.charts.stagePerformance.data.datasets[0].data = processingTimes;
            this.charts.stagePerformance.data.datasets[1].data = passRates;
            this.charts.stagePerformance.update('none');
        }
        
        // Update module availability chart
        if (this.charts.moduleAvailability && data.module_availability) {
            const availability = data.module_availability;
            this.charts.moduleAvailability.data.datasets[0].data = [
                (availability.delta_bic || 0) * 100,
                (availability.spectral_analysis || 0) * 100,
                (availability.radar_analysis || 0) * 100,
                (availability.thermal_ir || 0) * 100,
                (availability.gaia_analysis || 0) * 100
            ];
            this.charts.moduleAvailability.update('none');
        }
    }
    
    /**
     * Update detection statistics
     */
    updateDetectionStatistics(data) {
        const container = document.getElementById('detection-metrics');
        if (!container) return;
        
        const total = (data.accepted_objects || 0) + (data.rejected_objects || 0) + (data.expert_review_objects || 0);
        
        const metrics = [
            { 
                label: 'Accepted Objects', 
                value: data.accepted_objects || 0,
                percentage: total > 0 ? ((data.accepted_objects || 0) / total * 100).toFixed(1) + '%' : '0%'
            },
            { 
                label: 'Rejected Objects', 
                value: data.rejected_objects || 0,
                percentage: total > 0 ? ((data.rejected_objects || 0) / total * 100).toFixed(1) + '%' : '0%'
            },
            { 
                label: 'Expert Review', 
                value: data.expert_review_objects || 0,
                percentage: total > 0 ? ((data.expert_review_objects || 0) / total * 100).toFixed(1) + '%' : '0%'
            },
            { 
                label: 'Artificial Detections', 
                value: data.artificial_detections || 0,
                percentage: null
            }
        ];
        
        container.innerHTML = metrics.map(metric => `
            <div class="metric-item">
                <span class="metric-value">${metric.value}</span>
                <div class="metric-label">${metric.label}</div>
                ${metric.percentage ? `<div class="metric-change neutral">${metric.percentage}</div>` : ''}
            </div>
        `).join('');
        
        // Update recommendation distribution chart
        if (this.charts.recommendationDistribution) {
            this.charts.recommendationDistribution.data.datasets[0].data = [
                data.accepted_objects || 0,
                data.rejected_objects || 0,
                data.expert_review_objects || 0
            ];
            this.charts.recommendationDistribution.update('none');
        }
    }
    
    /**
     * Update alerts section
     */
    updateAlerts(data) {
        const alertList = document.getElementById('alert-list');
        const alertCount = document.getElementById('alert-count');
        
        if (!alertList || !data.recent_alerts) return;
        
        const alerts = data.recent_alerts.slice(0, 10); // Show latest 10 alerts
        const unresolved = alerts.filter(alert => !alert.resolved);
        
        if (alertCount) {
            alertCount.textContent = unresolved.length;
        }
        
        if (alerts.length === 0) {
            alertList.innerHTML = '<div class="loading">No recent alerts</div>';
            return;
        }
        
        alertList.innerHTML = alerts.map(alert => `
            <div class="alert-item ${alert.alert_level}" data-alert-id="${alert.alert_id}">
                <div class="alert-header">
                    <span class="alert-title">${alert.object_designation} - ${alert.artificial_probability ? (alert.artificial_probability * 100).toFixed(1) + '% artificial' : 'Detection Alert'}</span>
                    <span class="alert-time">${this.formatTimeAgo(alert.timestamp)}</span>
                </div>
                <div class="alert-message">
                    Confidence: ${(alert.confidence * 100).toFixed(1)}% | Modules: ${alert.detection_modules ? alert.detection_modules.join(', ') : 'Unknown'}
                </div>
                ${!alert.resolved ? `
                    <div class="alert-actions">
                        <button class="alert-button acknowledge" onclick="dashboard.acknowledgeAlert('${alert.alert_id}')">
                            Acknowledge
                        </button>
                        <button class="alert-button resolve" onclick="dashboard.resolveAlert('${alert.alert_id}')">
                            Resolve
                        </button>
                    </div>
                ` : '<div class="alert-message" style="color: #27ae60;">Resolved</div>'}
            </div>
        `).join('');
    }
    
    /**
     * Update system health metrics
     */
    updateSystemHealth(data) {
        const container = document.getElementById('health-metrics');
        if (!container || !data.system_health) return;
        
        const health = data.system_health;
        
        const metrics = [
            { 
                label: 'CPU Usage', 
                value: (health.cpu_usage_percent || 0).toFixed(1) + '%',
                color: health.cpu_usage_percent > 80 ? 'danger' : health.cpu_usage_percent > 60 ? 'warning' : 'success'
            },
            { 
                label: 'Memory Usage', 
                value: (health.memory_usage_percent || 0).toFixed(1) + '%',
                color: health.memory_usage_percent > 85 ? 'danger' : health.memory_usage_percent > 70 ? 'warning' : 'success'
            },
            { 
                label: 'Active Sessions', 
                value: health.active_validation_sessions || 0,
                color: null
            },
            { 
                label: 'Avg Processing Time', 
                value: (health.average_processing_time_ms || 0).toFixed(0) + 'ms',
                color: health.average_processing_time_ms > 2000 ? 'danger' : health.average_processing_time_ms > 1000 ? 'warning' : 'success'
            }
        ];
        
        container.innerHTML = metrics.map(metric => `
            <div class="metric-item">
                <span class="metric-value ${metric.color || ''}">${metric.value}</span>
                <div class="metric-label">${metric.label}</div>
            </div>
        `).join('');
    }
    
    /**
     * Add new validation result to scatter chart
     */
    addValidationResult(data) {
        if (!this.charts.validationScatter) return;
        
        const chart = this.charts.validationScatter;
        const score = 1 - (data.false_positive_probability || 0);
        const confidence = data.confidence || 0;
        const recommendation = data.recommendation || 'expert_review';
        
        // Determine dataset index based on recommendation
        let datasetIndex = 2; // Default to expert review
        if (recommendation === 'accept') datasetIndex = 0;
        else if (recommendation === 'reject') datasetIndex = 1;
        
        // Add point to appropriate dataset
        chart.data.datasets[datasetIndex].data.push({
            x: score,
            y: confidence,
            object_designation: data.object_designation,
            processing_time_ms: data.processing_time_ms,
            artificial_likelihood: data.artificial_likelihood || 0
        });
        
        // Limit number of points displayed (keep last 200)
        chart.data.datasets.forEach(dataset => {
            if (dataset.data.length > 200) {
                dataset.data.shift();
            }
        });
        
        chart.update('none');
    }
    
    /**
     * Show artificial object alert notification
     */
    showArtificialObjectAlert(data) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert-notification ${data.alert_level}`;
        notification.innerHTML = `
            <strong>Artificial Object Detected!</strong><br>
            ${data.object_designation} - ${(data.artificial_probability * 100).toFixed(1)}% confidence
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
        
        // Refresh alerts list
        this.loadAlerts();
    }
    
    /**
     * Switch dashboard view
     */
    switchView(view) {
        this.currentView = view;
        const cards = document.querySelectorAll('.dashboard-card');
        
        cards.forEach(card => {
            card.style.display = 'block'; // Show all cards by default
        });
        
        // Implement view-specific logic here if needed
        switch (view) {
            case 'validation':
                // Show validation-focused cards
                break;
            case 'detection':
                // Show detection-focused cards
                break;
            case 'alerts':
                // Show alerts-focused cards
                break;
            case 'analytics':
                // Show analytics-focused cards
                break;
        }
    }
    
    /**
     * Start update loops for periodic data refresh
     */
    startUpdateLoops() {
        // Refresh trends data every 30 seconds
        this.updateIntervals.trends = setInterval(async () => {
            try {
                const response = await fetch('/dashboard/api/visualization/trends');
                const data = await response.json();
                this.updateProcessingTrends(data.time_series);
            } catch (error) {
                console.error('Failed to update trends:', error);
            }
        }, 30000);
    }
    
    /**
     * Pause updates when page is not visible
     */
    pauseUpdates() {
        Object.values(this.updateIntervals).forEach(interval => {
            if (interval) clearInterval(interval);
        });
    }
    
    /**
     * Resume updates when page becomes visible
     */
    resumeUpdates() {
        this.startUpdateLoops();
    }
    
    /**
     * Update processing trends chart
     */
    updateProcessingTrends(data) {
        if (!this.charts.processingTrends || !data) return;
        
        const chart = this.charts.processingTrends;
        
        chart.data.labels = data.timestamps || [];
        chart.data.datasets[0].data = data.validation_rates || [];
        chart.data.datasets[1].data = data.average_processing_times || [];
        
        chart.update('none');
    }
    
    /**
     * Acknowledge an alert
     */
    async acknowledgeAlert(alertId) {
        try {
            const response = await fetch(`/dashboard/api/alerts/${alertId}/acknowledge`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.loadAlerts();
            } else {
                console.error('Failed to acknowledge alert');
            }
        } catch (error) {
            console.error('Error acknowledging alert:', error);
        }
    }
    
    /**
     * Resolve an alert
     */
    async resolveAlert(alertId) {
        try {
            const response = await fetch(`/dashboard/api/alerts/${alertId}/resolve`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.loadAlerts();
            } else {
                console.error('Failed to resolve alert');
            }
        } catch (error) {
            console.error('Error resolving alert:', error);
        }
    }
    
    /**
     * Load alerts from API
     */
    async loadAlerts() {
        try {
            const response = await fetch('/dashboard/api/alerts/artificial-objects');
            const data = await response.json();
            this.updateAlerts({ recent_alerts: data.alerts });
        } catch (error) {
            console.error('Failed to load alerts:', error);
        }
    }
    
    /**
     * Show error message
     */
    showError(message) {
        const container = document.getElementById('dashboard-container');
        if (container) {
            const errorElement = document.createElement('div');
            errorElement.className = 'error-message';
            errorElement.textContent = message;
            container.insertBefore(errorElement, container.firstChild);
            
            setTimeout(() => {
                if (errorElement.parentNode) {
                    errorElement.parentNode.removeChild(errorElement);
                }
            }, 5000);
        }
    }
    
    /**
     * Calculate change percentage
     */
    calculateChange(current, previous) {
        if (!previous || previous === 0) return null;
        
        const change = ((current - previous) / previous) * 100;
        const type = change > 0 ? 'positive' : change < 0 ? 'negative' : 'neutral';
        const text = `${change > 0 ? '+' : ''}${change.toFixed(1)}%`;
        
        return { type, text };
    }
    
    /**
     * Format timestamp as time ago
     */
    formatTimeAgo(timestamp) {
        const now = new Date();
        const time = new Date(timestamp);
        const diffMs = now - time;
        const diffMins = Math.floor(diffMs / 60000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
        return `${Math.floor(diffMins / 1440)}d ago`;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new ValidationDashboard();
});

// Export for use in HTML
window.ValidationDashboard = ValidationDashboard;