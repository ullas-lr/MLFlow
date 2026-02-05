#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard
Dash-based interactive dashboard for monitoring model performance
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from datetime import datetime
import threading
import time
from collections import deque
import logging

from src.model_client import OllamaClient
from src.monitoring import MetricsMonitor, SystemMonitor, AlertManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize monitoring
monitor = MetricsMonitor(max_history=100)
alert_manager = AlertManager()

# Create Dash app
app = dash.Dash(__name__)
app.title = "ML Model Monitoring"

# Layout
app.layout = html.Div([
    html.Div([
        html.H1("🔍 ML Model Monitoring Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.P("Real-time monitoring of Ollama model performance", style={'textAlign': 'center', 'color': '#7f8c8d'}),
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),
    
    # Stats cards
    html.Div([
        html.Div([
            html.H3("Total Requests", style={'color': '#3498db'}),
            html.H2(id='total-requests', children="0"),
        ], className='stat-card', style={'flex': '1', 'margin': '10px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        
        html.Div([
            html.H3("Success Rate", style={'color': '#2ecc71'}),
            html.H2(id='success-rate', children="0%"),
        ], className='stat-card', style={'flex': '1', 'margin': '10px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        
        html.Div([
            html.H3("Avg Latency", style={'color': '#e74c3c'}),
            html.H2(id='avg-latency', children="0.0s"),
        ], className='stat-card', style={'flex': '1', 'margin': '10px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        
        html.Div([
            html.H3("Avg Quality", style={'color': '#9b59b6'}),
            html.H2(id='avg-quality', children="0.0"),
        ], className='stat-card', style={'flex': '1', 'margin': '10px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    ], style={'display': 'flex', 'padding': '20px'}),
    
    # Alerts
    html.Div([
        html.H3("🚨 Active Alerts", style={'color': '#e74c3c'}),
        html.Div(id='alerts-container'),
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # Charts
    html.Div([
        html.Div([
            dcc.Graph(id='latency-chart'),
        ], style={'flex': '1', 'margin': '10px'}),
        
        html.Div([
            dcc.Graph(id='quality-chart'),
        ], style={'flex': '1', 'margin': '10px'}),
    ], style={'display': 'flex', 'padding': '20px'}),
    
    html.Div([
        html.Div([
            dcc.Graph(id='throughput-chart'),
        ], style={'flex': '1', 'margin': '10px'}),
        
        html.Div([
            dcc.Graph(id='system-chart'),
        ], style={'flex': '1', 'margin': '10px'}),
    ], style={'display': 'flex', 'padding': '20px'}),
    
    # Auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    )
])


@app.callback(
    [Output('total-requests', 'children'),
     Output('success-rate', 'children'),
     Output('avg-latency', 'children'),
     Output('avg-quality', 'children'),
     Output('alerts-container', 'children'),
     Output('latency-chart', 'figure'),
     Output('quality-chart', 'figure'),
     Output('throughput-chart', 'figure'),
     Output('system-chart', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """Update all dashboard components"""
    
    # Get current stats
    stats = monitor.get_current_stats()
    system_stats = SystemMonitor.get_system_stats()
    history = monitor.get_history(limit=50)
    
    # Update stats cards
    total_requests = stats.get('total_requests', 0)
    success_rate = f"{stats.get('success_rate', 0):.1%}"
    avg_latency = f"{stats.get('recent_avg_latency', 0):.2f}s"
    avg_quality = f"{stats.get('recent_avg_quality', 0):.3f}" if 'recent_avg_quality' in stats else "N/A"
    
    # Check alerts
    alerts = alert_manager.check_alerts(stats)
    active_alerts = alert_manager.get_active_alerts(minutes=5)
    
    if active_alerts:
        alerts_div = [
            html.Div([
                html.Span("⚠️ ", style={'fontSize': '20px'}),
                html.Span(f"[{alert['severity'].upper()}] {alert['message']}", style={'marginLeft': '10px'})
            ], style={'padding': '10px', 'margin': '5px', 'backgroundColor': '#ffe6e6', 'borderRadius': '4px'})
            for alert in active_alerts[-5:]  # Show last 5 alerts
        ]
    else:
        alerts_div = html.Div("✅ No active alerts", style={'color': '#27ae60', 'padding': '10px'})
    
    # Prepare data for charts
    if history:
        timestamps = [entry['timestamp'] for entry in history]
        latencies = [entry['latency'] for entry in history if entry['success']]
        quality_scores = [entry.get('quality_score', 0) for entry in history if entry['success'] and 'quality_score' in entry]
        tokens_per_sec = [entry.get('tokens_per_second', 0) for entry in history if entry['success']]
    else:
        timestamps = []
        latencies = []
        quality_scores = []
        tokens_per_sec = []
    
    # Latency chart
    latency_fig = go.Figure()
    latency_fig.add_trace(go.Scatter(
        x=timestamps[-30:] if len(timestamps) > 30 else timestamps,
        y=latencies[-30:] if len(latencies) > 30 else latencies,
        mode='lines+markers',
        name='Latency',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=6)
    ))
    latency_fig.update_layout(
        title='Response Latency (Last 30 Requests)',
        xaxis_title='Time',
        yaxis_title='Latency (seconds)',
        template='plotly_white',
        height=300
    )
    
    # Quality chart
    quality_fig = go.Figure()
    if quality_scores:
        quality_fig.add_trace(go.Scatter(
            x=timestamps[-30:] if len(timestamps) > 30 else timestamps,
            y=quality_scores[-30:] if len(quality_scores) > 30 else quality_scores,
            mode='lines+markers',
            name='Quality Score',
            line=dict(color='#9b59b6', width=2),
            marker=dict(size=6)
        ))
    quality_fig.update_layout(
        title='Quality Score (Last 30 Requests)',
        xaxis_title='Time',
        yaxis_title='Quality Score',
        template='plotly_white',
        height=300,
        yaxis=dict(range=[0, 1])
    )
    
    # Throughput chart
    throughput_fig = go.Figure()
    if tokens_per_sec:
        throughput_fig.add_trace(go.Scatter(
            x=timestamps[-30:] if len(timestamps) > 30 else timestamps,
            y=tokens_per_sec[-30:] if len(tokens_per_sec) > 30 else tokens_per_sec,
            mode='lines+markers',
            name='Tokens/sec',
            line=dict(color='#3498db', width=2),
            marker=dict(size=6),
            fill='tozeroy'
        ))
    throughput_fig.update_layout(
        title='Throughput (Tokens per Second)',
        xaxis_title='Time',
        yaxis_title='Tokens/sec',
        template='plotly_white',
        height=300
    )
    
    # System resources chart
    system_fig = go.Figure()
    system_fig.add_trace(go.Bar(
        x=['CPU', 'Memory', 'Disk'],
        y=[
            system_stats.get('cpu_percent', 0),
            system_stats.get('memory_percent', 0),
            system_stats.get('disk_percent', 0)
        ],
        marker_color=['#3498db', '#2ecc71', '#f39c12']
    ))
    system_fig.update_layout(
        title='System Resources Usage (%)',
        yaxis_title='Usage (%)',
        template='plotly_white',
        height=300,
        yaxis=dict(range=[0, 100])
    )
    
    return (
        total_requests,
        success_rate,
        avg_latency,
        avg_quality,
        alerts_div,
        latency_fig,
        quality_fig,
        throughput_fig,
        system_fig
    )


def simulate_requests():
    """Simulate requests for demo purposes"""
    import random
    from src.experiment_runner import ExperimentRunner
    
    logger.info("Starting request simulator (demo mode)...")
    
    client = OllamaClient()
    if not client.check_connection():
        logger.warning("Ollama not connected. Running in demo mode with simulated data.")
        # Simulate with dummy data
        while True:
            result = {
                "success": random.random() > 0.05,
                "latency": random.uniform(0.5, 3.0),
                "model": "llama2",
                "eval_count": random.randint(20, 100),
                "tokens_per_second": random.uniform(10, 30),
                "metrics": {
                    "quality_score": random.uniform(0.6, 0.95),
                    "coherence_score": random.uniform(0.6, 0.95),
                    "relevance_score": random.uniform(0.6, 0.95),
                }
            }
            monitor.record_request(result)
            time.sleep(random.uniform(2, 5))
    else:
        # Use real model
        runner = ExperimentRunner(client, experiment_name="dashboard_demo")
        test_prompts = [
            "What is machine learning?",
            "Explain neural networks.",
            "What is Python?",
            "Define artificial intelligence.",
        ]
        
        while True:
            prompt = random.choice(test_prompts)
            result = runner.run_single_experiment(prompt, log_to_mlflow=False)
            monitor.record_request(result)
            time.sleep(random.uniform(3, 8))


def main():
    """Main entry point"""
    import sys
    
    logger.info("="*60)
    logger.info("MONITORING DASHBOARD")
    logger.info("="*60)
    
    # Check if we should run in demo mode
    demo_mode = '--demo' in sys.argv
    
    if demo_mode:
        logger.info("\n🎯 Running in DEMO mode with simulated data")
        # Start simulator thread
        simulator_thread = threading.Thread(target=simulate_requests, daemon=True)
        simulator_thread.start()
    else:
        logger.info("\n📊 Running in LIVE mode")
        logger.info("   Connect to running experiments to see data")
        logger.info("   Or use --demo flag for simulated data")
    
    logger.info("\n🌐 Starting dashboard server...")
    logger.info("   URL: http://localhost:8050")
    logger.info("   Press Ctrl+C to stop\n")
    
    # Run dashboard
    app.run_server(debug=False, host='0.0.0.0', port=8050)


if __name__ == "__main__":
    main()
