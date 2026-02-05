# System Architecture

## Overview

This project implements a complete ML model validation and monitoring system for locally deployed LLMs using Ollama.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
├─────────────────────────────────────────────────────────────────┤
│  CLI Scripts      │  Jupyter Notebooks  │  Monitoring Dashboard │
│  - run_experiments│  - analysis.ipynb   │  - Dash (port 8050)  │
│  - quickstart.sh  │                     │  - Real-time charts  │
└─────────────┬───────────────────┬───────────────────┬───────────┘
              │                   │                   │
              ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ ExperimentRunner │  │ ValidationSuite  │  │  Monitoring  │ │
│  │                  │  │                  │  │              │ │
│  │ - Run tests      │  │ - Test suites    │  │ - Metrics    │ │
│  │ - Log to MLflow  │  │ - Benchmarks     │  │ - Alerts     │ │
│  │ - Track metrics  │  │ - Quality checks │  │ - Dashboard  │ │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘ │
│           │                     │                    │         │
│           └─────────────────────┼────────────────────┘         │
│                                 ▼                              │
│                    ┌──────────────────────┐                   │
│                    │  MetricsCollector    │                   │
│                    │                      │                   │
│                    │ - Coherence score    │                   │
│                    │ - Relevance score    │                   │
│                    │ - Safety score       │                   │
│                    │ - Performance metrics│                   │
│                    └──────────┬───────────┘                   │
│                               │                               │
└───────────────────────────────┼───────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INTEGRATION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐                   ┌──────────────────┐   │
│  │  OllamaClient    │                   │  MLflow Tracking │   │
│  │                  │                   │                  │   │
│  │ - API wrapper    │                   │ - Experiments    │   │
│  │ - Error handling │                   │ - Runs           │   │
│  │ - Retry logic    │                   │ - Metrics        │   │
│  └────────┬─────────┘                   │ - Artifacts      │   │
│           │                             └────────┬─────────┘   │
│           │                                      │             │
└───────────┼──────────────────────────────────────┼─────────────┘
            ▼                                      ▼
┌─────────────────────┐              ┌──────────────────────────┐
│   Ollama Server     │              │  Storage Layer           │
│   (localhost:11434) │              │                          │
│                     │              │  Local:                  │
│  ┌──────────────┐   │              │  - ./mlruns/             │
│  │   Models     │   │              │  - ./mlartifacts/        │
│  │              │   │              │                          │
│  │ - phi3:mini  │   │              │  Remote (Optional):      │
│  │ - llama2     │   │              │  - DagsHub               │
│  │ - mistral    │   │              │  - dagshub.com/user/repo │
│  │ - codellama  │   │              │                          │
│  └──────────────┘   │              └──────────────────────────┘
└─────────────────────┘
```

## Component Details

### 1. User Interface Layer

#### CLI Scripts
- **run_experiments.py**: Main entry point for running experiments
- **quickstart.sh**: Automated setup script
- **monitoring_dashboard.py**: Real-time monitoring dashboard

#### Jupyter Notebooks
- **analysis.ipynb**: Interactive data analysis and visualization

#### Monitoring Dashboard
- Built with Plotly Dash
- Real-time metrics display
- System resource monitoring
- Alert management

### 2. Application Layer

#### ExperimentRunner (`src/experiment_runner.py`)
- Orchestrates experiment execution
- Manages MLflow integration
- Handles batch processing
- Provides model comparison utilities

**Key Methods:**
```python
run_single_experiment()      # Run one experiment
run_batch_experiments()      # Run multiple experiments
compare_models()             # Compare different models
get_experiment_summary()     # Get summary statistics
```

#### ValidationSuite (`experiments/validation_suite.py`)
- Comprehensive test suites
- Multiple test categories (general, reasoning, coding, etc.)
- Quality validation
- Performance benchmarking

**Test Categories:**
- General Knowledge
- Reasoning
- Coding
- Creative
- Safety
- Performance

#### MetricsCollector (`src/metrics_collector.py`)
- Custom metric calculations
- Quality scoring
- Performance analysis

**Metrics:**
- Coherence Score (0-1)
- Relevance Score (0-1)
- Safety Score (0-1)
- Response Latency (seconds)
- Tokens per Second
- Overall Quality Score

#### Monitoring (`src/monitoring.py`)
- Real-time metrics tracking
- Alert management
- System resource monitoring
- Historical data storage

**Components:**
- MetricsMonitor: Track request metrics
- SystemMonitor: System resources
- AlertManager: Threshold-based alerts

### 3. Integration Layer

#### OllamaClient (`src/model_client.py`)
- Abstracts Ollama API
- Error handling
- Connection management
- Model listing

**Key Features:**
- Automatic retries
- Timeout handling
- Response parsing
- Metadata extraction

#### MLflow Tracking
- Experiment organization
- Run tracking
- Parameter logging
- Metric recording
- Artifact storage

**Tracked Data:**
- Parameters: model, temperature, max_tokens
- Metrics: quality, latency, tokens/sec
- Artifacts: prompts, responses, full results

### 4. Storage Layer

#### Local Storage
```
mlruns/
├── 0/                    # Default experiment
├── 1/                    # Named experiments
│   ├── meta.yaml
│   └── <run-id>/
│       ├── metrics/
│       ├── params/
│       ├── artifacts/
│       └── meta.yaml
└── ...
```

#### Remote Storage (DagsHub)
- Cloud-based MLflow server
- Web UI for viewing experiments
- Collaboration features
- Version control integration

## Data Flow

### Experiment Execution Flow

```
1. User Request
   ↓
2. ExperimentRunner
   - Parse parameters
   - Initialize MLflow run
   ↓
3. OllamaClient
   - Generate prompt request
   - Send to Ollama server
   ↓
4. Ollama Server
   - Load model
   - Generate response
   - Return with metadata
   ↓
5. MetricsCollector
   - Calculate quality scores
   - Compute performance metrics
   ↓
6. MLflow
   - Log parameters
   - Log metrics
   - Save artifacts
   ↓
7. Storage
   - Save to local/remote
   ↓
8. Return Results
   - Display to user
   - Update monitoring
```

### Monitoring Flow

```
Experiment → MetricsMonitor → Dashboard
                    ↓
            AlertManager → Notifications
                    ↓
            SystemMonitor → Resource Tracking
```

## Configuration

### Experiment Config (`config/experiment_config.yaml`)

```yaml
experiment:
  name: "model_validation_experiments"
  
models:
  - name: "llama2"
  - name: "phi3:mini"
  
parameters:
  temperature: [0.3, 0.7, 0.9]
  max_tokens: [256, 512]
  
validation:
  metrics:
    - coherence_score
    - relevance_score
    - safety_score
    - latency
    
monitoring:
  dashboard:
    refresh_interval: 5
    max_history: 100
```

### Environment Config (`.env`)

```bash
# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2

# MLflow
MLFLOW_TRACKING_URI=./mlruns

# DagsHub (Optional)
DAGSHUB_USERNAME=your_username
DAGSHUB_TOKEN=your_token
DAGSHUB_REPO=your_repo
```

## Deployment Options

### 1. Local Development
- Run everything locally
- No cloud dependencies
- Fast iteration

### 2. Team Collaboration (with DagsHub)
- Remote experiment tracking
- Shared results
- Version control integration

### 3. Production Monitoring
- Continuous validation
- Alert notifications
- Performance tracking

## Scalability

### Current Limitations
- Single machine deployment
- Sequential experiment execution
- Limited by Ollama server capacity

### Scaling Options

1. **Horizontal Scaling:**
   - Multiple Ollama servers
   - Load balancing
   - Distributed experiment execution

2. **Parallel Processing:**
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   with ThreadPoolExecutor(max_workers=4) as executor:
       results = executor.map(run_experiment, prompts)
   ```

3. **Cloud Deployment:**
   - Deploy Ollama on GPU instances
   - Use DagsHub for centralized tracking
   - Set up continuous monitoring

## Security Considerations

1. **API Keys:** Store in `.env` (not in version control)
2. **Model Access:** Local models (no external API calls)
3. **Data Privacy:** All data stays local unless using DagsHub
4. **Safety Filtering:** Built-in safety score metrics

## Performance Optimization

### Model Selection
- **Fast:** phi3:mini (~2GB, 10-20 tokens/sec)
- **Balanced:** llama2 (~4GB, 5-15 tokens/sec)
- **Quality:** mistral (~4GB, 5-15 tokens/sec)

### Tips
1. Use smaller models for testing
2. Limit max_tokens for faster responses
3. Cache common prompts
4. Batch similar experiments
5. Use GPU if available

## Monitoring and Observability

### Real-time Metrics
- Request rate
- Success rate
- Average latency
- Quality scores

### System Metrics
- CPU usage
- Memory usage
- Disk space
- Network I/O

### Alerts
- High latency (> threshold)
- High error rate (> threshold)
- Low quality scores (< threshold)
- System resource limits

## Extension Points

### Adding New Metrics
Edit `src/metrics_collector.py`:
```python
@staticmethod
def calculate_new_metric(text: str) -> float:
    # Implementation
    return score
```

### Adding New Models
```bash
ollama pull new-model
python run_experiments.py --model new-model
```

### Custom Validation Tests
Edit `experiments/benchmark_queries.json`:
```json
{
  "test_suites": [
    {
      "category": "custom",
      "queries": [...]
    }
  ]
}
```

### Integration with Other Tools
- Add Prometheus metrics export
- Integrate with Grafana
- Send alerts to Slack/Email
- Export to other ML platforms

## Technology Stack

- **Python 3.8+**: Core language
- **Ollama**: Local LLM deployment
- **MLflow 2.10**: Experiment tracking
- **DagsHub 0.3**: Remote tracking (optional)
- **Plotly Dash 2.14**: Monitoring dashboard
- **Pandas 2.2**: Data analysis
- **psutil 5.9**: System monitoring
- **Requests 2.31**: HTTP client

## Best Practices

1. **Version Control:** Track all code changes
2. **Documentation:** Keep README up to date
3. **Testing:** Run validation suite regularly
4. **Monitoring:** Keep dashboard running
5. **Backup:** Export MLflow data periodically
6. **Clean Code:** Follow PEP 8 standards
7. **Error Handling:** Log all errors
8. **Performance:** Profile slow components

## Future Enhancements

- [ ] Multi-GPU support
- [ ] Distributed experiment execution
- [ ] Advanced prompt templating
- [ ] Automated model selection
- [ ] Cost tracking
- [ ] A/B testing framework
- [ ] Integration with CI/CD
- [ ] Advanced analytics
- [ ] Custom alert channels
- [ ] Model fine-tuning support
