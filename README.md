# Local ML Model Validation and Monitoring

This project demonstrates how to perform model validation and monitoring locally using:
- **MLflow**: Experiment tracking and model registry
- **DagsHub**: Remote MLflow server for collaboration
- **Ollama**: Local LLM deployment
- **Custom Monitoring**: Observability and metrics tracking

## Architecture

```
┌─────────────┐
│   Ollama    │ ← Local LLM Model
│   Server    │
└──────┬──────┘
       │
       ↓
┌─────────────────────┐
│  Experiment Runner  │ ← Test queries & validation
└──────┬──────────────┘
       │
       ↓
┌─────────────┐      ┌──────────────┐
│   MLflow    │ ←──→ │   DagsHub    │
│   Local     │      │   Remote     │
└─────────────┘      └──────────────┘
       │
       ↓
┌─────────────────────┐
│  Monitoring/Metrics │ ← Observability
└─────────────────────┘
```

## Quick Start

### 1. Navigate to Project Directory

```bash
cd /Users/lakkurra/mlops/MLFlow
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

**macOS:**
```bash
brew install ollama
# OR download from https://ollama.ai
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Start Ollama and Pull a Model

```bash
# Start Ollama server (in a separate terminal)
ollama serve

# Pull a model (e.g., llama2, mistral, phi3)
ollama pull llama2
# or smaller models for testing
ollama pull phi3:mini
```

### 4. Configure DagsHub (Optional but Recommended)

1. Create a free account at [dagshub.com](https://dagshub.com)
2. Create a new repository
3. Get your DagsHub credentials:
   - Username: Your DagsHub username
   - Token: Generate from Settings → Tokens
4. Copy `.env.example` to `.env` and fill in your credentials

```bash
cp .env.example .env
# Edit .env with your DagsHub credentials
```

### 5. Run Experiments

```bash
# Run basic validation experiments
python run_experiments.py

# Run with custom parameters
python run_experiments.py --model llama2 --num-runs 10

# Run healthcare-specific demo
python demo_healthcare.py phi3:mini

# Run drift detection demo
python demo_drift_detection.py phi3:mini

# Launch drift detection dashboard
python drift_dashboard.py
```

## 🔍 Drift Detection (NEW!)

Detect and monitor data drift and model drift in your ML systems:

### Quick Test
```bash
# Verify drift detection works (30 seconds)
python test_drift_detector.py
```

### What's Included

**Data Drift Detection**:
- Prompt length distribution changes (Kolmogorov-Smirnov test)
- Keyword frequency shifts
- Category distribution changes (KL Divergence)

**Model Drift Detection**:
- Quality score degradation (Mann-Whitney U test)
- Safety score changes
- Latency increases
- Performance degradation alerts

**Healthcare-Specific Scenarios**:
- Seasonal disease patterns (summer → winter/flu season)
- Demographic shifts (age distribution changes)
- New medical terminology (COVID-19, new diseases)
- Treatment protocol updates

### Drift Detection Demo

```bash
# Run complete drift detection demo
python demo_drift_detection.py phi3:mini

# Launch drift visualization dashboard
python drift_dashboard.py
# Open http://localhost:8051
```

**View Results**:
- MLflow UI: Experiment "healthcare_drift_detection"
- Drift Dashboard: Interactive visualizations with alerts

### Documentation
- **Quick Start**: `DRIFT_DEMO_QUICKSTART.md` - Get started in 5 minutes
- **Complete Guide**: `DRIFT_DETECTION_GUIDE.md` - Detailed explanations
- **Session Demo**: `SESSION_DEMO_SCRIPT.md` - Presentation walkthrough

## Project Structure

```
mlops/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore file
├── config/
│   └── experiment_config.yaml   # Experiment configuration
├── src/
│   ├── __init__.py
│   ├── model_client.py         # Ollama client wrapper
│   ├── experiment_runner.py    # Experiment orchestration
│   ├── metrics_collector.py    # Custom metrics collection
│   └── monitoring.py           # Observability utilities
├── experiments/
│   ├── validation_suite.py     # Model validation tests
│   └── healthcare_queries.json # Healthcare test queries
├── run_experiments.py          # Main experiment runner
├── demo_healthcare.py          # Healthcare validation demo
├── demo_drift_detection.py     # Drift detection demo
├── drift_dashboard.py          # Drift visualization dashboard
└── test_drift_detector.py      # Drift detector tests

```

## Features

### 1. Experiment Tracking with MLflow
- Automatic logging of all experiments
- Parameter tracking (model, temperature, prompt)
- Metrics tracking (latency, token count, quality scores)
- Artifact storage (responses, logs)

### 2. DagsHub Integration
- Remote MLflow server
- Collaborative experiment viewing
- Version control for ML experiments
- Easy sharing of results

### 3. Model Validation
- Automated test suites
- Quality metrics (coherence, relevance, safety)
- Performance benchmarks (latency, throughput)
- Comparison across model versions

### 4. Observability
- Real-time metrics dashboard
- Response time tracking
- Error rate monitoring
- Resource utilization metrics

## Usage Examples

### Running Specific Experiments

```python
from src.experiment_runner import ExperimentRunner
from src.model_client import OllamaClient

# Initialize
client = OllamaClient(model="llama2")
runner = ExperimentRunner(client)

# Run single experiment
result = runner.run_experiment(
    prompt="What is machine learning?",
    temperature=0.7
)

# Run validation suite
results = runner.run_validation_suite()
```

### Viewing Results

1. **Local MLflow UI:**
   ```bash
   mlflow ui
   ```
   Visit: http://localhost:5000

2. **DagsHub (if configured):**
   Visit your DagsHub repository URL

### Drift Detection Dashboard

```bash
# Start drift detection dashboard
python drift_dashboard.py
```

Visit: http://localhost:8051

## Metrics Tracked

- **Performance Metrics:**
  - Response latency (ms)
  - Tokens per second
  - Total tokens generated
  - API success rate

- **Quality Metrics:**
  - Response length
  - Coherence score
  - Relevance score
  - Safety score

- **System Metrics:**
  - CPU usage
  - Memory usage
  - Request rate
  - Error rate

## Advanced Usage

### Custom Validation Tests

Create your own validation tests in `experiments/validation_suite.py`:

```python
def test_custom_behavior(client):
    response = client.generate("Your test prompt")
    assert len(response) > 0
    # Add your custom validations
```

### Custom Metrics

Add custom metrics in `src/metrics_collector.py`:

```python
def calculate_custom_metric(response):
    # Your custom logic
    return metric_value
```

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama server is running: `ollama serve`
- Check if model is available: `ollama list`
- Verify connection: `curl http://localhost:11434/api/tags`

### MLflow Issues
- Clear tracking directory: `rm -rf mlruns/`
- Check port availability: `lsof -i :5000`

### DagsHub Connection Issues
- Verify credentials in `.env`
- Check token permissions
- Ensure repository exists

## Next Steps

1. ✅ Set up basic infrastructure
2. ✅ Run initial experiments
3. 📊 Analyze results in MLflow UI
4. 🔄 Iterate on validation tests
5. 📈 Set up continuous monitoring
6. 🚀 Deploy best model

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DagsHub Documentation](https://dagshub.com/docs/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama Models](https://ollama.com/library)

## License

MIT
