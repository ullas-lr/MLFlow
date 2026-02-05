# Quick Reference Guide

## 📍 Project Location
```bash
cd /Users/lakkurra/mlops/MLFlow
```

## 🚀 Essential Commands

### First Time Setup
```bash
# 1. Navigate to project
cd /Users/lakkurra/mlops/MLFlow

# 2. Run automated setup
./quickstart.sh

# This will:
# - Check your environment
# - Install dependencies
# - Verify Ollama
# - Pull a model
# - Run a test
```

### Running Experiments
```bash
# Activate environment
source venv/bin/activate

# Run full validation suite
python run_experiments.py

# Run with specific model
python run_experiments.py --model phi3:mini
python run_experiments.py --model llama2

# Run with DagsHub tracking
python run_experiments.py --dagshub

# Quick test only
python run_experiments.py --quick-test
```

### Viewing Results

#### MLflow UI (Local)
```bash
# Start MLflow UI
mlflow ui

# Open browser: http://localhost:5000
```

#### Monitoring Dashboard
```bash
# With demo data
python monitoring_dashboard.py --demo

# Open browser: http://localhost:8050
```

#### Jupyter Notebook
```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/analysis.ipynb
```

### Ollama Commands
```bash
# Start Ollama server
ollama serve

# List available models
ollama list

# Pull a new model
ollama pull phi3:mini
ollama pull llama2
ollama pull mistral

# Test Ollama
curl http://localhost:11434/api/tags
```

### Git Commands
```bash
# Check status
git status

# Add all new files
git add .

# Commit changes
git commit -m "Add ML validation and monitoring system"

# Push to remote (if configured)
git push origin main
```

## 📁 Project Structure

```
MLFlow/
├── 📖 Documentation
│   ├── README.md              # Overview
│   ├── GETTING_STARTED.md     # 5-min quick start
│   ├── SETUP_GUIDE.md         # Detailed setup
│   ├── USAGE_EXAMPLES.md      # Code examples
│   ├── ARCHITECTURE.md        # System design
│   ├── PROJECT_INFO.md        # Project info
│   └── QUICK_REFERENCE.md     # This file
│
├── 🚀 Main Scripts
│   ├── quickstart.sh          # Automated setup
│   ├── run_experiments.py     # Run experiments
│   └── monitoring_dashboard.py # Dashboard
│
├── 💻 Source Code
│   └── src/
│       ├── model_client.py
│       ├── experiment_runner.py
│       ├── metrics_collector.py
│       └── monitoring.py
│
├── 🧪 Experiments
│   └── experiments/
│       ├── validation_suite.py
│       └── benchmark_queries.json
│
├── 📊 Analysis
│   └── notebooks/
│       └── analysis.ipynb
│
├── ⚙️ Configuration
│   ├── config/
│   │   └── experiment_config.yaml
│   ├── requirements.txt
│   ├── .env.example
│   └── .gitignore
│
└── 📦 Data (created at runtime)
    ├── data/
    ├── models/
    ├── mlruns/
    └── venv/
```

## 🎯 Common Tasks

### Test Different Models
```bash
# Pull models
ollama pull phi3:mini    # Fast (2GB)
ollama pull llama2       # Balanced (4GB)
ollama pull mistral      # Quality (4GB)

# Test each model
python run_experiments.py --model phi3:mini
python run_experiments.py --model llama2
python run_experiments.py --model mistral
```

### Compare Models in Python
```python
from src.model_client import OllamaClient
from src.experiment_runner import ExperimentRunner

client = OllamaClient()
runner = ExperimentRunner(client)

comparison = runner.compare_models(
    models=["phi3:mini", "llama2"],
    prompts=["What is AI?", "Explain ML"],
    temperature=0.7
)
```

### Add Custom Tests
```bash
# Edit test queries
nano experiments/benchmark_queries.json

# Run experiments
python run_experiments.py
```

### Setup DagsHub
```bash
# 1. Copy env template
cp .env.example .env

# 2. Edit with your credentials
nano .env

# 3. Run with DagsHub
python run_experiments.py --dagshub
```

## 🐛 Troubleshooting

### Ollama Not Running
```bash
# Check if running
curl http://localhost:11434/api/tags

# Start if not running
ollama serve
```

### Model Not Found
```bash
# List models
ollama list

# Pull missing model
ollama pull llama2
```

### Dependencies Issue
```bash
# Reinstall
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Port Already in Use
```bash
# MLflow (5000)
lsof -i :5000
kill -9 <PID>

# Dashboard (8050)
lsof -i :8050
kill -9 <PID>
```

## 📚 Documentation Guide

| Document | Purpose | When to Read |
|----------|---------|--------------|
| `GETTING_STARTED.md` | Quick 5-min start | First time |
| `SETUP_GUIDE.md` | Detailed setup | Having issues |
| `USAGE_EXAMPLES.md` | 18 code examples | Learning |
| `ARCHITECTURE.md` | System design | Understanding internals |
| `README.md` | Full overview | Reference |
| `QUICK_REFERENCE.md` | Command cheatsheet | Daily use |

## 💡 Pro Tips

1. **Start with phi3:mini** - It's fast and good for learning
2. **Keep MLflow UI open** - Monitor experiments in real-time
3. **Use the dashboard** - Great for observability
4. **Export results** - Run Jupyter notebooks for analysis
5. **Commit often** - Track your experiment configs in Git

## 🎓 Learning Path

### Day 1
```bash
./quickstart.sh
python run_experiments.py
mlflow ui
```

### Day 2
```bash
# Try different models
python run_experiments.py --model mistral

# Run dashboard
python monitoring_dashboard.py --demo
```

### Day 3
```bash
# Customize tests
nano experiments/benchmark_queries.json

# Analyze results
jupyter notebook notebooks/analysis.ipynb
```

### Day 4+
- Set up DagsHub
- Add custom metrics
- Run A/B tests
- Create continuous monitoring

## 🔗 Useful Links

- **Ollama**: https://ollama.ai
- **MLflow Docs**: https://mlflow.org/docs/latest/
- **DagsHub**: https://dagshub.com
- **Ollama Models**: https://ollama.com/library

---

**Current Location**: `/Users/lakkurra/mlops/MLFlow`

**Quick Start**: `./quickstart.sh`

**Help**: Read `GETTING_STARTED.md` or `SETUP_GUIDE.md`
