# Complete Setup Guide

This guide walks you through setting up the complete ML validation and monitoring system.

## Prerequisites

- Python 3.8 or higher
- macOS, Linux, or Windows (WSL recommended for Windows)
- At least 8GB RAM for running LLMs locally
- Internet connection for initial setup

## Step-by-Step Setup

### Step 1: Install Ollama

Ollama is required to run LLMs locally.

#### macOS
```bash
# Option 1: Using Homebrew
brew install ollama

# Option 2: Download from website
# Visit https://ollama.ai and download the installer
```

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Windows
Download from https://ollama.ai

### Step 2: Start Ollama and Pull a Model

```bash
# Start Ollama server (in a separate terminal)
ollama serve

# In another terminal, pull a model
# For beginners, phi3:mini is fast and small
ollama pull phi3:mini

# Or use llama2 (larger, but more capable)
ollama pull llama2

# List available models
ollama list
```

**Model Recommendations:**
- `phi3:mini` - Fast, small model (good for testing) ~2GB
- `llama2` - Balanced model (good all-around) ~4GB
- `mistral` - High quality responses ~4GB
- `codellama` - Best for code tasks ~4GB

### Step 3: Set Up Python Environment

```bash
# Navigate to project directory
cd /Users/lakkurra/mlops/MLFlow

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Configure DagsHub (Optional)

DagsHub provides a remote MLflow server for viewing experiments online.

1. **Create DagsHub Account:**
   - Go to https://dagshub.com
   - Sign up for a free account

2. **Create Repository:**
   - Click "New Repository"
   - Name it (e.g., "mlops-experiments")
   - Make it public or private

3. **Get Credentials:**
   - Go to Settings → Tokens
   - Create a new token
   - Copy your username and token

4. **Configure Environment:**
   ```bash
   # Copy the example env file
   cp .env.example .env
   
   # Edit .env with your credentials
   # Replace the placeholders:
   DAGSHUB_USERNAME=your_username
   DAGSHUB_TOKEN=your_token
   DAGSHUB_REPO=mlops-experiments
   MLFLOW_TRACKING_URI=https://dagshub.com/your_username/mlops-experiments.mlflow
   ```

**Note:** If you skip DagsHub setup, experiments will be tracked locally (which is fine for learning!).

### Step 5: Verify Setup

Run the quick test to ensure everything is working:

```bash
python run_experiments.py --quick-test
```

Expected output:
```
1. Checking Ollama connection...
   ✓ Connected
2. Checking available models...
   ✓ Found: phi3:mini, llama2
3. Checking model 'llama2'...
   ✓ Model available
4. Testing generation...
   ✓ Response: Hello! How can I assist you today?...
5. Testing MLflow...
   ✓ MLflow working

✅ All checks passed! Ready to run experiments.
```

## Running Your First Experiment

### Option 1: Run Full Validation Suite

```bash
# Run with default model (llama2)
python run_experiments.py

# Run with specific model
python run_experiments.py --model phi3:mini

# Run with DagsHub tracking
python run_experiments.py --dagshub
```

This will:
- Run comprehensive validation tests
- Track everything in MLflow
- Display results and metrics
- Save artifacts

### Option 2: Run Individual Tests

```python
# In Python or Jupyter notebook
from src.model_client import OllamaClient
from src.experiment_runner import ExperimentRunner

# Initialize
client = OllamaClient(model="phi3:mini")
runner = ExperimentRunner(client)

# Run single experiment
result = runner.run_single_experiment(
    prompt="What is machine learning?",
    temperature=0.7
)

print(f"Response: {result['response']}")
print(f"Quality: {result['metrics']['quality_score']:.3f}")
```

### Option 3: Run Monitoring Dashboard

```bash
# Run with demo data
python monitoring_dashboard.py --demo

# Then open: http://localhost:8050
```

## Viewing Results

### Local MLflow UI

```bash
# Start MLflow UI
mlflow ui

# Open in browser: http://localhost:5000
```

The MLflow UI shows:
- All experiment runs
- Parameters used
- Metrics tracked
- Artifacts saved
- Comparison tools

### DagsHub UI (if configured)

Visit: `https://dagshub.com/your_username/your_repo`

### Jupyter Notebook Analysis

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/analysis.ipynb
```

## Project Structure Explained

```
mlops/
├── README.md                    # Main documentation
├── SETUP_GUIDE.md              # This file
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .env                        # Your credentials (create this)
│
├── config/
│   └── experiment_config.yaml  # Experiment settings
│
├── src/                        # Source code
│   ├── model_client.py        # Ollama API wrapper
│   ├── experiment_runner.py   # Experiment orchestration
│   ├── metrics_collector.py   # Custom metrics
│   └── monitoring.py          # Monitoring utilities
│
├── experiments/                # Test suites
│   ├── validation_suite.py    # Validation tests
│   └── benchmark_queries.json # Test queries
│
├── run_experiments.py         # Main runner script
├── monitoring_dashboard.py    # Dashboard app
│
├── notebooks/                 # Jupyter notebooks
│   └── analysis.ipynb        # Analysis notebook
│
├── mlruns/                    # MLflow tracking (created automatically)
└── venv/                      # Virtual environment
```

## Common Issues and Solutions

### Issue: "Cannot connect to Ollama"

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

### Issue: "Model not found"

**Solution:**
```bash
# List available models
ollama list

# Pull the model you need
ollama pull llama2
```

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Port already in use"

**Solution:**
```bash
# For MLflow (port 5000)
# Find process
lsof -i :5000
# Kill it
kill -9 <PID>

# For Dashboard (port 8050)
lsof -i :8050
kill -9 <PID>
```

### Issue: Slow model responses

**Solution:**
- Use a smaller model: `ollama pull phi3:mini`
- Reduce max_tokens in queries
- Check system resources (CPU/RAM)

## Next Steps

1. **Run Initial Experiments:**
   ```bash
   python run_experiments.py --model phi3:mini
   ```

2. **View Results:**
   ```bash
   mlflow ui
   ```

3. **Analyze in Notebook:**
   - Open `notebooks/analysis.ipynb`
   - Run all cells
   - Explore visualizations

4. **Try Different Models:**
   ```bash
   # Pull and test another model
   ollama pull mistral
   python run_experiments.py --model mistral
   ```

5. **Customize Tests:**
   - Edit `experiments/benchmark_queries.json`
   - Add your own test categories
   - Run experiments again

6. **Set Up Monitoring:**
   ```bash
   python monitoring_dashboard.py --demo
   ```

## Advanced Usage

### Running Experiments Programmatically

```python
from src.model_client import OllamaClient
from src.experiment_runner import ExperimentRunner

client = OllamaClient(model="llama2")
runner = ExperimentRunner(client, experiment_name="my_experiments")

# Run batch experiments
prompts = [
    "Explain quantum computing",
    "What is blockchain?",
    "Describe machine learning"
]

results = runner.run_batch_experiments(
    prompts=prompts,
    temperatures=[0.5, 0.7, 0.9]
)
```

### Comparing Multiple Models

```python
models = ["phi3:mini", "llama2", "mistral"]
prompts = ["What is AI?", "Explain neural networks"]

comparison = runner.compare_models(
    models=models,
    prompts=prompts,
    temperature=0.7
)
```

### Custom Metrics

Edit `src/metrics_collector.py` to add your own metrics:

```python
@staticmethod
def calculate_custom_metric(response: str) -> float:
    # Your custom logic here
    return score
```

## Resources

- **Ollama:** https://ollama.ai
- **MLflow:** https://mlflow.org/docs/latest/
- **DagsHub:** https://dagshub.com/docs/
- **Plotly Dash:** https://dash.plotly.com/

## Getting Help

If you encounter issues:

1. Check the logs in the terminal
2. Review this guide's troubleshooting section
3. Check Ollama status: `ollama list`
4. Verify Python environment: `pip list`
5. Test individual components using `--quick-test`

## Summary Checklist

- [ ] Ollama installed and running
- [ ] Model pulled (e.g., `ollama pull phi3:mini`)
- [ ] Python virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] (Optional) DagsHub configured in `.env`
- [ ] Quick test passed (`python run_experiments.py --quick-test`)
- [ ] First experiment run successfully
- [ ] MLflow UI accessible (`mlflow ui`)
- [ ] Dashboard working (`python monitoring_dashboard.py --demo`)

Once all items are checked, you're ready to start experimenting! 🚀
