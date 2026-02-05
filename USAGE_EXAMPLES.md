# Usage Examples

This document provides practical examples for using the ML validation and monitoring system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Running Experiments](#running-experiments)
3. [Monitoring](#monitoring)
4. [Analysis](#analysis)
5. [Advanced Usage](#advanced-usage)
6. [DagsHub Integration](#dagshub-integration)

---

## Quick Start

### One-Command Setup (Recommended)

```bash
./quickstart.sh
```

This interactive script will:
- Check your environment
- Set up Python dependencies
- Verify Ollama installation
- Pull a model if needed
- Run a quick test

### Manual Setup

```bash
# 1. Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Start Ollama (in separate terminal)
ollama serve

# 3. Pull a model
ollama pull phi3:mini

# 4. Run quick test
python run_experiments.py --quick-test
```

---

## Running Experiments

### Example 1: Simple Single Experiment

```python
from src.model_client import OllamaClient
from src.experiment_runner import ExperimentRunner

# Initialize
client = OllamaClient(model="phi3:mini")
runner = ExperimentRunner(client)

# Run experiment
result = runner.run_single_experiment(
    prompt="What is machine learning?",
    temperature=0.7,
    max_tokens=200
)

print(f"Response: {result['response']}")
print(f"Quality Score: {result['metrics']['quality_score']:.3f}")
print(f"Latency: {result['metrics']['latency']:.3f}s")
```

### Example 2: Full Validation Suite

```bash
# Run with default model
python run_experiments.py

# Run with specific model
python run_experiments.py --model llama2

# Run with custom experiment name
python run_experiments.py --experiment-name "my_test"
```

### Example 3: Batch Experiments

```python
from src.model_client import OllamaClient
from src.experiment_runner import ExperimentRunner

client = OllamaClient(model="llama2")
runner = ExperimentRunner(client)

# Multiple prompts
prompts = [
    "Explain neural networks",
    "What is deep learning?",
    "Describe transformers"
]

# Multiple temperatures
temperatures = [0.3, 0.7, 0.9]

# Run all combinations
results = runner.run_batch_experiments(
    prompts=prompts,
    temperatures=temperatures,
    max_tokens=300
)

# Analyze results
for result in results:
    print(f"Temp: {result['temperature']}, Quality: {result['metrics']['quality_score']:.3f}")
```

### Example 4: Model Comparison

```python
from src.model_client import OllamaClient
from src.experiment_runner import ExperimentRunner

# Create runner
client = OllamaClient()  # Default model
runner = ExperimentRunner(client)

# Compare multiple models
models = ["phi3:mini", "llama2"]
prompts = [
    "What is AI?",
    "Explain machine learning",
    "What are neural networks?"
]

comparison = runner.compare_models(
    models=models,
    prompts=prompts,
    temperature=0.7
)

# Print comparison
for model, results in comparison.items():
    avg_quality = sum(r['metrics']['quality_score'] for r in results) / len(results)
    avg_latency = sum(r['metrics']['latency'] for r in results) / len(results)
    print(f"{model}: Quality={avg_quality:.3f}, Latency={avg_latency:.3f}s")
```

### Example 5: Custom Test Categories

Edit `experiments/benchmark_queries.json`:

```json
{
  "test_suites": [
    {
      "category": "my_custom_tests",
      "description": "My custom test suite",
      "queries": [
        {
          "prompt": "Your test prompt here",
          "expected_keywords": ["keyword1", "keyword2"],
          "max_tokens": 200
        }
      ]
    }
  ]
}
```

Then run:
```bash
python run_experiments.py
```

---

## Monitoring

### Example 6: Real-time Monitoring Dashboard

```bash
# Start dashboard with demo data
python monitoring_dashboard.py --demo

# Open in browser: http://localhost:8050
```

The dashboard shows:
- Total requests and success rate
- Average latency and quality
- Real-time charts
- System resource usage
- Active alerts

### Example 7: Programmatic Monitoring

```python
from src.monitoring import MetricsMonitor, SystemMonitor, AlertManager

# Initialize
monitor = MetricsMonitor()
alert_manager = AlertManager(
    latency_threshold=5.0,
    error_rate_threshold=0.1
)

# Record results
from src.model_client import OllamaClient

client = OllamaClient()
result = client.generate("Hello!")

# Add to monitor
monitor.record_request({
    "success": result["success"],
    "latency": result["latency"],
    "model": result["model"],
    "eval_count": result.get("eval_count", 0),
    "tokens_per_second": result.get("tokens_per_second", 0)
})

# Get stats
stats = monitor.get_current_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Success rate: {stats['success_rate']:.1%}")

# Check alerts
alerts = alert_manager.check_alerts(stats)
for alert in alerts:
    print(f"Alert: {alert['message']}")
```

### Example 8: Custom Metrics

Add to `src/metrics_collector.py`:

```python
@staticmethod
def calculate_sentiment_score(text: str) -> float:
    """Calculate sentiment score (example)"""
    positive_words = ['good', 'great', 'excellent', 'amazing']
    negative_words = ['bad', 'poor', 'terrible', 'awful']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count + negative_count == 0:
        return 0.5  # Neutral
    
    return positive_count / (positive_count + negative_count)
```

Use it:
```python
from src.metrics_collector import MetricsCollector

collector = MetricsCollector()
sentiment = collector.calculate_sentiment_score(response_text)
```

---

## Analysis

### Example 9: MLflow UI

```bash
# Start MLflow UI
mlflow ui

# Open: http://localhost:5000
```

In the UI:
1. View all experiments
2. Compare runs
3. Sort by metrics
4. Download artifacts
5. Export data

### Example 10: Jupyter Notebook Analysis

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/analysis.ipynb
# Run all cells to see visualizations
```

### Example 11: Programmatic Analysis

```python
import mlflow
import pandas as pd

# Set tracking URI
mlflow.set_tracking_uri("./mlruns")

# Get experiment
experiment = mlflow.get_experiment_by_name("model_validation")

# Get all runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.quality_score DESC"]
)

# Analyze
print("Top 5 runs by quality:")
print(runs[['params.model', 'metrics.quality_score', 'metrics.latency']].head())

# Filter runs
good_runs = runs[runs['metrics.quality_score'] > 0.8]
print(f"\nRuns with quality > 0.8: {len(good_runs)}")

# Export
runs.to_csv('analysis_results.csv', index=False)
```

### Example 12: Visualizations with Plotly

```python
import plotly.express as px
import mlflow
import pandas as pd

# Load runs
mlflow.set_tracking_uri("./mlruns")
experiment = mlflow.get_experiment_by_name("model_validation")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Quality vs Latency scatter
fig = px.scatter(
    runs,
    x='metrics.latency',
    y='metrics.quality_score',
    color='params.model',
    size='metrics.total_tokens',
    title='Quality vs Latency by Model'
)
fig.show()

# Temperature effect
fig = px.box(
    runs,
    x='params.temperature',
    y='metrics.quality_score',
    title='Quality Score Distribution by Temperature'
)
fig.show()
```

---

## Advanced Usage

### Example 13: Custom Validation Tests

Create `my_validation.py`:

```python
from src.model_client import OllamaClient
from src.metrics_collector import MetricsCollector

def validate_code_generation(client: OllamaClient):
    """Custom validation for code generation"""
    test_cases = [
        {
            "prompt": "Write a Python function to reverse a string",
            "expected_tokens": ["def", "return", "reverse"]
        },
        {
            "prompt": "Create a function to check if a number is prime",
            "expected_tokens": ["def", "prime", "return"]
        }
    ]
    
    results = []
    for test in test_cases:
        result = client.generate(test["prompt"], max_tokens=300)
        
        # Check if expected tokens are present
        score = sum(
            1 for token in test["expected_tokens"]
            if token in result["response"]
        ) / len(test["expected_tokens"])
        
        results.append({
            "prompt": test["prompt"],
            "score": score,
            "response": result["response"]
        })
    
    return results

# Run
client = OllamaClient(model="codellama")
results = validate_code_generation(client)

for r in results:
    print(f"Score: {r['score']:.2f} - {r['prompt'][:50]}...")
```

### Example 14: A/B Testing

```python
from src.model_client import OllamaClient
from src.experiment_runner import ExperimentRunner
import mlflow

def ab_test(model_a: str, model_b: str, prompts: list):
    """Compare two models on same prompts"""
    
    results = {"model_a": [], "model_b": []}
    
    # Test Model A
    client_a = OllamaClient(model=model_a)
    runner_a = ExperimentRunner(client_a, experiment_name="ab_test_a")
    
    for prompt in prompts:
        result = runner_a.run_single_experiment(prompt, temperature=0.7)
        results["model_a"].append(result)
    
    # Test Model B
    client_b = OllamaClient(model=model_b)
    runner_b = ExperimentRunner(client_b, experiment_name="ab_test_b")
    
    for prompt in prompts:
        result = runner_b.run_single_experiment(prompt, temperature=0.7)
        results["model_b"].append(result)
    
    # Compare
    avg_quality_a = sum(r['metrics']['quality_score'] for r in results["model_a"]) / len(results["model_a"])
    avg_quality_b = sum(r['metrics']['quality_score'] for r in results["model_b"]) / len(results["model_b"])
    
    print(f"{model_a} avg quality: {avg_quality_a:.3f}")
    print(f"{model_b} avg quality: {avg_quality_b:.3f}")
    print(f"Winner: {model_a if avg_quality_a > avg_quality_b else model_b}")
    
    return results

# Run A/B test
prompts = ["What is AI?", "Explain ML", "Define neural networks"]
results = ab_test("phi3:mini", "llama2", prompts)
```

### Example 15: Continuous Testing

```python
import time
from src.model_client import OllamaClient
from src.experiment_runner import ExperimentRunner

def continuous_testing(model: str, interval: int = 300):
    """Run continuous tests every N seconds"""
    
    client = OllamaClient(model=model)
    runner = ExperimentRunner(client, experiment_name="continuous_testing")
    
    test_prompt = "What is the current state of AI?"
    
    print(f"Starting continuous testing for {model}")
    print(f"Running test every {interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            result = runner.run_single_experiment(
                prompt=test_prompt,
                temperature=0.7,
                tags={"type": "continuous_test"}
            )
            
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Quality: {result['metrics']['quality_score']:.3f}, "
                  f"Latency: {result['metrics']['latency']:.3f}s")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nStopped continuous testing")

# Run
continuous_testing("phi3:mini", interval=60)  # Every minute
```

---

## DagsHub Integration

### Example 16: Setup DagsHub

```bash
# 1. Create .env file
cp .env.example .env

# 2. Edit .env with your credentials
# DAGSHUB_USERNAME=your_username
# DAGSHUB_TOKEN=your_token
# DAGSHUB_REPO=your_repo
# MLFLOW_TRACKING_URI=https://dagshub.com/your_username/your_repo.mlflow

# 3. Run experiments with DagsHub
python run_experiments.py --dagshub
```

### Example 17: View on DagsHub

After running experiments with `--dagshub`:

1. Go to: `https://dagshub.com/your_username/your_repo`
2. Click "Experiments" tab
3. View all tracked experiments
4. Share link with team
5. Compare runs online

### Example 18: Switch Between Local and Remote

```python
import mlflow
import os

# Use local
mlflow.set_tracking_uri("./mlruns")

# Use DagsHub
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("DAGSHUB_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")

# Now run experiments - they'll be tracked remotely
```

---

## Tips and Best Practices

### Performance Optimization

1. **Use smaller models for rapid iteration:**
   ```bash
   ollama pull phi3:mini  # Fast and efficient
   ```

2. **Limit max_tokens for quick tests:**
   ```python
   result = client.generate(prompt, max_tokens=50)
   ```

3. **Run experiments in parallel:**
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   with ThreadPoolExecutor(max_workers=3) as executor:
       futures = [executor.submit(run_experiment, prompt) for prompt in prompts]
       results = [f.result() for f in futures]
   ```

### Quality Improvement

1. **Temperature tuning:**
   - 0.1-0.3: More deterministic, factual
   - 0.7: Balanced (recommended)
   - 0.9-1.0: More creative

2. **Prompt engineering:**
   ```python
   # Be specific
   prompt = "Explain quantum computing in exactly 2 sentences using simple language."
   ```

3. **Multiple runs for consistency:**
   ```python
   results = [run_experiment(prompt) for _ in range(5)]
   avg_quality = sum(r['metrics']['quality_score'] for r in results) / 5
   ```

### Monitoring Best Practices

1. Set appropriate thresholds:
   ```python
   AlertManager(
       latency_threshold=5.0,      # 5 seconds
       error_rate_threshold=0.05,  # 5% errors
       quality_threshold=0.7       # 70% quality
   )
   ```

2. Regular health checks:
   ```bash
   # Add to crontab for hourly checks
   0 * * * * cd /path/to/mlops && python run_experiments.py --quick-test
   ```

---

## Troubleshooting Examples

### Debug Failed Experiments

```python
from src.model_client import OllamaClient

client = OllamaClient(model="llama2")

# Test connection
if client.check_connection():
    print("✓ Connected")
else:
    print("✗ Connection failed")
    print("Start Ollama with: ollama serve")

# Test generation
result = client.generate("Test", max_tokens=10)
if not result["success"]:
    print(f"Error: {result['error']}")
```

### Check MLflow Data

```python
import mlflow

mlflow.set_tracking_uri("./mlruns")

# List all experiments
for exp in mlflow.search_experiments():
    print(f"{exp.name}: {exp.experiment_id}")
    
    # Count runs
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    print(f"  Runs: {len(runs)}")
```

---

For more examples and detailed documentation, see:
- `README.md` - Project overview
- `SETUP_GUIDE.md` - Detailed setup instructions
- `config/experiment_config.yaml` - Configuration options
