# Getting Started with ML Model Validation & Monitoring

Welcome! This guide will help you get started in **5 minutes**.

## 🚀 Quick Start (Choose One)

### Option 1: Automated Setup (Recommended)

```bash
cd /Users/lakkurra/mlops/MLFlow
./quickstart.sh
```

This will automatically:
- ✅ Check your environment
- ✅ Install dependencies
- ✅ Verify Ollama
- ✅ Pull a model
- ✅ Run a test

### Option 2: Manual Setup

```bash
# 1. Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. In a separate terminal, start Ollama
ollama serve

# 3. Pull a model
ollama pull phi3:mini

# 4. Test it
python run_experiments.py --quick-test
```

## 📚 What's Included?

This project provides a complete system for:

1. **Model Deployment** - Run LLMs locally with Ollama
2. **Experiment Tracking** - Track all experiments with MLflow
3. **Validation** - Automated test suites for quality assurance
4. **Monitoring** - Real-time performance monitoring
5. **Observability** - Dashboards and metrics
6. **DagsHub Integration** - Optional cloud-based tracking

## 🎯 Your First Experiment

After setup, run your first experiment:

```bash
# Activate environment
source venv/bin/activate

# Run full validation suite
python run_experiments.py

# View results
mlflow ui
# Open: http://localhost:5000
```

## 📊 View Your Results

### MLflow UI (Local)

```bash
mlflow ui
```

Visit `http://localhost:5000` to see:
- All experiment runs
- Metrics and parameters
- Comparison charts
- Saved artifacts

### Monitoring Dashboard

```bash
python monitoring_dashboard.py --demo
```

Visit `http://localhost:8050` to see:
- Real-time metrics
- Performance charts
- System resources
- Active alerts

### Jupyter Analysis

```bash
jupyter notebook notebooks/analysis.ipynb
```

Interactive analysis with visualizations

## 🔧 Common Tasks

### Test a Different Model

```bash
# Pull the model
ollama pull mistral

# Run experiments
python run_experiments.py --model mistral
```

### Compare Models

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

Edit `experiments/benchmark_queries.json`:

```json
{
  "test_suites": [
    {
      "category": "my_tests",
      "queries": [
        {
          "prompt": "Your test here",
          "max_tokens": 200,
          "expected_keywords": ["keyword1", "keyword2"]
        }
      ]
    }
  ]
}
```

## 📖 Documentation

- **README.md** - Project overview and architecture
- **SETUP_GUIDE.md** - Detailed setup instructions
- **USAGE_EXAMPLES.md** - Code examples and recipes
- **This file** - Quick start guide

## 🛠 Project Structure

```
mlops/
├── README.md              # Overview
├── SETUP_GUIDE.md        # Setup instructions
├── USAGE_EXAMPLES.md     # Examples
├── GETTING_STARTED.md    # This file
├── quickstart.sh         # Automated setup
│
├── run_experiments.py    # Main experiment runner
├── monitoring_dashboard.py  # Monitoring dashboard
│
├── src/                  # Source code
│   ├── model_client.py   # Ollama integration
│   ├── experiment_runner.py  # Experiment logic
│   ├── metrics_collector.py  # Custom metrics
│   └── monitoring.py     # Monitoring tools
│
├── experiments/          # Test suites
│   ├── validation_suite.py
│   └── benchmark_queries.json
│
├── notebooks/            # Jupyter notebooks
│   └── analysis.ipynb
│
└── config/              # Configuration
    └── experiment_config.yaml
```

## 🎓 Learning Path

### Day 1: Setup and First Run
1. ✅ Run `./quickstart.sh`
2. ✅ Run `python run_experiments.py`
3. ✅ Explore MLflow UI

### Day 2: Understand the Code
1. 📖 Read `USAGE_EXAMPLES.md`
2. 🔬 Try different models
3. 📊 Run monitoring dashboard

### Day 3: Customize
1. ✏️ Add custom tests
2. 📝 Modify metrics
3. 🎨 Create custom analysis

### Day 4: Advanced
1. 🔗 Set up DagsHub
2. 🤖 Run continuous monitoring
3. 📈 A/B test models

## ❓ Common Questions

### Q: Which model should I use?

**For learning/testing:** `phi3:mini` (fast, 2GB)
**For quality:** `llama2` or `mistral` (slower, 4GB)
**For coding:** `codellama` (4GB)

```bash
ollama pull phi3:mini  # Start here
```

### Q: Where are experiments stored?

Locally in `./mlruns/` directory. View with:
```bash
mlflow ui
```

### Q: How do I use DagsHub?

1. Create account at https://dagshub.com
2. Create repository
3. Copy `.env.example` to `.env`
4. Add your credentials
5. Run with: `python run_experiments.py --dagshub`

### Q: How do I add my own metrics?

Edit `src/metrics_collector.py`:

```python
@staticmethod
def calculate_custom_metric(text: str) -> float:
    # Your logic here
    return score
```

### Q: Can I run experiments in parallel?

Yes! See `USAGE_EXAMPLES.md` for examples using ThreadPoolExecutor.

## 🐛 Troubleshooting

### "Cannot connect to Ollama"

```bash
# Start Ollama server
ollama serve
```

### "Model not found"

```bash
# List available models
ollama list

# Pull the model
ollama pull llama2
```

### "Port already in use"

```bash
# MLflow (port 5000)
lsof -i :5000
kill -9 <PID>

# Dashboard (port 8050)
lsof -i :8050
kill -9 <PID>
```

### Dependencies issues

```bash
# Reinstall
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 📞 Next Steps

1. **Run your first experiment:**
   ```bash
   python run_experiments.py
   ```

2. **Explore the results:**
   ```bash
   mlflow ui
   ```

3. **Try the monitoring dashboard:**
   ```bash
   python monitoring_dashboard.py --demo
   ```

4. **Read the examples:**
   Open `USAGE_EXAMPLES.md`

5. **Customize for your needs:**
   - Add custom tests
   - Modify metrics
   - Try different models

## 🎉 Success Checklist

- [ ] Ran `./quickstart.sh` successfully
- [ ] Saw "All checks passed!"
- [ ] Ran first experiment with `python run_experiments.py`
- [ ] Viewed results in MLflow UI (`mlflow ui`)
- [ ] Tried monitoring dashboard (`python monitoring_dashboard.py --demo`)
- [ ] Opened and explored `notebooks/analysis.ipynb`

Once all checked, you're ready to start experimenting! 🚀

## 🌟 Pro Tips

1. **Start small:** Use `phi3:mini` for fast iteration
2. **Monitor everything:** Keep the dashboard running
3. **Compare models:** Try multiple models on same prompts
4. **Version control:** Commit experiments to git
5. **Share results:** Use DagsHub for team collaboration

---

**Need help?** Check:
- `SETUP_GUIDE.md` for detailed setup
- `USAGE_EXAMPLES.md` for code examples
- `README.md` for architecture overview

Happy experimenting! 🎊
