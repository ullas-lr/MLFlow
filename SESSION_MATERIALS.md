# FDP Session: Cloud platforms, edge AI, model validation and monitoring

**Speaker:** Ullas Lakku Raghavendra, Adobe  
**Date:** June 2, 2026  
**Duration:** 9:30 AM - 11:15 AM

## 🎯 Session Overview

This session demonstrates a complete ML model validation and monitoring system suitable for:
- Healthcare applications
- Edge AI deployments
- Cloud-based systems
- Privacy-sensitive domains

## 🛠️ Technology Stack Demonstrated

### Core Components
- **Ollama** - Local LLM deployment (edge AI simulation)
- **MLflow** - Experiment tracking and model registry
- **Python** - Validation framework and metrics
- **Dash/Plotly** - Real-time monitoring dashboards

### Key Features
- ✅ Automated validation test suites
- ✅ Quality metrics (coherence, relevance, safety)
- ✅ Performance benchmarking
- ✅ Real-time monitoring
- ✅ Experiment tracking and comparison
- ✅ **Data drift detection**
- ✅ **Model drift detection**
- ✅ **Automated alerting**

## 📚 GitHub Repository

**Full source code available at:**
```
https://github.com/ullas-lr/MLFlow
```

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Install Ollama
brew install ollama  # macOS
# or download from https://ollama.ai

# Clone repository
git clone https://github.com/ullas-lr/MLFlow.git
cd MLFlow

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Ollama and pull a model
ollama serve
ollama pull phi3:mini
```

### 2. Run Validation

```bash
# General validation
python run_experiments.py --model phi3:mini

# Healthcare-specific demo
python demo_healthcare.py phi3:mini

# Drift detection demo
python demo_drift_detection.py phi3:mini
```

### 3. View Results

```bash
# MLflow UI
mlflow ui
# Open: http://localhost:5000

# Drift Detection Dashboard
python drift_dashboard.py
# Open: http://localhost:8051
```

## 🏥 Healthcare Use Cases

### 1. **Medical Chatbot Validation**
- Test medical knowledge accuracy
- Evaluate safety responses
- Measure response appropriateness

### 2. **Clinical Decision Support**
- Validate diagnostic suggestions
- Test treatment recommendations
- Assess risk calculations

### 3. **Medical Coding Assistance**
- Test ICD-10 code suggestions
- Validate CPT code accuracy
- Measure coding consistency

### 4. **Medical Report Analysis**
- Test information extraction
- Validate summarization quality
- Assess terminology understanding

## 📊 Validation Metrics Explained

### Quality Metrics
- **Coherence Score** (0-1): Text consistency and structure
- **Relevance Score** (0-1): Answer relevance to question
- **Safety Score** (0-1): Absence of harmful content
- **Overall Quality**: Weighted combination

### Performance Metrics
- **Latency**: Response time (seconds)
- **Throughput**: Tokens per second
- **Resource Usage**: CPU/Memory consumption

## 🔍 Drift Detection

### What is Drift?

**Data Drift** - Changes in input data distribution over time:
- Example: Summer (general health) → Winter (flu season)
- Impact: Model trained on summer data underperforms in winter

**Model Drift** - Changes in model performance over time:
- Example: Quality degrades from 0.92 to 0.72 (20% drop)
- Impact: Unreliable predictions, potentially harmful in healthcare

### Healthcare Drift Scenarios

#### 1. Seasonal Disease Patterns
```
Summer queries → Winter queries
General health → Respiratory illnesses
Action: Retrain with seasonal data
```

#### 2. Demographic Shifts
```
Mixed age → Aging population
20% geriatric → 40% geriatric
Action: Adjust training data proportions
```

#### 3. New Medical Events
```
Pre-pandemic → Post-pandemic
Traditional diseases → COVID-19 terminology
Action: Update with new medical literature
```

#### 4. Model Performance Degradation
```
Week 1 → Week 4
Quality: 0.92 → 0.72
Latency: 30s → 50s
Action: Investigate and retrain
```

### Statistical Tests Used

- **Kolmogorov-Smirnov Test**: Data distribution comparison
- **Mann-Whitney U Test**: Performance comparison
- **KL Divergence**: Category distribution shift
- **p-value < 0.05**: Statistical significance threshold

### Drift Detection Commands

```bash
# Run full drift demo
python demo_drift_detection.py phi3:mini

# Launch drift dashboard
python drift_dashboard.py
```

### Drift Monitoring Dashboard

The drift dashboard visualizes:
- Quality score trends over time
- Latency drift detection
- Distribution comparisons (baseline vs current)
- Real-time alerts
- Category shift visualization

### When to Act on Drift

**Critical (Immediate Action)**:
- Quality degradation > 20%
- Safety score < 0.90
- Latency > 2x baseline

**Warning (Monitor Closely)**:
- Quality degradation 10-20%
- Safety score 0.90-0.95
- Latency 1.5-2x baseline

**Info (Track Trends)**:
- Quality degradation 5-10%
- Minor latency increases
- Small distribution shifts

## 🔐 Privacy & Compliance

### HIPAA Compliance Considerations
- ✅ All processing happens locally
- ✅ No data transmission to external APIs
- ✅ Audit trail through MLflow
- ✅ Access control mechanisms
- ✅ Data encryption at rest

### GDPR Compliance
- ✅ Data minimization (local processing)
- ✅ Right to erasure (delete runs)
- ✅ Transparency (full logging)
- ✅ Data portability (export features)

## 🌐 Deployment Options

### Option 1: Local/Edge Deployment
```
Hospital Server → Ollama → Validation → Local MLflow
                     ↓
              No Internet Required
```

### Option 2: Hybrid Deployment
```
Edge Device → Local Inference
     ↓
Cloud → MLflow Tracking (metadata only)
```

### Option 3: Full Cloud Deployment
```
Cloud GPU → Model Serving
     ↓
Cloud MLflow → Full Tracking
     ↓
Dashboard → Monitoring
```

## 📖 Key Resources

### Documentation
- **Ollama Docs**: https://github.com/ollama/ollama
- **MLflow Docs**: https://mlflow.org/docs/latest/
- **Project README**: See repository

### Models Available
- **phi3:mini** (2GB) - Fast, efficient
- **llama2** (4GB) - Balanced performance
- **mistral** (4GB) - High quality
- **codellama** (4GB) - Code-specialized

### Example Notebooks
- `notebooks/analysis.ipynb` - Data analysis
- `experiments/validation_suite.py` - Test framework
- `demo_healthcare.py` - Healthcare demo

## 💡 Best Practices

### 1. **Start Small**
- Begin with a few validation tests
- Gradually expand test coverage
- Iterate based on results

### 2. **Define Success Criteria**
- Set quality thresholds (e.g., > 0.8)
- Define acceptable latency
- Establish safety requirements

### 3. **Automate Everything**
- Automated validation on model updates
- Continuous monitoring
- Alert on quality degradation

### 4. **Version Control**
- Track all experiments in MLflow
- Version your test suites
- Document changes

### 5. **Monitor in Production**
- Real-time performance tracking
- Quality drift detection
- User feedback integration

## 🎓 Learning Path

### Beginner
1. Run the quick test
2. Explore MLflow UI
3. Try different models

### Intermediate
1. Create custom test cases
2. Modify validation metrics
3. Set up monitoring alerts

### Advanced
1. Integrate with CI/CD
2. Deploy to production
3. Implement A/B testing
4. Build custom models

## 📞 Contact & Follow-up

**Speaker:** Ullas Lakku Raghavendra  
**Organization:** Adobe  
**Email:** [Your email]  
**LinkedIn:** [Your LinkedIn]  
**GitHub:** https://github.com/ullas-lr

## 🙏 Acknowledgments

- Ramaiah Institute of Technology
- Adobe Systems
- Open Source Community
- All FDP Participants

## 📝 Additional Notes

[Space for your notes during the session]

---

**Session Materials Version:** 1.0  
**Last Updated:** February 5, 2026  
**License:** MIT (for code samples)
