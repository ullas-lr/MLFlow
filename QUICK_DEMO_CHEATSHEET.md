# 📝 FDP Session Quick Cheatsheet

**Cloud Platforms, Edge AI, Model Validation & Monitoring**  
**One-page reference - keep open during session**

---

## 📋 Session Structure (105 min = 1h 45min)

1. **Introduction** (5 min)
2. **Cloud Platforms** (15 min) - Conceptual + Discussion
3. **Edge AI** (15 min) - Conceptual + Discussion
4. **Model Validation** (25 min) - Live Demo + Deep Dive
5. **Model Monitoring** (25 min) - Live Demo + Deep Dive
6. **Production & Compliance** (10 min)
7. **Wrap-up** (5 min)
8. **Q&A** (15 min)

---

## 🚀 Demo Commands (Parts 3 & 4)

### 1. Healthcare Validation
```bash
python demo_healthcare.py phi3:mini
```
*Show: Quality scores, safety checks, MLflow logging*

### 2. MLflow UI
```bash
mlflow ui
# http://localhost:5000
```
*Show: Experiments, runs, metrics, parameters*

### 3. Drift Detection
```bash
python demo_drift_detection.py phi3:mini
```
*Show: Seasonal drift, performance drift, statistical tests*

### 4. Drift Dashboard
```bash
python drift_dashboard.py
# http://localhost:8051
```
*Show: Real MLflow data, trends, alerts*

---

## 💬 Key Talking Points

### Part 1: Cloud Platforms
- "Three major players: AWS, Azure, GCP - all HIPAA-compliant"
- "Each has strengths: AWS (mature), Azure (enterprise), GCP (AI/ML)"
- "Healthcare requires BAA, specific configurations"

### Part 2: Edge AI
- "Edge = local processing, not cloud"
- "Critical for healthcare: privacy, latency, offline capability"
- "Real example: Brain hemorrhage detection in ER needs <5 seconds"
- "Hybrid is most common: inference at edge, tracking in cloud"

### Part 3: Model Validation
- "Testing quality, safety, coherence on healthcare queries"
- "Average quality: 93% - production-ready"
- "All tracked in MLflow - works on edge OR cloud"
- "Our demo runs on edge (Ollama locally)"

### Part 4: Model Monitoring
- "Two types: Data drift + Model drift"
- "Statistical tests: KS test, Mann-Whitney U, KL divergence"
- "Automated alerts when thresholds crossed"
- "Dashboard shows real data from 30+ runs"
- "Works on edge AND cloud!"

---

## 📊 Visual Flow

```
Healthcare Query
    ↓
phi3:mini Model
    ↓
Quality/Safety Metrics → MLflow
    ↓
Drift Detection
    ↓
Dashboard Alert
```

---

## ⏱️ Detailed Timing (105 min total)

**Introduction** (5 min)
- Welcome & agenda: 2 min
- Today's objectives: 3 min

**Part 1: Cloud Platforms** (15 min)
- Cloud overview (AWS/Azure/GCP): 5 min
- ML services comparison: 5 min
- Healthcare considerations: 3 min
- Discussion/Questions: 2 min

**Part 2: Edge AI** (15 min)
- What is Edge AI: 3 min
- Why critical for healthcare: 4 min
- Real-world use cases: 4 min
- Edge vs Cloud vs Hybrid: 2 min
- Discussion/Questions: 2 min

**Part 3: Model Validation** (25 min)
- Validation concepts: 3 min
- Live demo (healthcare validation): 8 min
- Explain results: 5 min
- MLflow UI walkthrough: 7 min
- Discussion/Questions: 2 min

**Part 4: Model Monitoring** (25 min)
- Drift concepts (data + model): 4 min
- Live demo (drift detection): 12 min
- Dashboard walkthrough: 7 min
- Discussion/Questions: 2 min

**Production & Compliance** (10 min)
- Deployment strategies: 4 min
- HIPAA compliance: 3 min
- Real-world impact: 3 min

**Wrap-up** (5 min)
- Key takeaways: 3 min
- Resources & next steps: 2 min

**Q&A** (15 min)

**Total: 105 minutes (9:30-11:15 AM)**

---

## 🎯 Key Messages

1. **Complete MLOps pipeline** - validation → tracking → monitoring
2. **Automated drift detection** - catches problems before they impact patients
3. **Production-ready** - HIPAA compliant, full audit trail
4. **Open source** - all code on GitHub

---

## 🔗 URLs

- MLflow: http://localhost:5000
- Drift Dashboard: http://localhost:8051
- GitHub: https://github.com/ullas-lr/MLFlow

---

## 🛟 Emergency Backup

If something breaks:
- Skip to next part
- Show screenshots (take these before session!)
- Explain concepts without demo
- Focus on MLflow UI (most reliable)
