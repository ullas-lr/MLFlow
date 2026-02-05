#!/usr/bin/env python3
"""
Healthcare AI Drift Detection Demo
Demonstrates data drift and model drift detection for FDP session

Scenarios covered:
1. Seasonal disease pattern drift (flu season)
2. Demographic shift drift (aging population)
3. Model performance degradation
4. Response quality drift over time
"""

import sys
import mlflow
from datetime import datetime
import time

from src.model_client import OllamaClient
from src.experiment_runner import ExperimentRunner
from src.drift_detector import (
    DataDriftDetector,
    ModelDriftDetector,
    HealthcareDriftScenarios,
    format_drift_report
)


def demo_scenario_1_seasonal_drift(model_name="phi3:mini"):
    """
    Scenario 1: Seasonal Disease Pattern Drift
    
    Simulates change from summer (general health) to winter (respiratory)
    """
    print("\n" + "="*80)
    print("SCENARIO 1: SEASONAL DISEASE PATTERN DRIFT")
    print("Simulating: Summer queries → Winter queries (Flu season)")
    print("="*80)
    
    # Get baseline and drifted data
    baseline_data, drifted_data = HealthcareDriftScenarios.simulate_seasonal_drift()
    
    print("\n📅 BASELINE (Summer - General Health Queries):")
    for i, q in enumerate(baseline_data, 1):
        print(f"  {i}. {q['prompt']}")
    
    print("\n❄️  CURRENT (Winter - Respiratory Focus):")
    for i, q in enumerate(drifted_data, 1):
        print(f"  {i}. {q['prompt']}")
    
    # Initialize detector
    detector = DataDriftDetector(baseline_data=baseline_data)
    
    # Detect category distribution drift
    current_categories = [d['category'] for d in drifted_data]
    drift_result = detector.detect_category_distribution_drift(current_categories)
    
    print("\n🔍 DRIFT ANALYSIS:")
    print(f"  Baseline Distribution: {drift_result.get('baseline_distribution', {})}")
    print(f"  Current Distribution:  {drift_result.get('current_distribution', {})}")
    print(f"  KL Divergence: {drift_result.get('kl_divergence', 0):.4f}")
    
    if drift_result.get('drift_detected'):
        print(f"\n  🚨 DRIFT DETECTED: Query patterns have shifted significantly!")
        print(f"     Action: Retrain model with respiratory-focused data")
    else:
        print(f"\n  ✅ No significant drift detected")
    
    return drift_result


def demo_scenario_2_model_drift(model_name="phi3:mini"):
    """
    Scenario 2: Model Performance Degradation
    
    Simulates model quality degradation over time
    """
    print("\n" + "="*80)
    print("SCENARIO 2: MODEL PERFORMANCE DRIFT")
    print("Simulating: Performance degradation detection")
    print("="*80)
    
    # Get baseline and degraded data
    baseline_runs, degraded_runs = HealthcareDriftScenarios.simulate_model_performance_drift()
    
    print("\n📊 BASELINE PERFORMANCE:")
    for run in baseline_runs[:3]:
        print(f"  Query: {run['prompt']}")
        print(f"    Quality: {run['metrics']['quality_score']:.3f}")
        print(f"    Latency: {run['metrics']['latency']:.1f}s")
    
    print("\n📉 CURRENT PERFORMANCE (Degraded):")
    for run in degraded_runs[:3]:
        print(f"  Query: {run['prompt']}")
        print(f"    Quality: {run['metrics']['quality_score']:.3f} (⬇️)")
        print(f"    Latency: {run['metrics']['latency']:.1f}s (⬆️)")
    
    # Initialize detector
    detector = ModelDriftDetector(baseline_runs=baseline_runs)
    
    # Run comprehensive drift detection
    drift_results = detector.detect_comprehensive_drift(degraded_runs)
    
    # Display report
    print(format_drift_report(drift_results))
    
    # Recommendations
    if drift_results['overall_drift_detected']:
        print("\n💡 RECOMMENDED ACTIONS:")
        for metric in drift_results['drifted_metrics']:
            check = drift_results['drift_checks'][metric]
            severity = check.get('severity', 'unknown')
            
            if metric == 'quality_score':
                if severity == 'high':
                    print("  🔴 Critical: Quality degraded significantly")
                    print("     → Immediate model retraining required")
                    print("     → Review recent data for issues")
                else:
                    print("  🟡 Warning: Quality declining")
                    print("     → Schedule model refresh")
            
            elif metric == 'latency':
                print("  🟠 Performance: Latency increased")
                print("     → Check system resources")
                print("     → Consider model optimization")
    
    return drift_results


def demo_scenario_3_real_time_drift(model_name="phi3:mini"):
    """
    Scenario 3: Real-time drift detection with actual model
    
    Runs actual queries and detects drift
    """
    print("\n" + "="*80)
    print("SCENARIO 3: REAL-TIME DRIFT DETECTION")
    print("Using actual Ollama model for drift monitoring")
    print("="*80)
    
    # Initialize client
    client = OllamaClient(model=model_name)
    runner = ExperimentRunner(
        client=client,
        experiment_name="drift_monitoring_demo"
    )
    
    # Baseline queries - general medical
    print("\n📊 Phase 1: Establishing Baseline (General Medical Queries)...")
    baseline_queries = [
        "What is blood pressure?",
        "Explain heart rate",
        "What is body temperature?",
    ]
    
    baseline_runs = []
    for prompt in baseline_queries:
        print(f"  Running: {prompt[:40]}...")
        result = runner.run_single_experiment(
            prompt=prompt,
            temperature=0.7,
            max_tokens=150,
            tags={"phase": "baseline", "drift_demo": "true"}
        )
        if result['success']:
            baseline_runs.append(result)
            print(f"    Quality: {result['metrics']['quality_score']:.3f}")
    
    # Wait a moment
    time.sleep(2)
    
    # Current queries - specialized focus (simulating drift)
    print("\n📊 Phase 2: Monitoring Current Queries (Specialized Focus)...")
    current_queries = [
        "What is cardiac arrhythmia?",
        "Explain atrial fibrillation",
        "What is ventricular tachycardia?",
    ]
    
    current_runs = []
    for prompt in current_queries:
        print(f"  Running: {prompt[:40]}...")
        result = runner.run_single_experiment(
            prompt=prompt,
            temperature=0.7,
            max_tokens=150,
            tags={"phase": "current", "drift_demo": "true"}
        )
        if result['success']:
            current_runs.append(result)
            print(f"    Quality: {result['metrics']['quality_score']:.3f}")
    
    # Detect drift
    print("\n🔍 Analyzing Drift...")
    
    # Model drift detection
    model_detector = ModelDriftDetector(baseline_runs=baseline_runs)
    drift_analysis = model_detector.detect_comprehensive_drift(current_runs)
    
    print(format_drift_report(drift_analysis))
    
    # Data drift detection
    data_detector = DataDriftDetector()
    data_detector.add_baseline_data([
        {"prompt": q, "category": "general"} for q in baseline_queries
    ])
    
    prompt_drift = data_detector.detect_prompt_length_drift(current_queries)
    
    print(f"\n📏 Prompt Length Drift:")
    print(f"  Baseline: {prompt_drift.get('baseline_mean', 0):.1f} ± {prompt_drift.get('baseline_std', 0):.1f} chars")
    print(f"  Current:  {prompt_drift.get('current_mean', 0):.1f} ± {prompt_drift.get('current_std', 0):.1f} chars")
    print(f"  Change:   {prompt_drift.get('percent_change', 0):.1f}%")
    print(f"  Status:   {'🚨 DRIFT' if prompt_drift.get('drift_detected') else '✅ OK'}")
    
    return drift_analysis


def main():
    """Main drift detection demo"""
    print("\n" + "="*80)
    print("HEALTHCARE AI DRIFT DETECTION DEMO")
    print("Session: Cloud platforms, edge AI, model validation and monitoring")
    print("By: Ullas Lakku Raghavendra, Adobe")
    print("="*80)
    
    model = sys.argv[1] if len(sys.argv) > 1 else "phi3:mini"
    
    print(f"\n🤖 Model: {model}")
    print("📋 Scenarios: 3")
    
    # Set MLflow experiment
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("drift_detection_demo")
    
    # Run scenarios
    try:
        # Scenario 1: Seasonal drift (simulated)
        scenario1 = demo_scenario_1_seasonal_drift(model)
        
        # Scenario 2: Model drift (simulated)
        scenario2 = demo_scenario_2_model_drift(model)
        
        # Scenario 3: Real-time drift (actual model)
        print("\n⏳ Running real-time drift detection with actual model...")
        print("   (This will take ~2-3 minutes)")
        scenario3 = demo_scenario_3_real_time_drift(model)
        
        # Log drift results to MLflow
        print("\n📊 Logging drift analysis to MLflow...")
        with mlflow.start_run(run_name=f"DRIFT_ANALYSIS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log scenario 1
            mlflow.log_param("scenario_1", "seasonal_drift")
            if scenario1.get('drift_detected'):
                mlflow.log_metric("seasonal_drift_detected", 1)
                mlflow.log_metric("seasonal_kl_divergence", scenario1.get('kl_divergence', 0))
            
            # Log scenario 2
            mlflow.log_param("scenario_2", "model_performance_drift")
            if scenario2.get('overall_drift_detected'):
                mlflow.log_metric("model_drift_detected", 1)
                mlflow.log_metric("num_drifted_metrics", len(scenario2.get('drifted_metrics', [])))
            
            # Log scenario 3
            mlflow.log_param("scenario_3", "real_time_drift")
            if scenario3.get('overall_drift_detected'):
                mlflow.log_metric("realtime_drift_detected", 1)
            
            mlflow.set_tag("demo_type", "drift_detection")
            mlflow.set_tag("domain", "healthcare")
        
        print("\n" + "="*80)
        print("DRIFT DETECTION DEMO COMPLETED")
        print("="*80)
        print("\n✅ All drift analyses logged to MLflow")
        print("   View at: http://localhost:5000")
        print("\n💡 Key Takeaways:")
        print("   • Drift detection is critical for production AI systems")
        print("   • Both data and model drift must be monitored")
        print("   • Healthcare requires continuous quality monitoring")
        print("   • MLflow tracks all drift metrics over time")
        print("\n" + "="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
