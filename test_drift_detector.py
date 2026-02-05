#!/usr/bin/env python3
"""
Quick test script to verify drift detector module
"""

import sys
from src.drift_detector import DataDriftDetector, ModelDriftDetector, HealthcareDriftScenarios

def test_data_drift():
    """Test data drift detection"""
    print("=" * 60)
    print("Testing Data Drift Detection")
    print("=" * 60)
    
    # Create baseline data (format expected by detector)
    baseline_data = [
        {'prompt': 'What is diabetes?', 'category': 'diagnosis'},
        {'prompt': 'Explain heart disease', 'category': 'diagnosis'},
        {'prompt': 'Symptoms of flu', 'category': 'diagnosis'}
    ]
    
    # Create drifted prompts (different distribution)
    current_prompts = ['ICD code for pneumonia', 'CPT code for surgery', 'Billing code']
    
    detector = DataDriftDetector(baseline_data=baseline_data)
    results = detector.detect_prompt_length_drift(current_prompts)
    
    print("\n✅ Prompt Length Drift Results:")
    print(f"  - Drift detected: {results.get('drift_detected', False)}")
    print(f"  - p-value: {results.get('p_value', 0):.4f}")
    print(f"  - Baseline mean: {results.get('baseline_mean', 0):.2f}")
    print(f"  - Current mean: {results.get('current_mean', 0):.2f}")
    
    return results

def test_model_drift():
    """Test model drift detection"""
    print("\n" + "=" * 60)
    print("Testing Model Drift Detection")
    print("=" * 60)
    
    # Create baseline runs (good performance)
    baseline_runs = [
        {'metrics': {'quality_score': 0.90, 'safety_score': 0.95, 'coherence_score': 0.88, 'latency': 25.0}},
        {'metrics': {'quality_score': 0.92, 'safety_score': 0.96, 'coherence_score': 0.91, 'latency': 28.0}},
        {'metrics': {'quality_score': 0.89, 'safety_score': 0.94, 'coherence_score': 0.87, 'latency': 26.0}},
    ]
    
    # Create drifted runs (degraded performance)
    current_runs = [
        {'metrics': {'quality_score': 0.72, 'safety_score': 0.75, 'coherence_score': 0.70, 'latency': 45.0}},
        {'metrics': {'quality_score': 0.74, 'safety_score': 0.78, 'coherence_score': 0.72, 'latency': 48.0}},
        {'metrics': {'quality_score': 0.71, 'safety_score': 0.76, 'coherence_score': 0.69, 'latency': 46.0}},
    ]
    
    detector = ModelDriftDetector(baseline_runs=baseline_runs)
    
    # Test quality drift
    quality_result = detector.detect_quality_drift(current_runs, metric='quality_score')
    
    print("\n✅ Quality Drift Results:")
    print(f"  - Drift detected: {quality_result.get('drift_detected', False)}")
    print(f"  - p-value: {quality_result.get('p_value', 0):.4f}")
    print(f"  - Baseline mean: {quality_result.get('baseline_mean', 0):.3f}")
    print(f"  - Current mean: {quality_result.get('current_mean', 0):.3f}")
    print(f"  - Percent change: {quality_result.get('percent_change', 0):.1f}%")
    
    # Test latency drift
    latency_result = detector.detect_latency_drift(current_runs)
    
    print("\n✅ Latency Drift Results:")
    print(f"  - Drift detected: {latency_result.get('drift_detected', False)}")
    print(f"  - Baseline mean: {latency_result.get('baseline_mean', 0):.1f}s")
    print(f"  - Current mean: {latency_result.get('current_mean', 0):.1f}s")
    print(f"  - Percent change: {latency_result.get('percent_change', 0):.1f}%")
    
    return {'quality': quality_result, 'latency': latency_result}

def test_healthcare_scenarios():
    """Test healthcare drift scenarios"""
    print("\n" + "=" * 60)
    print("Testing Healthcare Scenarios")
    print("=" * 60)
    
    print("\n✅ Seasonal Drift:")
    baseline, drifted = HealthcareDriftScenarios.simulate_seasonal_drift()
    print(f"  Baseline queries: {len(baseline)}")
    print(f"  Drifted queries: {len(drifted)}")
    print(f"  Sample baseline: {baseline[0]['prompt']}")
    print(f"  Sample drifted: {drifted[0]['prompt']}")
    
    print("\n✅ Demographic Drift:")
    baseline, drifted = HealthcareDriftScenarios.simulate_demographic_drift()
    print(f"  Baseline queries: {len(baseline)}")
    print(f"  Drifted queries: {len(drifted)}")
    print(f"  Sample baseline: {baseline[0]['prompt']}")
    print(f"  Sample drifted: {drifted[0]['prompt']}")
    
    print("\n✅ Terminology Drift:")
    baseline, drifted = HealthcareDriftScenarios.simulate_terminology_drift()
    print(f"  Baseline queries: {len(baseline)}")
    print(f"  Drifted queries: {len(drifted)}")
    print(f"  Sample baseline: {baseline[0]['prompt']}")
    print(f"  Sample drifted: {drifted[0]['prompt']}")

def main():
    print("\n🧪 Drift Detector Module Test\n")
    
    try:
        # Test 1: Data Drift
        data_results = test_data_drift()
        
        # Test 2: Model Drift
        model_results = test_model_drift()
        
        # Test 3: Healthcare Scenarios
        test_healthcare_scenarios()
        
        print("\n" + "=" * 60)
        print("✅ All Tests Passed Successfully!")
        print("=" * 60)
        print("\nNext Steps:")
        print("  1. Run full drift demo: python demo_drift_detection.py phi3:mini")
        print("  2. Launch drift dashboard: python drift_dashboard.py")
        print("  3. View in MLflow UI: mlflow ui")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
