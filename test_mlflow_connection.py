#!/usr/bin/env python3
"""
Quick test to verify MLflow connection for dashboard
"""

import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

def test_mlflow_connection():
    """Test if we can connect to MLflow and find experiments"""
    print("\n🧪 Testing MLflow Connection for Dashboard\n")
    print("="*60)
    
    try:
        # Connect to MLflow
        client = MlflowClient()
        print("✅ Connected to MLflow successfully")
        
        # List all experiments
        experiments = client.search_experiments()
        print(f"\n📊 Found {len(experiments)} experiments:\n")
        
        drift_experiments = []
        for exp in experiments:
            status = "✓" if exp.lifecycle_stage == "active" else "✗"
            print(f"  {status} {exp.name} (ID: {exp.experiment_id})")
            
            # Check if it's a drift-related experiment
            if 'drift' in exp.name.lower() or 'healthcare' in exp.name.lower():
                drift_experiments.append(exp)
        
        # Analyze drift experiments
        if drift_experiments:
            print(f"\n🎯 Found {len(drift_experiments)} drift-related experiments:\n")
            
            total_runs = 0
            for exp in drift_experiments:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=100
                )
                
                print(f"  📁 {exp.name}")
                print(f"     Runs: {len(runs)}")
                
                if len(runs) > 0:
                    # Show sample metrics
                    sample_run = runs[0]
                    metrics = sample_run.data.metrics
                    print(f"     Sample metrics: {', '.join(metrics.keys())}")
                    
                    # Show date range
                    if len(runs) > 1:
                        first_date = datetime.fromtimestamp(runs[-1].info.start_time / 1000.0)
                        last_date = datetime.fromtimestamp(runs[0].info.start_time / 1000.0)
                        print(f"     Date range: {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
                
                total_runs += len(runs)
                print()
            
            print("="*60)
            print(f"✅ Dashboard can load {total_runs} runs from MLflow")
            print("="*60)
            print("\n💡 Next steps:")
            print("   1. Run: python drift_dashboard.py")
            print("   2. Open: http://localhost:8051")
            print("   3. Look for: 'Data Source: MLflow (X runs)'")
            
        else:
            print("\n⚠️  No drift-related experiments found")
            print("\n💡 To create drift data:")
            print("   1. Run: python demo_drift_detection.py phi3:mini")
            print("   2. Then run: python drift_dashboard.py")
            print("   3. Dashboard will show real data!")
        
        return len(drift_experiments) > 0
        
    except Exception as e:
        print(f"❌ Error connecting to MLflow: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Check if mlruns directory exists")
        print("   - Try running an experiment first")
        return False

if __name__ == "__main__":
    success = test_mlflow_connection()
    print()
    exit(0 if success else 1)
