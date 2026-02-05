#!/usr/bin/env python3
"""
Healthcare-Specific Validation Demo
For FDP Session: Cloud platforms, edge AI, model validation and monitoring
By: Ullas Lakku Raghavendra, Adobe
"""

import sys
import json
from src.model_client import OllamaClient
from src.experiment_runner import ExperimentRunner
from src.metrics_collector import MetricsCollector

def run_healthcare_demo(model_name="phi3:mini"):
    """
    Run healthcare-specific validation demo
    
    This demonstrates:
    1. Medical knowledge validation
    2. Clinical reasoning assessment
    3. Safety and ethics evaluation
    4. Real-time monitoring
    """
    
    print("\n" + "="*70)
    print("HEALTHCARE AI MODEL VALIDATION DEMO")
    print("Session: Cloud platforms, edge AI, model validation and monitoring")
    print("="*70 + "\n")
    
    # Initialize
    client = OllamaClient(model=model_name)
    runner = ExperimentRunner(
        client=client,
        experiment_name="healthcare_validation_demo"
    )
    
    # Load healthcare queries
    with open('experiments/healthcare_queries.json', 'r') as f:
        data = json.load(f)
    
    print(f"🏥 Testing Model: {model_name}")
    print(f"📋 Test Categories: {len(data['test_suites'])}")
    print(f"✅ Total Tests: {sum(len(suite['queries']) for suite in data['test_suites'])}\n")
    
    all_results = []
    
    # Run each test suite
    for suite in data['test_suites']:
        category = suite['category']
        queries = suite['queries']
        
        print(f"\n{'='*70}")
        print(f"Category: {category.upper().replace('_', ' ')}")
        print(f"Description: {suite['description']}")
        print(f"{'='*70}\n")
        
        for i, query in enumerate(queries, 1):
            print(f"Test {i}/{len(queries)}: {query['prompt'][:60]}...")
            
            result = runner.run_single_experiment(
                prompt=query['prompt'],
                temperature=0.7,
                max_tokens=query.get('max_tokens', 200),
                tags={
                    'category': category,
                    'domain': 'healthcare',
                    'demo': 'fdp_session'
                }
            )
            
            if result['success']:
                metrics = result['metrics']
                print(f"  ✅ Quality: {metrics['quality_score']:.3f}")
                print(f"  ⏱️  Latency: {metrics['latency']:.2f}s")
                print(f"  🔒 Safety: {metrics['safety_score']:.3f}")
                print(f"  📝 Response: {result['response'][:100]}...\n")
                
                all_results.append({
                    'category': category,
                    'prompt': query['prompt'],
                    'quality': metrics['quality_score'],
                    'safety': metrics['safety_score'],
                    'latency': metrics['latency']
                })
            else:
                print(f"  ❌ Error: {result.get('error')}\n")
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70 + "\n")
    
    if all_results:
        avg_quality = sum(r['quality'] for r in all_results) / len(all_results)
        avg_safety = sum(r['safety'] for r in all_results) / len(all_results)
        avg_latency = sum(r['latency'] for r in all_results) / len(all_results)
        
        print(f"📊 Tests Completed: {len(all_results)}")
        print(f"🎯 Average Quality Score: {avg_quality:.3f}")
        print(f"🔒 Average Safety Score: {avg_safety:.3f}")
        print(f"⏱️  Average Latency: {avg_latency:.2f}s")
        print(f"\n✅ All results tracked in MLflow!")
        print(f"   View at: http://localhost:5000")
    
    print("\n" + "="*70)
    print("DEMO COMPLETED - Ready for Q&A!")
    print("="*70 + "\n")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "phi3:mini"
    run_healthcare_demo(model)
