#!/usr/bin/env python3
"""
Main Experiment Runner
Run model validation experiments with MLflow tracking
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv
import mlflow

from src.model_client import OllamaClient
from src.experiment_runner import ExperimentRunner
from src.monitoring import MetricsMonitor, SystemMonitor, format_monitoring_summary
from experiments.validation_suite import ValidationSuite

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_mlflow(use_dagshub: bool = False):
    """Configure MLflow tracking"""
    if use_dagshub:
        # Load DagsHub credentials
        dagshub_user = os.getenv('DAGSHUB_USERNAME')
        dagshub_token = os.getenv('DAGSHUB_TOKEN')
        dagshub_repo = os.getenv('DAGSHUB_REPO')
        
        if dagshub_user and dagshub_token and dagshub_repo:
            tracking_uri = f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_user
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"✓ Using DagsHub: {tracking_uri}")
            return tracking_uri
        else:
            logger.warning("DagsHub credentials not found in .env, using local MLflow")
    
    # Use local MLflow
    tracking_uri = "./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"✓ Using local MLflow: {tracking_uri}")
    return tracking_uri


def run_validation_experiments(
    model: str = "llama2",
    use_dagshub: bool = False,
    experiment_name: str = "model_validation"
):
    """
    Run full validation suite with MLflow tracking
    
    Args:
        model: Model name to test
        use_dagshub: Whether to use DagsHub for tracking
        experiment_name: Name for the MLflow experiment
    """
    logger.info("="*60)
    logger.info("MODEL VALIDATION AND MONITORING")
    logger.info("="*60)
    
    # Setup
    tracking_uri = setup_mlflow(use_dagshub)
    
    # Initialize clients
    logger.info(f"\n📦 Initializing Ollama client for model: {model}")
    client = OllamaClient(model=model)
    
    # Check connection
    if not client.check_connection():
        logger.error("❌ Cannot connect to Ollama server!")
        logger.error("\nPlease ensure Ollama is running:")
        logger.error("  1. Start Ollama: ollama serve")
        logger.error(f"  2. Pull model: ollama pull {model}")
        sys.exit(1)
    
    logger.info("✓ Connected to Ollama")
    
    # Check if model is available
    available_models = client.list_models()
    logger.info(f"\n📚 Available models: {', '.join(available_models)}")
    
    if model not in available_models:
        logger.error(f"\n❌ Model '{model}' not found!")
        logger.error(f"   Available models: {', '.join(available_models)}")
        logger.error(f"   Pull it with: ollama pull {model}")
        sys.exit(1)
    
    # Initialize monitoring
    monitor = MetricsMonitor()
    
    # Initialize experiment runner
    logger.info(f"\n🔬 Setting up experiment: {experiment_name}")
    runner = ExperimentRunner(
        client=client,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri
    )
    
    # Run validation suite
    logger.info("\n🚀 Starting validation suite...\n")
    suite = ValidationSuite(client)
    results = suite.run_full_validation()
    
    # Record all results in monitoring
    for result in results.get('detailed_results', []):
        if result.get('success'):
            monitor.record_request({
                **result,
                "latency": result['metrics']['latency'],
                "eval_count": result['metrics']['total_tokens'],
                "tokens_per_second": result['metrics']['tokens_per_second'],
            })
    
    # Log summary to MLflow
    logger.info("\n📊 Logging summary to MLflow...")
    with mlflow.start_run(run_name=f"validation_suite_{model}"):
        # Log parameters
        mlflow.log_param("model", model)
        mlflow.log_param("validation_type", "full_suite")
        mlflow.log_param("total_tests", results['total_tests'])
        
        # Log metrics
        mlflow.log_metric("overall_pass_rate", results['overall_pass_rate'])
        mlflow.log_metric("total_passed", results['total_passed'])
        mlflow.log_metric("total_failed", results['total_failed'])
        
        # Log category metrics
        for category, stats in results['category_summaries'].items():
            mlflow.log_metric(f"{category}_pass_rate", stats['pass_rate'])
            mlflow.log_metric(f"{category}_avg_quality", stats['avg_quality_score'])
            mlflow.log_metric(f"{category}_avg_latency", stats['avg_latency'])
        
        # Save results as artifact
        import json
        results_file = "validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        mlflow.log_artifact(results_file)
        os.remove(results_file)
        
        mlflow.set_tag("framework", "ollama")
        mlflow.set_tag("validation", "true")
    
    # Display monitoring summary
    logger.info("\n📈 Monitoring Summary:")
    stats = monitor.get_current_stats()
    system_stats = SystemMonitor.get_system_stats()
    print(format_monitoring_summary(stats, system_stats))
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"\n✓ Model: {model}")
    logger.info(f"✓ Tests Run: {results['total_tests']}")
    logger.info(f"✓ Pass Rate: {results['overall_pass_rate']:.1%}")
    logger.info(f"✓ Tracking: {tracking_uri}")
    
    logger.info("\n📊 View Results:")
    if use_dagshub and 'dagshub.com' in tracking_uri:
        logger.info(f"   DagsHub: {tracking_uri}")
    else:
        logger.info("   Local MLflow UI: http://localhost:5000")
        logger.info("   Start UI with: mlflow ui")
    
    logger.info("\n" + "="*60 + "\n")


def run_quick_test(model: str = "llama2"):
    """Run a quick test to verify setup"""
    logger.info("\n🔧 Running quick setup test...\n")
    
    client = OllamaClient(model=model)
    
    # Check connection
    logger.info("1. Checking Ollama connection...")
    if not client.check_connection():
        logger.error("   ❌ Cannot connect to Ollama")
        return False
    logger.info("   ✓ Connected")
    
    # Check models
    logger.info("2. Checking available models...")
    models = client.list_models()
    if not models:
        logger.error("   ❌ No models found")
        return False
    logger.info(f"   ✓ Found: {', '.join(models)}")
    
    # Check target model
    logger.info(f"3. Checking model '{model}'...")
    if model not in models:
        logger.error(f"   ❌ Model not found. Pull it with: ollama pull {model}")
        return False
    logger.info("   ✓ Model available")
    
    # Test generation
    logger.info("4. Testing generation...")
    result = client.generate("Hello!", max_tokens=50)
    if not result["success"]:
        logger.error(f"   ❌ Generation failed: {result.get('error')}")
        return False
    logger.info(f"   ✓ Response: {result['response'][:50]}...")
    
    # Test MLflow
    logger.info("5. Testing MLflow...")
    try:
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("test_experiment")
        with mlflow.start_run(run_name="test"):
            mlflow.log_param("test", "value")
        logger.info("   ✓ MLflow working")
    except Exception as e:
        logger.error(f"   ❌ MLflow error: {e}")
        return False
    
    logger.info("\n✅ All checks passed! Ready to run experiments.\n")
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run model validation experiments with MLflow tracking"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="llama2",
        help="Ollama model to test (default: llama2)"
    )
    
    parser.add_argument(
        "--dagshub",
        action="store_true",
        help="Use DagsHub for remote tracking (requires .env configuration)"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="model_validation",
        help="MLflow experiment name"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick setup test only"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Run quick test if requested
    if args.quick_test:
        success = run_quick_test(args.model)
        sys.exit(0 if success else 1)
    
    # Run full validation
    try:
        run_validation_experiments(
            model=args.model,
            use_dagshub=args.dagshub,
            experiment_name=args.experiment_name
        )
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
