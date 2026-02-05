"""
Experiment Runner
Orchestrates experiments and logs to MLflow
"""

import mlflow
import mlflow.pyfunc
from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime
import os

from src.model_client import OllamaClient
from src.metrics_collector import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrate and track model experiments"""
    
    def __init__(
        self,
        client: OllamaClient,
        experiment_name: str = "ollama_experiments",
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize experiment runner
        
        Args:
            client: Ollama client instance
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI (None for local)
        """
        self.client = client
        self.metrics_collector = MetricsCollector()
        
        # Set up MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("./mlruns")
        
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        
        logger.info(f"Experiment runner initialized: {experiment_name}")
    
    def run_single_experiment(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
        log_to_mlflow: bool = True
    ) -> Dict[str, Any]:
        """
        Run a single experiment
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            tags: Additional tags for MLflow
            log_to_mlflow: Whether to log to MLflow
            
        Returns:
            Dictionary with results and metrics
        """
        logger.info(f"Running experiment: {prompt[:50]}...")
        
        # Generate response
        result = self.client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if not result["success"]:
            logger.error(f"Experiment failed: {result.get('error', 'Unknown error')}")
            return result
        
        # Calculate metrics
        metrics = self.metrics_collector.calculate_all_metrics(
            prompt=prompt,
            response=result["response"],
            metadata=result
        )
        
        # Add to result
        result["metrics"] = metrics
        
        # Log to MLflow
        if log_to_mlflow:
            self._log_to_mlflow(
                prompt=prompt,
                result=result,
                metrics=metrics,
                temperature=temperature,
                max_tokens=max_tokens,
                tags=tags
            )
        
        return result
    
    def run_batch_experiments(
        self,
        prompts: List[str],
        temperatures: List[float] = [0.7],
        max_tokens: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run multiple experiments
        
        Args:
            prompts: List of prompts
            temperatures: List of temperatures to try
            max_tokens: Max tokens to generate
            tags: Additional tags for MLflow
            
        Returns:
            List of results
        """
        results = []
        
        total = len(prompts) * len(temperatures)
        logger.info(f"Running {total} experiments...")
        
        for i, prompt in enumerate(prompts):
            for temp in temperatures:
                logger.info(f"Experiment {len(results) + 1}/{total}")
                
                result = self.run_single_experiment(
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=max_tokens,
                    tags=tags
                )
                
                results.append(result)
        
        logger.info(f"Completed {len(results)} experiments")
        return results
    
    def _log_to_mlflow(
        self,
        prompt: str,
        result: Dict[str, Any],
        metrics: Dict[str, float],
        temperature: float,
        max_tokens: Optional[int],
        tags: Optional[Dict[str, str]]
    ):
        """Log experiment to MLflow"""
        
        # Create descriptive run name
        category = tags.get('category', 'general') if tags else 'general'
        # Use first few words of prompt for run name
        prompt_words = ' '.join(prompt.split()[:5])
        run_name = f"{category}_{prompt_words}"
        # Clean up the name (remove special characters)
        run_name = run_name.replace('?', '').replace(':', '').replace(',', '')
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("model", self.client.model)
            mlflow.log_param("temperature", temperature)
            mlflow.log_param("max_tokens", max_tokens or "default")
            mlflow.log_param("prompt_length", len(prompt))
            
            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            
            # Set default tags
            mlflow.set_tag("framework", "ollama")
            mlflow.set_tag("success", str(result["success"]))
            
            # Log artifacts
            # Save prompt and response as artifacts
            with open("prompt.txt", "w") as f:
                f.write(prompt)
            mlflow.log_artifact("prompt.txt")
            os.remove("prompt.txt")
            
            with open("response.txt", "w") as f:
                f.write(result.get("response", ""))
            mlflow.log_artifact("response.txt")
            os.remove("response.txt")
            
            # Save full result as JSON
            with open("result.json", "w") as f:
                # Remove non-serializable items
                serializable_result = {
                    k: v for k, v in result.items() 
                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }
                json.dump(serializable_result, f, indent=2)
            mlflow.log_artifact("result.json")
            os.remove("result.json")
            
            logger.info(f"Logged to MLflow: run_id={mlflow.active_run().info.run_id}")
    
    def compare_models(
        self,
        models: List[str],
        prompts: List[str],
        temperature: float = 0.7
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compare multiple models on the same prompts
        
        Args:
            models: List of model names
            prompts: List of prompts to test
            temperature: Temperature to use
            
        Returns:
            Dictionary mapping model names to results
        """
        comparison = {}
        
        for model_name in models:
            logger.info(f"\nTesting model: {model_name}")
            
            # Create client for this model
            client = OllamaClient(model=model_name, host=self.client.host)
            
            # Check if model is available
            available_models = client.list_models()
            if model_name not in available_models:
                logger.warning(f"Model {model_name} not found. Skipping...")
                logger.info(f"Available models: {available_models}")
                continue
            
            # Run experiments
            runner = ExperimentRunner(
                client=client,
                experiment_name=f"{self.experiment_name}_comparison"
            )
            
            results = []
            for prompt in prompts:
                result = runner.run_single_experiment(
                    prompt=prompt,
                    temperature=temperature,
                    tags={"comparison": "true", "model_name": model_name}
                )
                results.append(result)
            
            comparison[model_name] = results
        
        return comparison
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of current experiment"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if not experiment:
            return {"error": "Experiment not found"}
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=100
        )
        
        if len(runs) == 0:
            return {"total_runs": 0, "message": "No runs found"}
        
        summary = {
            "experiment_name": self.experiment_name,
            "experiment_id": experiment.experiment_id,
            "total_runs": len(runs),
            "metrics_summary": {
                "avg_latency": runs["metrics.latency"].mean() if "metrics.latency" in runs else None,
                "avg_quality_score": runs["metrics.quality_score"].mean() if "metrics.quality_score" in runs else None,
                "avg_tokens_per_second": runs["metrics.tokens_per_second"].mean() if "metrics.tokens_per_second" in runs else None,
            },
            "models_tested": runs["params.model"].unique().tolist() if "params.model" in runs else [],
        }
        
        return summary


if __name__ == "__main__":
    # Test experiment runner
    client = OllamaClient()
    
    if not client.check_connection():
        print("❌ Ollama is not running. Start it with: ollama serve")
        exit(1)
    
    runner = ExperimentRunner(client)
    
    # Run a simple experiment
    result = runner.run_single_experiment(
        prompt="What is the capital of France?",
        temperature=0.7
    )
    
    if result["success"]:
        print("\n✅ Experiment completed successfully!")
        print(f"Response: {result['response'][:200]}...")
        print(f"\nMetrics: {json.dumps(result['metrics'], indent=2)}")
    else:
        print(f"\n❌ Experiment failed: {result.get('error')}")
