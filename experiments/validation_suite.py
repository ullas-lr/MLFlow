"""
Validation Test Suite
Comprehensive tests for model validation
"""

import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
import logging

from src.model_client import OllamaClient
from src.metrics_collector import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationSuite:
    """Comprehensive validation test suite"""
    
    def __init__(self, client: OllamaClient):
        """
        Initialize validation suite
        
        Args:
            client: Ollama client instance
        """
        self.client = client
        self.metrics_collector = MetricsCollector()
        self.results = []
        
    def load_benchmark_queries(self, filepath: str = "experiments/benchmark_queries.json") -> Dict:
        """Load benchmark queries from JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Benchmark file not found: {filepath}")
            return {"test_suites": []}
    
    def run_category_tests(self, category: str, queries: List[Dict]) -> List[Dict[str, Any]]:
        """
        Run tests for a specific category
        
        Args:
            category: Test category name
            queries: List of query dictionaries
            
        Returns:
            List of test results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing Category: {category.upper()}")
        logger.info(f"{'='*60}")
        
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"\nTest {i}/{len(queries)}: {query['prompt'][:50]}...")
            
            # Run the query
            result = self.client.generate(
                prompt=query['prompt'],
                max_tokens=query.get('max_tokens', 256)
            )
            
            if result["success"]:
                # Calculate metrics
                metrics = self.metrics_collector.calculate_all_metrics(
                    prompt=query['prompt'],
                    response=result['response'],
                    metadata=result
                )
                
                # Check for expected keywords
                keyword_matches = self._check_keywords(
                    result['response'],
                    query.get('expected_keywords', [])
                )
                
                test_result = {
                    "category": category,
                    "prompt": query['prompt'],
                    "response": result['response'],
                    "success": True,
                    "metrics": metrics,
                    "keyword_match_rate": keyword_matches,
                    "passed": metrics['quality_score'] >= 0.5 and keyword_matches >= 0.3
                }
                
                # Log result
                status = "✅ PASS" if test_result["passed"] else "⚠️  FAIL"
                logger.info(f"{status} - Quality: {metrics['quality_score']:.3f}, "
                          f"Latency: {metrics['latency']:.3f}s, "
                          f"Keywords: {keyword_matches:.1%}")
                
            else:
                test_result = {
                    "category": category,
                    "prompt": query['prompt'],
                    "success": False,
                    "error": result.get('error'),
                    "passed": False
                }
                logger.error(f"❌ ERROR - {result.get('error')}")
            
            results.append(test_result)
        
        return results
    
    def run_full_validation(self) -> Dict[str, Any]:
        """
        Run full validation suite
        
        Returns:
            Summary of validation results
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING FULL VALIDATION SUITE")
        logger.info("="*60)
        
        # Load benchmark queries
        benchmark_data = self.load_benchmark_queries()
        test_suites = benchmark_data.get('test_suites', [])
        
        if not test_suites:
            logger.error("No test suites found!")
            return {"error": "No test suites loaded"}
        
        all_results = []
        category_summaries = {}
        
        # Run each test suite
        for suite in test_suites:
            category = suite['category']
            results = self.run_category_tests(category, suite['queries'])
            all_results.extend(results)
            
            # Calculate category summary
            passed = sum(1 for r in results if r.get('passed', False))
            total = len(results)
            avg_quality = sum(r['metrics']['quality_score'] for r in results if 'metrics' in r) / total if total > 0 else 0
            avg_latency = sum(r['metrics']['latency'] for r in results if 'metrics' in r) / total if total > 0 else 0
            
            category_summaries[category] = {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": passed / total if total > 0 else 0,
                "avg_quality_score": avg_quality,
                "avg_latency": avg_latency
            }
        
        # Overall summary
        total_tests = len(all_results)
        total_passed = sum(1 for r in all_results if r.get('passed', False))
        
        summary = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_tests - total_passed,
            "overall_pass_rate": total_passed / total_tests if total_tests > 0 else 0,
            "category_summaries": category_summaries,
            "detailed_results": all_results,
            "model": self.client.model,
        }
        
        # Print summary
        self._print_summary(summary)
        
        return summary
    
    def _check_keywords(self, response: str, expected_keywords: List[str]) -> float:
        """
        Check how many expected keywords are in the response
        
        Args:
            response: Model response
            expected_keywords: List of expected keywords
            
        Returns:
            Ratio of keywords found (0-1)
        """
        if not expected_keywords:
            return 1.0
        
        response_lower = response.lower()
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
        
        return matches / len(expected_keywords)
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print validation summary"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        print(f"\nModel: {summary['model']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['total_passed']} ({summary['overall_pass_rate']:.1%})")
        print(f"Failed: {summary['total_failed']}")
        
        print("\n📊 Category Breakdown:")
        for category, stats in summary['category_summaries'].items():
            print(f"\n  {category.upper()}:")
            print(f"    Tests: {stats['total_tests']}")
            print(f"    Pass Rate: {stats['pass_rate']:.1%}")
            print(f"    Avg Quality: {stats['avg_quality_score']:.3f}")
            print(f"    Avg Latency: {stats['avg_latency']:.3f}s")
        
        print("\n" + "="*60 + "\n")


def run_quick_validation(model: str = "llama2") -> Dict[str, Any]:
    """
    Quick validation with a subset of tests
    
    Args:
        model: Model name to test
        
    Returns:
        Validation summary
    """
    logger.info(f"Running quick validation for {model}...")
    
    client = OllamaClient(model=model)
    
    if not client.check_connection():
        return {"error": "Cannot connect to Ollama server"}
    
    # Quick tests
    quick_queries = [
        {"prompt": "What is 2+2?", "max_tokens": 50, "expected_keywords": ["4", "four"]},
        {"prompt": "What is the capital of France?", "max_tokens": 100, "expected_keywords": ["Paris"]},
        {"prompt": "Explain AI in one sentence.", "max_tokens": 100, "expected_keywords": ["artificial", "intelligence"]},
    ]
    
    suite = ValidationSuite(client)
    results = suite.run_category_tests("quick_test", quick_queries)
    
    passed = sum(1 for r in results if r.get('passed', False))
    
    return {
        "model": model,
        "total_tests": len(results),
        "passed": passed,
        "pass_rate": passed / len(results),
        "results": results
    }


if __name__ == "__main__":
    import sys
    
    model = sys.argv[1] if len(sys.argv) > 1 else "llama2"
    
    print(f"\nValidating model: {model}")
    print("Make sure Ollama is running and the model is pulled!\n")
    
    # Run full validation
    client = OllamaClient(model=model)
    suite = ValidationSuite(client)
    results = suite.run_full_validation()
    
    # Save results
    output_file = f"validation_results_{model}_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        # Remove response text for smaller file
        clean_results = {**results}
        for r in clean_results.get('detailed_results', []):
            if 'response' in r:
                r['response'] = r['response'][:200] + "..." if len(r['response']) > 200 else r['response']
        json.dump(clean_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
