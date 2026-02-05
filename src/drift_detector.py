"""
Data Drift and Model Drift Detection
Monitors changes in data distribution and model performance over time
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from datetime import datetime, timedelta
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Detect drift in input data distributions
    
    Uses statistical tests to identify when incoming data
    differs significantly from baseline/reference data
    """
    
    def __init__(self, baseline_data: Optional[List[Dict]] = None):
        """
        Initialize drift detector
        
        Args:
            baseline_data: Reference data to compare against
        """
        self.baseline_data = baseline_data or []
        self.drift_history = []
        
    def add_baseline_data(self, data: List[Dict]):
        """Add data to baseline"""
        self.baseline_data.extend(data)
        logger.info(f"Baseline updated: {len(self.baseline_data)} samples")
    
    def detect_prompt_length_drift(
        self,
        current_prompts: List[str],
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect drift in prompt length distribution
        
        Args:
            current_prompts: List of current prompts
            threshold: P-value threshold for drift detection
            
        Returns:
            Drift detection results
        """
        if not self.baseline_data:
            return {"error": "No baseline data available"}
        
        # Get baseline prompt lengths
        baseline_lengths = [len(d['prompt']) for d in self.baseline_data if 'prompt' in d]
        
        # Get current prompt lengths
        current_lengths = [len(p) for p in current_prompts]
        
        if not baseline_lengths or not current_lengths:
            return {"error": "Insufficient data"}
        
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(baseline_lengths, current_lengths)
        
        drift_detected = p_value < threshold
        
        result = {
            "metric": "prompt_length",
            "drift_detected": drift_detected,
            "p_value": p_value,
            "statistic": statistic,
            "threshold": threshold,
            "baseline_mean": np.mean(baseline_lengths),
            "baseline_std": np.std(baseline_lengths),
            "current_mean": np.mean(current_lengths),
            "current_std": np.std(current_lengths),
            "percent_change": ((np.mean(current_lengths) - np.mean(baseline_lengths)) / np.mean(baseline_lengths)) * 100,
            "timestamp": datetime.now().isoformat()
        }
        
        self.drift_history.append(result)
        
        if drift_detected:
            logger.warning(f"⚠️  DRIFT DETECTED in prompt length (p={p_value:.4f})")
        else:
            logger.info(f"✓ No drift in prompt length (p={p_value:.4f})")
        
        return result
    
    def detect_keyword_distribution_drift(
        self,
        current_prompts: List[str],
        keywords: List[str],
        threshold: float = 0.15
    ) -> Dict[str, Any]:
        """
        Detect drift in keyword frequency distribution
        
        Args:
            current_prompts: Current prompts
            keywords: Keywords to track (e.g., medical terms)
            threshold: Threshold for relative change
            
        Returns:
            Drift detection results
        """
        if not self.baseline_data:
            return {"error": "No baseline data available"}
        
        # Calculate keyword frequencies in baseline
        baseline_prompts = [d['prompt'] for d in self.baseline_data if 'prompt' in d]
        baseline_freqs = self._calculate_keyword_frequencies(baseline_prompts, keywords)
        
        # Calculate keyword frequencies in current data
        current_freqs = self._calculate_keyword_frequencies(current_prompts, keywords)
        
        # Compare distributions
        drifted_keywords = []
        for keyword in keywords:
            baseline_freq = baseline_freqs.get(keyword, 0)
            current_freq = current_freqs.get(keyword, 0)
            
            if baseline_freq > 0:
                relative_change = abs(current_freq - baseline_freq) / baseline_freq
                if relative_change > threshold:
                    drifted_keywords.append({
                        "keyword": keyword,
                        "baseline_freq": baseline_freq,
                        "current_freq": current_freq,
                        "relative_change": relative_change
                    })
        
        drift_detected = len(drifted_keywords) > 0
        
        result = {
            "metric": "keyword_distribution",
            "drift_detected": drift_detected,
            "drifted_keywords": drifted_keywords,
            "num_drifted": len(drifted_keywords),
            "threshold": threshold,
            "baseline_freqs": baseline_freqs,
            "current_freqs": current_freqs,
            "timestamp": datetime.now().isoformat()
        }
        
        if drift_detected:
            logger.warning(f"⚠️  DRIFT DETECTED in keyword distribution: {len(drifted_keywords)} keywords")
        else:
            logger.info("✓ No drift in keyword distribution")
        
        return result
    
    def _calculate_keyword_frequencies(self, prompts: List[str], keywords: List[str]) -> Dict[str, float]:
        """Calculate keyword frequencies in prompts"""
        if not prompts:
            return {}
        
        freqs = {}
        total_prompts = len(prompts)
        
        for keyword in keywords:
            count = sum(1 for prompt in prompts if keyword.lower() in prompt.lower())
            freqs[keyword] = count / total_prompts
        
        return freqs
    
    def detect_category_distribution_drift(
        self,
        current_categories: List[str],
        threshold: float = 0.15
    ) -> Dict[str, Any]:
        """
        Detect drift in query category distribution
        
        Args:
            current_categories: List of categories from current queries
            threshold: Threshold for Chi-square test
            
        Returns:
            Drift detection results
        """
        if not self.baseline_data:
            return {"error": "No baseline data available"}
        
        # Get baseline categories
        baseline_categories = [d.get('category', 'unknown') for d in self.baseline_data]
        
        # Calculate distributions
        baseline_dist = pd.Series(baseline_categories).value_counts(normalize=True).to_dict()
        current_dist = pd.Series(current_categories).value_counts(normalize=True).to_dict()
        
        # Calculate KL divergence (simple version)
        kl_div = 0
        for category in set(list(baseline_dist.keys()) + list(current_dist.keys())):
            p = baseline_dist.get(category, 0.001)
            q = current_dist.get(category, 0.001)
            kl_div += p * np.log(p / q)
        
        drift_detected = kl_div > threshold
        
        result = {
            "metric": "category_distribution",
            "drift_detected": drift_detected,
            "kl_divergence": kl_div,
            "threshold": threshold,
            "baseline_distribution": baseline_dist,
            "current_distribution": current_dist,
            "timestamp": datetime.now().isoformat()
        }
        
        if drift_detected:
            logger.warning(f"⚠️  DRIFT DETECTED in category distribution (KL={kl_div:.4f})")
        else:
            logger.info(f"✓ No drift in category distribution (KL={kl_div:.4f})")
        
        return result


class ModelDriftDetector:
    """
    Detect drift in model performance over time
    
    Monitors quality metrics, latency, and other performance indicators
    """
    
    def __init__(self, baseline_runs: Optional[List[Dict]] = None, window_size: int = 20):
        """
        Initialize model drift detector
        
        Args:
            baseline_runs: Reference model runs
            window_size: Number of recent runs to consider
        """
        self.baseline_runs = baseline_runs or []
        self.window_size = window_size
        self.drift_history = []
    
    def add_baseline_runs(self, runs: List[Dict]):
        """Add runs to baseline"""
        self.baseline_runs.extend(runs)
        logger.info(f"Baseline updated: {len(self.baseline_runs)} runs")
    
    def detect_quality_drift(
        self,
        current_runs: List[Dict],
        metric: str = 'quality_score',
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect drift in quality metrics
        
        Args:
            current_runs: Recent model runs
            metric: Metric to check (quality_score, safety_score, etc.)
            threshold: P-value threshold
            
        Returns:
            Drift detection results
        """
        if not self.baseline_runs:
            return {"error": "No baseline runs available"}
        
        # Extract metric values
        baseline_values = [
            run['metrics'][metric] 
            for run in self.baseline_runs 
            if 'metrics' in run and metric in run['metrics']
        ]
        
        current_values = [
            run['metrics'][metric]
            for run in current_runs
            if 'metrics' in run and metric in run['metrics']
        ]
        
        if not baseline_values or not current_values:
            return {"error": "Insufficient metric data"}
        
        # Statistical test (Mann-Whitney U test for non-parametric data)
        statistic, p_value = stats.mannwhitneyu(baseline_values, current_values, alternative='two-sided')
        
        drift_detected = p_value < threshold
        
        # Calculate performance degradation
        baseline_mean = np.mean(baseline_values)
        current_mean = np.mean(current_values)
        degradation = ((baseline_mean - current_mean) / baseline_mean) * 100
        
        result = {
            "metric": metric,
            "drift_detected": drift_detected,
            "p_value": p_value,
            "statistic": statistic,
            "threshold": threshold,
            "baseline_mean": baseline_mean,
            "baseline_std": np.std(baseline_values),
            "current_mean": current_mean,
            "current_std": np.std(current_values),
            "degradation_percent": degradation,
            "severity": self._assess_severity(degradation),
            "timestamp": datetime.now().isoformat()
        }
        
        self.drift_history.append(result)
        
        if drift_detected:
            logger.warning(f"⚠️  MODEL DRIFT DETECTED in {metric} (p={p_value:.4f}, degradation={degradation:.1f}%)")
        else:
            logger.info(f"✓ No drift in {metric} (p={p_value:.4f})")
        
        return result
    
    def detect_latency_drift(
        self,
        current_runs: List[Dict],
        threshold_percent: float = 20.0
    ) -> Dict[str, Any]:
        """
        Detect drift in response latency
        
        Args:
            current_runs: Recent model runs
            threshold_percent: Percentage increase threshold
            
        Returns:
            Drift detection results
        """
        if not self.baseline_runs:
            return {"error": "No baseline runs available"}
        
        baseline_latencies = [
            run['metrics']['latency']
            for run in self.baseline_runs
            if 'metrics' in run and 'latency' in run['metrics']
        ]
        
        current_latencies = [
            run['metrics']['latency']
            for run in current_runs
            if 'metrics' in run and 'latency' in run['metrics']
        ]
        
        if not baseline_latencies or not current_latencies:
            return {"error": "Insufficient latency data"}
        
        baseline_mean = np.mean(baseline_latencies)
        current_mean = np.mean(current_latencies)
        
        percent_increase = ((current_mean - baseline_mean) / baseline_mean) * 100
        drift_detected = percent_increase > threshold_percent
        
        result = {
            "metric": "latency",
            "drift_detected": drift_detected,
            "baseline_mean": baseline_mean,
            "current_mean": current_mean,
            "percent_increase": percent_increase,
            "threshold_percent": threshold_percent,
            "severity": self._assess_severity(percent_increase),
            "timestamp": datetime.now().isoformat()
        }
        
        if drift_detected:
            logger.warning(f"⚠️  LATENCY DRIFT DETECTED: {percent_increase:.1f}% increase")
        else:
            logger.info(f"✓ No latency drift ({percent_increase:.1f}% change)")
        
        return result
    
    def detect_comprehensive_drift(
        self,
        current_runs: List[Dict]
    ) -> Dict[str, Any]:
        """
        Run comprehensive drift detection across all metrics
        
        Args:
            current_runs: Recent model runs
            
        Returns:
            Complete drift analysis
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "num_baseline_runs": len(self.baseline_runs),
            "num_current_runs": len(current_runs),
            "drift_checks": {}
        }
        
        # Check quality drift
        quality_drift = self.detect_quality_drift(current_runs, metric='quality_score')
        results['drift_checks']['quality_score'] = quality_drift
        
        # Check safety drift
        safety_drift = self.detect_quality_drift(current_runs, metric='safety_score')
        results['drift_checks']['safety_score'] = safety_drift
        
        # Check coherence drift
        coherence_drift = self.detect_quality_drift(current_runs, metric='coherence_score')
        results['drift_checks']['coherence_score'] = coherence_drift
        
        # Check latency drift
        latency_drift = self.detect_latency_drift(current_runs)
        results['drift_checks']['latency'] = latency_drift
        
        # Overall assessment
        drift_detected = any(
            check.get('drift_detected', False) 
            for check in results['drift_checks'].values()
        )
        
        results['overall_drift_detected'] = drift_detected
        results['drifted_metrics'] = [
            metric for metric, check in results['drift_checks'].items()
            if check.get('drift_detected', False)
        ]
        
        return results
    
    def _assess_severity(self, degradation: float) -> str:
        """Assess severity of drift"""
        abs_deg = abs(degradation)
        if abs_deg < 5:
            return "low"
        elif abs_deg < 15:
            return "medium"
        else:
            return "high"


class HealthcareDriftScenarios:
    """
    Healthcare-specific drift scenarios for demonstration
    
    Simulates common drift scenarios in healthcare AI:
    - Seasonal disease patterns (flu season)
    - Demographic shifts (aging population)
    - New medical terminology (COVID-19)
    - Treatment protocol changes
    """
    
    @staticmethod
    def simulate_seasonal_drift() -> Tuple[List[Dict], List[Dict]]:
        """
        Simulate seasonal disease pattern drift
        
        Returns:
            (baseline_data, drifted_data)
        """
        # Summer baseline - general health queries
        baseline_queries = [
            {"prompt": "What are symptoms of dehydration?", "category": "general"},
            {"prompt": "How to treat sunburn?", "category": "general"},
            {"prompt": "What is heat exhaustion?", "category": "general"},
            {"prompt": "Symptoms of allergies?", "category": "general"},
            {"prompt": "How to manage diabetes?", "category": "chronic"},
        ]
        
        # Winter drift - respiratory illness queries increase
        drifted_queries = [
            {"prompt": "What are flu symptoms?", "category": "respiratory"},
            {"prompt": "How to treat common cold?", "category": "respiratory"},
            {"prompt": "When to get flu vaccine?", "category": "respiratory"},
            {"prompt": "Pneumonia symptoms?", "category": "respiratory"},
            {"prompt": "COVID-19 testing guidelines?", "category": "respiratory"},
        ]
        
        return baseline_queries, drifted_queries
    
    @staticmethod
    def simulate_demographic_drift() -> Tuple[List[Dict], List[Dict]]:
        """
        Simulate demographic shift (aging population)
        
        Returns:
            (baseline_data, drifted_data)
        """
        # Baseline - mixed age demographics
        baseline_queries = [
            {"prompt": "What are pediatric vaccination schedules?", "category": "pediatric"},
            {"prompt": "How to manage teenage acne?", "category": "pediatric"},
            {"prompt": "What are symptoms of arthritis?", "category": "geriatric"},
            {"prompt": "How to maintain fitness in 30s?", "category": "adult"},
        ]
        
        # Drifted - more geriatric queries (aging population)
        drifted_queries = [
            {"prompt": "What medications for osteoporosis?", "category": "geriatric"},
            {"prompt": "How to prevent falls in elderly?", "category": "geriatric"},
            {"prompt": "What are dementia warning signs?", "category": "geriatric"},
            {"prompt": "Managing multiple medications in seniors?", "category": "geriatric"},
        ]
        
        return baseline_queries, drifted_queries
    
    @staticmethod
    def simulate_terminology_drift() -> Tuple[List[Dict], List[Dict]]:
        """
        Simulate new medical terminology (e.g., pandemic)
        
        Returns:
            (baseline_data, drifted_data)
        """
        # Pre-pandemic baseline
        baseline_queries = [
            {"prompt": "What is influenza?", "category": "infectious"},
            {"prompt": "How does vaccination work?", "category": "preventive"},
            {"prompt": "What is pneumonia?", "category": "infectious"},
        ]
        
        # Post-pandemic drift - new terminology
        drifted_queries = [
            {"prompt": "What is long COVID?", "category": "infectious"},
            {"prompt": "mRNA vaccine effectiveness?", "category": "preventive"},
            {"prompt": "COVID-19 variant symptoms?", "category": "infectious"},
            {"prompt": "What is contact tracing?", "category": "preventive"},
        ]
        
        return baseline_queries, drifted_queries
    
    @staticmethod
    def simulate_model_performance_drift() -> Tuple[List[Dict], List[Dict]]:
        """
        Simulate model performance degradation
        
        Returns:
            (baseline_results, degraded_results)
        """
        # Baseline - good performance
        baseline_results = [
            {
                "prompt": "What is diabetes?",
                "category": "medical_knowledge",
                "metrics": {
                    "quality_score": 0.92,
                    "safety_score": 1.0,
                    "coherence_score": 0.95,
                    "latency": 25.0
                }
            },
            {
                "prompt": "Symptoms of hypertension?",
                "category": "medical_knowledge",
                "metrics": {
                    "quality_score": 0.89,
                    "safety_score": 1.0,
                    "coherence_score": 0.91,
                    "latency": 28.0
                }
            },
            {
                "prompt": "How to manage asthma?",
                "category": "clinical_reasoning",
                "metrics": {
                    "quality_score": 0.91,
                    "safety_score": 1.0,
                    "coherence_score": 0.93,
                    "latency": 30.0
                }
            },
        ]
        
        # Degraded - lower quality, higher latency
        degraded_results = [
            {
                "prompt": "What is diabetes?",
                "category": "medical_knowledge",
                "metrics": {
                    "quality_score": 0.72,  # 20% degradation
                    "safety_score": 0.95,
                    "coherence_score": 0.75,
                    "latency": 45.0  # 80% increase
                }
            },
            {
                "prompt": "Symptoms of hypertension?",
                "category": "medical_knowledge",
                "metrics": {
                    "quality_score": 0.68,
                    "safety_score": 0.92,
                    "coherence_score": 0.70,
                    "latency": 52.0
                }
            },
            {
                "prompt": "How to manage asthma?",
                "category": "clinical_reasoning",
                "metrics": {
                    "quality_score": 0.70,
                    "safety_score": 0.94,
                    "coherence_score": 0.73,
                    "latency": 55.0
                }
            },
        ]
        
        return baseline_results, degraded_results


def format_drift_report(drift_results: Dict[str, Any]) -> str:
    """Format drift detection results for display"""
    lines = []
    lines.append("\n" + "="*70)
    lines.append("DRIFT DETECTION REPORT")
    lines.append("="*70)
    
    lines.append(f"\n📅 Timestamp: {drift_results.get('timestamp', 'N/A')}")
    lines.append(f"📊 Baseline Runs: {drift_results.get('num_baseline_runs', 0)}")
    lines.append(f"📊 Current Runs: {drift_results.get('num_current_runs', 0)}")
    
    # Overall status
    if drift_results.get('overall_drift_detected'):
        lines.append(f"\n🚨 DRIFT DETECTED in: {', '.join(drift_results.get('drifted_metrics', []))}")
    else:
        lines.append(f"\n✅ NO SIGNIFICANT DRIFT DETECTED")
    
    # Detailed checks
    lines.append("\n" + "-"*70)
    lines.append("DETAILED ANALYSIS")
    lines.append("-"*70)
    
    for metric_name, check in drift_results.get('drift_checks', {}).items():
        if 'error' in check:
            continue
            
        status = "🚨 DRIFT" if check.get('drift_detected') else "✅ OK"
        lines.append(f"\n{status} {metric_name.upper()}:")
        
        if 'baseline_mean' in check:
            lines.append(f"  Baseline: {check['baseline_mean']:.3f} ± {check.get('baseline_std', 0):.3f}")
            lines.append(f"  Current:  {check['current_mean']:.3f} ± {check.get('current_std', 0):.3f}")
            
            if 'degradation_percent' in check:
                deg = check['degradation_percent']
                arrow = "📉" if deg > 0 else "📈"
                lines.append(f"  Change:   {arrow} {abs(deg):.1f}%")
            
            if 'severity' in check:
                lines.append(f"  Severity: {check['severity'].upper()}")
    
    lines.append("\n" + "="*70)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("DRIFT DETECTION DEMO")
    print("="*70)
    
    # 1. Model Performance Drift
    print("\n1. Simulating Model Performance Drift...")
    baseline_runs, degraded_runs = HealthcareDriftScenarios.simulate_model_performance_drift()
    
    detector = ModelDriftDetector(baseline_runs=baseline_runs)
    drift_results = detector.detect_comprehensive_drift(degraded_runs)
    
    print(format_drift_report(drift_results))
    
    # 2. Data Distribution Drift
    print("\n2. Simulating Seasonal Data Drift...")
    baseline_data, drifted_data = HealthcareDriftScenarios.simulate_seasonal_drift()
    
    data_detector = DataDriftDetector(baseline_data=baseline_data)
    
    # Extract prompts
    current_prompts = [d['prompt'] for d in drifted_data]
    current_categories = [d['category'] for d in drifted_data]
    
    # Detect category drift
    category_drift = data_detector.detect_category_distribution_drift(current_categories)
    
    print(f"\n📊 Category Distribution Drift:")
    print(f"  Baseline: {category_drift.get('baseline_distribution', {})}")
    print(f"  Current:  {category_drift.get('current_distribution', {})}")
    print(f"  KL Divergence: {category_drift.get('kl_divergence', 0):.4f}")
    print(f"  Drift: {'YES' if category_drift.get('drift_detected') else 'NO'}")
