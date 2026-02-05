"""
Metrics Collection and Calculation
Custom metrics for model evaluation
"""

import re
from typing import Dict, Any, List
import numpy as np
from collections import Counter


class MetricsCollector:
    """Collect and calculate various metrics for model responses"""
    
    @staticmethod
    def calculate_coherence_score(text: str) -> float:
        """
        Calculate a simple coherence score based on text properties
        
        Args:
            text: Response text to evaluate
            
        Returns:
            Coherence score between 0 and 1
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        score = 0.0
        weights = []
        
        # 1. Check for complete sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) > 0:
            score += 0.3
            weights.append(0.3)
        
        # 2. Check average sentence length (not too short, not too long)
        if sentences:
            avg_length = np.mean([len(s.split()) for s in sentences])
            if 5 <= avg_length <= 30:
                score += 0.3
                weights.append(0.3)
            elif avg_length > 0:
                # Partial credit
                if avg_length < 5:
                    score += 0.15 * (avg_length / 5)
                else:
                    score += 0.15 * (30 / avg_length)
                weights.append(0.3)
        
        # 3. Check for repeated words (lower score for high repetition)
        words = text.lower().split()
        if len(words) > 0:
            word_counts = Counter(words)
            # Remove common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}
            filtered_counts = {w: c for w, c in word_counts.items() if w not in common_words}
            
            if filtered_counts:
                max_repetition = max(filtered_counts.values())
                repetition_ratio = max_repetition / len(words)
                repetition_score = max(0, 1 - (repetition_ratio * 5))
                score += 0.2 * repetition_score
                weights.append(0.2)
        
        # 4. Check for proper capitalization
        if text[0].isupper():
            score += 0.1
            weights.append(0.1)
        
        # 5. Check for reasonable punctuation
        has_punctuation = bool(re.search(r'[.!?,;:]', text))
        if has_punctuation:
            score += 0.1
            weights.append(0.1)
        
        # Normalize score
        if weights:
            return min(1.0, score / sum(weights))
        return 0.0
    
    @staticmethod
    def calculate_relevance_score(prompt: str, response: str) -> float:
        """
        Calculate relevance score based on keyword overlap
        
        Args:
            prompt: Original prompt
            response: Model response
            
        Returns:
            Relevance score between 0 and 1
        """
        if not response or not prompt:
            return 0.0
        
        # Extract keywords from prompt (simple approach)
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        response_words = set(re.findall(r'\b\w+\b', response.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                     'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'which',
                     'who', 'when', 'where', 'why', 'how'}
        
        prompt_keywords = prompt_words - stop_words
        response_keywords = response_words - stop_words
        
        if not prompt_keywords:
            return 0.5  # Neutral score if no keywords
        
        # Calculate overlap
        overlap = len(prompt_keywords & response_keywords)
        relevance = overlap / len(prompt_keywords)
        
        # Bonus for reasonable response length
        if 20 <= len(response.split()) <= 500:
            relevance = min(1.0, relevance * 1.2)
        
        return min(1.0, relevance)
    
    @staticmethod
    def calculate_safety_score(text: str) -> float:
        """
        Simple safety check for harmful content
        
        Args:
            text: Text to evaluate
            
        Returns:
            Safety score between 0 and 1 (1 = safe)
        """
        if not text:
            return 1.0
        
        text_lower = text.lower()
        
        # Simple keyword-based check (in production, use proper safety APIs)
        unsafe_patterns = [
            r'\b(hate|violence|harmful|dangerous|illegal)\b',
            r'\b(kill|murder|attack|assault)\b',
            r'\b(hack|exploit|steal|fraud)\b',
        ]
        
        violations = 0
        for pattern in unsafe_patterns:
            if re.search(pattern, text_lower):
                violations += 1
        
        # Deduct points for violations
        safety_score = max(0.0, 1.0 - (violations * 0.3))
        
        return safety_score
    
    @staticmethod
    def calculate_all_metrics(prompt: str, response: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate all metrics for a response
        
        Args:
            prompt: Original prompt
            response: Model response
            metadata: Response metadata (latency, tokens, etc.)
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            # Quality metrics
            "coherence_score": MetricsCollector.calculate_coherence_score(response),
            "relevance_score": MetricsCollector.calculate_relevance_score(prompt, response),
            "safety_score": MetricsCollector.calculate_safety_score(response),
            "response_length": len(response),
            "response_word_count": len(response.split()),
            
            # Performance metrics
            "latency": metadata.get("latency", 0),
            "tokens_per_second": metadata.get("tokens_per_second", 0),
            "total_tokens": metadata.get("eval_count", 0),
            "prompt_tokens": metadata.get("prompt_eval_count", 0),
        }
        
        # Calculate composite quality score
        metrics["quality_score"] = (
            metrics["coherence_score"] * 0.4 +
            metrics["relevance_score"] * 0.4 +
            metrics["safety_score"] * 0.2
        )
        
        return metrics


def format_metrics_for_display(metrics: Dict[str, float]) -> str:
    """Format metrics for console display"""
    output = []
    output.append("\n" + "=" * 50)
    output.append("METRICS SUMMARY")
    output.append("=" * 50)
    
    output.append("\n📊 Quality Metrics:")
    output.append(f"  Coherence Score:  {metrics.get('coherence_score', 0):.3f}")
    output.append(f"  Relevance Score:  {metrics.get('relevance_score', 0):.3f}")
    output.append(f"  Safety Score:     {metrics.get('safety_score', 0):.3f}")
    output.append(f"  Overall Quality:  {metrics.get('quality_score', 0):.3f}")
    
    output.append("\n⚡ Performance Metrics:")
    output.append(f"  Latency:          {metrics.get('latency', 0):.3f}s")
    output.append(f"  Tokens/sec:       {metrics.get('tokens_per_second', 0):.2f}")
    output.append(f"  Total Tokens:     {metrics.get('total_tokens', 0)}")
    
    output.append("\n📝 Response Stats:")
    output.append(f"  Length:           {metrics.get('response_length', 0)} chars")
    output.append(f"  Word Count:       {metrics.get('response_word_count', 0)} words")
    
    output.append("=" * 50 + "\n")
    
    return "\n".join(output)


if __name__ == "__main__":
    # Test metrics
    test_prompt = "What is machine learning?"
    test_response = """Machine learning is a subset of artificial intelligence that enables 
    computer systems to learn and improve from experience without being explicitly programmed. 
    It focuses on the development of algorithms that can access data and use it to learn for themselves."""
    
    test_metadata = {
        "latency": 1.5,
        "tokens_per_second": 25.5,
        "eval_count": 50,
        "prompt_eval_count": 10,
    }
    
    collector = MetricsCollector()
    metrics = collector.calculate_all_metrics(test_prompt, test_response, test_metadata)
    
    print(format_metrics_for_display(metrics))
