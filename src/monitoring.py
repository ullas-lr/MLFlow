"""
Monitoring and Observability Utilities
Real-time monitoring of model performance
"""

import time
import psutil
from collections import deque
from typing import Dict, Any, List, Optional
from datetime import datetime
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsMonitor:
    """Monitor and track metrics in real-time"""
    
    def __init__(self, max_history: int = 100):
        """
        Initialize metrics monitor
        
        Args:
            max_history: Maximum number of entries to keep in history
        """
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        # Aggregated metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0
        self.total_tokens = 0
        
    def record_request(self, result: Dict[str, Any]):
        """
        Record a new request result
        
        Args:
            result: Result dictionary from experiment
        """
        with self._lock:
            timestamp = datetime.now()
            
            entry = {
                "timestamp": timestamp,
                "success": result.get("success", False),
                "latency": result.get("latency", 0),
                "model": result.get("model", "unknown"),
                "tokens": result.get("eval_count", 0),
                "tokens_per_second": result.get("tokens_per_second", 0),
            }
            
            # Add metrics if available
            if "metrics" in result:
                entry.update({
                    "quality_score": result["metrics"].get("quality_score", 0),
                    "coherence_score": result["metrics"].get("coherence_score", 0),
                    "relevance_score": result["metrics"].get("relevance_score", 0),
                })
            
            self.history.append(entry)
            
            # Update aggregated metrics
            self.total_requests += 1
            if entry["success"]:
                self.successful_requests += 1
                self.total_latency += entry["latency"]
                self.total_tokens += entry["tokens"]
            else:
                self.failed_requests += 1
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self._lock:
            if self.total_requests == 0:
                return {
                    "total_requests": 0,
                    "message": "No requests recorded yet"
                }
            
            uptime = time.time() - self.start_time
            success_rate = self.successful_requests / self.total_requests
            
            stats = {
                "uptime_seconds": uptime,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "error_rate": 1 - success_rate,
            }
            
            # Calculate averages
            if self.successful_requests > 0:
                stats["avg_latency"] = self.total_latency / self.successful_requests
                stats["avg_tokens"] = self.total_tokens / self.successful_requests
                stats["requests_per_minute"] = (self.successful_requests / uptime) * 60
            
            # Recent metrics (last 10 requests)
            recent = list(self.history)[-10:]
            if recent:
                recent_successful = [r for r in recent if r["success"]]
                if recent_successful:
                    stats["recent_avg_latency"] = sum(r["latency"] for r in recent_successful) / len(recent_successful)
                    stats["recent_avg_tokens_per_sec"] = sum(r["tokens_per_second"] for r in recent_successful) / len(recent_successful)
                    
                    if "quality_score" in recent_successful[0]:
                        stats["recent_avg_quality"] = sum(r["quality_score"] for r in recent_successful) / len(recent_successful)
            
            return stats
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get history of requests
        
        Args:
            limit: Number of recent entries to return (None for all)
            
        Returns:
            List of request entries
        """
        with self._lock:
            history_list = list(self.history)
            if limit:
                return history_list[-limit:]
            return history_list
    
    def clear_history(self):
        """Clear all history"""
        with self._lock:
            self.history.clear()
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.total_latency = 0.0
            self.total_tokens = 0
            self.start_time = time.time()
            logger.info("Monitor history cleared")


class SystemMonitor:
    """Monitor system resources"""
    
    @staticmethod
    def get_system_stats() -> Dict[str, Any]:
        """Get current system statistics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024 ** 3),
            "memory_total_gb": psutil.virtual_memory().total / (1024 ** 3),
            "disk_percent": psutil.disk_usage('/').percent,
        }
    
    @staticmethod
    def get_process_stats() -> Dict[str, Any]:
        """Get current process statistics"""
        process = psutil.Process()
        
        return {
            "process_cpu_percent": process.cpu_percent(interval=0.1),
            "process_memory_mb": process.memory_info().rss / (1024 ** 2),
            "process_threads": process.num_threads(),
        }


class AlertManager:
    """Manage alerts for monitoring thresholds"""
    
    def __init__(
        self,
        latency_threshold: float = 10.0,
        error_rate_threshold: float = 0.1,
        quality_threshold: float = 0.5
    ):
        """
        Initialize alert manager
        
        Args:
            latency_threshold: Alert if latency exceeds this (seconds)
            error_rate_threshold: Alert if error rate exceeds this (0-1)
            quality_threshold: Alert if quality score below this (0-1)
        """
        self.latency_threshold = latency_threshold
        self.error_rate_threshold = error_rate_threshold
        self.quality_threshold = quality_threshold
        
        self.alerts = []
        self._lock = threading.Lock()
    
    def check_alerts(self, stats: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Check if any alerts should be triggered
        
        Args:
            stats: Current statistics dictionary
            
        Returns:
            List of alert dictionaries
        """
        new_alerts = []
        
        # Check latency
        if stats.get("recent_avg_latency", 0) > self.latency_threshold:
            new_alerts.append({
                "type": "high_latency",
                "severity": "warning",
                "message": f"Average latency ({stats['recent_avg_latency']:.2f}s) exceeds threshold ({self.latency_threshold}s)",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check error rate
        if stats.get("error_rate", 0) > self.error_rate_threshold:
            new_alerts.append({
                "type": "high_error_rate",
                "severity": "critical",
                "message": f"Error rate ({stats['error_rate']:.2%}) exceeds threshold ({self.error_rate_threshold:.2%})",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check quality
        if "recent_avg_quality" in stats and stats["recent_avg_quality"] < self.quality_threshold:
            new_alerts.append({
                "type": "low_quality",
                "severity": "warning",
                "message": f"Quality score ({stats['recent_avg_quality']:.2f}) below threshold ({self.quality_threshold})",
                "timestamp": datetime.now().isoformat()
            })
        
        with self._lock:
            self.alerts.extend(new_alerts)
            # Keep only last 100 alerts
            self.alerts = self.alerts[-100:]
        
        return new_alerts
    
    def get_active_alerts(self, minutes: int = 5) -> List[Dict[str, str]]:
        """Get alerts from the last N minutes"""
        cutoff = datetime.now().timestamp() - (minutes * 60)
        
        with self._lock:
            return [
                alert for alert in self.alerts
                if datetime.fromisoformat(alert["timestamp"]).timestamp() > cutoff
            ]


def format_monitoring_summary(stats: Dict[str, Any], system_stats: Dict[str, Any]) -> str:
    """Format monitoring summary for console display"""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("MONITORING DASHBOARD")
    lines.append("=" * 60)
    
    # Request stats
    lines.append("\n📊 Request Statistics:")
    lines.append(f"  Total Requests:     {stats.get('total_requests', 0)}")
    lines.append(f"  Successful:         {stats.get('successful_requests', 0)}")
    lines.append(f"  Failed:             {stats.get('failed_requests', 0)}")
    lines.append(f"  Success Rate:       {stats.get('success_rate', 0):.1%}")
    
    # Performance stats
    if "avg_latency" in stats:
        lines.append("\n⚡ Performance:")
        lines.append(f"  Avg Latency:        {stats.get('avg_latency', 0):.3f}s")
        lines.append(f"  Recent Latency:     {stats.get('recent_avg_latency', 0):.3f}s")
        lines.append(f"  Avg Tokens/sec:     {stats.get('recent_avg_tokens_per_sec', 0):.2f}")
        lines.append(f"  Requests/min:       {stats.get('requests_per_minute', 0):.2f}")
    
    # Quality stats
    if "recent_avg_quality" in stats:
        lines.append("\n🎯 Quality:")
        lines.append(f"  Recent Quality:     {stats.get('recent_avg_quality', 0):.3f}")
    
    # System stats
    lines.append("\n💻 System Resources:")
    lines.append(f"  CPU Usage:          {system_stats.get('cpu_percent', 0):.1f}%")
    lines.append(f"  Memory Usage:       {system_stats.get('memory_percent', 0):.1f}%")
    lines.append(f"  Memory Available:   {system_stats.get('memory_available_gb', 0):.2f} GB")
    
    lines.append("\n" + "=" * 60 + "\n")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test monitoring
    monitor = MetricsMonitor()
    alert_manager = AlertManager()
    
    # Simulate some requests
    import random
    
    for i in range(10):
        result = {
            "success": random.random() > 0.1,
            "latency": random.uniform(0.5, 3.0),
            "model": "llama2",
            "eval_count": random.randint(20, 100),
            "tokens_per_second": random.uniform(10, 30),
            "metrics": {
                "quality_score": random.uniform(0.6, 0.95),
                "coherence_score": random.uniform(0.6, 0.95),
                "relevance_score": random.uniform(0.6, 0.95),
            }
        }
        monitor.record_request(result)
        time.sleep(0.1)
    
    # Get stats
    stats = monitor.get_current_stats()
    system_stats = SystemMonitor.get_system_stats()
    
    print(format_monitoring_summary(stats, system_stats))
    
    # Check alerts
    alerts = alert_manager.check_alerts(stats)
    if alerts:
        print("🚨 Active Alerts:")
        for alert in alerts:
            print(f"  [{alert['severity'].upper()}] {alert['message']}")
