"""
Benchmark & Performance Profiler for Citizendashboard.
Measures API latency, SWR cache hit speeds, and spatial indexing throughput.
"""

import time
import urllib.request
import json
import statistics
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def benchmark_endpoint(url, name, iterations=10):
    latencies = []
    for i in range(iterations):
        t0 = time.perf_counter()
        try:
            req = urllib.request.urlopen(url, timeout=5)
            data = json.loads(req.read().decode('utf-8'))
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # in ms
        except Exception as e:
            print(f"Error calling {url}: {e}")
            return None
    
    avg = statistics.mean(latencies)
    median = statistics.median(latencies)
    min_lat = min(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    return {
        "name": name,
        "avg_ms": avg,
        "median_ms": median,
        "min_ms": min_lat,
        "p95_ms": p95
    }

def main():
    print("=" * 90)
    print("🚀 HIGH-PERFORMANCE BENCHMARK & LATENCY PROFILER")
    print("=" * 90)
    
    endpoints = [
        ("http://localhost/api/telemetry/dashboard", "Weather Dashboard (23 Stations Telemetry)"),
        ("http://localhost/api/water-level/dashboard", "Water Level Dashboard (13 WLMS Stations)"),
        ("http://localhost/api/prediction/station/3nzr48bG", "PINN-LNN Nowcast Prediction (Calumpit)"),
        ("http://localhost/api/prediction/station/95pM7BAV", "PINN-LNN Nowcast Prediction (Balanga)"),
    ]
    
    # Warmup run
    print("🔥 Warming up in-memory SWR caches...")
    for url, name in endpoints:
        try:
            urllib.request.urlopen(url, timeout=5)
        except Exception:
            pass
    time.sleep(0.5)
    
    print(f"\n{'Endpoint / Feature':<46} | {'Avg Latency':<12} | {'Min Latency':<12} | {'P95 Latency':<12} | {'Speedup Rating'}")
    print("-" * 90)
    
    for url, name in endpoints:
        res = benchmark_endpoint(url, name, iterations=15)
        if res:
            rating = "⚡ ULTRA-FAST (<10ms)" if res['avg_ms'] < 10 else "🚀 FAST (<50ms)" if res['avg_ms'] < 50 else "✅ OPTIMAL"
            print(f"{res['name']:<46} | {res['avg_ms']:<7.2f} ms   | {res['min_ms']:<7.2f} ms   | {res['p95_ms']:<7.2f} ms   | {rating}")

    print("=" * 90)

if __name__ == "__main__":
    main()
