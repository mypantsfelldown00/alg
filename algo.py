# Military-Grade Trading System - FINAL BUILD
# ✅ Ultra-low latency + Full Production Features
# Author: BE SKY & GPT-5

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import logging
import os
import sys
import time
import json
import requests
import warnings
import threading
import psutil
from dataclasses import dataclass
from collections import deque
from numba import jit
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from datetime import datetime

warnings.filterwarnings("ignore")
load_dotenv()

# ==================== CONFIG ====================
@dataclass
class Config:
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1/chat/completions"
    MAX_WORKERS: int = 30
    REQUEST_TIMEOUT: int = 3
    PERIOD: str = "1d"
    INTERVAL: str = "1m"
    CACHE_TTL: int = 20
    CACHE_SIZE: int = 2000
    FAILURE_THRESHOLD: int = 5
    RECOVERY_TIMEOUT: int = 60
    LOG_LEVEL: str = "INFO"

config = Config()

# ==================== LOGGER ====================
class PerfLogger:
    def __init__(self):
        self.logger = logging.getLogger("TradingSystem")
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL))
        self.metrics = {
            "latencies": deque(maxlen=10000),
            "errors": deque(maxlen=1000),
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self.lock = threading.Lock()

    def log_latency(self, duration):
        with self.lock:
            self.metrics["latencies"].append(duration)

    def log_error(self, err, ctx):
        with self.lock:
            self.metrics["errors"].append({"time": time.time(), "error": err, "context": ctx})
        self.logger.error(f"{err} | Context: {ctx}")

perf = PerfLogger()

# ==================== NUMBA INDICATORS ====================
@jit(nopython=True, fastmath=True, cache=True)
def rsi(arr, period=14):
    if len(arr) < period + 1: return 50.0
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1)+gains[i])/period
        avg_loss = (avg_loss*(period-1)+losses[i])/period
    return 100.0 if avg_loss == 0 else 100.0 - (100/(1+avg_gain/avg_loss))

@jit(nopython=True, fastmath=True, cache=True)
def macd(arr):
    if len(arr) < 26: return 0
    ema12 = np.mean(arr[-12:])
    ema26 = np.mean(arr[-26:])
    return ema12 - ema26

@jit(nopython=True, fastmath=True, cache=True)
def sma(arr, period=20):
    if len(arr) < period: return np.mean(arr)
    return np.mean(arr[-period:])

@jit(nopython=True, fastmath=True, cache=True)
def bollinger(arr, period=20, num_std=2):
    if len(arr) < period: return (0,0)
    mean = np.mean(arr[-period:])
    std = np.std(arr[-period:])
    return mean+num_std*std, mean-num_std*std

# ==================== ANALYZER ====================
class Analyzer:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

    def fetch(self, ticker, interval=None, period=None):
        try:
            df = yf.download(ticker, period=period or config.PERIOD, interval=interval or config.INTERVAL, progress=False)
            return df if df is not None and not df.empty else None
        except Exception as e:
            perf.log_error(str(e), {"ticker": ticker})
            return None

    def analyze(self, ticker, stream=False):
        start = time.perf_counter_ns()
        df = self.fetch(ticker)
        if df is None:
            return {"success": False, "error": "no_data"}

        close = df["Close"].values.astype(np.float64)
        vol = df["Volume"].values.astype(np.float64)

        # Indicators
        val_rsi = rsi(close)
        if stream: yield "progress", {"step": "RSI", "value": float(val_rsi)}

        val_macd = macd(close)
        if stream: yield "progress", {"step": "MACD", "value": float(val_macd)}

        val_sma = sma(close, 20)
        val_boll_up, val_boll_low = bollinger(close)
        if stream: yield "progress", {"step": "Bollinger", "upper": float(val_boll_up), "lower": float(val_boll_low)}

        avg_vol = np.mean(vol[-20:])
        vol_spike = vol[-1] > 2*avg_vol
        if stream: yield "progress", {"step": "Volume", "spike": bool(vol_spike)}

        # Scoring
        score = 0
        if val_rsi < 30: score += 1
        if val_rsi > 70: score -= 1
        if val_macd > 0: score += 1
        if vol_spike: score += 1
        signal = "bullish" if score > 0 else "bearish"
        if stream: yield "progress", {"step": "Scoring", "score": score, "signal": signal}

        latency = round((time.perf_counter_ns()-start)/1e6,3)
        perf.log_latency(latency)

        result = {
            "ticker": ticker,
            "price": float(close[-1]),
            "indicators": {
                "rsi": float(val_rsi),
                "macd": float(val_macd),
                "sma20": float(val_sma),
                "bollinger": {"upper": float(val_boll_up), "lower": float(val_boll_low)},
                "volume_spike": bool(vol_spike),
            },
            "scoring": {"score": score, "signal": signal},
            "performance": {"latency_ms": latency},
            "success": True
        }

        if stream: yield "final", result
        else: return result

analyzer = Analyzer()

# ==================== FLASK API ====================
app = Flask(__name__)
CORS(app)

@app.route("/")
def root():
    return jsonify({"status": "ok", "name": "Military-Grade Trading System"})

@app.route("/api/v3/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json() or {}
    ticker = data.get("ticker", "").upper()
    if not ticker:
        return jsonify({"success": False, "error": "ticker_required"}), 400
    return jsonify(analyzer.analyze(ticker))

@app.route("/api/v3/stream")
def api_stream():
    ticker = request.args.get("ticker", "").upper()
    if not ticker:
        return jsonify({"success": False, "error": "ticker_required"}), 400

    def gen():
        for ev, payload in analyzer.analyze(ticker, stream=True):
            yield f"event: {ev}\ndata: {json.dumps(payload)}\n\n"
    return Response(stream_with_context(gen()), mimetype="text/event-stream")

@app.route("/api/v3/batch", methods=["POST"])
def api_batch():
    data = request.get_json() or {}
    tickers = data.get("tickers", [])
    results = {}
    with analyzer.executor as ex:
        futures = {ex.submit(analyzer.analyze, t): t for t in tickers}
        for f in as_completed(futures):
            results[futures[f]] = f.result()
    return jsonify(results)

@app.route("/api/v3/health")
def api_health():
    return jsonify({
        "status": "ok",
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent,
        "threads": threading.active_count(),
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/api/v3/metrics")
def api_metrics():
    latencies = list(perf.metrics["latencies"])
    def pct(p): return round(np.percentile(latencies, p),3) if latencies else 0
    return jsonify({
        "count": len(latencies),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "cache_hits": perf.metrics["cache_hits"],
        "cache_misses": perf.metrics["cache_misses"]
    })

# ==================== RUN ====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
