"""
Bot Scalping v20.0.1 — REVERSED LOGIC (CONTRARIAN)
====================================================
- Entry direction inverted (LONG <-> SHORT)
- HardSL becomes TP, ExtremeTP becomes SL
- Original loss becomes profit and vice versa
"""

import os
import time
import math
import threading
import queue
import json
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

from dotenv import load_dotenv
from binance.client import Client
import ta

load_dotenv()
client = Client(os.getenv("API_KEY"), os.getenv("API_SECRET"))
client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

LEVERAGE = 20
ORDER_USDT = 2.0
MAX_POSITIONS = 3

# Risk Management (original multipliers, will be swapped in live_open)
ATR_MULT_SL = 1.2      # original SL distance multiplier
ATR_MULT_TP = 2.5      # original TP distance multiplier
MIN_RR_RATIO = 1.8
MAX_SL_PCT = 0.015
MAX_TP_PCT = 0.04

# ═══════════════════════════════════════════════════════════════════════════
#  SYMBOLS (sama seperti original)
# ═══════════════════════════════════════════════════════════════════════════
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT",
    "LINKUSDT", "MATICUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT",
    "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
    "SUIUSDT", "SEIUSDT", "FETUSDT", "WLDUSDT", "AAVEUSDT",
    "ORDIUSDT", "TONUSDT", "1000PEPEUSDT", "WIFUSDT", "JUPUSDT",
    "FTMUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT", "APEUSDT",
    "CRVUSDT", "1000SHIBUSDT", "COMPUSDT", "MKRUSDT", "SNXUSDT",
]
SYMBOLS = list(dict.fromkeys(SYMBOLS))
# Scanning
SCAN_INTERVAL = 2.0
MONITOR_INT = 1.0
BATCH_SIZE = 15
MAX_WORKERS = 5
SLOT_FILL_INT = 0.01

# Scoring & Filter
MIN_SCORE = 55
MIN_GAP = 10
SLIPPAGE_GUARD = 0.0015
TTL_5M = 2

# Kill Switch
DAILY_LOSS = -20.0
CONSEC_MAX = 15
CONSEC_PAUSE = 10

# Learning
LEARNING_WINDOW = 200
MIN_TRADES_FOR_WEIGHT = 20

# ═══════════════════════════════════════════════════════════════════════════
#  MARKET REGIME DETECTION (tidak berubah)
# ═══════════════════════════════════════════════════════════════════════════

class MarketRegime:
    REGIME_TRENDING_BULL = "TRENDING_BULL"
    REGIME_TRENDING_BEAR = "TRENDING_BEAR"
    REGIME_RANGE = "RANGE"
    REGIME_VOLATILE = "VOLATILE"
    REGIME_EXHAUSTION = "EXHAUSTION"
    
    @staticmethod
    def detect(df: pd.DataFrame) -> Tuple[str, float, float]:
        if df is None or len(df) < 55:
            return MarketRegime.REGIME_RANGE, 0, 0
        row = df.iloc[-2]
        prev = df.iloc[-3]
        close = row["close"]
        e5, e9, e21, e50 = row["e5"], row["e9"], row["e21"], row["e50"]
        atr = row["atr"]
        atr_prev = prev["atr"]
        adx = row["adx"]
        bull_stack = close > e5 > e9 > e21 > e50
        bear_stack = close < e5 < e9 < e21 < e50
        mild_bull = close > e9 > e21
        mild_bear = close < e9 < e21
        strong_trend = adx > 25
        very_strong_trend = adx > 35
        atr_expand = (atr / atr_prev) > 1.2 if atr_prev > 0 else False
        atr_collapse = (atr / atr_prev) < 0.8 if atr_prev > 0 else False
        m5 = row["m5"]
        m5_prev = prev["m5"]
        decelerating = (abs(m5) < abs(m5_prev)) if not np.isnan(m5_prev) else False
        
        if very_strong_trend and bull_stack:
            regime = MarketRegime.REGIME_TRENDING_BULL
            strength = min(adx, 100)
            bias = 1.0
        elif very_strong_trend and bear_stack:
            regime = MarketRegime.REGIME_TRENDING_BEAR
            strength = min(adx, 100)
            bias = -1.0
        elif strong_trend and (bull_stack or mild_bull):
            regime = MarketRegime.REGIME_TRENDING_BULL
            strength = min(adx, 80)
            bias = 0.7
        elif strong_trend and (bear_stack or mild_bear):
            regime = MarketRegime.REGIME_TRENDING_BEAR
            strength = min(adx, 80)
            bias = -0.7
        elif atr_expand and adx < 20:
            regime = MarketRegime.REGIME_VOLATILE
            strength = 50
            bias = 0
        elif (atr_collapse and decelerating) or (adx > 20 and adx < 35 and decelerating):
            regime = MarketRegime.REGIME_EXHAUSTION
            strength = 40
            bias = 1 if m5 > 0 else -1
        else:
            regime = MarketRegime.REGIME_RANGE
            strength = 30
            bias = 0
        return regime, strength, bias


# ═══════════════════════════════════════════════════════════════════════════
#  EXHAUSTION CONFIRMATION LAYER (tidak berubah)
# ═══════════════════════════════════════════════════════════════════════════

class ExhaustionConfirmation:
    @staticmethod
    def check_short_exhaustion(df: pd.DataFrame) -> Tuple[bool, int, List[str]]:
        if df is None or len(df) < 55:
            return False, 0, []
        row = df.iloc[-2]
        prev = df.iloc[-3]
        prev2 = df.iloc[-4]
        conditions = []
        reasons = []
        # 1. RSI > 75
        if row["rsi"] > 75:
            conditions.append(True)
            reasons.append(f"RSI_{row['rsi']:.0f}>75")
        else:
            conditions.append(False)
        # 2. RSI Divergence
        high_price = max(df["high"].iloc[-10:])
        high_rsi = max(df["rsi"].iloc[-10:])
        if row["close"] >= high_price * 0.99 and row["rsi"] < high_rsi - 3:
            conditions.append(True)
            reasons.append("RSI_Div")
        else:
            conditions.append(False)
        # 3. MACD Divergence
        high_macd = max(df["mh"].iloc[-10:])
        if row["close"] >= high_price * 0.99 and row["mh"] < high_macd - 0.5*row["atr"]:
            conditions.append(True)
            reasons.append("MACD_Div")
        else:
            conditions.append(False)
        # 4. Volume Climax
        if row["vr"] > 2.0:
            conditions.append(True)
            reasons.append(f"VolClimax_{row['vr']:.1f}x")
        else:
            conditions.append(False)
        # 5. Delta Volume Climax
        vol_ratio = row["vr"]
        vol_prev = prev["vr"] if not np.isnan(prev["vr"]) else 1
        if vol_ratio > 1.8 and vol_ratio > vol_prev * 1.2:
            conditions.append(True)
            reasons.append("DeltaVolClimax")
        else:
            conditions.append(False)
        # 6. Long Upper Wick
        body = abs(row["close"] - row["open"])
        upper_wick = row["high"] - max(row["close"], row["open"])
        if upper_wick > body * 1.5 and upper_wick > row["atr"] * 0.3:
            conditions.append(True)
            reasons.append("LongUpperWick")
        else:
            conditions.append(False)
        # 7. ATR Expansion lalu Collapse
        atr_series = df["atr"].iloc[-10:]
        atr_peak = atr_series.max()
        atr_now = row["atr"]
        if atr_peak > atr_series.iloc[-5] * 1.3 and atr_now < atr_peak * 0.8:
            conditions.append(True)
            reasons.append("ATR_ExpCollapse")
        else:
            conditions.append(False)
        # 8. Momentum Deceleration
        m5 = row["m5"]
        m5_prev = prev["m5"]
        if m5 > 0.002 and m5 < m5_prev * 0.7:
            conditions.append(True)
            reasons.append("MomDecel")
        else:
            conditions.append(False)
        # 9. Orderflow Reversal
        br = row["br"]
        br_peak = max(df["br"].iloc[-10:])
        if br < br_peak - 0.1 and br_peak > 0.6:
            conditions.append(True)
            reasons.append("OrderflowRev")
        else:
            conditions.append(False)
        count = sum(conditions)
        return count >= 3, count, reasons
    
    @staticmethod
    def check_long_exhaustion(df: pd.DataFrame) -> Tuple[bool, int, List[str]]:
        if df is None or len(df) < 55:
            return False, 0, []
        row = df.iloc[-2]
        prev = df.iloc[-3]
        conditions = []
        reasons = []
        # 1. RSI < 25
        if row["rsi"] < 25:
            conditions.append(True)
            reasons.append(f"RSI_{row['rsi']:.0f}<25")
        else:
            conditions.append(False)
        # 2. RSI Divergence bullish
        low_price = min(df["low"].iloc[-10:])
        low_rsi = min(df["rsi"].iloc[-10:])
        if row["close"] <= low_price * 1.01 and row["rsi"] > low_rsi + 3:
            conditions.append(True)
            reasons.append("RSI_Div_Bull")
        else:
            conditions.append(False)
        # 3. MACD Divergence bullish
        low_macd = min(df["mh"].iloc[-10:])
        if row["close"] <= low_price * 1.01 and row["mh"] > low_macd + 0.5*row["atr"]:
            conditions.append(True)
            reasons.append("MACD_Div_Bull")
        else:
            conditions.append(False)
        # 4. Volume Climax
        if row["vr"] > 2.0:
            conditions.append(True)
            reasons.append(f"VolClimax_{row['vr']:.1f}x")
        else:
            conditions.append(False)
        # 5. Delta Volume Climax
        vol_ratio = row["vr"]
        vol_prev = prev["vr"] if not np.isnan(prev["vr"]) else 1
        if vol_ratio > 1.8 and vol_ratio > vol_prev * 1.2:
            conditions.append(True)
            reasons.append("DeltaVolClimax")
        else:
            conditions.append(False)
        # 6. Long Lower Wick
        body = abs(row["close"] - row["open"])
        lower_wick = min(row["close"], row["open"]) - row["low"]
        if lower_wick > body * 1.5 and lower_wick > row["atr"] * 0.3:
            conditions.append(True)
            reasons.append("LongLowerWick")
        else:
            conditions.append(False)
        # 7. ATR Expansion lalu Collapse
        atr_series = df["atr"].iloc[-10:]
        atr_peak = atr_series.max()
        atr_now = row["atr"]
        if atr_peak > atr_series.iloc[-5] * 1.3 and atr_now < atr_peak * 0.8:
            conditions.append(True)
            reasons.append("ATR_ExpCollapse")
        else:
            conditions.append(False)
        # 8. Momentum Deceleration
        m5 = row["m5"]
        m5_prev = prev["m5"]
        if m5 < -0.002 and m5 > m5_prev * 0.7:
            conditions.append(True)
            reasons.append("MomDecel_Bull")
        else:
            conditions.append(False)
        # 9. Orderflow Reversal
        br = row["br"]
        br_trough = min(df["br"].iloc[-10:])
        if br > br_trough + 0.1 and br_trough < 0.4:
            conditions.append(True)
            reasons.append("OrderflowRev_Bull")
        else:
            conditions.append(False)
        count = sum(conditions)
        return count >= 3, count, reasons


# ═══════════════════════════════════════════════════════════════════════════
#  SELF-LEARNING SIGNAL WEIGHTING (tidak berubah)
# ═══════════════════════════════════════════════════════════════════════════

class SignalWeights:
    def __init__(self):
        self.weights = {
            "ema_bull_stack": 35, "ema_mild_bull": 26, "ema_weak_bull": 14,
            "mom_strong": 30, "mom_moderate": 20,
            "macd_cross_up": 22, "macd_strengthen": 15,
            "orderflow_buy_climax": 25, "orderflow_buy_high": 14,
            "rsi_extreme_ob": 25, "rsi_high": 12,
            "ema_bear_stack": 35, "ema_mild_bear": 26, "ema_weak_bear": 14,
            "mom_strong_neg": 30, "mom_moderate_neg": 20,
            "macd_cross_down": 22, "macd_strengthen_neg": 15,
            "orderflow_sell_climax": 25, "orderflow_sell_high": 14,
            "rsi_extreme_os": 25, "rsi_low": 12,
        }
        self.history = defaultdict(list)
        self.adaptive_enabled = True
    
    def record_outcome(self, signals: List[str], won: bool):
        for sig in signals:
            base_sig = sig.split('[')[0].strip()
            if base_sig in self.weights:
                self.history[base_sig].append(1 if won else 0)
                if len(self.history[base_sig]) > LEARNING_WINDOW:
                    self.history[base_sig] = self.history[base_sig][-LEARNING_WINDOW:]
    
    def get_adjusted_weight(self, signal_name: str) -> float:
        if not self.adaptive_enabled:
            return self.weights.get(signal_name, 10)
        base = signal_name.split('[')[0].strip()
        hist = self.history.get(base, [])
        if len(hist) < MIN_TRADES_FOR_WEIGHT:
            return self.weights.get(base, 10)
        win_rate = sum(hist) / len(hist)
        factor = max(0.5, min(1.5, 0.5 + win_rate))
        return self.weights.get(base, 10) * factor


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL SCORING (tidak berubah, direction asli)
# ═══════════════════════════════════════════════════════════════════════════

class SignalScorer:
    def __init__(self, signal_weights: SignalWeights):
        self.weights = signal_weights
    
    def get_signal(self, df: pd.DataFrame, symbol: str = None) -> Tuple[Optional[str], int, List[str], float, float, float, str, float]:
        if df is None or len(df) < 55:
            return None, 0, [], 0.0, 0.0, 0.0, "UNKNOWN", 0.0
        
        regime, strength, bias = MarketRegime.detect(df)
        long_score, long_signals = self._score_long(df)
        short_score, short_signals = self._score_short(df)
        
        is_exhausted_short = False
        exhaustion_count_short = 0
        exhaustion_reasons_short = []
        is_exhausted_long = False
        exhaustion_count_long = 0
        exhaustion_reasons_long = []
        
        if regime in (MarketRegime.REGIME_RANGE, MarketRegime.REGIME_EXHAUSTION, MarketRegime.REGIME_VOLATILE):
            is_exhausted_short, exhaustion_count_short, exhaustion_reasons_short = ExhaustionConfirmation.check_short_exhaustion(df)
            is_exhausted_long, exhaustion_count_long, exhaustion_reasons_long = ExhaustionConfirmation.check_long_exhaustion(df)
        
        atr = df["atr"].iloc[-2]
        
        if regime == MarketRegime.REGIME_TRENDING_BULL:
            if long_score >= MIN_SCORE:
                return "LONG", long_score, long_signals, atr, 0, 0, regime, bias
            else:
                return None, max(long_score, short_score), [], atr, 0, 0, regime, bias
        elif regime == MarketRegime.REGIME_TRENDING_BEAR:
            if short_score >= MIN_SCORE:
                return "SHORT", short_score, short_signals, atr, 0, 0, regime, bias
            else:
                return None, max(long_score, short_score), [], atr, 0, 0, regime, bias
        elif regime == MarketRegime.REGIME_RANGE:
            if short_score > long_score and short_score >= MIN_SCORE and is_exhausted_short:
                return "SHORT", short_score, short_signals + exhaustion_reasons_short, atr, 0, 0, regime, bias
            elif long_score > short_score and long_score >= MIN_SCORE and is_exhausted_long:
                return "LONG", long_score, long_signals + exhaustion_reasons_long, atr, 0, 0, regime, bias
            else:
                return None, max(long_score, short_score), [], atr, 0, 0, regime, bias
        elif regime == MarketRegime.REGIME_EXHAUSTION:
            if short_score > long_score and short_score >= MIN_SCORE and exhaustion_count_short >= 2:
                return "SHORT", short_score, short_signals + exhaustion_reasons_short, atr, 0, 0, regime, bias
            elif long_score > short_score and long_score >= MIN_SCORE and exhaustion_count_long >= 2:
                return "LONG", long_score, long_signals + exhaustion_reasons_long, atr, 0, 0, regime, bias
            else:
                return None, max(long_score, short_score), [], atr, 0, 0, regime, bias
        elif regime == MarketRegime.REGIME_VOLATILE:
            if short_score > long_score and short_score >= MIN_SCORE + 10 and is_exhausted_short:
                return "SHORT", short_score, short_signals + exhaustion_reasons_short, atr, 0, 0, regime, bias
            elif long_score > short_score and long_score >= MIN_SCORE + 10 and is_exhausted_long:
                return "LONG", long_score, long_signals + exhaustion_reasons_long, atr, 0, 0, regime, bias
            else:
                return None, max(long_score, short_score), [], atr, 0, 0, regime, bias
        else:
            return None, 0, [], atr, 0, 0, regime, bias
    
    def _score_long(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        row = df.iloc[-2]
        prev = df.iloc[-3]
        prev2 = df.iloc[-4]
        score = 0
        signals = []
        p, e5, e9, e21, e50 = row["close"], row["e5"], row["e9"], row["e21"], row["e50"]
        if p < e5 < e9 < e21 < e50:
            w = self.weights.get_adjusted_weight("ema_bear_stack")
            score += w
            signals.append(f"EMA5↓[{w:.0f}]")
        elif p < e5 < e9 < e21:
            w = self.weights.get_adjusted_weight("ema_mild_bear")
            score += w
            signals.append(f"EMA4↓[{w:.0f}]")
        elif p < e5 < e9:
            w = self.weights.get_adjusted_weight("ema_weak_bear")
            score += w
            signals.append(f"EMA3↓[{w:.0f}]")
        m5 = row["m5"]
        if m5 < -0.003:
            w = self.weights.get_adjusted_weight("mom_strong_neg")
            score += w
            signals.append(f"Mom{m5*100:.1f}%↓[{w:.0f}]")
        elif m5 < -0.002:
            w = self.weights.get_adjusted_weight("mom_moderate_neg")
            score += w
            signals.append(f"Mom{m5*100:.1f}%↓[{w:.0f}]")
        mh = row["mh"]
        mh_p = prev["mh"]
        mh_p2 = prev2["mh"]
        if mh_p >= 0 and mh < 0:
            w = self.weights.get_adjusted_weight("macd_cross_down")
            score += w
            signals.append(f"MACD_X↓[{w:.0f}]")
        elif mh < 0 and mh < mh_p < mh_p2:
            w = self.weights.get_adjusted_weight("macd_strengthen_neg")
            score += w
            signals.append(f"MACD↓↓[{w:.0f}]")
        br = row["br"]
        if br < 0.44:
            w = self.weights.get_adjusted_weight("orderflow_sell_climax")
            score += w
            signals.append(f"SellClimax{1-br:.0%}[{w:.0f}]")
        elif br < 0.48:
            w = self.weights.get_adjusted_weight("orderflow_sell_high")
            score += w
            signals.append(f"Sell{1-br:.0%}[{w:.0f}]")
        rsi = row["rsi"]
        if rsi < 32:
            w = self.weights.get_adjusted_weight("rsi_extreme_os")
            score += w
            signals.append(f"RSI{rsi:.0f}OS[{w:.0f}]")
        elif rsi < 40:
            w = self.weights.get_adjusted_weight("rsi_low")
            score += w
            signals.append(f"RSI{rsi:.0f}Lo[{w:.0f}]")
        return score, signals
    
    def _score_short(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        row = df.iloc[-2]
        prev = df.iloc[-3]
        prev2 = df.iloc[-4]
        score = 0
        signals = []
        p, e5, e9, e21, e50 = row["close"], row["e5"], row["e9"], row["e21"], row["e50"]
        if p > e5 > e9 > e21 > e50:
            w = self.weights.get_adjusted_weight("ema_bull_stack")
            score += w
            signals.append(f"EMA5↑[{w:.0f}]")
        elif p > e5 > e9 > e21:
            w = self.weights.get_adjusted_weight("ema_mild_bull")
            score += w
            signals.append(f"EMA4↑[{w:.0f}]")
        elif p > e5 > e9:
            w = self.weights.get_adjusted_weight("ema_weak_bull")
            score += w
            signals.append(f"EMA3↑[{w:.0f}]")
        m5 = row["m5"]
        if m5 > 0.003:
            w = self.weights.get_adjusted_weight("mom_strong")
            score += w
            signals.append(f"Mom+{m5*100:.1f}%↑[{w:.0f}]")
        elif m5 > 0.002:
            w = self.weights.get_adjusted_weight("mom_moderate")
            score += w
            signals.append(f"Mom+{m5*100:.1f}%↑[{w:.0f}]")
        mh = row["mh"]
        mh_p = prev["mh"]
        mh_p2 = prev2["mh"]
        if mh_p <= 0 and mh > 0:
            w = self.weights.get_adjusted_weight("macd_cross_up")
            score += w
            signals.append(f"MACD_X↑[{w:.0f}]")
        elif mh > 0 and mh > mh_p > mh_p2:
            w = self.weights.get_adjusted_weight("macd_strengthen")
            score += w
            signals.append(f"MACD↑↑[{w:.0f}]")
        br = row["br"]
        if br > 0.56:
            w = self.weights.get_adjusted_weight("orderflow_buy_climax")
            score += w
            signals.append(f"BuyClimax{br:.0%}[{w:.0f}]")
        elif br > 0.52:
            w = self.weights.get_adjusted_weight("orderflow_buy_high")
            score += w
            signals.append(f"Buy{br:.0%}[{w:.0f}]")
        rsi = row["rsi"]
        if rsi > 68:
            w = self.weights.get_adjusted_weight("rsi_extreme_ob")
            score += w
            signals.append(f"RSI{rsi:.0f}OB[{w:.0f}]")
        elif rsi > 60:
            w = self.weights.get_adjusted_weight("rsi_high")
            score += w
            signals.append(f"RSI{rsi:.0f}Hi[{w:.0f}]")
        return score, signals


# ═══════════════════════════════════════════════════════════════════════════
#  RISK MANAGER (digunakan untuk menghitung level asli sebelum ditukar)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
#  RISK MANAGER (Diperbarui untuk minimal TP 0.1% Net)
# ═══════════════════════════════════════════════════════════════════════════

class RiskManager:
    @staticmethod
    def calculate_sl_tp(entry_price: float, atr: float, direction: str) -> Tuple[float, float, float, float]:
        sl_distance = ATR_MULT_SL * atr
        tp_distance = ATR_MULT_TP * atr
        
        # --- PERBAIKAN: MINIMAL TP BERSIH DARI FEE ---
        # Karena REVERSED LOGIC, sl_distance asli menjadi actual TP bot nantinya.
        # Fee taker 0.05% per trade = 0.1% round-trip. 
        # Biar profit bersih minimal 0.1%, pergerakan harga (gross) harus minimal 0.2% (0.002).
        min_tp_gross = entry_price * 0.002
        
        if sl_distance < min_tp_gross:
            sl_distance = min_tp_gross
        # ---------------------------------------------
        
        rr = tp_distance / sl_distance
        if rr < MIN_RR_RATIO:
            tp_distance = sl_distance * MIN_RR_RATIO
            
        max_sl = entry_price * MAX_SL_PCT
        max_tp = entry_price * MAX_TP_PCT
        
        sl_distance = min(sl_distance, max_sl)
        tp_distance = min(tp_distance, max_tp)
        
        if direction == "LONG":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
            sl_pct = sl_distance / entry_price
            tp_pct = tp_distance / entry_price
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
            sl_pct = sl_distance / entry_price
            tp_pct = tp_distance / entry_price
            
        return sl_price, tp_price, sl_pct, tp_pct
    
# ═══════════════════════════════════════════════════════════════════════════
#  TRADE RECORDER & LEARNING LAYER (tidak berubah)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    won: bool
    regime: str
    signals: List[str]
    score: float
    atr_entry: float
    sl_pct: float
    tp_pct: float
    hold_seconds: float
    timestamp: float = field(default_factory=time.time)


class LearningLayer:
    def __init__(self, signal_weights: SignalWeights):
        self.signal_weights = signal_weights
        self.trades: List[TradeRecord] = []
        self.stats_by_regime = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
        self.stats_by_signal_combo = defaultdict(lambda: {"wins": 0, "losses": 0})
        self.stats_by_symbol = defaultdict(lambda: {"wins": 0, "losses": 0})
    
    def add_trade(self, trade: TradeRecord):
        self.trades.append(trade)
        regime = trade.regime
        self.stats_by_regime[regime]["wins"] += 1 if trade.won else 0
        self.stats_by_regime[regime]["losses"] += 0 if trade.won else 1
        self.stats_by_regime[regime]["pnl"] += trade.pnl
        sig_key = "|".join(sorted([s.split('[')[0].strip() for s in trade.signals[:3]]))
        self.stats_by_signal_combo[sig_key]["wins"] += 1 if trade.won else 0
        self.stats_by_signal_combo[sig_key]["losses"] += 0 if trade.won else 1
        self.stats_by_symbol[trade.symbol]["wins"] += 1 if trade.won else 0
        self.stats_by_symbol[trade.symbol]["losses"] += 0 if trade.won else 1
        self.signal_weights.record_outcome(trade.signals, trade.won)
        if len(self.trades) > 1000:
            self.trades = self.trades[-500:]
    
    def get_winrate_by_regime(self, regime: str) -> float:
        stats = self.stats_by_regime[regime]
        total = stats["wins"] + stats["losses"]
        return stats["wins"] / total if total > 0 else 0.5
    
    def get_global_winrate(self) -> float:
        total_wins = sum(s["wins"] for s in self.stats_by_regime.values())
        total_losses = sum(s["losses"] for s in self.stats_by_regime.values())
        total = total_wins + total_losses
        return total_wins / total if total > 0 else 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  BOT STATE & UTILITIES (tidak berubah)
# ═══════════════════════════════════════════════════════════════════════════

_precision_cache = {}
_ohlcv_cache = {}
_ticker_cache = {}
_ticker_ts = 0
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_rescan_q = queue.Queue()
_hot_syms = deque(maxlen=30)

_macro = {"btc": "UNKNOWN"}
_ks = {"active": False, "reason": "", "resume": 0, "consec": 0, "daily": 0.0, "day_reset": 0}
_stats = {
    "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "best": 0.0, "worst": 0.0,
    "extreme_tp": 0, "hard_sl": 0, "force": 0, "btc_block": 0,
    "hist": deque(maxlen=200), "start": time.time(),
}

def get_precision(symbol):
    if symbol in _precision_cache: return _precision_cache[symbol]
    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                prec = int(s['quantityPrecision'])
                _precision_cache[symbol] = prec
                return prec
    except: pass
    return 2

def qty(symbol, price):
    raw_qty = (ORDER_USDT * LEVERAGE) / price
    prec = get_precision(symbol)
    return round(raw_qty, prec)

def price_live(symbol):
    try: return float(client.futures_symbol_ticker(symbol=symbol)["price"])
    except: return 0.0

def tickers_all():
    global _ticker_cache, _ticker_ts
    now = time.time()
    if now - _ticker_ts < 2 and _ticker_cache: return _ticker_cache
    try:
        raw = client.futures_ticker()
        _ticker_cache = {
            t["symbol"]: {
                "pct": float(t["priceChangePercent"]),
                "vol": float(t["quoteVolume"]),
                "last": float(t["lastPrice"])
            } for t in raw
        }
        _ticker_ts = now
        return _ticker_cache
    except: return _ticker_cache

def ohlcv(symbol, interval, limit=100):
    key, now = (symbol, interval), time.time()
    ttl = TTL_5M
    if key in _ohlcv_cache and now - _ohlcv_cache[key][0] < ttl:
        return _ohlcv_cache[key][1]
    try:
        kl = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(kl, columns=["time","open","high","low","close","volume",
                                        "ct","qv","trades","tbbase","tbquote","ignore"])
        for c in ["open","high","low","close","volume","tbbase","tbquote"]:
            df[c] = df[c].astype(float)
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        df["mh"] = ta.trend.MACD(df["close"], 12, 26, 9).macd_diff()
        df["e5"] = ta.trend.EMAIndicator(df["close"], 5).ema_indicator()
        df["e9"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
        df["e21"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
        df["e50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
        df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14).adx()
        df["vm"] = df["volume"].rolling(20).mean()
        df["vr"] = df["volume"] / df["vm"].replace(0, 1)
        df["br"] = df["tbbase"] / df["volume"].replace(0, 1)
        df["body"] = abs(df["close"] - df["open"])
        df["rng"] = df["high"] - df["low"]
        df["br2"] = df["body"] / df["rng"].replace(0, 1)
        df["m5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5)
        df["m3"] = (df["close"] - df["close"].shift(3)) / df["close"].shift(3)
        _ohlcv_cache[key] = (now, df)
        return df
    except:
        return _ohlcv_cache.get(key, (None, None))[1]

def ks_check():
    k, now = _ks, time.time()
    if k["active"] and now >= k["resume"]:
        k["active"] = False; k["consec"] = 0
    if k["active"]: return True, k["reason"]
    day = now - (now % 86400)
    if day > k["day_reset"]: k["daily"] = 0.0; k["day_reset"] = day
    if k["daily"] <= DAILY_LOSS:
        k["active"] = True
        k["reason"] = f"daily({k['daily']:.2f})"
        k["resume"] = day + 86400
        return True, k["reason"]
    if k["consec"] >= CONSEC_MAX:
        k["active"] = True
        k["reason"] = f"consec({k['consec']})"
        k["resume"] = now + CONSEC_PAUSE
        return True, k["reason"]
    return False, ""

def ks_upd(pnl):
    _ks["daily"] += pnl
    _ks["consec"] = 0 if pnl >= 0 else _ks["consec"] + 1

# ═══════════════════════════════════════════════════════════════════════════
#  TRADING STATE
# ═══════════════════════════════════════════════════════════════════════════

live_positions = {}
trade_log = []
signal_weights = SignalWeights()
scorer = SignalScorer(signal_weights)
learning = LearningLayer(signal_weights)

# ═══════════════════════════════════════════════════════════════════════════
#  CORE TRADING FUNCTIONS (dengan logika kebalikan)
# ═══════════════════════════════════════════════════════════════════════════

def live_open(orig_direction, score, sigs, price, atr, regime, bias, sym):
    """
    Membuka posisi dengan logika kebalikan:
    - orig_direction adalah sinyal asli dari scorer (LONG/SHORT)
    - Entry dibalik: jika orig_direction LONG → buka SHORT, jika SHORT → buka LONG
    - HardSL (stop loss asli) menjadi TP, ExtremeTP (take profit asli) menjadi SL
    """
    with _lock:
        if sym in live_positions or len(live_positions) >= MAX_POSITIONS:
            return
        live_positions[sym] = {"_r": True}
    
    px_now = price_live(sym)
    if px_now > 0:
        slip = abs(px_now - price) / price
        if slip > SLIPPAGE_GUARD:
            with _lock: live_positions.pop(sym, None)
            return
        price = px_now
    
    try:
        q_val = qty(sym, price)
    except:
        with _lock: live_positions.pop(sym, None)
        return
    
    # Hitung level SL/TP asli berdasarkan arah asli
    sl_orig, tp_orig, sl_pct_orig, tp_pct_orig = RiskManager.calculate_sl_tp(price, atr, orig_direction)
    
    # Tentukan arah yang sebenarnya dibuka (kebalikan dari asli)
    if orig_direction == "LONG":
        actual_side = "SHORT"
        # Untuk SHORT, SL harus di atas entry, TP di bawah entry
        # Gunakan level asli: tp_orig (awalnya di atas entry untuk LONG) menjadi SL (di atas entry)
        # sl_orig (awalnya di bawah entry) menjadi TP (di bawah entry)
        sl_price = tp_orig   # karena tp_orig > price
        tp_price = sl_orig   # karena sl_orig < price
    else:  # orig_direction == "SHORT"
        actual_side = "LONG"
        # Untuk LONG, SL di bawah entry, TP di atas entry
        # tp_orig (awalnya di bawah entry untuk SHORT) menjadi SL (di bawah entry)
        # sl_orig (awalnya di atas entry) menjadi TP (di atas entry)
        sl_price = tp_orig   # tp_orig < price
        tp_price = sl_orig   # sl_orig > price
    
    # Hitung persentase untuk logging
    sl_pct = abs(sl_price - price) / price
    tp_pct = abs(tp_price - price) / price
    # Pastikan RR tetap >= MIN_RR_RATIO setelah swap (tidak wajib karena sudah dipenuhi asli)
    
    pos = {
        "side": actual_side,
        "entry": price,
        "qty": q_val,
        "open_time": time.time(),
        "score": score,
        "sigs": sigs,
        "atr": atr,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "regime": regime,
        "bias": bias,
        "orig_direction": orig_direction   # simpan untuk referensi
    }
    with _lock: live_positions[sym] = pos
    
    d = "🟢" if actual_side == "LONG" else "🔴"
    print(f"\n  {d} [REVERSED] {sym} {actual_side} (orig {orig_direction}) @{price:.6g} | SL:{sl_pct*100:.2f}% TP:{tp_pct*100:.2f}% | Regime:{regime}")
    print(f"       Original signals: {' | '.join(sigs[:5])}")
    _stats["trades"] += 1

def live_close(sym, reason, price=None):
    with _lock:
        pos = live_positions.pop(sym, None)
    if pos is None or pos.get("_r"): return
    
    if price is None: price = price_live(sym)
    if price == 0: return
    
    side, entry, q_val = pos["side"], pos["entry"], pos["qty"]
    gross_pnl = (price - entry) * q_val if side == "LONG" else (entry - price) * q_val
    fee_rate = 0.0005
    total_fee = (entry * q_val + price * q_val) * fee_rate
    pnl = gross_pnl - total_fee
    pct = (price - entry) / entry * 100 if side == "LONG" else (entry - price) / entry * 100
    hold = time.time() - pos["open_time"]
    won = pnl >= 0
    e = "🟢" if won else "🔴"
    
    print(f"  {e} [REVERSED] {sym} {side} CLOSE — {reason}")
    print(f"     {entry:.6g}→{price:.6g} ({pct:+.3f}%) hold:{hold:.0f}s | PnL:{pnl:+.5f}U")
    
    trade = TradeRecord(
        symbol=sym, direction=side, entry_price=entry, exit_price=price,
        pnl=pnl, won=won, regime=pos.get("regime", "UNKNOWN"),
        signals=pos.get("sigs", []), score=pos.get("score", 0),
        atr_entry=pos.get("atr", 0), sl_pct=pos.get("sl_pct", 0), tp_pct=pos.get("tp_pct", 0),
        hold_seconds=hold
    )
    learning.add_trade(trade)
    
    _stats["pnl"] += pnl
    _stats["hist"].append(pnl)
    ks_upd(pnl)
    if won:
        _stats["wins"] += 1
        if pnl > _stats["best"]: _stats["best"] = pnl
    else:
        _stats["losses"] += 1
        if pnl < _stats["worst"]: _stats["worst"] = pnl
    
    if "TP" in reason: _stats["extreme_tp"] += 1
    elif "SL" in reason: _stats["hard_sl"] += 1
    
    trade_log.append({
        "sym": sym, "side": side, "entry": round(entry, 7), "exit": round(price, 7),
        "pnl": round(pnl, 5), "reason": reason, "hold": int(hold),
    })
    _hot_syms.appendleft(sym)
    _rescan_q.put(1)
    print_inline()

def monitor_positions():
    for sym in list(live_positions.keys()):
        pos = live_positions.get(sym)
        if pos is None or pos.get("_r"): continue
        px = price_live(sym)
        if px == 0: continue
        side = pos["side"]
        sl_px = pos["sl_price"]
        tp_px = pos["tp_price"]
        if side == "LONG":
            if px <= sl_px:
                live_close(sym, "HardSL", px); continue
            if px >= tp_px:
                live_close(sym, "ExtremeTP", px); continue
        else:
            if px >= sl_px:
                live_close(sym, "HardSL", px); continue
            if px <= tp_px:
                live_close(sym, "ExtremeTP", px); continue

# ═══════════════════════════════════════════════════════════════════════════
#  SCANNER THREAD (tidak berubah)
# ═══════════════════════════════════════════════════════════════════════════

def scan_one(sym):
    try:
        time.sleep(0.002)
        df = ohlcv(sym, Client.KLINE_INTERVAL_5MINUTE, 100)
        if df is None: return None
        df_ta = df.copy()
        required = ["rsi","mh","e5","e9","e21","e50","atr","adx","vr","br","m5","br2"]
        if not all(col in df_ta.columns for col in required):
            df_ta = run_ta(df_ta)
        px = df_ta["close"].iloc[-2]
        atr = df_ta["atr"].iloc[-2]
        if px == 0 or np.isnan(atr): return None
        direction, score, sigs, atr_val, _, _, regime, bias = scorer.get_signal(df_ta, sym)
        if direction is None: return None
        px_live = price_live(sym)
        if px_live == 0: return None
        return (sym, direction, score, sigs, px_live, atr_val, regime, bias)
    except Exception as e:
        return None

def run_ta(df):
    if "rsi" not in df.columns:
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        df["mh"] = ta.trend.MACD(df["close"], 12, 26, 9).macd_diff()
        df["e5"] = ta.trend.EMAIndicator(df["close"], 5).ema_indicator()
        df["e9"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
        df["e21"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
        df["e50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
        df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14).adx()
        df["vm"] = df["volume"].rolling(20).mean()
        df["vr"] = df["volume"] / df["vm"].replace(0, 1)
        df["br"] = df["tbbase"] / df["volume"].replace(0, 1)
        df["body"] = abs(df["close"] - df["open"])
        df["rng"] = df["high"] - df["low"]
        df["br2"] = df["body"] / df["rng"].replace(0, 1)
        df["m5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5)
    return df

def scan_batch(syms):
    res = []
    fut = {_executor.submit(scan_one, s): s for s in syms[:BATCH_SIZE]}
    for f in as_completed(fut, timeout=5):
        try:
            if r := f.result(timeout=1): res.append(r)
        except: pass
    return res

def top_movers(syms, n=30):
    tk, ss = tickers_all(), set(syms)
    mv = [(s, abs(d["pct"])) for s, d in tk.items() if s in ss]
    return [s for s, _ in sorted(mv, key=lambda x: x[1], reverse=True)[:n]]

# ═══════════════════════════════════════════════════════════════════════════
#  PRINTING
# ═══════════════════════════════════════════════════════════════════════════

def print_inline():
    n = _stats["wins"] + _stats["losses"]
    wr = _stats["wins"] / n * 100 if n else 0
    pnl, e = _stats["pnl"], "💚" if _stats["pnl"] >= 0 else "🔴"
    print(f"       ┌ [v20.1 REVERSED] {n}T WR:{wr:.0f}% W:{_stats['wins']} L:{_stats['losses']} {e}PnL:{pnl:+.4f}U")
    print(f"       └ ExTP:{_stats['extreme_tp']} HardSL:{_stats['hard_sl']} | Regime WR: {learning.get_winrate_by_regime('TRENDING_BULL'):.0%}")

def print_full():
    n = _stats["wins"] + _stats["losses"]
    wr = _stats["wins"] / n * 100 if n else 0
    pnl = _stats["pnl"]
    sess = (time.time() - _stats["start"]) / 3600
    tph = n / sess if sess > 0 else 0
    e = "💚" if pnl >= 0 else "🔴"
    print(f"\n  {'─'*70}")
    print(f"    ✅ REVERSED LOGIC v20.1 — CONTRARIAN ADAPTIVE TRADING")
    print(f"    🎯 {n}T WR:{wr:.0f}% W:{_stats['wins']} L:{_stats['losses']} ({tph:.1f}T/hr)")
    print(f"    {e} PnL Net:{pnl:+.5f}U Best:{_stats['best']:+.5f} Worst:{_stats['worst']:+.5f}")
    print(f"    💰 ExtremeTP:{_stats['extreme_tp']} HardSL:{_stats['hard_sl']}")
    print(f"    📊 Learning: Global WR {learning.get_global_winrate():.1%}")
    print(f"    ⚙️  TP/SL: SWAPPED (HardSL<->ExtremeTP) + Direction Reversed")
    if trade_log:
        print(f"    📋 Last 5:")
        for t in trade_log[-5:]:
            em = "🟢" if t["pnl"] > 0 else "🔴"
            print(f"       {em} {t['sym']:<16} {t['side']} {t['pnl']:+.5f}U {t['hold']}s — {t['reason']}")
    print(f"  {'─'*70}")

# ═══════════════════════════════════════════════════════════════════════════
#  THREADS
# ═══════════════════════════════════════════════════════════════════════════

def t_monitor():
    while True:
        try:
            if live_positions:
                monitor_positions()
        except: pass
        time.sleep(MONITOR_INT)

def t_slot_filler(syms):
    scan_idx = 0
    n_bat = max(1, math.ceil(len(syms) / BATCH_SIZE))
    while True:
        try:
            slots = MAX_POSITIONS - len(live_positions)
            if slots <= 0 or ks_check()[0]:
                time.sleep(SLOT_FILL_INT)
                continue
            hot = [s for s in _hot_syms if s not in live_positions]
            mv = top_movers(syms, 30)
            mv = [s for s in mv if s not in live_positions]
            bs = scan_idx * BATCH_SIZE
            reg = [s for s in syms[bs:bs+BATCH_SIZE] if s not in live_positions and s not in mv]
            scan_idx = (scan_idx + 1) % n_bat
            scan_list = list(dict.fromkeys(hot[:5] + mv[:20] + reg[:15]))[:BATCH_SIZE]
            if not scan_list:
                time.sleep(SLOT_FILL_INT)
                continue
            res = scan_batch(scan_list)
            if res:
                res.sort(key=lambda x: x[2], reverse=True)
                for r in res[:slots]:
                    if len(live_positions) >= MAX_POSITIONS: break
                    sym, orig_dir, sc, sg, px, atr, regime, bias = r
                    # Panggil live_open dengan orig_dir asli, fungsi akan membalik otomatis
                    live_open(orig_dir, sc, sg, px, atr, regime, bias, sym)
        except: pass
        time.sleep(SLOT_FILL_INT)

def t_rescan(syms):
    while True:
        try:
            _rescan_q.get(timeout=5)
            time.sleep(0.05)
            slots = MAX_POSITIONS - len(live_positions)
            if slots <= 0 or ks_check()[0]: continue
            hot = [s for s in _hot_syms if s not in live_positions]
            rest = [s for s in syms if s not in live_positions and s not in hot]
            res = scan_batch((hot + rest)[:30])
            if res:
                res.sort(key=lambda x: x[2], reverse=True)
                for r in res[:slots]:
                    if len(live_positions) >= MAX_POSITIONS: break
                    sym, orig_dir, sc, sg, px, atr, regime, bias = r
                    live_open(orig_dir, sc, sg, px, atr, regime, bias, sym)
        except: pass

def t_macro():
    while True:
        try:
            df_btc = ohlcv("BTCUSDT", Client.KLINE_INTERVAL_5MINUTE, 80)
            if df_btc is not None:
                regime, _, _ = MarketRegime.detect(df_btc)
                _macro["btc"] = regime
        except: pass
        time.sleep(10)

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_bot():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  ✅ REVERSED LOGIC v20.1 — CONTRARIAN ADAPTIVE TRADING ENGINE     ║")
    print("║  ✅ Entry dibalik (LONG<->SHORT)                                 ║")
    print("║  ✅ HardSL menjadi TP, ExtremeTP menjadi SL                       ║")
    print("║  ✅ Loss asli berubah jadi profit dan sebaliknya                  ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    try:
        valid = {s["symbol"] for s in client.futures_exchange_info()["symbols"] if s["status"] == "TRADING"}
        syms = list(dict.fromkeys([s for s in SYMBOLS if s in valid]))
    except:
        syms = list(dict.fromkeys(SYMBOLS))
    print(f"  📋 {len(syms)} simbol aktif terpantau")
    threading.Thread(target=t_monitor, daemon=True).start()
    threading.Thread(target=t_slot_filler, args=(syms,), daemon=True).start()
    threading.Thread(target=t_rescan, args=(syms,), daemon=True).start()
    threading.Thread(target=t_macro, daemon=True).start()
    time.sleep(2)
    tickers_all()
    cycle = 0
    while True:
        cycle += 1
        slots = MAX_POSITIONS - len(live_positions)
        print(f"\n{'═'*62}")
        print(f"  #{cycle} {time.strftime('%H:%M:%S')} BTC Regime:{_macro['btc']} ({len(live_positions)}/{MAX_POSITIONS}) PnL:{_stats['pnl']:+.4f}U")
        if (k := ks_check())[0]:
            print(f"  🚨 KS:{k[1]}")
        elif slots == 0:
            print(f"  ✅ Slots full")
        else:
            print(f"  🔍 {slots} slot kosong — Adaptive scanning (REVERSED)...")
        if cycle % 30 == 0:
            print_full()
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    run_bot()
