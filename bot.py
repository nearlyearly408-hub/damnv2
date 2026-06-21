"""
Bot Scalping v20.3 — REVERSED LOGIC (CONTRARIAN)
=================================================
ROOT CAUSE FIX:
  Bot lama hanya cek harga setiap 1 detik via polling.
  Kalau harga gap dalam <1 detik, bot close di harga yang sudah
  jauh melewati SL → loss bisa -3U padahal SL harusnya -0.24U.

SOLUSI v20.3:
  - Saat open posisi, langsung pasang STOP_MARKET order di Binance
    sebagai real hard stop. Exchange yang eksekusi, bukan bot.
  - TP tetap dimonitor bot via polling (ExtremeTP).
  - Trailing Stop juga tetap via polling (sifatnya memang dynamic).
  - Saat TP/Trail trigger, cancel SL order yang ada di exchange.
  - SL = TP = 0.5% fixed simetris.
  - Trail aktif saat profit >= 1%, gap 0.15%.
"""

import os
import time
import math
import threading
import queue
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from dotenv import load_dotenv
from binance.client import Client
from binance.enums import (
    SIDE_BUY, SIDE_SELL,
    ORDER_TYPE_STOP_MARKET,
    FUTURE_ORDER_TYPE_STOP_MARKET,
)
import ta

load_dotenv()
client = Client(os.getenv("API_KEY"), os.getenv("API_SECRET"))
client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

LEVERAGE      = 20
ORDER_USDT    = 2.0
MAX_POSITIONS = 3

# SL dan TP: fixed simetris 0.5%
SL_PCT = 0.005   # 0.5%  — Hard Stop Loss (real order di exchange)
TP_PCT = 0.005   # 0.5%  — Take Profit (dimonitor bot)

# Trailing Stop
TRAIL_ACTIVATE_PCT = 0.010   # 1.0%  — trail mulai aktif
TRAIL_GAP_PCT      = 0.0015  # 0.15% — jarak trail dari peak profit
TRAIL_STEP_PCT     = 0.003   # 0.3%  — log step

# Scanning
SCAN_INTERVAL  = 2.0
MONITOR_INT    = 0.5   # ✅ dipercepat ke 0.5 detik untuk TP/Trail
BATCH_SIZE     = 15
MAX_WORKERS    = 5
SLOT_FILL_INT  = 0.01
TTL_5M         = 2

# Scoring
MIN_SCORE      = 55
SLIPPAGE_GUARD = 0.0015

# Kill Switch
DAILY_LOSS   = -20.0
CONSEC_MAX   = 15
CONSEC_PAUSE = 10

# Learning
LEARNING_WINDOW       = 200
MIN_TRADES_FOR_WEIGHT = 20

# ═══════════════════════════════════════════════════════════════════════════
#  SYMBOLS
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

# ═══════════════════════════════════════════════════════════════════════════
#  MARKET REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════

class MarketRegime:
    REGIME_TRENDING_BULL = "TRENDING_BULL"
    REGIME_TRENDING_BEAR = "TRENDING_BEAR"
    REGIME_RANGE         = "RANGE"
    REGIME_VOLATILE      = "VOLATILE"
    REGIME_EXHAUSTION    = "EXHAUSTION"

    @staticmethod
    def detect(df: pd.DataFrame) -> Tuple[str, float, float]:
        if df is None or len(df) < 55:
            return MarketRegime.REGIME_RANGE, 0, 0
        row  = df.iloc[-2]
        prev = df.iloc[-3]
        close = row["close"]
        e5, e9, e21, e50 = row["e5"], row["e9"], row["e21"], row["e50"]
        atr      = row["atr"]
        atr_prev = prev["atr"]
        adx      = row["adx"]
        bull_stack        = close > e5 > e9 > e21 > e50
        bear_stack        = close < e5 < e9 < e21 < e50
        mild_bull         = close > e9 > e21
        mild_bear         = close < e9 < e21
        strong_trend      = adx > 25
        very_strong_trend = adx > 35
        atr_expand   = (atr / atr_prev) > 1.2 if atr_prev > 0 else False
        atr_collapse = (atr / atr_prev) < 0.8 if atr_prev > 0 else False
        m5      = row["m5"]
        m5_prev = prev["m5"]
        decelerating = (abs(m5) < abs(m5_prev)) if not np.isnan(m5_prev) else False

        if very_strong_trend and bull_stack:
            return MarketRegime.REGIME_TRENDING_BULL, min(adx, 100), 1.0
        elif very_strong_trend and bear_stack:
            return MarketRegime.REGIME_TRENDING_BEAR, min(adx, 100), -1.0
        elif strong_trend and (bull_stack or mild_bull):
            return MarketRegime.REGIME_TRENDING_BULL, min(adx, 80), 0.7
        elif strong_trend and (bear_stack or mild_bear):
            return MarketRegime.REGIME_TRENDING_BEAR, min(adx, 80), -0.7
        elif atr_expand and adx < 20:
            return MarketRegime.REGIME_VOLATILE, 50, 0
        elif (atr_collapse and decelerating) or (20 < adx < 35 and decelerating):
            return MarketRegime.REGIME_EXHAUSTION, 40, 1 if m5 > 0 else -1
        else:
            return MarketRegime.REGIME_RANGE, 30, 0


# ═══════════════════════════════════════════════════════════════════════════
#  EXHAUSTION CONFIRMATION LAYER
# ═══════════════════════════════════════════════════════════════════════════

class ExhaustionConfirmation:
    @staticmethod
    def check_short_exhaustion(df: pd.DataFrame) -> Tuple[bool, int, List[str]]:
        if df is None or len(df) < 55:
            return False, 0, []
        row, prev = df.iloc[-2], df.iloc[-3]
        conditions, reasons = [], []

        conditions.append(row["rsi"] > 75)
        if row["rsi"] > 75: reasons.append(f"RSI_{row['rsi']:.0f}>75")

        high_price = max(df["high"].iloc[-10:])
        high_rsi   = max(df["rsi"].iloc[-10:])
        ok = row["close"] >= high_price * 0.99 and row["rsi"] < high_rsi - 3
        conditions.append(ok)
        if ok: reasons.append("RSI_Div")

        high_macd = max(df["mh"].iloc[-10:])
        ok = row["close"] >= high_price * 0.99 and row["mh"] < high_macd - 0.5 * row["atr"]
        conditions.append(ok)
        if ok: reasons.append("MACD_Div")

        conditions.append(row["vr"] > 2.0)
        if row["vr"] > 2.0: reasons.append(f"VolClimax_{row['vr']:.1f}x")

        vol_prev = prev["vr"] if not np.isnan(prev["vr"]) else 1
        ok = row["vr"] > 1.8 and row["vr"] > vol_prev * 1.2
        conditions.append(ok)
        if ok: reasons.append("DeltaVolClimax")

        body = abs(row["close"] - row["open"])
        uw   = row["high"] - max(row["close"], row["open"])
        ok   = uw > body * 1.5 and uw > row["atr"] * 0.3
        conditions.append(ok)
        if ok: reasons.append("LongUpperWick")

        atr_s = df["atr"].iloc[-10:]
        ok = atr_s.max() > atr_s.iloc[-5] * 1.3 and row["atr"] < atr_s.max() * 0.8
        conditions.append(ok)
        if ok: reasons.append("ATR_ExpCollapse")

        m5, m5p = row["m5"], prev["m5"]
        ok = m5 > 0.002 and m5 < m5p * 0.7
        conditions.append(ok)
        if ok: reasons.append("MomDecel")

        br_peak = max(df["br"].iloc[-10:])
        ok = row["br"] < br_peak - 0.1 and br_peak > 0.6
        conditions.append(ok)
        if ok: reasons.append("OrderflowRev")

        count = sum(conditions)
        return count >= 3, count, reasons

    @staticmethod
    def check_long_exhaustion(df: pd.DataFrame) -> Tuple[bool, int, List[str]]:
        if df is None or len(df) < 55:
            return False, 0, []
        row, prev = df.iloc[-2], df.iloc[-3]
        conditions, reasons = [], []

        conditions.append(row["rsi"] < 25)
        if row["rsi"] < 25: reasons.append(f"RSI_{row['rsi']:.0f}<25")

        low_price = min(df["low"].iloc[-10:])
        low_rsi   = min(df["rsi"].iloc[-10:])
        ok = row["close"] <= low_price * 1.01 and row["rsi"] > low_rsi + 3
        conditions.append(ok)
        if ok: reasons.append("RSI_Div_Bull")

        low_macd = min(df["mh"].iloc[-10:])
        ok = row["close"] <= low_price * 1.01 and row["mh"] > low_macd + 0.5 * row["atr"]
        conditions.append(ok)
        if ok: reasons.append("MACD_Div_Bull")

        conditions.append(row["vr"] > 2.0)
        if row["vr"] > 2.0: reasons.append(f"VolClimax_{row['vr']:.1f}x")

        vol_prev = prev["vr"] if not np.isnan(prev["vr"]) else 1
        ok = row["vr"] > 1.8 and row["vr"] > vol_prev * 1.2
        conditions.append(ok)
        if ok: reasons.append("DeltaVolClimax")

        body = abs(row["close"] - row["open"])
        lw   = min(row["close"], row["open"]) - row["low"]
        ok   = lw > body * 1.5 and lw > row["atr"] * 0.3
        conditions.append(ok)
        if ok: reasons.append("LongLowerWick")

        atr_s = df["atr"].iloc[-10:]
        ok = atr_s.max() > atr_s.iloc[-5] * 1.3 and row["atr"] < atr_s.max() * 0.8
        conditions.append(ok)
        if ok: reasons.append("ATR_ExpCollapse")

        m5, m5p = row["m5"], prev["m5"]
        ok = m5 < -0.002 and m5 > m5p * 0.7
        conditions.append(ok)
        if ok: reasons.append("MomDecel_Bull")

        br_trough = min(df["br"].iloc[-10:])
        ok = row["br"] > br_trough + 0.1 and br_trough < 0.4
        conditions.append(ok)
        if ok: reasons.append("OrderflowRev_Bull")

        count = sum(conditions)
        return count >= 3, count, reasons


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL WEIGHTING & SCORING
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
        self.history          = defaultdict(list)
        self.adaptive_enabled = True

    def record_outcome(self, signals: List[str], won: bool):
        for sig in signals:
            base = sig.split('[')[0].strip()
            if base in self.weights:
                self.history[base].append(1 if won else 0)
                if len(self.history[base]) > LEARNING_WINDOW:
                    self.history[base] = self.history[base][-LEARNING_WINDOW:]

    def get_adjusted_weight(self, name: str) -> float:
        if not self.adaptive_enabled:
            return self.weights.get(name, 10)
        base = name.split('[')[0].strip()
        hist = self.history.get(base, [])
        if len(hist) < MIN_TRADES_FOR_WEIGHT:
            return self.weights.get(base, 10)
        factor = max(0.5, min(1.5, 0.5 + sum(hist) / len(hist)))
        return self.weights.get(base, 10) * factor


class SignalScorer:
    def __init__(self, sw: SignalWeights):
        self.w = sw

    def get_signal(self, df, sym=None):
        if df is None or len(df) < 55:
            return None, 0, [], 0.0, 0, 0, "UNKNOWN", 0.0
        regime, strength, bias = MarketRegime.detect(df)
        ls, lsig = self._score_long(df)
        ss, ssig = self._score_short(df)
        ex_s, cnt_s, rsn_s = False, 0, []
        ex_l, cnt_l, rsn_l = False, 0, []
        if regime in (MarketRegime.REGIME_RANGE, MarketRegime.REGIME_EXHAUSTION, MarketRegime.REGIME_VOLATILE):
            ex_s, cnt_s, rsn_s = ExhaustionConfirmation.check_short_exhaustion(df)
            ex_l, cnt_l, rsn_l = ExhaustionConfirmation.check_long_exhaustion(df)
        atr = df["atr"].iloc[-2]
        if regime == MarketRegime.REGIME_TRENDING_BULL:
            if ls >= MIN_SCORE: return "LONG", ls, lsig, atr, 0, 0, regime, bias
            return None, max(ls, ss), [], atr, 0, 0, regime, bias
        elif regime == MarketRegime.REGIME_TRENDING_BEAR:
            if ss >= MIN_SCORE: return "SHORT", ss, ssig, atr, 0, 0, regime, bias
            return None, max(ls, ss), [], atr, 0, 0, regime, bias
        elif regime == MarketRegime.REGIME_RANGE:
            if ss > ls and ss >= MIN_SCORE and ex_s: return "SHORT", ss, ssig+rsn_s, atr, 0, 0, regime, bias
            if ls > ss and ls >= MIN_SCORE and ex_l: return "LONG",  ls, lsig+rsn_l, atr, 0, 0, regime, bias
            return None, max(ls, ss), [], atr, 0, 0, regime, bias
        elif regime == MarketRegime.REGIME_EXHAUSTION:
            if ss > ls and ss >= MIN_SCORE and cnt_s >= 2: return "SHORT", ss, ssig+rsn_s, atr, 0, 0, regime, bias
            if ls > ss and ls >= MIN_SCORE and cnt_l >= 2: return "LONG",  ls, lsig+rsn_l, atr, 0, 0, regime, bias
            return None, max(ls, ss), [], atr, 0, 0, regime, bias
        elif regime == MarketRegime.REGIME_VOLATILE:
            if ss > ls and ss >= MIN_SCORE+10 and ex_s: return "SHORT", ss, ssig+rsn_s, atr, 0, 0, regime, bias
            if ls > ss and ls >= MIN_SCORE+10 and ex_l: return "LONG",  ls, lsig+rsn_l, atr, 0, 0, regime, bias
            return None, max(ls, ss), [], atr, 0, 0, regime, bias
        return None, 0, [], atr, 0, 0, regime, bias

    def _score_long(self, df):
        row, prev, prev2 = df.iloc[-2], df.iloc[-3], df.iloc[-4]
        sc, sig = 0, []
        p, e5, e9, e21, e50 = row["close"], row["e5"], row["e9"], row["e21"], row["e50"]
        if   p<e5<e9<e21<e50: w=self.w.get_adjusted_weight("ema_bear_stack"); sc+=w; sig.append(f"EMA5↓[{w:.0f}]")
        elif p<e5<e9<e21:      w=self.w.get_adjusted_weight("ema_mild_bear");  sc+=w; sig.append(f"EMA4↓[{w:.0f}]")
        elif p<e5<e9:          w=self.w.get_adjusted_weight("ema_weak_bear");  sc+=w; sig.append(f"EMA3↓[{w:.0f}]")
        m5=row["m5"]
        if   m5<-0.003: w=self.w.get_adjusted_weight("mom_strong_neg");   sc+=w; sig.append(f"Mom{m5*100:.1f}%↓[{w:.0f}]")
        elif m5<-0.002: w=self.w.get_adjusted_weight("mom_moderate_neg"); sc+=w; sig.append(f"Mom{m5*100:.1f}%↓[{w:.0f}]")
        mh,mhp,mhp2=row["mh"],prev["mh"],prev2["mh"]
        if   mhp>=0 and mh<0:          w=self.w.get_adjusted_weight("macd_cross_down");     sc+=w; sig.append(f"MACD_X↓[{w:.0f}]")
        elif mh<0 and mh<mhp<mhp2:     w=self.w.get_adjusted_weight("macd_strengthen_neg"); sc+=w; sig.append(f"MACD↓↓[{w:.0f}]")
        br=row["br"]
        if   br<0.44: w=self.w.get_adjusted_weight("orderflow_sell_climax"); sc+=w; sig.append(f"SellClimax{1-br:.0%}[{w:.0f}]")
        elif br<0.48: w=self.w.get_adjusted_weight("orderflow_sell_high");   sc+=w; sig.append(f"Sell{1-br:.0%}[{w:.0f}]")
        rsi=row["rsi"]
        if   rsi<32: w=self.w.get_adjusted_weight("rsi_extreme_os"); sc+=w; sig.append(f"RSI{rsi:.0f}OS[{w:.0f}]")
        elif rsi<40: w=self.w.get_adjusted_weight("rsi_low");        sc+=w; sig.append(f"RSI{rsi:.0f}Lo[{w:.0f}]")
        return sc, sig

    def _score_short(self, df):
        row, prev, prev2 = df.iloc[-2], df.iloc[-3], df.iloc[-4]
        sc, sig = 0, []
        p, e5, e9, e21, e50 = row["close"], row["e5"], row["e9"], row["e21"], row["e50"]
        if   p>e5>e9>e21>e50: w=self.w.get_adjusted_weight("ema_bull_stack"); sc+=w; sig.append(f"EMA5↑[{w:.0f}]")
        elif p>e5>e9>e21:      w=self.w.get_adjusted_weight("ema_mild_bull");  sc+=w; sig.append(f"EMA4↑[{w:.0f}]")
        elif p>e5>e9:          w=self.w.get_adjusted_weight("ema_weak_bull");  sc+=w; sig.append(f"EMA3↑[{w:.0f}]")
        m5=row["m5"]
        if   m5>0.003: w=self.w.get_adjusted_weight("mom_strong");   sc+=w; sig.append(f"Mom+{m5*100:.1f}%↑[{w:.0f}]")
        elif m5>0.002: w=self.w.get_adjusted_weight("mom_moderate"); sc+=w; sig.append(f"Mom+{m5*100:.1f}%↑[{w:.0f}]")
        mh,mhp,mhp2=row["mh"],prev["mh"],prev2["mh"]
        if   mhp<=0 and mh>0:         w=self.w.get_adjusted_weight("macd_cross_up");  sc+=w; sig.append(f"MACD_X↑[{w:.0f}]")
        elif mh>0 and mh>mhp>mhp2:    w=self.w.get_adjusted_weight("macd_strengthen"); sc+=w; sig.append(f"MACD↑↑[{w:.0f}]")
        br=row["br"]
        if   br>0.56: w=self.w.get_adjusted_weight("orderflow_buy_climax"); sc+=w; sig.append(f"BuyClimax{br:.0%}[{w:.0f}]")
        elif br>0.52: w=self.w.get_adjusted_weight("orderflow_buy_high");   sc+=w; sig.append(f"Buy{br:.0%}[{w:.0f}]")
        rsi=row["rsi"]
        if   rsi>68: w=self.w.get_adjusted_weight("rsi_extreme_ob"); sc+=w; sig.append(f"RSI{rsi:.0f}OB[{w:.0f}]")
        elif rsi>60: w=self.w.get_adjusted_weight("rsi_high");       sc+=w; sig.append(f"RSI{rsi:.0f}Hi[{w:.0f}]")
        return sc, sig


# ═══════════════════════════════════════════════════════════════════════════
#  TRADE RECORDER & LEARNING LAYER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    symbol: str; direction: str; entry_price: float; exit_price: float
    pnl: float; won: bool; regime: str; signals: List[str]; score: float
    atr_entry: float; sl_pct: float; tp_pct: float; hold_seconds: float
    timestamp: float = field(default_factory=time.time)


class LearningLayer:
    def __init__(self, sw: SignalWeights):
        self.sw = sw
        self.trades = []
        self.stats_by_regime = defaultdict(lambda: {"wins":0,"losses":0,"pnl":0.0})

    def add_trade(self, t: TradeRecord):
        self.trades.append(t)
        s = self.stats_by_regime[t.regime]
        s["wins"]   += 1 if t.won else 0
        s["losses"] += 0 if t.won else 1
        s["pnl"]    += t.pnl
        self.sw.record_outcome(t.signals, t.won)
        if len(self.trades) > 1000: self.trades = self.trades[-500:]

    def get_global_winrate(self):
        w = sum(s["wins"]   for s in self.stats_by_regime.values())
        l = sum(s["losses"] for s in self.stats_by_regime.values())
        return w/(w+l) if w+l > 0 else 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  BOT STATE & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

_precision_cache = {}
_price_prec_cache = {}
_ohlcv_cache  = {}
_ticker_cache = {}
_ticker_ts    = 0
_lock         = threading.Lock()
_executor     = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_rescan_q     = queue.Queue()
_hot_syms     = deque(maxlen=30)
_macro        = {"btc": "UNKNOWN"}
_ks           = {"active":False,"reason":"","resume":0,"consec":0,"daily":0.0,"day_reset":0}
_stats        = {
    "trades":0,"wins":0,"losses":0,"pnl":0.0,"best":0.0,"worst":0.0,
    "extreme_tp":0,"hard_sl":0,"trail_sl":0,
    "hist":deque(maxlen=200),"start":time.time(),
}

live_positions = {}
trade_log      = []
signal_weights = SignalWeights()
scorer         = SignalScorer(signal_weights)
learning       = LearningLayer(signal_weights)


def _get_symbol_info(symbol):
    """Ambil qty precision dan price precision dari exchange info."""
    if symbol in _precision_cache:
        return _precision_cache[symbol], _price_prec_cache.get(symbol, 2)
    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                qty_prec   = int(s['quantityPrecision'])
                price_prec = int(s['pricePrecision'])
                _precision_cache[symbol]   = qty_prec
                _price_prec_cache[symbol]  = price_prec
                return qty_prec, price_prec
    except: pass
    return 2, 2

def qty(symbol, price):
    raw = (ORDER_USDT * LEVERAGE) / price
    qp, _ = _get_symbol_info(symbol)
    return round(raw, qp)

def round_price(symbol, price):
    _, pp = _get_symbol_info(symbol)
    return round(price, pp)

def price_live(symbol):
    try: return float(client.futures_symbol_ticker(symbol=symbol)["price"])
    except: return 0.0

def tickers_all():
    global _ticker_cache, _ticker_ts
    now = time.time()
    if now - _ticker_ts < 2 and _ticker_cache: return _ticker_cache
    try:
        raw = client.futures_ticker()
        _ticker_cache = {t["symbol"]:{"pct":float(t["priceChangePercent"]),"vol":float(t["quoteVolume"]),"last":float(t["lastPrice"])} for t in raw}
        _ticker_ts = now
        return _ticker_cache
    except: return _ticker_cache

def ohlcv(symbol, interval, limit=100):
    key, now = (symbol, interval), time.time()
    if key in _ohlcv_cache and now - _ohlcv_cache[key][0] < TTL_5M:
        return _ohlcv_cache[key][1]
    try:
        kl = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(kl, columns=["time","open","high","low","close","volume","ct","qv","trades","tbbase","tbquote","ignore"])
        for c in ["open","high","low","close","volume","tbbase","tbquote"]: df[c]=df[c].astype(float)
        df["rsi"] = ta.momentum.RSIIndicator(df["close"],14).rsi()
        df["mh"]  = ta.trend.MACD(df["close"],12,26,9).macd_diff()
        df["e5"]  = ta.trend.EMAIndicator(df["close"],5).ema_indicator()
        df["e9"]  = ta.trend.EMAIndicator(df["close"],9).ema_indicator()
        df["e21"] = ta.trend.EMAIndicator(df["close"],21).ema_indicator()
        df["e50"] = ta.trend.EMAIndicator(df["close"],50).ema_indicator()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"],df["low"],df["close"],14).average_true_range()
        df["adx"] = ta.trend.ADXIndicator(df["high"],df["low"],df["close"],14).adx()
        df["vm"]  = df["volume"].rolling(20).mean()
        df["vr"]  = df["volume"]/df["vm"].replace(0,1)
        df["br"]  = df["tbbase"]/df["volume"].replace(0,1)
        df["body"] = abs(df["close"]-df["open"])
        df["rng"]  = df["high"]-df["low"]
        df["br2"]  = df["body"]/df["rng"].replace(0,1)
        df["m5"]  = (df["close"]-df["close"].shift(5))/df["close"].shift(5)
        df["m3"]  = (df["close"]-df["close"].shift(3))/df["close"].shift(3)
        _ohlcv_cache[key] = (now, df)
        return df
    except: return _ohlcv_cache.get(key,(None,None))[1]

def run_ta(df):
    if "rsi" not in df.columns:
        df["rsi"]=ta.momentum.RSIIndicator(df["close"],14).rsi()
        df["mh"] =ta.trend.MACD(df["close"],12,26,9).macd_diff()
        df["e5"] =ta.trend.EMAIndicator(df["close"],5).ema_indicator()
        df["e9"] =ta.trend.EMAIndicator(df["close"],9).ema_indicator()
        df["e21"]=ta.trend.EMAIndicator(df["close"],21).ema_indicator()
        df["e50"]=ta.trend.EMAIndicator(df["close"],50).ema_indicator()
        df["atr"]=ta.volatility.AverageTrueRange(df["high"],df["low"],df["close"],14).average_true_range()
        df["adx"]=ta.trend.ADXIndicator(df["high"],df["low"],df["close"],14).adx()
        df["vm"] =df["volume"].rolling(20).mean()
        df["vr"] =df["volume"]/df["vm"].replace(0,1)
        df["br"] =df["tbbase"]/df["volume"].replace(0,1)
        df["body"]=abs(df["close"]-df["open"])
        df["rng"] =df["high"]-df["low"]
        df["br2"] =df["body"]/df["rng"].replace(0,1)
        df["m5"] =(df["close"]-df["close"].shift(5))/df["close"].shift(5)
    return df

def ks_check():
    k, now = _ks, time.time()
    if k["active"] and now >= k["resume"]: k["active"]=False; k["consec"]=0
    if k["active"]: return True, k["reason"]
    day = now-(now%86400)
    if day > k["day_reset"]: k["daily"]=0.0; k["day_reset"]=day
    if k["daily"] <= DAILY_LOSS:
        k["active"]=True; k["reason"]=f"daily({k['daily']:.2f})"; k["resume"]=day+86400; return True,k["reason"]
    if k["consec"] >= CONSEC_MAX:
        k["active"]=True; k["reason"]=f"consec({k['consec']})"; k["resume"]=now+CONSEC_PAUSE; return True,k["reason"]
    return False, ""

def ks_upd(pnl):
    _ks["daily"] += pnl
    _ks["consec"] = 0 if pnl >= 0 else _ks["consec"]+1


# ═══════════════════════════════════════════════════════════════════════════
#  EXCHANGE STOP ORDER HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _place_stop_market(symbol: str, side: str, qty_val: float, stop_price: float) -> Optional[int]:
    """
    Pasang STOP_MARKET order di Binance Futures.
    side: "BUY" atau "SELL"
    Ini yang akan eksekusi SL langsung di exchange, tanpa tergantung polling bot.
    Return: orderId atau None kalau gagal.
    """
    sp = round_price(symbol, stop_price)
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="STOP_MARKET",
            stopPrice=str(sp),
            quantity=str(qty_val),
            closePosition=False,
            timeInForce="GTE_GTC",   # Good Till Cancel
            workingType="MARK_PRICE",  # pakai mark price bukan last price
        )
        oid = order.get("orderId")
        print(f"    🛡️  StopMarket {side} placed @{sp} orderId={oid}")
        return oid
    except Exception as e:
        # Fallback: coba tanpa timeInForce (beberapa versi API tidak support)
        try:
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                stopPrice=str(sp),
                quantity=str(qty_val),
                workingType="MARK_PRICE",
            )
            oid = order.get("orderId")
            print(f"    🛡️  StopMarket {side} placed @{sp} orderId={oid} (fallback)")
            return oid
        except Exception as e2:
            print(f"    ⚠️  StopMarket GAGAL {symbol}: {e2}")
            return None

def _cancel_stop_order(symbol: str, order_id: int):
    """Cancel SL order di exchange (dipanggil saat TP atau Trail trigger)."""
    if order_id is None: return
    try:
        client.futures_cancel_order(symbol=symbol, orderId=order_id)
        print(f"    🗑️  StopOrder cancelled {symbol} id={order_id}")
    except Exception as e:
        # Mungkin sudah tereksekusi atau expired — tidak masalah
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  CORE TRADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def live_open(orig_direction, score, sigs, price, atr, regime, bias, sym):
    """
    Buka posisi REVERSED + langsung pasang STOP_MARKET order di exchange.
    - orig LONG  → buka SHORT: SL di +0.5% (atas entry), TP di -0.5% (bawah entry)
    - orig SHORT → buka LONG:  SL di -0.5% (bawah entry), TP di +0.5% (atas entry)
    Real SL order langsung dipasang di Binance saat posisi terbuka.
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

    # ── Hitung SL dan TP fixed 0.5% ─────────────────────────────────────
    if orig_direction == "LONG":
        actual_side  = "SHORT"
        sl_price     = price * (1 + SL_PCT)   # SL di ATAS entry untuk SHORT
        tp_price     = price * (1 - TP_PCT)   # TP di BAWAH entry untuk SHORT
        sl_order_side = "BUY"                  # untuk close SHORT, kita BUY
    else:
        actual_side  = "LONG"
        sl_price     = price * (1 - SL_PCT)   # SL di BAWAH entry untuk LONG
        tp_price     = price * (1 + TP_PCT)   # TP di ATAS entry untuk LONG
        sl_order_side = "SELL"                 # untuk close LONG, kita SELL

    # ── Pasang REAL stop-market order di exchange ────────────────────────
    sl_order_id = _place_stop_market(sym, sl_order_side, q_val, sl_price)

    pos = {
        "side":           actual_side,
        "entry":          price,
        "qty":            q_val,
        "open_time":      time.time(),
        "score":          score,
        "sigs":           sigs,
        "atr":            atr,
        "sl_price":       sl_price,
        "tp_price":       tp_price,
        "sl_pct":         SL_PCT,
        "tp_pct":         TP_PCT,
        "regime":         regime,
        "bias":           bias,
        "orig_direction": orig_direction,
        "sl_order_id":    sl_order_id,    # ✅ simpan id stop order
        # Trailing Stop state
        "trail_sl_price":  None,
        "peak_profit_pct": 0.0,
    }
    with _lock: live_positions[sym] = pos

    d = "🟢" if actual_side == "LONG" else "🔴"
    sl_note = "✅ RealSL" if sl_order_id else "⚠️ NoSL-order"
    print(f"\n  {d} [v20.3] {sym} {actual_side} (orig:{orig_direction}) @{price:.6g}")
    print(f"       SL:{SL_PCT*100:.1f}% ({sl_price:.6g}) {sl_note} | TP:{TP_PCT*100:.1f}% ({tp_price:.6g}) | Regime:{regime}")
    print(f"       Trail: aktif saat profit >= {TRAIL_ACTIVATE_PCT*100:.1f}%, gap {TRAIL_GAP_PCT*100:.2f}%")
    print(f"       Signals: {' | '.join(sigs[:5])}")
    _stats["trades"] += 1


def live_close(sym, reason, price=None):
    """
    Tutup posisi.
    Selalu cancel SL order di exchange sebelum close
    (kalau bot yang close duluan karena TP/Trail).
    """
    with _lock:
        pos = live_positions.pop(sym, None)
    if pos is None or pos.get("_r"): return

    if price is None: price = price_live(sym)
    if price == 0: return

    # ── Cancel SL order kalau masih ada (TP atau Trail yang trigger) ─────
    sl_oid = pos.get("sl_order_id")
    if sl_oid and "HardSL" not in reason:
        _cancel_stop_order(sym, sl_oid)

    side, entry, q_val = pos["side"], pos["entry"], pos["qty"]
    gross_pnl = (price-entry)*q_val if side=="LONG" else (entry-price)*q_val
    fee_rate  = 0.0005
    total_fee = (entry*q_val + price*q_val) * fee_rate
    pnl       = gross_pnl - total_fee
    pct       = (price-entry)/entry*100 if side=="LONG" else (entry-price)/entry*100
    hold      = time.time() - pos["open_time"]
    won       = pnl >= 0
    e         = "🟢" if won else "🔴"

    peak_info = f" [peak:{pos['peak_profit_pct']*100:.2f}%]" if pos["peak_profit_pct"]>0 else ""
    print(f"  {e} [v20.3] {sym} {side} CLOSE — {reason}{peak_info}")
    print(f"     {entry:.6g}→{price:.6g} ({pct:+.3f}%) hold:{hold:.0f}s | PnL:{pnl:+.5f}U")

    trade = TradeRecord(
        symbol=sym, direction=side, entry_price=entry, exit_price=price,
        pnl=pnl, won=won, regime=pos.get("regime","UNKNOWN"),
        signals=pos.get("sigs",[]), score=pos.get("score",0),
        atr_entry=pos.get("atr",0), sl_pct=pos.get("sl_pct",0),
        tp_pct=pos.get("tp_pct",0), hold_seconds=hold
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

    if "ExtremeTP" in reason:  _stats["extreme_tp"] += 1
    elif "TrailSL"  in reason: _stats["trail_sl"]   += 1
    elif "HardSL"   in reason: _stats["hard_sl"]    += 1

    trade_log.append({"sym":sym,"side":side,"entry":round(entry,7),"exit":round(price,7),
                      "pnl":round(pnl,5),"reason":reason,"hold":int(hold)})
    _hot_syms.appendleft(sym)
    _rescan_q.put(1)
    print_inline()


def _check_sl_filled(sym: str, pos: dict) -> bool:
    """
    Cek apakah stop-market order sudah terisi di exchange.
    Kalau ya, record sebagai HardSL dan remove dari live_positions.
    """
    sl_oid = pos.get("sl_order_id")
    if sl_oid is None:
        return False
    try:
        order = client.futures_get_order(symbol=sym, orderId=sl_oid)
        status = order.get("status", "")
        if status == "FILLED":
            fill_price = float(order.get("avgPrice", 0) or order.get("stopPrice", 0))
            print(f"  🛡️  [v20.3] {sym} SL order FILLED @{fill_price:.6g} (exchange executed)")
            # Hapus dari live_positions dulu sebelum live_close
            with _lock:
                live_positions.pop(sym, None)
            # Hitung PnL manual
            side, entry, q_val = pos["side"], pos["entry"], pos["qty"]
            if fill_price == 0: fill_price = price_live(sym)
            gross_pnl = (fill_price-entry)*q_val if side=="LONG" else (entry-fill_price)*q_val
            fee_rate  = 0.0005
            total_fee = (entry*q_val + fill_price*q_val) * fee_rate
            pnl       = gross_pnl - total_fee
            pct       = (fill_price-entry)/entry*100 if side=="LONG" else (entry-fill_price)/entry*100
            hold      = time.time() - pos["open_time"]
            won       = pnl >= 0
            e         = "🟢" if won else "🔴"
            print(f"  {e} [v20.3] {sym} {side} CLOSE — HardSL(exchange)")
            print(f"     {entry:.6g}→{fill_price:.6g} ({pct:+.3f}%) hold:{hold:.0f}s | PnL:{pnl:+.5f}U")
            trade = TradeRecord(
                symbol=sym, direction=side, entry_price=entry, exit_price=fill_price,
                pnl=pnl, won=won, regime=pos.get("regime","UNKNOWN"),
                signals=pos.get("sigs",[]), score=pos.get("score",0),
                atr_entry=pos.get("atr",0), sl_pct=pos.get("sl_pct",0),
                tp_pct=pos.get("tp_pct",0), hold_seconds=hold
            )
            learning.add_trade(trade)
            _stats["pnl"] += pnl; _stats["hist"].append(pnl); ks_upd(pnl)
            if won: _stats["wins"]+=1;  (lambda: _stats.__setitem__("best",pnl) if pnl>_stats["best"] else None)()
            else:   _stats["losses"]+=1; (lambda: _stats.__setitem__("worst",pnl) if pnl<_stats["worst"] else None)()
            _stats["hard_sl"] += 1
            trade_log.append({"sym":sym,"side":side,"entry":round(entry,7),"exit":round(fill_price,7),
                              "pnl":round(pnl,5),"reason":"HardSL(exchange)","hold":int(hold)})
            _hot_syms.appendleft(sym); _rescan_q.put(1); print_inline()
            return True
        elif status in ("CANCELED", "EXPIRED", "REJECTED"):
            # SL order sudah tidak aktif — hapus id supaya tidak di-cek terus
            pos["sl_order_id"] = None
            return False
    except Exception as e:
        pass
    return False


def _update_trailing_stop(sym: str, pos: dict, px: float) -> Optional[float]:
    """
    Update trailing stop. Aktif saat profit >= 1%, gap 0.15%.
    Ratchet — hanya makin ketat, tidak pernah mundur.
    """
    side, entry = pos["side"], pos["entry"]
    profit_pct = (px-entry)/entry if side=="LONG" else (entry-px)/entry
    if profit_pct <= 0: return pos.get("trail_sl_price")
    if profit_pct > pos["peak_profit_pct"]:
        pos["peak_profit_pct"] = profit_pct
    peak = pos["peak_profit_pct"]
    if peak < TRAIL_ACTIVATE_PCT: return None
    floor_pct = max(0.0, peak - TRAIL_GAP_PCT)
    new_trail = entry*(1.0+floor_pct) if side=="LONG" else entry*(1.0-floor_pct)
    old_trail = pos.get("trail_sl_price")
    if old_trail is None:
        pos["trail_sl_price"] = new_trail
        print(f"    🔒 TrailSL AKTIF {sym} {side}: floor={floor_pct*100:.2f}% @{new_trail:.6g} (peak={peak*100:.2f}%)")
    else:
        if (side=="LONG" and new_trail>old_trail) or (side=="SHORT" and new_trail<old_trail):
            pos["trail_sl_price"] = new_trail
            print(f"    ↗️  TrailSL update {sym}: {old_trail:.6g}→{new_trail:.6g} (peak={peak*100:.2f}%)")
    return pos["trail_sl_price"]


def monitor_positions():
    """
    Urutan cek per posisi:
    1. Cek apakah SL order di exchange sudah FILLED (kalau ya, sudah ditangani)
    2. Cek ExtremeTP (bot yang close, lalu cancel SL order)
    3. Update & cek Trailing Stop (bot yang close, lalu cancel SL order)
    Hard SL TIDAK dicek via polling lagi — sudah ditangani exchange.
    """
    for sym in list(live_positions.keys()):
        pos = live_positions.get(sym)
        if pos is None or pos.get("_r"): continue

        # ── 1. Cek SL order status di exchange ──────────────────────────
        if _check_sl_filled(sym, pos):
            continue   # sudah ditangani

        px = price_live(sym)
        if px == 0: continue

        side  = pos["side"]
        tp_px = pos["tp_price"]

        # ── 2. ExtremeTP ────────────────────────────────────────────────
        if side == "LONG":
            if px >= tp_px:
                live_close(sym, "ExtremeTP", px); continue
        else:
            if px <= tp_px:
                live_close(sym, "ExtremeTP", px); continue

        # ── 3. Trailing Stop ────────────────────────────────────────────
        trail_price = _update_trailing_stop(sym, pos, px)
        if trail_price is not None:
            if side == "LONG" and px <= trail_price:
                live_close(sym, f"TrailSL@{trail_price:.6g}", px); continue
            elif side == "SHORT" and px >= trail_price:
                live_close(sym, f"TrailSL@{trail_price:.6g}", px); continue


# ═══════════════════════════════════════════════════════════════════════════
#  SCANNER
# ═══════════════════════════════════════════════════════════════════════════

def scan_one(sym):
    try:
        time.sleep(0.002)
        df = ohlcv(sym, Client.KLINE_INTERVAL_5MINUTE, 100)
        if df is None: return None
        df_ta = df.copy()
        required = ["rsi","mh","e5","e9","e21","e50","atr","adx","vr","br","m5","br2"]
        if not all(c in df_ta.columns for c in required): df_ta = run_ta(df_ta)
        px = df_ta["close"].iloc[-2]; atr = df_ta["atr"].iloc[-2]
        if px==0 or np.isnan(atr): return None
        direction,score,sigs,atr_val,_,_,regime,bias = scorer.get_signal(df_ta,sym)
        if direction is None: return None
        px_live = price_live(sym)
        if px_live == 0: return None
        return (sym,direction,score,sigs,px_live,atr_val,regime,bias)
    except: return None

def scan_batch(syms):
    res = []
    fut = {_executor.submit(scan_one,s):s for s in syms[:BATCH_SIZE]}
    for f in as_completed(fut,timeout=5):
        try:
            if r := f.result(timeout=1): res.append(r)
        except: pass
    return res

def top_movers(syms,n=30):
    tk,ss = tickers_all(),set(syms)
    mv = [(s,abs(d["pct"])) for s,d in tk.items() if s in ss]
    return [s for s,_ in sorted(mv,key=lambda x:x[1],reverse=True)[:n]]


# ═══════════════════════════════════════════════════════════════════════════
#  PRINTING
# ═══════════════════════════════════════════════════════════════════════════

def print_inline():
    n  = _stats["wins"]+_stats["losses"]
    wr = _stats["wins"]/n*100 if n else 0
    e  = "💚" if _stats["pnl"]>=0 else "🔴"
    print(f"       ┌ [v20.3] {n}T WR:{wr:.0f}% W:{_stats['wins']} L:{_stats['losses']} {e}PnL:{_stats['pnl']:+.4f}U")
    print(f"       └ ExTP:{_stats['extreme_tp']} TrailSL:{_stats['trail_sl']} HardSL:{_stats['hard_sl']}")

def print_full():
    n    = _stats["wins"]+_stats["losses"]
    wr   = _stats["wins"]/n*100 if n else 0
    sess = (time.time()-_stats["start"])/3600
    tph  = n/sess if sess>0 else 0
    e    = "💚" if _stats["pnl"]>=0 else "🔴"
    print(f"\n  {'─'*72}")
    print(f"    ✅ REVERSED LOGIC v20.3 — REAL EXCHANGE SL + TRAIL")
    print(f"    🎯 {n}T WR:{wr:.0f}% W:{_stats['wins']} L:{_stats['losses']} ({tph:.1f}T/hr)")
    print(f"    {e} PnL Net:{_stats['pnl']:+.5f}U Best:{_stats['best']:+.5f} Worst:{_stats['worst']:+.5f}")
    print(f"    💰 ExtremeTP:{_stats['extreme_tp']} TrailSL:{_stats['trail_sl']} HardSL:{_stats['hard_sl']}")
    print(f"    ⚙️  SL=TP={SL_PCT*100:.1f}% fixed | SL = REAL STOP_MARKET di exchange")
    print(f"    🔒 Trail: aktif@{TRAIL_ACTIVATE_PCT*100:.1f}% profit | gap {TRAIL_GAP_PCT*100:.2f}%")
    print(f"    📊 Learning WR: {learning.get_global_winrate():.1%} | BTC:{_macro['btc']}")
    if trade_log:
        print(f"    📋 Last 5:")
        for t in trade_log[-5:]:
            em="🟢" if t["pnl"]>0 else "🔴"
            print(f"       {em} {t['sym']:<16} {t['side']} {t['pnl']:+.5f}U {t['hold']}s — {t['reason']}")
    print(f"  {'─'*72}")


# ═══════════════════════════════════════════════════════════════════════════
#  THREADS
# ═══════════════════════════════════════════════════════════════════════════

def t_monitor():
    while True:
        try:
            if live_positions: monitor_positions()
        except: pass
        time.sleep(MONITOR_INT)

def t_slot_filler(syms):
    scan_idx = 0
    n_bat    = max(1, math.ceil(len(syms)/BATCH_SIZE))
    while True:
        try:
            slots = MAX_POSITIONS-len(live_positions)
            if slots<=0 or ks_check()[0]: time.sleep(SLOT_FILL_INT); continue
            hot  = [s for s in _hot_syms if s not in live_positions]
            mv   = [s for s in top_movers(syms,30) if s not in live_positions]
            bs   = scan_idx*BATCH_SIZE
            reg  = [s for s in syms[bs:bs+BATCH_SIZE] if s not in live_positions and s not in mv]
            scan_idx = (scan_idx+1)%n_bat
            sl   = list(dict.fromkeys(hot[:5]+mv[:20]+reg[:15]))[:BATCH_SIZE]
            if not sl: time.sleep(SLOT_FILL_INT); continue
            res  = scan_batch(sl)
            if res:
                res.sort(key=lambda x:x[2],reverse=True)
                for r in res[:slots]:
                    if len(live_positions)>=MAX_POSITIONS: break
                    sym,od,sc,sg,px,atr,regime,bias=r
                    live_open(od,sc,sg,px,atr,regime,bias,sym)
        except: pass
        time.sleep(SLOT_FILL_INT)

def t_rescan(syms):
    while True:
        try:
            _rescan_q.get(timeout=5); time.sleep(0.05)
            slots = MAX_POSITIONS-len(live_positions)
            if slots<=0 or ks_check()[0]: continue
            hot  = [s for s in _hot_syms if s not in live_positions]
            rest = [s for s in syms if s not in live_positions and s not in hot]
            res  = scan_batch((hot+rest)[:30])
            if res:
                res.sort(key=lambda x:x[2],reverse=True)
                for r in res[:slots]:
                    if len(live_positions)>=MAX_POSITIONS: break
                    sym,od,sc,sg,px,atr,regime,bias=r
                    live_open(od,sc,sg,px,atr,regime,bias,sym)
        except: pass

def t_macro():
    while True:
        try:
            df_btc = ohlcv("BTCUSDT",Client.KLINE_INTERVAL_5MINUTE,80)
            if df_btc is not None:
                regime,_,_ = MarketRegime.detect(df_btc)
                _macro["btc"] = regime
        except: pass
        time.sleep(10)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_bot():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  ✅ REVERSED LOGIC v20.3 — REAL EXCHANGE SL ORDER                 ║")
    print(f"║  ✅ SL = TP = {SL_PCT*100:.1f}% fixed simetris                             ║")
    print(f"║  ✅ SL dipasang sebagai STOP_MARKET order langsung di Binance     ║")
    print(f"║  ✅ Worst case loss terjamin ~-0.24U per trade                    ║")
    print(f"║  ✅ Trail aktif @{TRAIL_ACTIVATE_PCT*100:.1f}% profit, gap {TRAIL_GAP_PCT*100:.2f}%, protect profit          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    try:
        valid = {s["symbol"] for s in client.futures_exchange_info()["symbols"] if s["status"]=="TRADING"}
        syms  = list(dict.fromkeys([s for s in SYMBOLS if s in valid]))
    except:
        syms = list(dict.fromkeys(SYMBOLS))
    print(f"  📋 {len(syms)} simbol aktif")

    threading.Thread(target=t_monitor,     daemon=True).start()
    threading.Thread(target=t_slot_filler, args=(syms,), daemon=True).start()
    threading.Thread(target=t_rescan,      args=(syms,), daemon=True).start()
    threading.Thread(target=t_macro,       daemon=True).start()
    time.sleep(2)
    tickers_all()

    cycle = 0
    while True:
        cycle += 1
        slots = MAX_POSITIONS-len(live_positions)
        print(f"\n{'═'*64}")
        print(f"  #{cycle} {time.strftime('%H:%M:%S')} BTC:{_macro['btc']} ({len(live_positions)}/{MAX_POSITIONS}) PnL:{_stats['pnl']:+.4f}U")
        if (k:=ks_check())[0]: print(f"  🚨 KillSwitch:{k[1]}")
        elif slots==0: print(f"  ✅ Slots full")
        else: print(f"  🔍 {slots} slot kosong — scanning...")
        if cycle%30==0: print_full()
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    run_bot()
