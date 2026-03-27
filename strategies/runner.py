import sys
import os
import pandas as pd
import numpy as np
import traceback

# Global data store
_prefetched_data = {}

def set_prefetched_data(data_dict):
    global _prefetched_data
    _prefetched_data = data_dict

# ────────────────────────────────────────────────────────────
# ORTAK YARDIMCI FONKSİYONLAR
# ────────────────────────────────────────────────────────────

def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def _macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def _sma(close, period):
    return close.rolling(window=period).mean()

def _ema(close, period):
    return close.ewm(span=period, adjust=False).mean()

def _bollinger(close, period=20, std=2):
    mid = close.rolling(period).mean()
    band = close.rolling(period).std()
    return mid - std*band, mid, mid + std*band

def _atr(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _price_levels(df, atr_mult_stop=2.0, target_pct=0.05):
    """
    Hisse için giriş, stop ve hedef fiyat hesaplar.
    - entry_price: son kapanış fiyatı
    - stop_loss: ATR tabanlı dinamik stop (son 20 günün en düşüğü - ATR ile hesaplanır)
    - target_price: %5 kar hedefi (ATR veya dirençle sınırlandırılır)
    """
    try:
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()

        entry = float(close.iloc[-1])
        atr = float(_atr(high, low, close, 14).iloc[-1])
        low20 = float(low.rolling(20).min().iloc[-1])

        # Stop: son 20 günün en düşüğü veya ATR*2 (hangisi daha yakınsa)
        atr_stop = entry - atr_mult_stop * atr
        stop = max(atr_stop, low20 * 0.99)

        # Hedef: %5 veya son 20 günün en yüksek seviyesi (hangisi yakınsa)
        high20 = float(high.rolling(20).max().iloc[-1])
        pct_target = entry * (1 + target_pct)
        target = min(pct_target, high20) if high20 > entry else pct_target

        risk = entry - stop
        reward = target - entry
        rr = round(reward / risk, 2) if risk > 0 else 0.0

        change_1d = round((entry / float(close.iloc[-2]) - 1) * 100, 2)

        return {
            "entry_price": round(entry, 2),
            "stop_loss": round(stop, 2),
            "target_price": round(target, 2),
            "risk_reward": rr,
            "change_1d": change_1d,
            "atr": round(atr, 2),
        }
    except Exception:
        return {
            "entry_price": 0, "stop_loss": 0, "target_price": 0,
            "risk_reward": 0, "change_1d": 0, "atr": 0
        }

# ────────────────────────────────────────────────────────────
# STRAT 1 – BİRLEŞİK TARAMA
# ────────────────────────────────────────────────────────────
def _run_birlesik(data_dict):
    results = []
    for symbol, df in data_dict.items():
        try:
            if len(df) < 30:
                continue
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            volume = df['Volume'].squeeze()

            rsi = _rsi(close).iloc[-1]
            macd, sig = _macd(close)
            bb_low, bb_mid, bb_high = _bollinger(close)

            price = float(close.iloc[-1])
            bb_l = float(bb_low.iloc[-1])
            bb_m = float(bb_mid.iloc[-1])
            macd_v = float(macd.iloc[-1])
            sig_v = float(sig.iloc[-1])
            macd_prev = float(macd.iloc[-2])
            sig_prev = float(sig.iloc[-2])
            vol_avg = float(volume.rolling(10).mean().iloc[-1])
            vol_last = float(volume.iloc[-1])

            score = 0
            reasons = []

            change_10d = (price / float(close.iloc[-11]) - 1) * 100
            change_5d = (price / float(close.iloc[-6]) - 1) * 100

            if change_10d < -5:
                score += 2
                reasons.append(f"10g düşüş %{change_10d:.1f}")
            if change_5d > 1:
                score += 2
                reasons.append(f"5g toparlanma +%{change_5d:.1f}")
            if 25 < rsi < 45:
                score += 2
                reasons.append(f"RSI:{rsi:.0f} dönüş bölgesi")
            if macd_v > sig_v and macd_prev <= sig_prev:
                score += 3
                reasons.append("MACD al sinyali")
            elif macd_v > macd_prev:
                score += 1
                reasons.append("MACD yukarı")
            if price < bb_m and price > bb_l:
                score += 2
                reasons.append("BB dip bölgesi")
            if vol_avg > 0 and vol_last > vol_avg * 1.5:
                score += 2
                reasons.append(f"Hacim artışı {vol_last/vol_avg:.1f}x")

            if score >= 6:
                levels = _price_levels(df)
                results.append({"symbol": symbol, "score": score,
                                 "details": " | ".join(reasons), **levels})
        except Exception:
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]


# ────────────────────────────────────────────────────────────
# STRAT 2 – GELİŞMİŞ YEŞİL TARAMA
# ────────────────────────────────────────────────────────────
def _run_gelismis_yesil(data_dict):
    results = []
    for symbol, df in data_dict.items():
        try:
            if len(df) < 50:
                continue
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            volume = df['Volume'].squeeze()

            rsi = _rsi(close)
            macd, sig = _macd(close)
            sma20 = _sma(close, 20)
            sma50 = _sma(close, 50)
            ema9 = _ema(close, 9)

            price = float(close.iloc[-1])
            rsi_now = float(rsi.iloc[-1])
            rsi_prev = float(rsi.iloc[-2])
            macd_now = float(macd.iloc[-1])
            sig_now = float(sig.iloc[-1])
            macd_prev = float(macd.iloc[-2])
            sig_prev = float(sig.iloc[-2])
            sma20_now = float(sma20.iloc[-1])
            sma50_now = float(sma50.iloc[-1])
            ema9_now = float(ema9.iloc[-1])
            vol_avg = float(volume.rolling(20).mean().iloc[-1])
            vol_last = float(volume.iloc[-1])

            score = 0
            reasons = []

            if price > ema9_now:
                score += 2; reasons.append("Fiyat EMA9 üstünde")
            if sma20_now > sma50_now:
                score += 2; reasons.append("SMA20>SMA50 (trendin yukarı)")
            if 30 < rsi_now < 65 and rsi_now > rsi_prev:
                score += 2; reasons.append(f"RSI:{rsi_now:.0f} yükseliyor")
            if macd_now > sig_now and macd_prev <= sig_prev:
                score += 4; reasons.append("MACD kesişim al sinyali")
            elif macd_now > sig_now and macd_now > 0:
                score += 2; reasons.append("MACD pozitif bölgede")
            change_1d = (float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100
            if vol_avg > 0 and vol_last > vol_avg * 1.2 and change_1d > 0:
                score += 2; reasons.append(f"Hacimli yükseliş ({vol_last/vol_avg:.1f}x)")

            if score >= 7:
                levels = _price_levels(df)
                results.append({"symbol": symbol, "score": score,
                                 "details": " | ".join(reasons), **levels})
        except Exception:
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]


# ────────────────────────────────────────────────────────────
# STRAT 3 – MİNERVİNİ TREND ŞABLONU
# ────────────────────────────────────────────────────────────
def _run_minervini(data_dict):
    results = []
    for symbol, df in data_dict.items():
        try:
            if len(df) < 200:
                continue
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()

            price = float(close.iloc[-1])
            ma50 = float(_sma(close, 50).iloc[-1])
            ma150 = float(_sma(close, 150).iloc[-1])
            ma200 = float(_sma(close, 200).iloc[-1])
            ma200_prev = float(_sma(close, 200).iloc[-21])
            rsi_val = float(_rsi(close).iloc[-1])
            high_52w = float(high.rolling(252).max().iloc[-1])
            low_52w = float(low.rolling(252).min().iloc[-1])

            score = 0
            checks = []

            if price > ma150 and price > ma200:
                score += 1; checks.append("Fiyat>MA150,MA200")
            if ma150 > ma200:
                score += 1; checks.append("MA150>MA200")
            if ma200 > ma200_prev:
                score += 1; checks.append("MA200 yükseliyor")
            if ma50 > ma150 and ma50 > ma200:
                score += 1; checks.append("MA50>MA150,MA200")
            if price > ma50:
                score += 1; checks.append("Fiyat>MA50")
            if price >= low_52w * 1.25:
                score += 1; checks.append("52h dibinin +%25 üstünde")
            if price >= high_52w * 0.75:
                score += 1; checks.append("52h zirvesinin %75 üstünde")
            if rsi_val >= 70:
                score += 1; checks.append(f"RSI≥70 ({rsi_val:.0f})")

            if score >= 6:
                levels = _price_levels(df, atr_mult_stop=2.5, target_pct=0.08)
                results.append({"symbol": symbol, "score": score,
                                 "details": " | ".join(checks), **levels})
        except Exception:
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]


# ────────────────────────────────────────────────────────────
# STRAT 4 – MUM FORMASYONU
# ────────────────────────────────────────────────────────────
def _run_mum(data_dict):
    results = []
    for symbol, df in data_dict.items():
        try:
            if len(df) < 10:
                continue
            o = df['Open'].squeeze(); h = df['High'].squeeze()
            l = df['Low'].squeeze(); c = df['Close'].squeeze()

            patterns = []
            last_o = float(o.iloc[-1]); last_c = float(c.iloc[-1])
            last_h = float(h.iloc[-1]); last_l = float(l.iloc[-1])
            prev_o = float(o.iloc[-2]); prev_c = float(c.iloc[-2])
            prev_h = float(h.iloc[-2])

            body = abs(last_c - last_o)
            total = last_h - last_l
            if total == 0:
                continue
            upper = last_h - max(last_o, last_c)
            lower = min(last_o, last_c) - last_l

            if (prev_c < prev_o and last_c > last_o and
                    body <= 0.3 * total and lower >= 2 * body and upper <= 0.1 * total):
                patterns.append("Çekiç (Hammer)")
            if (prev_c < prev_o and last_c > last_o and
                    last_c > prev_o and last_o < prev_c and
                    (last_c - last_o) > (prev_o - prev_c) * 1.1):
                patterns.append("Yutan Boğa (Bullish Engulfing)")
            if (last_c > last_o and last_c > prev_h and
                    (last_c - last_o) / total > 0.6 and prev_c < prev_o):
                patterns.append("Boğa Hamlesi (Bullish Thrust)")
            if len(c) >= 3:
                c1=float(c.iloc[-1]); c2=float(c.iloc[-2]); c3=float(c.iloc[-3])
                o1=float(o.iloc[-1]); o2=float(o.iloc[-2]); o3=float(o.iloc[-3])
                if c1>o1 and c2>o2 and c3>o3 and c1>c2>c3 and o1>o2>o3:
                    patterns.append("Üç Beyaz Asker (Three White Soldiers)")

            if patterns:
                levels = _price_levels(df)
                results.append({"symbol": symbol, "score": len(patterns) * 25,
                                 "details": " | ".join(patterns), **levels})
        except Exception:
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]


# ────────────────────────────────────────────────────────────
# STRAT 5 – SIKIŞMİŞ HİSSELER (Bollinger Squeeze)
# ────────────────────────────────────────────────────────────
def _run_sikismis(data_dict):
    results = []
    for symbol, df in data_dict.items():
        try:
            if len(df) < 30:
                continue
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()

            bb_low, bb_mid, bb_high = _bollinger(close, 20)
            bb_width = (bb_high - bb_low) / (bb_mid + 1e-10)
            current_width = float(bb_width.iloc[-1])
            avg_width = float(bb_width.rolling(50).mean().iloc[-1])

            kc_mid = _ema(close, 20)
            atr = _atr(high, low, close, 14)
            kc_low = kc_mid - 1.5 * atr
            kc_high = kc_mid + 1.5 * atr
            squeeze = float(bb_low.iloc[-1]) > float(kc_low.iloc[-1]) and float(bb_high.iloc[-1]) < float(kc_high.iloc[-1])

            score = 0; reasons = []
            if avg_width > 0 and current_width < avg_width * 0.7:
                score += 4; reasons.append(f"BB sıkışması (genişlik:{current_width:.3f} < ort:{avg_width:.3f})")
            if squeeze:
                score += 4; reasons.append("Keltner squeeze aktif")
            price_range = float((close.rolling(5).max() / close.rolling(5).min() - 1).iloc[-1] * 100)
            if price_range < 3:
                score += 2; reasons.append(f"5g düşük oynaklık (%{price_range:.1f})")

            if score >= 6:
                levels = _price_levels(df)
                results.append({"symbol": symbol, "score": score,
                                 "details": " | ".join(reasons), **levels})
        except Exception:
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]


# ────────────────────────────────────────────────────────────
# STRAT 6 – TEPKİ TARAMASI 1
# ────────────────────────────────────────────────────────────
def _run_tepki1(data_dict):
    results = []
    for symbol, df in data_dict.items():
        try:
            if len(df) < 20:
                continue
            close = df['Close'].squeeze()
            volume = df['Volume'].squeeze()

            rsi = _rsi(close)
            bb_low, bb_mid, bb_high = _bollinger(close)
            macd, sig = _macd(close)

            price = float(close.iloc[-1])
            rsi_now = float(rsi.iloc[-1]); rsi_prev = float(rsi.iloc[-2])
            change_5d = (price / float(close.iloc[-6]) - 1) * 100
            change_1d = (price / float(close.iloc[-2]) - 1) * 100
            vol_avg = float(volume.rolling(10).mean().iloc[-1])
            vol_last = float(volume.iloc[-1])
            macd_now = float(macd.iloc[-1]); macd_prev = float(macd.iloc[-2])

            score = 0; reasons = []
            if change_5d < -5:
                score += 3; reasons.append(f"Düşüş:%{change_5d:.2f}")
            elif change_5d < -3:
                score += 2; reasons.append(f"Düşüş:%{change_5d:.2f}")
            if change_1d > 1.5:
                score += 4; reasons.append(f"Güçlü tepki +%{change_1d:.2f}")
            elif change_1d > 0.5:
                score += 2; reasons.append(f"Tepki +%{change_1d:.2f}")
            if rsi_prev < 35 and rsi_now > rsi_prev:
                score += 4; reasons.append(f"RSI dönüşü ({rsi_prev:.0f}->{rsi_now:.0f})")
            elif rsi_now < 40:
                score += 2; reasons.append(f"RSI aşırı satım ({rsi_now:.0f})")
            if price < float(bb_low.iloc[-1]) * 1.02:
                score += 2; reasons.append("BB alt bandına yakın")
            if macd_now > macd_prev:
                score += 2; reasons.append("MACD yukarı")
            if vol_avg > 0 and vol_last > vol_avg * 1.3:
                score += 2; reasons.append(f"Hacim artışı {vol_last/vol_avg:.1f}x")

            if score >= 10:
                levels = _price_levels(df, atr_mult_stop=1.5, target_pct=0.05)
                results.append({"symbol": symbol, "score": score,
                                 "details": f"Düşüş:%{change_5d:.2f} -> Toparlanma Sinyali", **levels})
        except Exception:
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:15]


# ────────────────────────────────────────────────────────────
# STRAT 7 – TEPKİ TARAMASI 2 (EMA Desteği)
# ────────────────────────────────────────────────────────────
def _run_tepki2(data_dict):
    results = []
    for symbol, df in data_dict.items():
        try:
            if len(df) < 50:
                continue
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()

            ema20 = _ema(close, 20); ema50 = _ema(close, 50)
            rsi = _rsi(close)

            price = float(close.iloc[-1]); prev_price = float(close.iloc[-2])
            ema20_now = float(ema20.iloc[-1]); ema50_now = float(ema50.iloc[-1])
            rsi_now = float(rsi.iloc[-1]); rsi_prev = float(rsi.iloc[-2])
            change_5d = (price / float(close.iloc[-6]) - 1) * 100

            score = 0; reasons = []
            touched_ema20 = abs(float(low.iloc[-2]) - ema20_now) / ema20_now < 0.02
            touched_ema50 = abs(float(low.iloc[-2]) - ema50_now) / ema50_now < 0.02
            if touched_ema20 and price > prev_price:
                score += 5; reasons.append(f"EMA20 desteğinden döndü ({ema20_now:.2f})")
            if touched_ema50 and price > prev_price:
                score += 5; reasons.append(f"EMA50 desteğinden döndü ({ema50_now:.2f})")
            if rsi_now > rsi_prev and rsi_now < 60:
                score += 3; reasons.append(f"RSI yükseliyor ({rsi_prev:.0f}->{rsi_now:.0f})")
            if change_5d < -3 and price > prev_price:
                score += 3; reasons.append(f"Düşüş sonrası tepki (%{change_5d:.1f})")

            if score >= 8:
                levels = _price_levels(df, atr_mult_stop=1.5, target_pct=0.05)
                results.append({"symbol": symbol, "score": score,
                                 "details": " | ".join(reasons), **levels})
        except Exception:
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:15]


# ────────────────────────────────────────────────────────────
# STRAT 8 – YEŞİL TARAMA
# ────────────────────────────────────────────────────────────
def _run_yesil(data_dict):
    results = []
    for symbol, df in data_dict.items():
        try:
            if len(df) < 30:
                continue
            close = df['Close'].squeeze()
            volume = df['Volume'].squeeze()

            rsi = _rsi(close); macd, sig = _macd(close); sma20 = _sma(close, 20)
            price = float(close.iloc[-1])
            rsi_now = float(rsi.iloc[-1])
            macd_now = float(macd.iloc[-1]); sig_now = float(sig.iloc[-1])
            macd_prev = float(macd.iloc[-2]); sig_prev = float(sig.iloc[-2])
            sma20_now = float(sma20.iloc[-1])
            change_20d = (price / float(close.iloc[-21]) - 1) * 100
            change_1d = (price / float(close.iloc[-2]) - 1) * 100
            vol_now = float(volume.iloc[-1]); vol_avg = float(volume.rolling(20).mean().iloc[-1])

            score = 0; reasons = []
            if change_20d < -10 and change_1d > 1:
                score += 5; reasons.append(f"Düşüş sonrası yeşil başlangıç (%{change_20d:.1f})")
            if macd_now > sig_now and macd_prev <= sig_prev:
                score += 5; reasons.append("MACD yeşil kesiş")
            if 35 < rsi_now < 65:
                score += 2; reasons.append(f"RSI:{rsi_now:.0f}")
            prev_close = float(close.iloc[-2])
            if prev_close < sma20_now and price > sma20_now:
                score += 4; reasons.append("SMA20 yukarı kırdı")
            if vol_avg > 0 and vol_now > vol_avg * 1.5 and change_1d > 0:
                score += 3; reasons.append(f"Hacimli yükseliş {vol_now/vol_avg:.1f}x")

            if score >= 9:
                levels = _price_levels(df)
                results.append({"symbol": symbol, "score": score,
                                 "details": " | ".join(reasons), **levels})
        except Exception:
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]


# ────────────────────────────────────────────────────────────
# STRAT 9 – MOMENTUM
# ────────────────────────────────────────────────────────────
_momentum_predictor = None

def get_momentum_predictor():
    global _momentum_predictor
    if _momentum_predictor is None:
        import sys
        import os
        module_path = os.path.join(os.path.dirname(__file__), "momentum_module")
        if module_path not in sys.path:
            sys.path.append(module_path)
        from .momentum_module.momentum_predictor import MomentumPredictor
        _momentum_predictor = MomentumPredictor()
    return _momentum_predictor


def _run_momentum(data_dict):
    try:
        predictor = get_momentum_predictor()
        # Automatically discover >30% rising stocks from last month and train model
        predictor.setup_and_train_if_needed(data_dict)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Momentum predictor setup error: {e}")
        return []

    # ── Orijinal momentum_predictor strateji eşikleri ──────────────────
    MIN_LAYER1_PROB    = 60.0   # ml_prob eşiği   (Katman 1)
    MIN_PATTERN_SCORE  = 60.0   # pattern eşiği   (Katman 1)
    MIN_SELF_SIM       = 60.0   # self-sim eşiği  (Katman 2)

    results = []
    for symbol, df in data_dict.items():
        try:
            if len(df) < 100:
                continue

            prediction = predictor.predict_momentum(symbol, df)
            if not prediction:
                continue

            layer1_prob   = prediction['ml_probability']
            pattern_score = prediction['pattern_score']
            self_sim      = prediction['self_similarity']

            # ── KATMAN 1: ikisi de geçmeli (AND) ───────────────────────
            if layer1_prob < MIN_LAYER1_PROB or pattern_score < MIN_PATTERN_SCORE:
                continue

            # ── KATMAN 2: self-benzerlik eşiği ─────────────────────────
            # self_sim == 0 → yetersiz geçmiş veri, Katman 2 atlanır
            if self_sim > 0 and self_sim < MIN_SELF_SIM:
                continue

            # ── Birleşik skor (orijinal formül) ────────────────────────
            # combined = prob*0.4 + pattern*0.3 + self_sim*0.3
            # (orijinalde similarity_avg*0.1 de vardı; burada pattern'a dahil)
            combined_score = (
                layer1_prob   * 0.4 +
                pattern_score * 0.3 +
                self_sim      * 0.3
            )

            reasons = [
                f"ML İhtimali: %{layer1_prob:.1f}",
                f"Patern Uyumu: {pattern_score:.1f}/100",
            ]
            if self_sim > 0:
                reasons.append(f"Kendi Geçmişi (Self): {self_sim:.1f}/100")

            levels = _price_levels(df, atr_mult_stop=1.5, target_pct=0.08)
            results.append({
                "symbol": symbol,
                "score":  combined_score,
                "details": f"Momentum Kombine Skor: {combined_score:.1f}/100 | " + " | ".join(reasons),
                **levels
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:15]


# ────────────────────────────────────────────────────────────
# STRAT 10 – TAVAN TAHMİNİ
# ────────────────────────────────────────────────────────────
_tavan_predictor = None

def get_tavan_predictor():
    global _tavan_predictor
    if _tavan_predictor is None:
        import sys
        import os
        module_path = os.path.join(os.path.dirname(__file__), "tavan_module")
        if module_path not in sys.path:
            sys.path.append(module_path)
        from strategies.tavan_module.tavan_predictor import CeilingPredictor
        _tavan_predictor = CeilingPredictor()
    return _tavan_predictor


def _run_tavan(data_dict):
    try:
        predictor = get_tavan_predictor()
        from strategies.tavan_module.advanced_indicators import AdvancedIndicators
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Tavan predictor error: {e}")
        return []

    results = []
    for symbol, df in data_dict.items():
        try:
            if len(df) < 100:  # ML models require at least 100 days of history
                continue
            
            # 1. Tüm indikatörleri heapla (70+)
            indicators = AdvancedIndicators.calculate_all_indicators(df)
            
            # 2. Patern benzerliği skoru
            pattern_score = predictor.pattern_analyzer.calculate_similarity_score(indicators)
            
            # 3. ML Lojistik Regresyon / Ensemble ihtimali
            ml_probability = predictor.ml_trainer.predict_ceiling_probability(indicators)
            
            # 4. Kendi geçmişine (Self Analyzer) göre benzerlik
            symbol_clean = symbol.replace('.IS', '')
            self_analysis = predictor.self_analyzer.calculate_self_similarity(symbol_clean, indicators)
            self_similarity = self_analysis['similarity_score']
            self_confidence_emoji = self_analysis['confidence_emoji']
            
            # 5. Kombine Skor Matematiği
            combined_score = (ml_probability * 100 * 0.4) + (pattern_score * 0.3) + (self_similarity * 0.3)
            
            # Karar koşulları
            if ml_probability >= 0.60 or combined_score >= 60:
                reasons = []
                reasons.append(f"ML İhtimali: %{ml_probability*100:.1f}")
                reasons.append(f"Genel Patern Uyumu: {pattern_score:.1f}/100")
                reasons.append(f"Kendi Geçmişi (Self): {self_similarity:.1f}/100 {self_confidence_emoji}")
                
                # Detaylar
                pre_sqz = indicators.get('pre_ceiling_squeeze', 0)
                vol_break = indicators.get('volume_breakout_score', 0)
                rsi = indicators.get('rsi_14', 0)
                
                if pre_sqz > 50: reasons.append(f"Tavan Öncesi Sıkışma: {pre_sqz:.0f}")
                if vol_break > 60: reasons.append(f"Hacim Breakout: {vol_break:.0f}")
                if rsi > 70: reasons.append(f"RSI Yüksek: {rsi:.0f}")
                
                levels = _price_levels(df, atr_mult_stop=2.0, target_pct=0.07)
                results.append({
                    "symbol": symbol,
                    "score": combined_score,
                    "details": f"Tavan Kombine Skor: {combined_score:.1f}/100 | " + " | ".join(reasons),
                    **levels
                })
        except Exception:
            continue
            
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:15]


# ────────────────────────────────────────────────────────────
# STRAT 11 – SPEK TARAYICI (son/main.py SpekAnalyzer mantığı)
# Minervini Trend Şablonu + Spek Uçuş Puanı kombinasyonu
# ────────────────────────────────────────────────────────────
def _run_spek(data_dict):
    results = []
    MIN_AVG_VOLUME_TL = 1_000_000
    MAX_UPPER_WICK_RATIO = 0.4
    MAX_CONSECUTIVE_UP_DAYS = 5
    MAX_RVOL_FOR_SAFETY = 5.0
    
    for symbol, df in data_dict.items():
        try:
            if len(df) < 260:
                continue

            # 1. Temel Fiyat Verileri ve Göstergeler
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            open_ = df['Open'].squeeze()
            volume = df['Volume'].squeeze()

            price = float(close.iloc[-1])
            prev_close = float(close.iloc[-2]) if len(close) > 1 else price
            open_val = float(open_.iloc[-1])
            high_val = float(high.iloc[-1])
            low_val = float(low.iloc[-1])

            ma50 = float(_sma(close, 50).iloc[-1])
            ma150 = float(_sma(close, 150).iloc[-1])
            ma200 = float(_sma(close, 200).iloc[-1])
            ma200_prev = float(_sma(close, 200).iloc[-31]) if len(close) > 230 else ma200
            
            high_52w = float(close.rolling(260).max().iloc[-1])
            low_52w = float(close.rolling(260).min().iloc[-1])

            tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
            atr20 = float(tr.rolling(20).mean().iloc[-1])
            atr60 = float(tr.rolling(60).mean().iloc[-1])
            vcp_ratio = atr20 / atr60 if atr60 > 0 else 1.0

            high_30d = float(high.rolling(30).max().iloc[-1])
            low_30d = float(low.rolling(30).min().iloc[-1])
            range_30d = (high_30d - low_30d) / low_30d * 100 if low_30d > 0 else 0

            vol_ma50 = float(volume.rolling(50).mean().iloc[-1])
            recent_volume_avg = float(volume.iloc[-5:].mean())

            returns = close.pct_change()

            # 2. Minervini Trend Şablonu
            cond_trend = [
                price > ma150, price > ma200, ma150 > ma200,
                (ma200 - ma200_prev) > 0, 
                ma50 > ma150, price > ma50,
                price >= high_52w * 0.75, price >= low_52w * 1.30
            ]
            trend_score = sum(cond_trend)
            vcp_score = sum([vcp_ratio < 0.95, range_30d < 30])
            volume_score = 1 if recent_volume_avg > vol_ma50 * 0.8 else 0

            # Sadece trend'i sağlam hisselerde spek tara
            if trend_score < 6:
                continue
                
            # 3. Spek Derin Feature Hesaplamaları
            def calc_roc(p): 
                return (price / float(close.iloc[-p-1]) - 1) * 100 if len(close) > p else 0
            
            roc_1d = calc_roc(1)
            roc_3d = calc_roc(3)
            roc_5d = calc_roc(5)
            roc_10d = calc_roc(10)

            high_20d = high.rolling(20).max()
            high_20d_max = float(high_20d.iloc[-1])
            high_55d_max = float(high.rolling(55).max().iloc[-1])

            dist_20d = (price / high_20d_max - 1) * 100 if high_20d_max > 0 else 0
            dist_55d = (price / high_55d_max - 1) * 100 if high_55d_max > 0 else 0

            breakout_days = close > high_20d.shift(1)
            breakout_vol_ratio = 0.0
            if breakout_days.iloc[-5:].any():
                last_breakout_idx = breakout_days[breakout_days].index[-1]
                b_vol = float(volume.loc[last_breakout_idx])
                avg_b_vol = float(volume.rolling(20).mean().loc[last_breakout_idx])
                breakout_vol_ratio = b_vol / avg_b_vol if avg_b_vol > 0 else 0

            vol_mean_20 = float(volume.rolling(20).mean().iloc[-1])
            rvol_20 = float(volume.iloc[-1] / vol_mean_20) if vol_mean_20 > 0 else 1.0
            vol_mean_60 = float(volume.rolling(60).mean().iloc[-1])
            vol_std_60 = float(volume.rolling(60).std().iloc[-1])
            vol_zscore = (float(volume.iloc[-1]) - vol_mean_60) / vol_std_60 if vol_std_60 > 0 else 0

            up_vol = float(volume[returns > 0].tail(20).mean()) if not volume[returns > 0].empty else 1.0
            down_vol = float(volume[returns < 0].tail(20).mean()) if not volume[returns < 0].empty else 1.0
            up_vol_dom = up_vol / down_vol if down_vol > 0 else 1.0

            atr_pct = (atr20 / price) * 100 if price > 0 else 0

            bb_period = 20
            bb_middle = close.rolling(bb_period).mean()
            bb_std_dev = close.rolling(bb_period).std()
            bb_upper = bb_middle + (bb_std_dev * 2)
            bb_lower = bb_middle - (bb_std_dev * 2)
            bb_width = float(((bb_upper - bb_lower) / bb_middle * 100).iloc[-1])

            atr_20_prev = float(tr.rolling(20).mean().iloc[-6])
            atr_expansion = (atr20 / atr_20_prev - 1) * 100 if atr_20_prev > 0 else 0

            avg_vol_tl = float((close * volume).tail(20).mean())
            
            slippage_risk = 0
            if avg_vol_tl < MIN_AVG_VOLUME_TL: slippage_risk = 10
            elif avg_vol_tl < MIN_AVG_VOLUME_TL * 3: slippage_risk = 5

            candle_body_top = max(open_val, price)
            candle_range = high_val - low_val
            if candle_range == 0: candle_range = 1e-10
            upper_wick_ratio = (high_val - candle_body_top) / candle_range

            close_pos = (price - low_val) / candle_range
            blowoff_risk = 10 if float(returns.iloc[-1]) > 0.05 and rvol_20 > 3 and close_pos < 0.3 else 0

            gap_up = (open_val - prev_close) / prev_close if prev_close > 0 else 0
            gap_reversal = 10 if gap_up > 0.03 and price < open_val else 0

            consecutive_up = 0
            for i in range(1, min(10, len(returns))):
                if float(returns.iloc[-i]) > 0: consecutive_up += 1
                else: break
            mean_reversion_risk = 10 if consecutive_up > MAX_CONSECUTIVE_UP_DAYS else 0

            market_regime_score = 5

            # 4. Spek Mimarisi Uçuş Skoru
            mom_sc = 0
            if roc_5d > 10: mom_sc += 10
            elif roc_5d > 5: mom_sc += 5
            if dist_20d > -2: mom_sc += 10
            elif dist_20d > -5: mom_sc += 5
            if breakout_vol_ratio > 1.5: mom_sc += 10
            elif breakout_vol_ratio > 1.2: mom_sc += 5
            mom_sc = min(mom_sc, 30)

            vol_sc = 0
            if 1.5 < rvol_20 < 4: vol_sc += 10
            elif 1.2 < rvol_20 < 1.5: vol_sc += 5
            if vol_zscore > 2: vol_sc += 8
            elif vol_zscore > 1: vol_sc += 4
            if up_vol_dom > 1.3: vol_sc += 7
            elif up_vol_dom > 1.1: vol_sc += 3
            vol_sc = min(vol_sc, 25)

            sqz_sc = 0
            if atr_expansion > 10: sqz_sc += 10
            elif atr_expansion > 5: sqz_sc += 5
            if bb_width < 5: sqz_sc += 10
            elif bb_width < 8: sqz_sc += 5
            sqz_sc = min(sqz_sc, 20)

            liq_sc = 0
            if avg_vol_tl > MIN_AVG_VOLUME_TL * 5: liq_sc = 10
            elif avg_vol_tl > MIN_AVG_VOLUME_TL * 2: liq_sc = 7
            elif avg_vol_tl > MIN_AVG_VOLUME_TL: liq_sc = 5
            liq_sc = max(liq_sc - slippage_risk, 0)

            mkt_sc = min(market_regime_score * 1.5, 15)

            pen_sc = blowoff_risk + gap_reversal + mean_reversion_risk
            if upper_wick_ratio > MAX_UPPER_WICK_RATIO: pen_sc += 10
            if rvol_20 > MAX_RVOL_FOR_SAFETY: pen_sc += 10
            pen_sc = min(pen_sc, 30)

            flight_score = max(0, min(100, mom_sc + vol_sc + sqz_sc + liq_sc + mkt_sc - pen_sc))

            # 5. No-Trade Taraması İçin Ret Kuralları
            reasons = []
            if rvol_20 > 5 and upper_wick_ratio > 0.4:
                reasons.append("PUMP RİSKİ")
            if avg_vol_tl < MIN_AVG_VOLUME_TL and atr_pct > 5:
                reasons.append("SLİPPAGE RİSKİ")
            if consecutive_up > 5 and rvol_20 < 1:
                reasons.append("DAĞITIM RİSKİ")
            if blowoff_risk > 0:
                reasons.append("BLOW-OFF TOP")
            if flight_score < 40:
                reasons.append("DÜŞÜK SKOR")

            if len(reasons) > 0:
                continue

            # 6. Analiz Cümleleri (Neden Uçabilir)
            positives = []
            if roc_5d > 10: positives.append(f"Momentum 5G: +%{roc_5d:.1f}")
            if rvol_20 > 1.5: positives.append(f"RVOL: {rvol_20:.1f}x")
            if dist_20d > -2: positives.append(f"Zirveye %{abs(dist_20d):.1f}")
            if atr_expansion > 10: positives.append(f"ATR Exp: +%{atr_expansion:.1f}")
            if breakout_vol_ratio > 1.5: positives.append("Kırılım Teyidi")

            # 7. İşlem Seviyeleri (Tetik/Direnç/Hedefler)
            entry = round(price * 1.01, 2)
            stop = round(entry - (atr20 * 2.0), 2)
            target_1 = round(entry * 1.15, 2)
            target_2 = round(entry * 1.40, 2)

            results.append({
                "symbol": symbol,
                "score": flight_score,
                "details": f"Spek Skor: {flight_score:.1f}/100 | " + " | ".join(positives[:4]),
                "Entry_Price": entry,
                "Stop_Loss": stop,
                "Target_1": target_1,
                "Target_2": target_2
            })

        except Exception as e:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]


# ────────────────────────────────────────────────────────────
# STRAT 12 – TGYESIL
# ────────────────────────────────────────────────────────────
def _run_tgyesil(data_dict):
    results = []
    for symbol, df in data_dict.items():
        try:
            if len(df) < 200:
                continue
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            volume = df['Volume'].squeeze()

            price = float(close.iloc[-1])
            ma50 = float(_sma(close, 50).iloc[-1])
            ma150 = float(_sma(close, 150).iloc[-1])
            ma200 = float(_sma(close, 200).iloc[-1])
            ma200_prev = float(_sma(close, 200).iloc[-2])
            
            high_52w = float(high.rolling(252).max().iloc[-1])
            low_52w = float(low.rolling(252).min().iloc[-1])

            # Trend Şablonu (Trend Template)
            cond1 = price > ma150
            cond2 = price > ma200
            cond3 = ma150 > ma200
            cond4 = ma200 > ma200_prev  # Slope > 0
            cond5 = ma50 > ma150
            cond6 = price > ma50
            cond7 = price >= high_52w * 0.75
            cond8 = price >= low_52w * 1.30
            
            trend_score = sum([cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8])

            # VCP Kriterleri
            def range_pct(period):
                h = float(high.rolling(period).max().iloc[-1])
                l = float(low.rolling(period).min().iloc[-1])
                return ((h - l) / l) * 100 if l > 0 else 100

            range_15 = range_pct(15)
            range_30 = range_pct(30)
            range_50 = range_pct(50)
            
            cond10 = any(r < 30 for r in [range_15, range_30, range_50])
            cond9 = range_15 < range_50  # VCP Contraction proxy
            vcp_score = sum([cond9, cond10])

            # Hacim Kriterleri (Sadece çok güçlü, güçlü, orta)
            latest_volume = float(volume.iloc[-1])
            previous_volume = float(volume.iloc[-2]) if len(volume) > 1 else latest_volume
            volume_ma50 = float(volume.rolling(50).mean().iloc[-1])
            
            volume_increase = ((latest_volume - previous_volume) / previous_volume * 100) if previous_volume > 0 else 0
            ma50_ratio = latest_volume / volume_ma50 if volume_ma50 > 0 else 1.0
            
            is_valid_breakout = False
            breakout_quality = ''
            
            if ma50_ratio >= 1.5 and volume_increase >= 50:
                is_valid_breakout = True
                breakout_quality = 'MÜKEMMEL 🏆'
            elif ma50_ratio >= 1.2 and volume_increase >= 40:
                is_valid_breakout = True
                breakout_quality = 'İYİ ✅'
            elif ma50_ratio >= 1.0 and volume_increase >= 20:
                is_valid_breakout = True
                breakout_quality = 'ORTA 📊'

            # Eğer hacim kriteri orta değerin altındaysa sinyal iptal
            if not is_valid_breakout:
                continue
                
            volume_score = 1
            
            # Toplam Puan
            total_score = trend_score + vcp_score + volume_score
            
            if trend_score == 8 and vcp_score == 2 and volume_score == 1:
                signal = 'GÜÇLÜ GİR'
            elif trend_score >= 7 and vcp_score >= 1 and volume_score == 1:
                signal = 'GİR'
            else:
                continue  # İZLE ve BEKLE sinyalleri iptal edildi, sadece GİR ve GÜÇLÜ GİR
                
            reasons = [f"Sinyal: {signal}", f"Trend:{trend_score}/8", f"Hacim: {breakout_quality}"]
            
            levels = _price_levels(df, atr_mult_stop=2.0, target_pct=0.10)
            
            results.append({
                "symbol": symbol,
                "score": total_score,
                "details": " | ".join(reasons),
                **levels
            })
            
        except Exception:
            continue
            
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]


# ────────────────────────────────────────────────────────────
# ANA DISPATCHER
# ────────────────────────────────────────────────────────────
def run_strategy(strategy_id, data_dict):
    set_prefetched_data(data_dict)
    try:
        dispatch = {
            "strat1_birlesik":       _run_birlesik,
            "strat2_gelismis_yesil": _run_gelismis_yesil,
            "strat3_minervini":      _run_minervini,
            "strat4_mum":            _run_mum,
            "strat5_sikismis":       _run_sikismis,
            "strat6_tepki1":         _run_tepki1,
            "strat7_tepki2":         _run_tepki2,
            "strat8_yesil":          _run_yesil,
            "strat9_momentum":       _run_momentum,
            "strat10_tavan":         _run_tavan,
            "strat11_spek":          _run_spek,
            "strat12_tgyesil":  _run_tgyesil,
        }
        fn = dispatch.get(strategy_id)
        return fn(data_dict) if fn else []
    except Exception:
        traceback.print_exc()
        return []

