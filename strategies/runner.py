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
<<<<<<< HEAD
            if rsi_val >= 70:
                score += 1; checks.append(f"RSI≥70 ({rsi_val:.0f})")
=======
            if rsi_val >= 60:
                score += 1; checks.append(f"RSI≥60 ({rsi_val:.0f})")
>>>>>>> 9e439e475d3267be52c01eac93d6a8e0814baba5

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
def _run_momentum(data_dict):
    results = []
    for symbol, df in data_dict.items():
        try:
            if len(df) < 20:
                continue
            close = df['Close'].squeeze(); volume = df['Volume'].squeeze()
            high = df['High'].squeeze(); low = df['Low'].squeeze()

            rsi = _rsi(close); atr = _atr(high, low, close, 14)
            price = float(close.iloc[-1]); rsi_now = float(rsi.iloc[-1])
            atr_now = float(atr.iloc[-1])
            vol_now = float(volume.iloc[-1]); vol_avg = float(volume.rolling(10).mean().iloc[-1])
            change_1d = (price / float(close.iloc[-2]) - 1) * 100
            change_3d = (price / float(close.iloc[-4]) - 1) * 100

            score = 0; reasons = []
            if change_1d > 3:
                score += 4; reasons.append(f"+%{change_1d:.1f} 1g momentum")
            elif change_1d > 1.5:
                score += 2; reasons.append(f"+%{change_1d:.1f} 1g yükseliş")
            if change_3d > 5:
                score += 3; reasons.append(f"+%{change_3d:.1f} 3g trend")
            if vol_avg > 0 and vol_now > vol_avg * 2:
                score += 4; reasons.append(f"Hacim patlaması {vol_now/vol_avg:.1f}x")
            elif vol_avg > 0 and vol_now > vol_avg * 1.5:
                score += 2; reasons.append(f"Hacim artışı {vol_now/vol_avg:.1f}x")
            if 50 < rsi_now < 75:
                score += 2; reasons.append(f"RSI momentum bölgesi ({rsi_now:.0f})")
            if atr_now / price > 0.02:
                score += 1; reasons.append("Yeterli volatilite")

            if score >= 8:
                levels = _price_levels(df, atr_mult_stop=1.5, target_pct=0.06)
                results.append({"symbol": symbol, "score": score,
                                 "details": " | ".join(reasons), **levels})
        except Exception:
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]


# ────────────────────────────────────────────────────────────
# STRAT 10 – TAVAN TAHMİNİ
# ────────────────────────────────────────────────────────────
def _run_tavan(data_dict):
    results = []
    for symbol, df in data_dict.items():
        try:
            if len(df) < 10:
                continue
            close = df['Close'].squeeze(); volume = df['Volume'].squeeze()
            high = df['High'].squeeze(); low = df['Low'].squeeze()

            price = float(close.iloc[-1]); prev_price = float(close.iloc[-2])
            change_1d = (price / prev_price - 1) * 100
            change_3d = (price / float(close.iloc[-4]) - 1) * 100
            vol_now = float(volume.iloc[-1]); vol_avg = float(volume.rolling(5).mean().iloc[-2])
            rsi_now = float(_rsi(close).iloc[-1])
            high_now = float(high.iloc[-1]); low_now = float(low.iloc[-1])
            close_position = (price - low_now) / (high_now - low_now + 1e-10)

            score = 0; reasons = []
            if change_1d > 5 and close_position > 0.85:
                score += 5; reasons.append(f"Güçlü yükseliş+%{change_1d:.1f}, Tavan kapanışı")
            if change_3d > 10:
                score += 3; reasons.append(f"3g trend +%{change_3d:.1f}")
            if vol_avg > 0 and vol_now > vol_avg * 2.5:
                score += 4; reasons.append(f"Hacim x{vol_now/vol_avg:.1f} (spek ilgi)")
            elif vol_avg > 0 and vol_now > vol_avg * 1.5:
                score += 2; reasons.append(f"Yüksek hacim x{vol_now/vol_avg:.1f}")
            if rsi_now > 65:
                score += 2; reasons.append(f"RSI yüksek ({rsi_now:.0f})")

            if score >= 8:
                levels = _price_levels(df, atr_mult_stop=2.0, target_pct=0.07)
                results.append({"symbol": symbol, "score": score,
                                 "details": f"Tavan İhtimali (Skor:{score}) | " + " | ".join(reasons),
                                 **levels})
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
    # Likidite eşiği: günlük min 1M TL ciro
    MIN_VOLUME_TL = 1_000_000

    for symbol, df in data_dict.items():
        try:
            if len(df) < 60:
                continue

            close = df['Close'].squeeze()
            high  = df['High'].squeeze()
            low   = df['Low'].squeeze()
            volume = df['Volume'].squeeze()
            open_  = df['Open'].squeeze()

            # ── Temel fiyat değerleri ──────────────────────
            price      = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            open_val   = float(open_.iloc[-1])
            high_val   = float(high.iloc[-1])
            low_val    = float(low.iloc[-1])

            # ── Hareketli ortalamalar ──────────────────────
            ma50  = float(_sma(close, 50).iloc[-1])
            ma150 = float(_sma(close, 150).iloc[-1]) if len(df) >= 150 else ma50
            ma200 = float(_sma(close, 200).iloc[-1]) if len(df) >= 200 else ma150

            # ── Momentum (ROC) ─────────────────────────────
            roc_5d  = (price / float(close.iloc[-6])  - 1) * 100 if len(df) > 6  else 0
            roc_3d  = (price / float(close.iloc[-4])  - 1) * 100 if len(df) > 4  else 0
            roc_10d = (price / float(close.iloc[-11]) - 1) * 100 if len(df) > 11 else 0

            # ── Hacim ──────────────────────────────────────
            vol_ma20 = float(volume.rolling(20).mean().iloc[-1])
            vol_ma50 = float(volume.rolling(50).mean().iloc[-1])
            vol_now  = float(volume.iloc[-1])
            rvol_20  = vol_now / vol_ma20 if vol_ma20 > 0 else 1.0
            vol_zscore = (vol_now - float(volume.rolling(60).mean().iloc[-1])) / \
                         (float(volume.rolling(60).std().iloc[-1]) + 1e-10)

            # yükseliş / düşüş günlerinde hacim dağılımı
            returns = close.pct_change()
            up_vol   = float(volume[returns > 0].tail(20).mean()) + 1e-10
            down_vol = float(volume[returns < 0].tail(20).mean()) + 1e-10
            up_vol_dom = up_vol / down_vol

            # ── ATR ve BB ─────────────────────────────────
            atr14   = float(_atr(high, low, close, 14).iloc[-1])
            atr5    = float(_atr(high, low, close, 5).iloc[-1])
            atr_exp = (atr5 / atr14 - 1) * 100 if atr14 > 0 else 0
            atr_pct = atr14 / price * 100 if price > 0 else 0

            bb_l, bb_m, bb_h = _bollinger(close, 20)
            bb_width = float((bb_h - bb_l) / (bb_m + 1e-10) * 100).real
            bb_width = float(bb_width) if not (bb_width != bb_width) else 10  # nan guard

            # ── Likidite ───────────────────────────────────
            avg_vol_tl = float((close * volume).tail(20).mean())

            # ── Risk Faktörleri ────────────────────────────
            candle_range  = high_val - low_val + 1e-10
            upper_wick    = high_val - max(open_val, price)
            upper_wick_ratio = upper_wick / candle_range

            # Ardışık yükseliş günleri
            consec_up = 0
            for i in range(1, min(10, len(returns))):
                if float(returns.iloc[-i]) > 0:
                    consec_up += 1
                else:
                    break

            # gap + zayıf kapanış
            gap_up_pct = (open_val - prev_close) / prev_close if prev_close > 0 else 0
            gap_reversal = gap_up_pct > 0.03 and price < open_val
            close_pos    = (price - low_val) / candle_range
            blowoff      = roc_5d > 10 and rvol_20 > 3 and close_pos < 0.3

            # 20d zirveye mesafe
            high_20d = float(high.rolling(20).max().iloc[-1])
            dist_20d = (price / high_20d - 1) * 100 if high_20d > 0 else 0

            # ── PUANLAMA (0-100) ───────────────────────────
            # Momentum (max 30)
            mom = 0
            if roc_5d > 10: mom += 10
            elif roc_5d > 5: mom += 5
            if dist_20d > -2: mom += 10
            elif dist_20d > -5: mom += 5
            mom = min(mom, 30)

            # Hacim (max 25)
            vol_sc = 0
            if 1.5 < rvol_20 < 4: vol_sc += 10
            elif 1.2 < rvol_20 < 1.5: vol_sc += 5
            if vol_zscore > 2: vol_sc += 8
            elif vol_zscore > 1: vol_sc += 4
            if up_vol_dom > 1.3: vol_sc += 7
            elif up_vol_dom > 1.1: vol_sc += 3
            vol_sc = min(vol_sc, 25)

            # Sıkışma kırılımı (max 20)
            sqz = 0
            if atr_exp > 10: sqz += 10
            elif atr_exp > 5: sqz += 5
            if bb_width < 5: sqz += 10
            elif bb_width < 8: sqz += 5
            sqz = min(sqz, 20)

            # Likidite (max 10)
            liq = 0
            if avg_vol_tl > MIN_VOLUME_TL * 5: liq = 10
            elif avg_vol_tl > MIN_VOLUME_TL * 2: liq = 7
            elif avg_vol_tl > MIN_VOLUME_TL: liq = 5
            liq = max(liq, 0)

            # Market (sabit 7.5 — tek hisseden piyasa bilgisi yok)
            mkt = 7.5

            # Ceza
            pen = 0
            if blowoff: pen += 10
            if gap_reversal: pen += 10
            if consec_up > 5: pen += 10
            if upper_wick_ratio > 0.4: pen += 10
            if rvol_20 > 5: pen += 10
            if avg_vol_tl < MIN_VOLUME_TL: pen += 10
            pen = min(pen, 30)

            total = max(0, min(100, mom + vol_sc + sqz + liq + mkt - pen))

            # Minimum Minervini koşulları (piyasa trendi)
            minervini_ok = price > ma50 and price > ma150

            if total >= 45 and minervini_ok and avg_vol_tl >= MIN_VOLUME_TL:
                reasons = []
                if roc_5d > 5:  reasons.append(f"5g momentum +%{roc_5d:.1f}")
                if rvol_20 > 1.2: reasons.append(f"RVOL:{rvol_20:.1f}x")
                if dist_20d > -5: reasons.append(f"20g zirveye {dist_20d:.1f}%")
                if sqz >= 10:   reasons.append("Sıkışma kırılımı")
                if up_vol_dom > 1.2: reasons.append(f"Alım hacmi {up_vol_dom:.1f}x")

                levels = _price_levels(df, atr_mult_stop=2.0, target_pct=0.12)
                results.append({
                    "symbol": symbol,
                    "score": round(total),
                    "details": f"Spek Skor:{round(total)}/100 | " + " | ".join(reasons),
                    **levels
                })
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]


# ────────────────────────────────────────────────────────────
<<<<<<< HEAD
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
=======
>>>>>>> 9e439e475d3267be52c01eac93d6a8e0814baba5
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
<<<<<<< HEAD
            "strat12_tgyesil":  _run_tgyesil,
=======
>>>>>>> 9e439e475d3267be52c01eac93d6a8e0814baba5
        }
        fn = dispatch.get(strategy_id)
        return fn(data_dict) if fn else []
    except Exception:
        traceback.print_exc()
        return []

