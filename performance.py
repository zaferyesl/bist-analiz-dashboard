"""
Genişletilmiş Strateji Performans Analizi
------------------------------------------
Her gün biriken results_YYYYMMDD.json dosyalarından:
  - T+1, T+2, T+3, T+5, T+10, T+22 getiri (sinyalden n gün sonraki kapanış)
  - Pencere içi maksimum getiri (max gain, optimal çıkış noktası)
  - Kar eşiği başarı oranları: kaç sinyal %3, %5, %8, %10'a ulaştı ve kaç günde?
  - isabet oranı, ortalama getiri, beklenen değer (EV = win_rate × avg_return)
Günlük cache: results/performance_cache.json
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import yfinance as yf

# ── Yollar ─────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_FILE  = os.path.join(RESULTS_DIR, "performance_cache.json")

# ── Strateji Etiketleri ─────────────────────────────────────
STRATEGY_LABELS = {
    "strat1_birlesik":       "Birleşik Tarama",
    "strat2_gelismis_yesil": "Gelişmiş Yeşil",
    "strat3_minervini":      "Minervini",
    "strat4_mum":            "Mum Formasyonları",
    "strat5_sikismis":       "Sıkışmış Hisseler",
    "strat6_tepki1":         "Tepki Taraması 1",
    "strat7_tepki2":         "Tepki Taraması 2",
    "strat8_yesil":          "Yeşil Tarama",
    "strat9_momentum":       "Momentum",
    "strat10_tavan":         "Tavan Tahmini",
    "strat11_spek":          "Spek Tarayıcı",
}

# ── Analiz Parametreleri ────────────────────────────────────
HORIZONS  = [1, 2, 3, 5, 10, 22]          # İş günü sayısı
THRESHOLDS = [3.0, 5.0, 8.0, 10.0]        # % kar eşikleri
MAX_WINDOW = 22                            # Max pencere (aylık)


# ────────────────────────────────────────────────────────────
# VERİ YÜKLEME
# ────────────────────────────────────────────────────────────
def _load_all_results() -> List[Dict]:
    """Tüm results_YYYYMMDD.json dosyalarını sırala ve yükle."""
    files = sorted([
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("results_") and f.endswith(".json")
        and "performance" not in f and "cache" not in f
    ])
    records = []
    for fname in files:
        try:
            with open(os.path.join(RESULTS_DIR, fname), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if "date" not in data or "strategies" not in data:
                continue
            sig_date = datetime.strptime(data["date"][:10], "%Y-%m-%d").date()
            records.append({"date": sig_date, "strategies": data["strategies"]})
        except Exception:
            continue
    return records


# ────────────────────────────────────────────────────────────
# FİYAT VERİSİ ÇEKME
# ────────────────────────────────────────────────────────────
def _fetch_price_series(symbol: str, sig_date, max_days: int = MAX_WINDOW) -> Optional[List[float]]:
    """
    Sinyal tarihinden itibaren max_days iş günü kapanış fiyatlarını döndürür.
    İlk eleman = sinyal günü kapanışı (giriş fiyatı).
    """
    try:
        end = sig_date + timedelta(days=max_days * 2 + 10)
        if end > datetime.today().date():
            end = datetime.today().date()
        if end <= sig_date:
            return None

        df = yf.download(symbol, start=str(sig_date), end=str(end),
                         auto_adjust=True, progress=False, actions=False)
        if df is None or df.empty or len(df) < 2:
            return None

        close = df["Close"].squeeze()
        prices = [float(close.iloc[i]) for i in range(min(max_days + 1, len(close)))]
        return prices if len(prices) >= 2 else None
    except Exception:
        return None


# ────────────────────────────────────────────────────────────
# TEK SİNYAL ANALİZİ
# ────────────────────────────────────────────────────────────
def _analyze_signal(prices: List[float]) -> Dict:
    """
    Bir sinyalin fiyat serisinden tüm metrikleri hesapla.
    prices[0] = giriş (sinyal günü kapanışı)
    """
    entry = prices[0]
    if entry <= 0:
        return {}

    result = {}

    # Her horizon için getiri
    horizon_returns = {}
    for h in HORIZONS:
        if len(prices) > h:
            ret = (prices[h] / entry - 1) * 100
            horizon_returns[h] = round(ret, 2)
        else:
            horizon_returns[h] = None
    result["horizon_returns"] = horizon_returns

    # Pencere içi maksimum getiri ve optimal çıkış günü
    max_ret = 0.0
    optimal_day = 0
    for i in range(1, len(prices)):
        ret = (prices[i] / entry - 1) * 100
        if ret > max_ret:
            max_ret = ret
            optimal_day = i
    result["max_gain"] = round(max_ret, 2)
    result["optimal_exit_day"] = optimal_day

    # Kar eşiği analizi: kaç günde hedefe ulaşıldı?
    threshold_results = {}
    for thr in THRESHOLDS:
        hit_day = None
        for i in range(1, len(prices)):
            if (prices[i] / entry - 1) * 100 >= thr:
                hit_day = i
                break
        threshold_results[thr] = hit_day   # None = ulaşılamadı
    result["threshold_hits"] = threshold_results

    return result


# ────────────────────────────────────────────────────────────
# STRATEJİ TOPLAMA
# ────────────────────────────────────────────────────────────
def _build_strategy_stats(records: List[Dict]) -> Dict:
    """Tüm sinyaller için analiz yap, strateji bazında topla."""

    # raw_signals[strat_id] = [{ symbol, date, signal_data }, ...]
    raw_signals = defaultdict(list)

    for rec in records:
        sig_date = rec["date"]
        for strat_id, signals in rec["strategies"].items():
            if strat_id not in STRATEGY_LABELS:
                continue
            if not isinstance(signals, list):
                continue
            for item in signals:
                sym = item.get("symbol", "")
                if not sym:
                    continue
                if not sym.endswith(".IS"):
                    sym += ".IS"
                raw_signals[strat_id].append({
                    "symbol": sym,
                    "date": sig_date,
                })

    # Her strateji için fiyat çek + analiz et
    strategy_stats = {}
    for strat_id, sigs in raw_signals.items():
        analyzed = []
        for sig in sigs:
            prices = _fetch_price_series(sig["symbol"], sig["date"])
            if not prices:
                continue
            metrics = _analyze_signal(prices)
            if metrics:
                metrics["symbol"] = sig["symbol"].replace(".IS", "")
                metrics["date"]   = str(sig["date"])
                analyzed.append(metrics)

        strategy_stats[strat_id] = _aggregate_strategy(analyzed)
        strategy_stats[strat_id]["label"] = STRATEGY_LABELS.get(strat_id, strat_id)
        strategy_stats[strat_id]["signal_count"] = len(analyzed)

    return strategy_stats


def _aggregate_strategy(signals: List[Dict]) -> Dict:
    """Tek strateji için tüm sinyallerin istatistiklerini çıkar."""
    if not signals:
        return {
            "horizons": {h: None for h in HORIZONS},
            "thresholds": {t: {"hit_rate": None, "avg_days": None} for t in THRESHOLDS},
            "max_gain": {"avg": None, "best": None, "avg_optimal_day": None},
            "expected_value": None,
            "top_stocks": [],
        }

    n = len(signals)

    # ── Horizon getirileri ──
    horizons = {}
    for h in HORIZONS:
        vals = [s["horizon_returns"][h] for s in signals if s["horizon_returns"].get(h) is not None]
        if vals:
            avg_ret = round(sum(vals) / len(vals), 2)
            win_rate = round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1)
            horizons[h] = {
                "avg_return": avg_ret,
                "win_rate": win_rate,
                "sample": len(vals),
                "ev": round(win_rate / 100 * avg_ret, 2)   # Beklenen Değer
            }
        else:
            horizons[h] = None

    # ── Kar eşiği analizi ──
    thresholds = {}
    for thr in THRESHOLDS:
        hits = [s["threshold_hits"][thr] for s in signals if s["threshold_hits"][thr] is not None]
        hit_rate = round(len(hits) / n * 100, 1)
        avg_days = round(sum(hits) / len(hits), 1) if hits else None
        min_days = min(hits) if hits else None
        thresholds[thr] = {
            "hit_rate": hit_rate,       # % kaç sinyalde bu eşiğe ulaşıldı
            "avg_days": avg_days,       # Ortalama kaç günde
            "min_days": min_days,       # En kısa sürede
        }

    # ── Max getiri ──
    max_gains = [s["max_gain"] for s in signals]
    opt_days  = [s["optimal_exit_day"] for s in signals if s["optimal_exit_day"] > 0]
    max_gain_stats = {
        "avg": round(sum(max_gains) / len(max_gains), 2) if max_gains else None,
        "best": round(max(max_gains), 2) if max_gains else None,
        "avg_optimal_day": round(sum(opt_days) / len(opt_days), 1) if opt_days else None,
    }

    # ── Beklenen Değer (T+5 bazında genel EV) ──
    h5 = horizons.get(5)
    ev_5 = h5["ev"] if h5 else None

    # ── En iyi hisseler (max_gain bazında) ──
    stock_map = defaultdict(list)
    for s in signals:
        stock_map[s["symbol"]].append(s["max_gain"])
    top_stocks = sorted(
        [{"symbol": sym, "avg_max_gain": round(sum(v)/len(v), 2), "count": len(v)}
         for sym, v in stock_map.items()],
        key=lambda x: x["avg_max_gain"],
        reverse=True
    )[:6]

    return {
        "horizons": horizons,
        "thresholds": thresholds,
        "max_gain": max_gain_stats,
        "expected_value_5d": ev_5,
        "top_stocks": top_stocks,
    }


# ────────────────────────────────────────────────────────────
# ANA ENTRY POINT
# ────────────────────────────────────────────────────────────
def get_performance(force_refresh: bool = False) -> Dict:
    """Cache'e bak, yoksa hesapla ve kaydet."""
    if not force_refresh and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            cache_date = cache.get("computed_at", "")[:10]
            if cache_date == str(datetime.today().date()):
                return cache
        except Exception:
            pass

    records = _load_all_results()
    if not records:
        return {"error": "Henüz kayıtlı analiz sonucu yok. Lütfen önce analizi çalıştırın."}

    stats = _build_strategy_stats(records)

    # Strateji sıralaması: 5 günlük EV bazında
    ranked = sorted(
        [(sid, s) for sid, s in stats.items()],
        key=lambda x: (x[1].get("expected_value_5d") or -999),
        reverse=True
    )

    cache_data = {
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "record_count": len(records),
        "date_range": {
            "start": str(records[0]["date"]),
            "end":   str(records[-1]["date"])
        },
        "horizons_available": HORIZONS,
        "thresholds_available": THRESHOLDS,
        "strategies": {sid: s for sid, s in ranked},
        "strategy_ranking": [sid for sid, _ in ranked],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2, default=str)

    return cache_data
