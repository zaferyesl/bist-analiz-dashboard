"""
Sistem Konfigürasyonu
"""

import json
import os
import logging

logger = logging.getLogger(__name__)


class Config:
    """Genel sistem ayarları"""
    
    # Uçuş tanımı
    TARGET_RETURN_MIN = 15.0
    TARGET_RETURN_MAX = 40.0
    TARGET_DAYS = 5
    
    # Likidite
    MIN_AVG_VOLUME_TL = 1_000_000
    MAX_AVG_VOLUME_TL = 50_000_000
    
    # Volatilite
    MIN_ATR_PCT = 2.0
    MAX_ATR_PCT = 8.0
    
    # RVOL
    MIN_RVOL = 1.2
    MAX_RVOL_FOR_SAFETY = 5.0
    
    # Tuzak filtreleri
    MAX_UPPER_WICK_RATIO = 0.4
    MAX_CONSECUTIVE_UP_DAYS = 5
    
    # Risk
    MAX_POSITION_SIZE_PCT = 10.0
    STOP_LOSS_ATR_MULTIPLIER = 2.0
    
    # Backtest
    COMMISSION_PCT = 0.2
    SLIPPAGE_PCT = 0.3
    INITIAL_CAPITAL = 100_000
    
    # ML
    ML_MIN_SAMPLES = 50
    ML_TRAIN_TEST_SPLIT = 0.8
    
    # Walk-Forward
    WF_TRAIN_WINDOW = 120
    WF_TEST_WINDOW = 30
    WF_STEP = 15
    
    @classmethod
    def load_from_file(cls, filename='config.json'):
        """JSON'dan yükle"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        setattr(cls, key, value)
                logger.info(f"Konfigürasyon yüklendi: {filename}")
        except Exception as e:
            logger.error(f"Konfigürasyon yüklenemedi: {e}")
    
    @classmethod
    def save_to_file(cls, filename='config.json'):
        """JSON'a kaydet"""
        try:
            data = {k: v for k, v in cls.__dict__.items() if not k.startswith('_') and k.isupper()}
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Konfigürasyon kaydedildi: {filename}")
        except Exception as e:
            logger.error(f"Konfigürasyon kaydedilemedi: {e}")


class CeilingConfig:
    """Tavan tahmin özel ayarları"""
    
    # Eşik değerleri
    HIGH_PROBABILITY_THRESHOLD = 0.7
    MEDIUM_PROBABILITY_THRESHOLD = 0.6
    LOW_PROBABILITY_THRESHOLD = 0.5
    
    # Pattern benzerlik
    HIGH_SIMILARITY_THRESHOLD = 80
    MEDIUM_SIMILARITY_THRESHOLD = 60
    
    # Kombine skor
    EXCELLENT_SCORE = 80
    GOOD_SCORE = 70
    FAIR_SCORE = 60
    
    # Veri gereksinimleri
    MIN_CEILING_SAMPLES = 30
    MIN_DATA_DAYS = 100
    LOOKBACK_DAYS = 365
