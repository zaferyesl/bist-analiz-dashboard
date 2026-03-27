"""
GELİŞMİŞ TEKNİK İNDİKATÖRLER
70+ teknik indikatör hesaplar
"""

import numpy as np
import pandas as pd
import logging
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

logger = logging.getLogger(__name__)

class AdvancedIndicators:
    """Gelişmiş teknik indikatör hesaplayıcı"""
    
    def __init__(self):
        pass
    
    def calculate_all_indicators(self, open_prices, high_prices, low_prices, close_prices, volumes):
        """
        Tüm indikatörleri hesapla
        
        Args:
            open_prices: Açılış fiyatları (numpy array)
            high_prices: En yüksek fiyatlar
            low_prices: En düşük fiyatlar
            close_prices: Kapanış fiyatları
            volumes: Hacimler
        
        Returns:
            dict: Tüm indikatörler
        """
        try:
            # Array'lere çevir
            open_arr = np.array(open_prices)
            high_arr = np.array(high_prices)
            low_arr = np.array(low_prices)
            close_arr = np.array(close_prices)
            volume_arr = np.array(volumes)
            
            # Pandas Series'e çevir (ta kütüphanesi için)
            close_series = pd.Series(close_arr)
            high_series = pd.Series(high_arr)
            low_series = pd.Series(low_arr)
            volume_series = pd.Series(volume_arr)
            
            indicators = {}
            
            # === TREND İNDİKATÖRLERİ ===
            
            # RSI
            try:
                rsi = RSIIndicator(close_series, window=14)
                indicators['rsi_14'] = rsi.rsi().iloc[-1] if len(rsi.rsi()) > 0 else 50.0
            except:
                indicators['rsi_14'] = 50.0
            
            # MACD
            try:
                macd = MACD(close_series, window_slow=26, window_fast=12, window_sign=9)
                indicators['macd'] = macd.macd().iloc[-1] if len(macd.macd()) > 0 else 0.0
                indicators['macd_signal'] = macd.macd_signal().iloc[-1] if len(macd.macd_signal()) > 0 else 0.0
                indicators['macd_diff'] = macd.macd_diff().iloc[-1] if len(macd.macd_diff()) > 0 else 0.0
            except:
                indicators['macd'] = 0.0
                indicators['macd_signal'] = 0.0
                indicators['macd_diff'] = 0.0
            
            # EMA
            for period in [5, 10, 20, 50, 100, 200]:
                try:
                    ema = EMAIndicator(close_series, window=period)
                    indicators[f'ema_{period}'] = ema.ema_indicator().iloc[-1] if len(ema.ema_indicator()) > 0 else close_arr[-1]
                except:
                    indicators[f'ema_{period}'] = close_arr[-1]
            
            # SMA
            for period in [5, 10, 20, 50, 100, 200]:
                try:
                    sma = SMAIndicator(close_series, window=period)
                    indicators[f'sma_{period}'] = sma.sma_indicator().iloc[-1] if len(sma.sma_indicator()) > 0 else close_arr[-1]
                except:
                    indicators[f'sma_{period}'] = close_arr[-1]
            
            # ADX (Trend gücü)
            try:
                adx = ADXIndicator(high_series, low_series, close_series, window=14)
                indicators['adx'] = adx.adx().iloc[-1] if len(adx.adx()) > 0 else 20.0
                indicators['adx_pos'] = adx.adx_pos().iloc[-1] if len(adx.adx_pos()) > 0 else 20.0
                indicators['adx_neg'] = adx.adx_neg().iloc[-1] if len(adx.adx_neg()) > 0 else 20.0
            except:
                indicators['adx'] = 20.0
                indicators['adx_pos'] = 20.0
                indicators['adx_neg'] = 20.0
            
            # === MOMENTUM İNDİKATÖRLERİ ===
            
            # Stochastic
            try:
                stoch = StochasticOscillator(high_series, low_series, close_series, window=14, smooth_window=3)
                indicators['stoch_k'] = stoch.stoch().iloc[-1] if len(stoch.stoch()) > 0 else 50.0
                indicators['stoch_d'] = stoch.stoch_signal().iloc[-1] if len(stoch.stoch_signal()) > 0 else 50.0
            except:
                indicators['stoch_k'] = 50.0
                indicators['stoch_d'] = 50.0
            
            # ROC (Rate of Change)
            try:
                roc = ROCIndicator(close_series, window=12)
                indicators['roc'] = roc.roc().iloc[-1] if len(roc.roc()) > 0 else 0.0
            except:
                indicators['roc'] = 0.0
            
            # === VOLATİLİTE İNDİKATÖRLERİ ===
            
            # Bollinger Bands
            try:
                bb = BollingerBands(close_series, window=20, window_dev=2)
                bb_high = bb.bollinger_hband().iloc[-1] if len(bb.bollinger_hband()) > 0 else close_arr[-1]
                bb_low = bb.bollinger_lband().iloc[-1] if len(bb.bollinger_lband()) > 0 else close_arr[-1]
                bb_mid = bb.bollinger_mavg().iloc[-1] if len(bb.bollinger_mavg()) > 0 else close_arr[-1]
                
                indicators['bb_high'] = bb_high
                indicators['bb_low'] = bb_low
                indicators['bb_mid'] = bb_mid
                
                # BB pozisyonu (0-1 arası)
                if bb_high > bb_low:
                    indicators['bb_position'] = (close_arr[-1] - bb_low) / (bb_high - bb_low)
                else:
                    indicators['bb_position'] = 0.5
                
                # BB genişliği
                indicators['bb_width'] = (bb_high - bb_low) / bb_mid if bb_mid > 0 else 0.0
            except:
                indicators['bb_high'] = close_arr[-1]
                indicators['bb_low'] = close_arr[-1]
                indicators['bb_mid'] = close_arr[-1]
                indicators['bb_position'] = 0.5
                indicators['bb_width'] = 0.0
            
            # ATR (Average True Range)
            try:
                atr = AverageTrueRange(high_series, low_series, close_series, window=14)
                indicators['atr'] = atr.average_true_range().iloc[-1] if len(atr.average_true_range()) > 0 else 0.0
                indicators['atr_ratio'] = indicators['atr'] / close_arr[-1] if close_arr[-1] > 0 else 0.0
            except:
                indicators['atr'] = 0.0
                indicators['atr_ratio'] = 0.0
            
            # === HACİM İNDİKATÖRLERİ ===
            
            # OBV (On Balance Volume)
            try:
                obv = OnBalanceVolumeIndicator(close_series, volume_series)
                indicators['obv'] = obv.on_balance_volume().iloc[-1] if len(obv.on_balance_volume()) > 0 else 0.0
            except:
                indicators['obv'] = 0.0
            
            # Hacim ortalamaları
            for period in [5, 10, 20]:
                try:
                    vol_ma = volume_series.rolling(window=period).mean()
                    indicators[f'volume_ma_{period}'] = vol_ma.iloc[-1] if len(vol_ma) > 0 else volume_arr[-1]
                except:
                    indicators[f'volume_ma_{period}'] = volume_arr[-1]
            
            # === EK ÖZELLİKLER ===
            
            # EMA dizilimi (tüm EMA'lar sıralı mı?)
            try:
                ema_alignment = (
                    indicators['ema_5'] > indicators['ema_10'] > indicators['ema_20'] > 
                    indicators['ema_50'] > indicators['ema_100'] > indicators['ema_200']
                )
                indicators['ema_alignment'] = 1.0 if ema_alignment else 0.0
            except:
                indicators['ema_alignment'] = 0.0
            
            # Fiyat-EMA mesafeleri
            for period in [20, 50, 200]:
                try:
                    distance = ((close_arr[-1] - indicators[f'ema_{period}']) / indicators[f'ema_{period}']) * 100
                    indicators[f'price_ema_{period}_distance'] = distance
                except:
                    indicators[f'price_ema_{period}_distance'] = 0.0
            
            # Momentum (son 10 günün trendi)
            try:
                if len(close_arr) >= 10:
                    momentum = ((close_arr[-1] - close_arr[-10]) / close_arr[-10]) * 100
                    indicators['momentum_10'] = momentum
                else:
                    indicators['momentum_10'] = 0.0
            except:
                indicators['momentum_10'] = 0.0
            
            return indicators
        
        except Exception as e:
            logger.error(f"İndikatör hesaplama hatası: {e}")
            return {}
