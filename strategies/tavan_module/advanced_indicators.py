"""
Gelişmiş Teknik İndikatörler (70+)
yfinance 1.0 uyumlu
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class AdvancedIndicators:
    """70+ teknik indikatör hesaplama"""
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> Dict[str, float]:
        """Tüm indikatörleri hesapla - yfinance 1.0 uyumlu"""
        
        indicators = {}
        
        try:
            # Multi-index kontrolü
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Series çıkarma
            close = data['Close']
            high = data['High']
            low = data['Low']
            open_price = data['Open']
            volume = data['Volume']
            
            # MOMENTUM İNDİKATÖRLERİ
            indicators['rsi_14'] = AdvancedIndicators._calculate_rsi(close, 14)
            indicators['rsi_9'] = AdvancedIndicators._calculate_rsi(close, 9)
            indicators['rsi_25'] = AdvancedIndicators._calculate_rsi(close, 25)
            
            indicators['stoch_k'], indicators['stoch_d'] = AdvancedIndicators._calculate_stochastic(high, low, close)
            
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = AdvancedIndicators._calculate_macd(close)
            
            indicators['williams_r'] = AdvancedIndicators._calculate_williams_r(high, low, close)
            indicators['cci'] = AdvancedIndicators._calculate_cci(high, low, close)
            
            for period in [1, 3, 5, 10, 20]:
                indicators[f'roc_{period}'] = AdvancedIndicators._calculate_roc(close, period)
            
            indicators['momentum_10'] = AdvancedIndicators._calculate_momentum(close, 10)
            
            # TREND İNDİKATÖRLERİ
            indicators['adx'], indicators['plus_di'], indicators['minus_di'] = AdvancedIndicators._calculate_adx(high, low, close)
            indicators['aroon_up'], indicators['aroon_down'] = AdvancedIndicators._calculate_aroon(high, low)
            
            # VOLATİLİTE İNDİKATÖRLERİ
            indicators['atr_14'] = AdvancedIndicators._calculate_atr(high, low, close, 14)
            indicators['atr_20'] = AdvancedIndicators._calculate_atr(high, low, close, 20)
            
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'], indicators['bb_width'], indicators['bb_pct'] = AdvancedIndicators._calculate_bollinger_bands(close)
            
            indicators['hist_volatility'] = AdvancedIndicators._calculate_historical_volatility(close)
            
            # HACİM İNDİKATÖRLERİ
            indicators['obv'] = AdvancedIndicators._calculate_obv(close, volume)
            indicators['vpt'] = AdvancedIndicators._calculate_vpt(close, volume)
            indicators['ad_line'] = AdvancedIndicators._calculate_ad_line(high, low, close, volume)
            indicators['cmf'] = AdvancedIndicators._calculate_cmf(high, low, close, volume)
            indicators['mfi'] = AdvancedIndicators._calculate_mfi(high, low, close, volume)
            indicators['vwap'] = AdvancedIndicators._calculate_vwap(high, low, close, volume)
            indicators['volume_oscillator'] = AdvancedIndicators._calculate_volume_oscillator(volume)
            
            # GAP ANALİZİ
            indicators['gap_size'], indicators['gap_type'] = AdvancedIndicators._detect_gap(data)
            indicators['gap_fill_prob'] = AdvancedIndicators._calculate_gap_fill_probability(data)
            
            # VOLUME SENTIMENT BREAKOUT
            indicators['volume_breakout_score'] = AdvancedIndicators._calculate_volume_breakout(volume, close)
            indicators['compression_score'] = AdvancedIndicators._calculate_compression_score(high, low, close)
            indicators['breakout_strength'] = AdvancedIndicators._calculate_breakout_strength(high, low, close, volume)
            
            # FLOOR/CEILING PATTERNS
            indicators['support_strength'], indicators['resistance_strength'] = AdvancedIndicators._calculate_sr_strength(high, low, close)
            indicators['distance_to_ceiling'] = AdvancedIndicators._calculate_distance_to_ceiling(high, close)
            indicators['ceiling_breakout_prob'] = AdvancedIndicators._calculate_ceiling_breakout_prob(high, close, volume)
            
            # PATTERN RECOGNITION
            indicators['hammer'] = AdvancedIndicators._detect_hammer(open_price, high, low, close)
            indicators['shooting_star'] = AdvancedIndicators._detect_shooting_star(open_price, high, low, close)
            indicators['doji'] = AdvancedIndicators._detect_doji(open_price, close, high, low)
            indicators['engulfing_bullish'] = AdvancedIndicators._detect_bullish_engulfing(data)
            
            # TAVAN ÖZEL İNDİKATÖRLER
            indicators['pre_ceiling_squeeze'] = AdvancedIndicators._calculate_pre_ceiling_squeeze(high, low, close, volume)
            indicators['institutional_accumulation'] = AdvancedIndicators._calculate_institutional_accumulation(close, volume)
            indicators['retail_fomo'] = AdvancedIndicators._calculate_retail_fomo(close, volume)
            indicators['smart_money_divergence'] = AdvancedIndicators._calculate_smart_money_divergence(close, volume)
            indicators['ceiling_velocity'] = AdvancedIndicators._calculate_ceiling_velocity(close)
            
            # MULTI-TIMEFRAME
            indicators['trend_alignment'] = AdvancedIndicators._calculate_trend_alignment(close)
            indicators['volume_profile_score'] = AdvancedIndicators._calculate_volume_profile(close, volume)
            
            # STATISTICAL
            indicators['z_score'] = AdvancedIndicators._calculate_z_score(close)
            indicators['skewness'] = AdvancedIndicators._calculate_skewness(close.pct_change())
            indicators['kurtosis'] = AdvancedIndicators._calculate_kurtosis(close.pct_change())
            
        except Exception as e:
            logger.error(f"İndikatör hesaplama hatası: {e}")
        
        return indicators
    
    @staticmethod
    def _calculate_rsi(close: pd.Series, period: int = 14) -> float:
        """RSI hesapla"""
        try:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50.0
    
    @staticmethod
    def _calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[float, float]:
        """Stochastic Oscillator"""
        try:
            lowest_low = low.rolling(window=period).min()
            highest_high = high.rolling(window=period).max()
            k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            d = k.rolling(window=3).mean()
            return float(k.iloc[-1]), float(d.iloc[-1])
        except:
            return 50.0, 50.0
    
    @staticmethod
    def _calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """MACD"""
        try:
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            macd_hist = macd - macd_signal
            return float(macd.iloc[-1]), float(macd_signal.iloc[-1]), float(macd_hist.iloc[-1])
        except:
            return 0.0, 0.0, 0.0
    
    @staticmethod
    def _calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Williams %R"""
        try:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            wr = -100 * (highest_high - close) / (highest_high - lowest_low)
            return float(wr.iloc[-1])
        except:
            return -50.0
    
    @staticmethod
    def _calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> float:
        """Commodity Channel Index"""
        try:
            tp = (high + low + close) / 3
            sma = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            cci = (tp - sma) / (0.015 * mad)
            return float(cci.iloc[-1])
        except:
            return 0.0
    
    @staticmethod
    def _calculate_roc(close: pd.Series, period: int) -> float:
        """Rate of Change"""
        try:
            if len(close) < period + 1:
                return 0.0
            roc = ((close.iloc[-1] - close.iloc[-period-1]) / close.iloc[-period-1]) * 100
            return float(roc)
        except:
            return 0.0
    
    @staticmethod
    def _calculate_momentum(close: pd.Series, period: int) -> float:
        """Momentum"""
        try:
            if len(close) < period + 1:
                return 0.0
            momentum = close.iloc[-1] - close.iloc[-period-1]
            return float(momentum)
        except:
            return 0.0
    
    @staticmethod
    def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[float, float, float]:
        """Average Directional Index"""
        try:
            tr = pd.DataFrame({
                'hl': high - low,
                'hc': abs(high - close.shift(1)),
                'lc': abs(low - close.shift(1))
            }).max(axis=1)
            
            atr = tr.rolling(window=period).mean()
            
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return float(adx.iloc[-1]), float(plus_di.iloc[-1]), float(minus_di.iloc[-1])
        except:
            return 20.0, 20.0, 20.0
    
    @staticmethod
    def _calculate_aroon(high: pd.Series, low: pd.Series, period: int = 25) -> Tuple[float, float]:
        """Aroon Indicator"""
        try:
            aroon_up = high.rolling(window=period).apply(lambda x: x.argmax()) / period * 100
            aroon_down = low.rolling(window=period).apply(lambda x: x.argmin()) / period * 100
            return float(aroon_up.iloc[-1]), float(aroon_down.iloc[-1])
        except:
            return 50.0, 50.0
    
    @staticmethod
    def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
        """Average True Range"""
        try:
            tr = pd.DataFrame({
                'hl': high - low,
                'hc': abs(high - close.shift(1)),
                'lc': abs(low - close.shift(1))
            }).max(axis=1)
            
            atr = tr.rolling(window=period).mean()
            return float(atr.iloc[-1])
        except:
            return 0.0
    
    @staticmethod
    def _calculate_bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[float, float, float, float, float]:
        """Bollinger Bands"""
        try:
            middle = close.rolling(window=period).mean()
            std_dev = close.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            width = ((upper - lower) / middle) * 100
            pct = (close - lower) / (upper - lower)
            
            return float(upper.iloc[-1]), float(middle.iloc[-1]), float(lower.iloc[-1]), float(width.iloc[-1]), float(pct.iloc[-1])
        except:
            c = float(close.iloc[-1])
            return c, c, c, 0.0, 0.5
    
    @staticmethod
    def _calculate_historical_volatility(close: pd.Series, period: int = 20) -> float:
        """Historical Volatility"""
        try:
            returns = close.pct_change()
            volatility = returns.rolling(window=period).std() * np.sqrt(252) * 100
            return float(volatility.iloc[-1])
        except:
            return 0.0
    
    @staticmethod
    def _calculate_obv(close: pd.Series, volume: pd.Series) -> float:
        """On Balance Volume"""
        try:
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            return float(obv.iloc[-1])
        except:
            return 0.0
    
    @staticmethod
    def _calculate_vpt(close: pd.Series, volume: pd.Series) -> float:
        """Volume Price Trend"""
        try:
            vpt = (volume * close.pct_change()).fillna(0).cumsum()
            return float(vpt.iloc[-1])
        except:
            return 0.0
    
    @staticmethod
    def _calculate_ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> float:
        """Accumulation/Distribution Line"""
        try:
            clv = ((close - low) - (high - close)) / (high - low)
            clv = clv.fillna(0)
            ad = (clv * volume).cumsum()
            return float(ad.iloc[-1])
        except:
            return 0.0
    
    @staticmethod
    def _calculate_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> float:
        """Chaikin Money Flow"""
        try:
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)
            mfv = mfm * volume
            cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
            return float(cmf.iloc[-1])
        except:
            return 0.0
    
    @staticmethod
    def _calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> float:
        """Money Flow Index"""
        try:
            tp = (high + low + close) / 3
            mf = tp * volume
            
            positive_mf = mf.where(tp > tp.shift(1), 0).rolling(window=period).sum()
            negative_mf = mf.where(tp < tp.shift(1), 0).rolling(window=period).sum()
            
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            return float(mfi.iloc[-1])
        except:
            return 50.0
    
    @staticmethod
    def _calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> float:
        """Volume Weighted Average Price"""
        try:
            tp = (high + low + close) / 3
            vwap = (tp * volume).cumsum() / volume.cumsum()
            return float(vwap.iloc[-1])
        except:
            return float(close.iloc[-1])
    
    @staticmethod
    def _calculate_volume_oscillator(volume: pd.Series, short: int = 5, long: int = 10) -> float:
        """Volume Oscillator"""
        try:
            short_ma = volume.rolling(window=short).mean()
            long_ma = volume.rolling(window=long).mean()
            vo = ((short_ma - long_ma) / long_ma) * 100
            return float(vo.iloc[-1])
        except:
            return 0.0
    
    @staticmethod
    def _detect_gap(data: pd.DataFrame) -> Tuple[float, str]:
        """Gap Detection"""
        try:
            open_price = float(data['Open'].iloc[-1])
            prev_close = float(data['Close'].iloc[-2])
            
            gap_size = ((open_price - prev_close) / prev_close) * 100
            
            if gap_size > 2:
                gap_type = "GAP_UP"
            elif gap_size < -2:
                gap_type = "GAP_DOWN"
            else:
                gap_type = "NO_GAP"
            
            return float(gap_size), gap_type
        except:
            return 0.0, "NO_GAP"
    
    @staticmethod
    def _calculate_gap_fill_probability(data: pd.DataFrame) -> float:
        """Gap Fill Probability"""
        try:
            open_price = float(data['Open'].iloc[-1])
            prev_close = float(data['Close'].iloc[-2])
            current_close = float(data['Close'].iloc[-1])
            
            gap_size = abs(open_price - prev_close)
            filled = abs(current_close - prev_close)
            
            if gap_size > 0:
                fill_pct = min(filled / gap_size, 1.0) * 100
            else:
                fill_pct = 0.0
            
            return float(fill_pct)
        except:
            return 0.0
    
    @staticmethod
    def _calculate_volume_breakout(volume: pd.Series, close: pd.Series) -> float:
        """Volume Sentiment Breakout Score"""
        try:
            avg_volume = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            
            price_change = close.pct_change().iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if price_change > 0 and volume_ratio > 1.5:
                score = min(volume_ratio * 20, 100)
            else:
                score = 0
            
            return float(score)
        except:
            return 0.0
    
    @staticmethod
    def _calculate_compression_score(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Price Compression Score"""
        try:
            recent_range = (high.tail(20).max() - low.tail(20).min()) / close.iloc[-1] * 100
            current_range = (high.tail(5).max() - low.tail(5).min()) / close.iloc[-1] * 100
            
            if recent_range > 0:
                compression = (1 - (current_range / recent_range)) * 100
            else:
                compression = 0
            
            return float(max(compression, 0))
        except:
            return 0.0
    
    @staticmethod
    def _calculate_breakout_strength(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> float:
        """Breakout Strength"""
        try:
            high_20 = high.rolling(20).max().iloc[-2]
            
            if close.iloc[-1] > high_20:
                avg_volume = volume.rolling(20).mean().iloc[-1]
                volume_confirmation = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1.0
                
                price_momentum = (close.iloc[-1] - high_20) / high_20 * 100
                
                strength = (volume_confirmation * 30) + (price_momentum * 10)
                return float(min(strength, 100))
            else:
                return 0.0
        except:
            return 0.0
    
    @staticmethod
    def _calculate_sr_strength(high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[float, float]:
        """Support/Resistance Strength"""
        try:
            support = low.tail(60).min()
            resistance = high.tail(60).max()
            
            support_touches = ((low.tail(60) - support).abs() < (support * 0.02)).sum()
            resistance_touches = ((high.tail(60) - resistance).abs() < (resistance * 0.02)).sum()
            
            support_strength = min(support_touches * 10, 100)
            resistance_strength = min(resistance_touches * 10, 100)
            
            return float(support_strength), float(resistance_strength)
        except:
            return 0.0, 0.0
    
    @staticmethod
    def _calculate_distance_to_ceiling(high: pd.Series, close: pd.Series) -> float:
        """Distance to Ceiling"""
        try:
            ceiling = high.tail(60).max()
            current = close.iloc[-1]
            
            distance = ((ceiling - current) / current) * 100
            return float(distance)
        except:
            return 100.0
    
    @staticmethod
    def _calculate_ceiling_breakout_prob(high: pd.Series, close: pd.Series, volume: pd.Series) -> float:
        """Ceiling Breakout Probability"""
        try:
            ceiling = high.tail(60).max()
            current = close.iloc[-1]
            
            distance_pct = ((ceiling - current) / current) * 100
            
            avg_volume = volume.rolling(20).mean().iloc[-1]
            volume_increase = volume.iloc[-1] / avg_volume if avg_volume > 0 else 1.0
            
            if distance_pct < 3 and volume_increase > 1.3:
                prob = 80
            elif distance_pct < 5 and volume_increase > 1.1:
                prob = 60
            elif distance_pct < 10:
                prob = 30
            else:
                prob = 10
            
            return float(prob)
        except:
            return 0.0
    
    @staticmethod
    def _detect_hammer(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> int:
        """Hammer Pattern"""
        try:
            o = float(open_price.iloc[-1])
            h = float(high.iloc[-1])
            l = float(low.iloc[-1])
            c = float(close.iloc[-1])
            
            body = abs(c - o)
            lower_shadow = min(o, c) - l
            upper_shadow = h - max(o, c)
            
            if lower_shadow > body * 2 and upper_shadow < body * 0.5:
                return 1
            return 0
        except:
            return 0
    
    @staticmethod
    def _detect_shooting_star(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> int:
        """Shooting Star Pattern"""
        try:
            o = float(open_price.iloc[-1])
            h = float(high.iloc[-1])
            l = float(low.iloc[-1])
            c = float(close.iloc[-1])
            
            body = abs(c - o)
            lower_shadow = min(o, c) - l
            upper_shadow = h - max(o, c)
            
            if upper_shadow > body * 2 and lower_shadow < body * 0.5:
                return 1
            return 0
        except:
            return 0
    
    @staticmethod
    def _detect_doji(open_price: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series) -> int:
        """Doji Pattern"""
        try:
            o = float(open_price.iloc[-1])
            c = float(close.iloc[-1])
            h = float(high.iloc[-1])
            l = float(low.iloc[-1])
            
            body = abs(c - o)
            total_range = h - l
            
            if total_range > 0 and body / total_range < 0.1:
                return 1
            return 0
        except:
            return 0
    
    @staticmethod
    def _detect_bullish_engulfing(data: pd.DataFrame) -> int:
        """Bullish Engulfing Pattern"""
        try:
            if len(data) < 2:
                return 0
            
            prev_o = float(data['Open'].iloc[-2])
            prev_c = float(data['Close'].iloc[-2])
            curr_o = float(data['Open'].iloc[-1])
            curr_c = float(data['Close'].iloc[-1])
            
            if prev_c < prev_o and curr_c > curr_o and curr_o < prev_c and curr_c > prev_o:
                return 1
            return 0
        except:
            return 0
    
    @staticmethod
    def _calculate_pre_ceiling_squeeze(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> float:
        """Pre-Ceiling Squeeze Score"""
        try:
            ceiling = high.tail(60).max()
            distance = ((ceiling - close.iloc[-1]) / close.iloc[-1]) * 100
            
            recent_range = (high.tail(5).max() - low.tail(5).min()) / close.iloc[-1] * 100
            avg_range = (high.tail(20).max() - low.tail(20).min()) / close.iloc[-1] * 100
            
            compression = (1 - (recent_range / avg_range)) * 100 if avg_range > 0 else 0
            
            avg_volume = volume.rolling(20).mean().iloc[-1]
            volume_increase = (volume.iloc[-1] / avg_volume - 1) * 100 if avg_volume > 0 else 0
            
            score = 0
            if distance < 5:
                score += 40
            elif distance < 10:
                score += 20
            
            if compression > 50:
                score += 30
            elif compression > 30:
                score += 15
            
            if volume_increase > 50:
                score += 30
            elif volume_increase > 20:
                score += 15
            
            return float(score)
        except:
            return 0.0
    
    @staticmethod
    def _calculate_institutional_accumulation(close: pd.Series, volume: pd.Series) -> float:
        """Institutional Accumulation Score"""
        try:
            price_up = close.diff() > 0
            high_volume = volume > volume.rolling(20).mean()
            
            institutional_days = (price_up & high_volume).tail(20).sum()
            
            score = min(institutional_days * 5, 100)
            return float(score)
        except:
            return 0.0
    
    @staticmethod
    def _calculate_retail_fomo(close: pd.Series, volume: pd.Series) -> float:
        """Retail FOMO Score"""
        try:
            recent_volume = volume.tail(3).mean()
            avg_volume = volume.rolling(20).mean().iloc[-1]
            
            volume_spike = (recent_volume / avg_volume - 1) * 100 if avg_volume > 0 else 0
            
            price_increase = (close.iloc[-1] / close.iloc[-4] - 1) * 100
            
            if volume_spike > 100 and price_increase > 5:
                score = 80
            elif volume_spike > 50 and price_increase > 3:
                score = 50
            else:
                score = 20
            
            return float(score)
        except:
            return 0.0
    
    @staticmethod
    def _calculate_smart_money_divergence(close: pd.Series, volume: pd.Series) -> float:
        """Smart Money Divergence"""
        try:
            price_trend = (close.iloc[-1] / close.iloc[-11] - 1) * 100
            volume_trend = (volume.tail(5).mean() / volume.iloc[-15:-10].mean() - 1) * 100
            
            if price_trend > 0 and volume_trend < -20:
                divergence = abs(price_trend) + abs(volume_trend)
            else:
                divergence = 0
            
            return float(min(divergence, 100))
        except:
            return 0.0
    
    @staticmethod
    def _calculate_ceiling_velocity(close: pd.Series) -> float:
        """Ceiling Approach Velocity"""
        try:
            ceiling = close.tail(60).max()
            
            distance_5_days_ago = ((ceiling - close.iloc[-6]) / close.iloc[-6]) * 100
            distance_now = ((ceiling - close.iloc[-1]) / close.iloc[-1]) * 100
            
            velocity = distance_5_days_ago - distance_now
            
            return float(velocity)
        except:
            return 0.0
    
    @staticmethod
    def _calculate_trend_alignment(close: pd.Series) -> float:
        """Multi-Timeframe Trend Alignment"""
        try:
            ema8 = close.ewm(span=8).mean().iloc[-1]
            ema20 = close.ewm(span=20).mean().iloc[-1]
            ema50 = close.ewm(span=50).mean().iloc[-1]
            
            current = close.iloc[-1]
            
            score = 0
            if current > ema8:
                score += 33
            if current > ema20:
                score += 33
            if current > ema50:
                score += 34
            
            return float(score)
        except:
            return 0.0
    
    @staticmethod
    def _calculate_volume_profile(close: pd.Series, volume: pd.Series) -> float:
        """Volume Profile Score"""
        try:
            price_levels = pd.cut(close.tail(60), bins=10)
            volume_by_level = volume.tail(60).groupby(price_levels).sum()
            
            max_volume_level = volume_by_level.idxmax()
            
            if close.iloc[-1] > max_volume_level.right:
                score = 70
            elif close.iloc[-1] > max_volume_level.left:
                score = 50
            else:
                score = 30
            
            return float(score)
        except:
            return 50.0
    
    @staticmethod
    def _calculate_z_score(close: pd.Series, period: int = 20) -> float:
        """Z-Score"""
        try:
            mean = close.rolling(window=period).mean().iloc[-1]
            std = close.rolling(window=period).std().iloc[-1]
            
            if std > 0:
                z_score = (close.iloc[-1] - mean) / std
            else:
                z_score = 0
            
            return float(z_score)
        except:
            return 0.0
    
    @staticmethod
    def _calculate_skewness(returns: pd.Series, period: int = 20) -> float:
        """Skewness"""
        try:
            skew = returns.tail(period).skew()
            return float(skew)
        except:
            return 0.0
    
    @staticmethod
    def _calculate_kurtosis(returns: pd.Series, period: int = 20) -> float:
        """Kurtosis"""
        try:
            kurt = returns.tail(period).kurtosis()
            return float(kurt)
        except:
            return 0.0
