"""
KENDİ İÇİNDE ANALİZ MODÜLÜ
Hissenin kendi geçmiş yükselişlerini analiz eder
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SelfAnalyzer:
    """Hissenin kendi içinde analiz motoru"""
    
    def __init__(self, data_fetcher, momentum_analyzer):
        self.data_fetcher = data_fetcher
        self.momentum_analyzer = momentum_analyzer
    
    def analyze_stock_self_pattern(self, symbol, min_rise_pct=15, lookback_days=365):
        """
        Hissenin kendi geçmiş yükselişlerini analiz et
        
        Args:
            symbol: Hisse sembolü
            min_rise_pct: Minimum yükseliş yüzdesi
            lookback_days: Geriye bakış süresi
        
        Returns:
            dict: Analiz sonucu
        """
        try:
            logger.info(f"🔍 Kendi içinde analiz: {symbol}")
            
            # 1. Geçmiş yükselişleri bul
            historical_rises = self.data_fetcher.find_historical_rises(
                symbol,
                lookback_days=lookback_days,
                min_rise_pct=min_rise_pct,
                min_duration=5,
                max_duration=30
            )
            
            if not historical_rises or len(historical_rises) < 2:
                logger.warning(f"  ✗ Yetersiz geçmiş yükseliş: {symbol} ({len(historical_rises) if historical_rises else 0})")
                return {
                    'success': False,
                    'reason': 'insufficient_history',
                    'historical_rises_count': len(historical_rises) if historical_rises else 0
                }
            
            logger.info(f"  ✓ {len(historical_rises)} geçmiş yükseliş bulundu")
            
            # 2. Her yükselişten önceki durumu analiz et
            pre_rise_features = []
            
            for idx, rise in enumerate(historical_rises[:5], 1):  # En fazla 5 yükseliş
                # Yükselişten 1-3 gün önce
                pre_date = rise['start_date'] - timedelta(days=2)
                
                logger.debug(f"    Analiz {idx}: {rise['start_date'].date()} öncesi")
                
                features = self.momentum_analyzer.analyze_stock_at_date(
                    symbol,
                    pre_date,
                    lookback_days=200
                )
                
                if features:
                    features['future_rise_pct'] = rise['rise_pct']
                    features['rise_duration'] = rise['duration']
                    pre_rise_features.append(features)
            
            if len(pre_rise_features) < 2:
                logger.warning(f"  ✗ Yetersiz analiz verisi: {symbol}")
                return {
                    'success': False,
                    'reason': 'insufficient_data',
                    'analyzed_rises': len(pre_rise_features)
                }
            
            logger.info(f"  ✓ {len(pre_rise_features)} yükseliş öncesi analiz edildi")
            
            # 3. Ortak paternleri bul
            common_pattern = self._find_self_common_patterns(pre_rise_features)
            
            # 4. Şu anki durumu analiz et
            current_features = self.momentum_analyzer.analyze_stock_at_date(
                symbol,
                datetime.now(),
                lookback_days=200
            )
            
            if not current_features:
                logger.warning(f"  ✗ Güncel analiz başarısız: {symbol}")
                return {
                    'success': False,
                    'reason': 'current_analysis_failed'
                }
            
            # 5. Benzerlik skoru hesapla
            similarity_score = self._calculate_self_similarity(
                current_features,
                pre_rise_features,
                common_pattern
            )
            
            # 6. Tahmin yap
            prediction = self._predict_self_rise(
                similarity_score,
                historical_rises,
                common_pattern
            )
            
            logger.info(f"  ✓ Kendi analiz tamamlandı: {symbol}")
            logger.info(f"    Benzerlik: {similarity_score:.1f}%")
            logger.info(f"    Tahmin: {prediction['probability']:.1f}%")
            
            return {
                'success': True,
                'symbol': symbol,
                'historical_rises_count': len(historical_rises),
                'analyzed_rises_count': len(pre_rise_features),
                'common_pattern': common_pattern,
                'current_features': current_features,
                'similarity_score': similarity_score,
                'prediction': prediction,
                'historical_rises': historical_rises[:5]  # En fazla 5 tanesini döndür
            }
        
        except Exception as e:
            logger.error(f"Kendi analiz hatası ({symbol}): {e}")
            return {
                'success': False,
                'reason': 'error',
                'error': str(e)
            }
    
    def _find_self_common_patterns(self, pre_rise_features):
        """Geçmiş yükselişlerin ortak özelliklerini bul"""
        try:
            df = pd.DataFrame(pre_rise_features)
            
            # Sadece numerik sütunlar
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Önemli özellikler
            important_features = [
                'rsi_14', 'macd_diff', 'stoch_k', 'bb_position',
                'volume_ratio', 'price_change_20d', 'adx',
                'ema_alignment', 'momentum_10', 'atr_ratio'
            ]
            
            pattern = {}
            
            for feat in important_features:
                if feat in numeric_cols:
                    values = df[feat].dropna()
                    
                    if len(values) > 0:
                        pattern[feat] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'median': float(values.median())
                        }
            
            return pattern
        
        except Exception as e:
            logger.error(f"Pattern bulma hatası: {e}")
            return {}
    
    def _calculate_self_similarity(self, current_features, pre_rise_features, common_pattern):
        """Şu anki durum ile geçmiş yükseliş öncesi durumların benzerliğini hesapla"""
        try:
            if not common_pattern:
                return 0.0
            
            scores = []
            
            for feat, pattern in common_pattern.items():
                current_value = current_features.get(feat, 0)
                
                if pd.isna(current_value):
                    continue
                
                # Ortalama ve standart sapma
                mean = pattern['mean']
                std = pattern['std']
                
                if std == 0:
                    # Standart sapma 0 ise, değer ortalamaya eşit mi?
                    score = 100.0 if abs(current_value - mean) < 0.01 else 0.0
                else:
                    # Z-score hesapla
                    z_score = abs((current_value - mean) / std)
                    
                    # Z-score'u benzerlik skoruna çevir
                    # z=0 -> %100, z=1 -> %60, z=2 -> %30, z=3 -> %10
                    if z_score <= 1:
                        score = 100.0 - (z_score * 40)
                    elif z_score <= 2:
                        score = 60.0 - ((z_score - 1) * 30)
                    elif z_score <= 3:
                        score = 30.0 - ((z_score - 2) * 20)
                    else:
                        score = max(0, 10.0 - ((z_score - 3) * 5))
                
                scores.append(score)
            
            if scores:
                return float(np.mean(scores))
            else:
                return 0.0
        
        except Exception as e:
            logger.error(f"Benzerlik hesaplama hatası: {e}")
            return 0.0
    
    def _predict_self_rise(self, similarity_score, historical_rises, common_pattern):
        """Benzerlik skoruna göre yükseliş tahmini yap"""
        try:
            # Olasılık = benzerlik skoru (zaten 0-100 arası)
            probability = similarity_score
            
            # Geçmiş yükselişlerin ortalaması
            rise_pcts = [r['rise_pct'] for r in historical_rises]
            durations = [r['duration'] for r in historical_rises]
            
            avg_rise = np.mean(rise_pcts)
            std_rise = np.std(rise_pcts)
            avg_duration = np.mean(durations)
            
            # Tahmini yükseliş aralığı
            if probability >= 80:
                # Yüksek benzerlik - üst çeyrek
                estimated_rise_min = avg_rise - 0.5 * std_rise
                estimated_rise_max = avg_rise + 1.0 * std_rise
            elif probability >= 70:
                # Orta-yüksek benzerlik
                estimated_rise_min = avg_rise - 0.7 * std_rise
                estimated_rise_max = avg_rise + 0.7 * std_rise
            elif probability >= 60:
                # Orta benzerlik
                estimated_rise_min = avg_rise - 1.0 * std_rise
                estimated_rise_max = avg_rise + 0.5 * std_rise
            else:
                # Düşük benzerlik
                estimated_rise_min = avg_rise - 1.5 * std_rise
                estimated_rise_max = avg_rise
            
            # Negatif olmasın
            estimated_rise_min = max(0, estimated_rise_min)
            estimated_rise_max = max(estimated_rise_min + 5, estimated_rise_max)
            
            return {
                'probability': float(probability),
                'estimated_rise_min': float(estimated_rise_min),
                'estimated_rise_max': float(estimated_rise_max),
                'estimated_duration': int(avg_duration),
                'historical_avg_rise': float(avg_rise),
                'historical_std_rise': float(std_rise),
                'confidence': 'high' if probability >= 80 else 'medium' if probability >= 70 else 'low'
            }
        
        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return {
                'probability': 0.0,
                'estimated_rise_min': 0.0,
                'estimated_rise_max': 0.0,
                'estimated_duration': 0,
                'confidence': 'none'
            }
