"""
SELF ANALYZER - Hissenin Kendi Geçmişi ile Karşılaştırma
Her hisse için: "Bu hisse daha önce tavan yaptığında nasıl bir durumdaydı?"
TXT dosyasındaki tavan tarihlerini kullanır, tavan gününün 1 GÜN ÖNCESİNİ analiz eder.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import yfinance as yf
from advanced_indicators import AdvancedIndicators
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

logger = logging.getLogger(__name__)


class SelfAnalyzer:
    """Hissenin kendi geçmişi ile karşılaştırma motoru"""
    
    def __init__(self, ceiling_threshold: float = 9.0):
        """
        Args:
            ceiling_threshold: Tavan eşiği (varsayılan %9)
        """
        self.self_patterns = {}  # {symbol: [pattern1, pattern2, ...]}
        self.ceiling_threshold = ceiling_threshold
    
    def analyze_stock_ceiling_history(self, symbol: str, ceiling_dates: List[str]) -> List[Dict]:
        """
        Bir hissenin geçmişteki tavan günlerinin 1 GÜN ÖNCESİNİ analiz et
        (TXT'de zaten sadece tavan günleri var, 1 gün öncesi otomatik olarak tavan değil)
        
        Args:
            symbol: Hisse sembolü (örn: THYAO)
            ceiling_dates: Tavan yapma tarihleri listesi (TXT'den)
        
        Returns:
            Tavan öncesi indikatör listesi
        """
        logger.info(f"📊 {symbol} için kendi geçmişi analiz ediliyor...")
        logger.info(f"   Tavan sayısı: {len(ceiling_dates)}")
        logger.info(f"   Tavan tarihleri: {ceiling_dates}")
        
        patterns = []
        
        try:
            # 2 yıllık veri çek
            symbol_yahoo = symbol if symbol.endswith('.IS') else symbol + '.IS'
            ticker = yf.Ticker(symbol_yahoo)
            
            hist = ticker.history(
                period='2y',
                auto_adjust=True,
                actions=False
            )
            
            # Multi-index kontrolü
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.droplevel(1)
            
            if hist.empty or len(hist) < 100:
                logger.warning(f"   {symbol}: Yetersiz veri")
                return []
            
            logger.info(f"   ✓ {len(hist)} günlük veri alındı")
            
            # ✅ Timezone'ları normalize et
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            
            # Her tavan tarihi için
            for ceiling_date_str in ceiling_dates:
                try:
                    ceiling_dt = pd.to_datetime(ceiling_date_str).tz_localize(None)
                    
                    logger.info(f"\n   🎯 Tavan tarihi: {ceiling_date_str}")
                    
                    # Tavan tarihini veri setinde bul
                    if ceiling_dt not in hist.index:
                        # En yakın tarihi bul
                        closest_idx = hist.index.get_indexer([ceiling_dt], method='nearest')[0]
                        ceiling_dt = hist.index[closest_idx]
                        logger.warning(f"      Tavan tarihi bulunamadı, en yakın: {ceiling_dt.strftime('%Y-%m-%d')}")
                    
                    # Tavan tarihinin index'ini bul
                    ceiling_index = hist.index.get_loc(ceiling_dt)
                    
                    if ceiling_index <= 1:
                        logger.warning(f"      Tavan tarihi çok erken, atlanıyor")
                        continue
                    
                    # ✅ BASİT MANTIK: 1 GÜN ÖNCE (1 işlem günü önce)
                    analysis_index = ceiling_index - 1
                    analysis_date = hist.index[analysis_index]
                    
                    days_before = (ceiling_dt - analysis_date).days
                    logger.info(f"      ✓ Analiz günü: {analysis_date.strftime('%Y-%m-%d')} (tavana {days_before} gün kala)")
                    
                    if analysis_index < 100:
                        logger.warning(f"      Yetersiz geçmiş veri")
                        continue
                    
                    # O tarihe kadar olan veriyi al (analysis_date dahil)
                    hist_until_date = hist.iloc[:analysis_index + 1]
                    
                    if len(hist_until_date) < 100:
                        logger.warning(f"      Yetersiz geçmiş veri ({len(hist_until_date)} gün)")
                        continue
                    
                    # İndikatörleri hesapla
                    indicators = AdvancedIndicators.calculate_all_indicators(hist_until_date)
                    
                    # Metadata ekle
                    indicators['ceiling_date'] = ceiling_date_str
                    indicators['analysis_date'] = analysis_date.strftime('%Y-%m-%d')
                    indicators['days_before_ceiling'] = days_before
                    
                    # Tavan günündeki yükselişi hesapla
                    ceiling_close = float(hist['Close'].iloc[ceiling_index])
                    prev_close = float(hist['Close'].iloc[analysis_index])
                    ceiling_rise = ((ceiling_close - prev_close) / prev_close) * 100
                    indicators['ceiling_rise_pct'] = ceiling_rise
                    
                    patterns.append(indicators)
                    
                    logger.info(f"      ✅ Başarıyla analiz edildi (Tavan yükselişi: %{ceiling_rise:.2f})")
                
                except Exception as e:
                    logger.error(f"      ❌ {ceiling_date_str} analiz hatası: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            logger.info(f"\n   ✅ {symbol}: {len(patterns)} tavan öncesi pattern bulundu")
            
            # Kaydet
            if patterns:
                self.self_patterns[symbol] = patterns
            
            return patterns
        
        except Exception as e:
            logger.error(f"   ❌ {symbol} geçmiş analiz hatası: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def calculate_self_similarity(self, symbol: str, current_indicators: Dict) -> Dict:
        """
        Mevcut durumun, hissenin kendi tavan öncesi durumlarına benzerliği
        
        Args:
            symbol: Hisse sembolü
            current_indicators: Şu anki indikatörler
        
        Returns:
            Benzerlik skoru ve detaylar
        """
        if symbol not in self.self_patterns:
            logger.warning(f"{symbol}: Kendi geçmişi yok")
            return {
                'similarity_score': 0.0,
                'max_similarity': 0.0,
                'matched_patterns': 0,
                'confidence': 'low',
                'confidence_emoji': '❌',
                'pattern_details': [],
                'reason': 'Kendi geçmişi bulunamadı'
            }
        
        patterns = self.self_patterns[symbol]
        
        if not patterns:
            return {
                'similarity_score': 0.0,
                'max_similarity': 0.0,
                'matched_patterns': 0,
                'confidence': 'low',
                'confidence_emoji': '❌',
                'pattern_details': [],
                'reason': 'Pattern yok'
            }
        
        logger.info(f"🔍 {symbol}: Kendi geçmişi ile karşılaştırılıyor...")
        logger.info(f"   Karşılaştırılacak pattern: {len(patterns)}")
        
        # Özellik vektörlerini hazırla
        feature_names = [k for k in current_indicators.keys() 
                        if isinstance(current_indicators[k], (int, float)) 
                        and not np.isnan(current_indicators[k]) 
                        and not np.isinf(current_indicators[k])]
        
        current_vector = np.array([current_indicators.get(f, 0) for f in feature_names]).reshape(1, -1)
        
        # Sıfır varyans kontrolü
        if np.std(current_vector) == 0:
            logger.warning(f"   {symbol}: Sıfır varyans, benzerlik hesaplanamıyor")
            return {
                'similarity_score': 0.0,
                'max_similarity': 0.0,
                'matched_patterns': 0,
                'confidence': 'low',
                'confidence_emoji': '❌',
                'pattern_details': [],
                'reason': 'Sıfır varyans'
            }
        
        # Her pattern ile benzerlik hesapla
        similarities = []
        pattern_details = []
        
        for i, pattern in enumerate(patterns):
            pattern_vector = np.array([pattern.get(f, 0) for f in feature_names]).reshape(1, -1)
            
            # Cosine similarity
            try:
                sim = cosine_similarity(current_vector, pattern_vector)[0][0]
                sim = max(0, min(1, sim))  # 0-1 arası normalize
                
                similarities.append(sim)
                
                pattern_details.append({
                    'ceiling_date': pattern.get('ceiling_date', 'unknown'),
                    'analysis_date': pattern.get('analysis_date', 'unknown'),
                    'days_before_ceiling': pattern.get('days_before_ceiling', 0),
                    'ceiling_rise_pct': pattern.get('ceiling_rise_pct', 0),
                    'similarity': float(sim * 100),
                    'key_matches': self._compare_key_indicators(current_indicators, pattern)
                })
                
                logger.info(f"   Pattern {i+1} ({pattern.get('ceiling_date', 'unknown')}): "
                          f"Benzerlik %{sim*100:.1f} | "
                          f"Analiz günü: {pattern.get('analysis_date', 'unknown')} | "
                          f"Tavana {pattern.get('days_before_ceiling', 0)} gün kala")
            
            except Exception as e:
                logger.error(f"   Pattern {i+1} benzerlik hatası: {e}")
                continue
        
        if not similarities:
            return {
                'similarity_score': 0.0,
                'max_similarity': 0.0,
                'matched_patterns': 0,
                'confidence': 'low',
                'confidence_emoji': '❌',
                'pattern_details': [],
                'reason': 'Benzerlik hesaplanamadı'
            }
        
        # İstatistikler
        avg_similarity = np.mean(similarities) * 100
        max_similarity = np.max(similarities) * 100
        min_similarity = np.min(similarities) * 100
        
        # Güven seviyesi (max similarity'ye göre)
        if max_similarity >= 85:
            confidence = 'very_high'
            confidence_emoji = '⭐⭐⭐'
        elif max_similarity >= 75:
            confidence = 'high'
            confidence_emoji = '⭐⭐'
        elif max_similarity >= 65:
            confidence = 'medium'
            confidence_emoji = '⭐'
        else:
            confidence = 'low'
            confidence_emoji = '❌'
        
        logger.info(f"   ✅ Ortalama benzerlik: %{avg_similarity:.1f}")
        logger.info(f"   ✅ Maksimum benzerlik: %{max_similarity:.1f}")
        logger.info(f"   ✅ Güven seviyesi: {confidence} {confidence_emoji}")
        
        # En benzer pattern'i bul
        best_pattern_index = np.argmax(similarities)
        best_pattern = pattern_details[best_pattern_index]
        
        return {
            'similarity_score': float(avg_similarity),
            'max_similarity': float(max_similarity),
            'min_similarity': float(min_similarity),
            'matched_patterns': len(similarities),
            'confidence': confidence,
            'confidence_emoji': confidence_emoji,
            'pattern_details': pattern_details,
            'best_match': best_pattern,
            'reason': f"{len(similarities)} geçmiş tavan ile karşılaştırıldı"
        }
    
    def _compare_key_indicators(self, current: Dict, pattern: Dict) -> Dict:
        """Anahtar göstergeleri karşılaştır"""
        key_indicators = [
            'rsi_14', 'volume_breakout_score', 'pre_ceiling_squeeze',
            'ceiling_breakout_prob', 'distance_to_ceiling', 
            'institutional_accumulation', 'trend_alignment',
            'compression_score', 'breakout_strength'
        ]
        
        matches = {}
        
        for indicator in key_indicators:
            current_val = current.get(indicator, 0)
            pattern_val = pattern.get(indicator, 0)
            
            if pattern_val != 0:
                diff_pct = abs((current_val - pattern_val) / pattern_val) * 100
                
                if diff_pct < 10:
                    match_level = 'excellent'
                    match_emoji = '🟢'
                elif diff_pct < 20:
                    match_level = 'good'
                    match_emoji = '🟡'
                elif diff_pct < 30:
                    match_level = 'fair'
                    match_emoji = '🟠'
                else:
                    match_level = 'poor'
                    match_emoji = '🔴'
                
                matches[indicator] = {
                    'current': float(current_val),
                    'pattern': float(pattern_val),
                    'diff_pct': float(diff_pct),
                    'match_level': match_level,
                    'match_emoji': match_emoji
                }
        
        return matches
    
    def get_self_pattern_summary(self, symbol: str) -> Dict:
        """Hissenin kendi pattern özetini al"""
        if symbol not in self.self_patterns:
            return {
                'symbol': symbol,
                'pattern_count': 0,
                'message': 'Kendi geçmişi bulunamadı'
            }
        
        patterns = self.self_patterns[symbol]
        
        if not patterns:
            return {
                'symbol': symbol,
                'pattern_count': 0,
                'message': 'Pattern yok'
            }
        
        # Ortak özellikler
        df = pd.DataFrame(patterns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = {
            'symbol': symbol,
            'pattern_count': len(patterns),
            'ceiling_dates': [p.get('ceiling_date', 'unknown') for p in patterns],
            'analysis_dates': [p.get('analysis_date', 'unknown') for p in patterns],
            'avg_days_before': float(df['days_before_ceiling'].mean()) if 'days_before_ceiling' in df else 0,
            'avg_ceiling_rise': float(df['ceiling_rise_pct'].mean()) if 'ceiling_rise_pct' in df else 0,
            'common_features': {}
        }
        
        for col in numeric_cols:
            values = df[col].dropna()
            
            if len(values) > 0:
                summary['common_features'][col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max())
                }
        
        return summary
    
    def save_patterns(self, filename=None):
        if filename is None:
            import os
            filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'self_patterns.json')
        """Paternleri kaydet"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # JSON serializable hale getir
            serializable = {}
            for symbol, patterns in self.self_patterns.items():
                serializable[symbol] = []
                for pattern in patterns:
                    clean_pattern = {}
                    for k, v in pattern.items():
                        if isinstance(v, (int, float, str)):
                            if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                                clean_pattern[k] = v
                    serializable[symbol].append(clean_pattern)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Self patterns kaydedildi: {filename}")
        
        except Exception as e:
            logger.error(f"Self patterns kaydetme hatası: {e}")
    
    def load_patterns(self, filename=None):
        if filename is None:
            import os
            filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'self_patterns.json')
        """Paternleri yükle"""
        try:
            if not os.path.exists(filename):
                logger.info(f"Self patterns dosyası bulunamadı: {filename}")
                return False
            
            with open(filename, 'r', encoding='utf-8') as f:
                self.self_patterns = json.load(f)
            
            logger.info(f"✓ Self patterns yüklendi: {len(self.self_patterns)} hisse")
            return True
        
        except Exception as e:
            logger.error(f"Self patterns yükleme hatası: {e}")
            return False
