"""
PATTERN EŞLEŞTİRME MODÜLÜ
Yükselen hisselerin ortak özelliklerini bulur ve benzerlik hesaplar
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json

logger = logging.getLogger(__name__)

class PatternMatcher:
    """Pattern eşleştirme ve benzerlik hesaplama"""
    
    def __init__(self):
        self.common_patterns = {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def find_common_patterns(self, rising_stocks_features):
        """
        Yükselen hisselerin ortak özelliklerini bul
        
        Args:
            rising_stocks_features: List of dicts, her hissenin özellikleri
        
        Returns:
            dict: Ortak paternler
        """
        try:
            if not rising_stocks_features or len(rising_stocks_features) == 0:
                logger.error("Özellik verisi bulunamadı")
                return {}
            
            logger.info(f"🔍 {len(rising_stocks_features)} hissenin ortak paternleri aranıyor...")
            
            # DataFrame'e çevir
            df = pd.DataFrame(rising_stocks_features)
            
            # Sadece numerik sütunları al
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Hariç tutulacak sütunlar
            exclude_cols = ['data_points', 'rise_pct', 'future_rise']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if len(numeric_cols) == 0:
                logger.error("Numerik özellik bulunamadı")
                return {}
            
            patterns = {}
            
            # Her özellik için ortak aralık bul
            for col in numeric_cols:
                values = df[col].dropna()
                
                if len(values) == 0:
                    continue
                
                # İstatistikler
                mean_val = values.mean()
                std_val = values.std()
                min_val = values.min()
                max_val = values.max()
                median_val = values.median()
                q25 = values.quantile(0.25)
                q75 = values.quantile(0.75)
                
                # Ortak aralık (IQR bazlı)
                iqr = q75 - q25
                range_min = max(min_val, q25 - 0.5 * iqr)
                range_max = min(max_val, q75 + 0.5 * iqr)
                
                patterns[col] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'min': float(min_val),
                    'max': float(max_val),
                    'median': float(median_val),
                    'q25': float(q25),
                    'q75': float(q75),
                    'range_min': float(range_min),
                    'range_max': float(range_max),
                    'target_range': [float(range_min), float(range_max)]
                }
            
            self.common_patterns = patterns
            self.feature_names = numeric_cols
            
            # Önemli paternleri logla
            logger.info("✓ Ortak paternler bulundu:")
            
            important_features = [
                'rsi_14', 'volume_ratio', 'macd_diff', 
                'bb_position', 'ema_alignment', 'price_change_20d',
                'adx', 'stoch_k', 'momentum_10'
            ]
            
            for feat in important_features:
                if feat in patterns:
                    p = patterns[feat]
                    logger.info(f"  • {feat}: {p['range_min']:.2f} - {p['range_max']:.2f} "
                              f"(ort: {p['mean']:.2f}, med: {p['median']:.2f})")
            
            return patterns
        
        except Exception as e:
            logger.error(f"Pattern bulma hatası: {e}")
            return {}
    
    def calculate_similarity(self, stock_features, rising_stocks_features):
        """
        Bir hissenin yükselen hisselerle benzerliğini hesapla
        
        Args:
            stock_features: dict, analiz edilen hissenin özellikleri
            rising_stocks_features: list of dicts, yükselen hisselerin özellikleri
        
        Returns:
            dict: Benzerlik skorları
        """
        try:
            if not stock_features or not rising_stocks_features:
                return {
                    'average_similarity': 0.0,
                    'max_similarity': 0.0,
                    'similar_stocks': []
                }
            
            # Feature vektörlerini oluştur
            feature_names = self.feature_names
            
            if not feature_names:
                # İlk kez çağrılıyorsa feature isimlerini al
                feature_names = [k for k in stock_features.keys() 
                               if isinstance(stock_features[k], (int, float)) and 
                               k not in ['data_points', 'rise_pct', 'future_rise']]
                self.feature_names = feature_names
            
            # Hedef hissenin vektörü
            stock_vector = []
            for f in feature_names:
                val = stock_features.get(f, 0)
                if pd.isna(val) or np.isinf(val):
                    val = 0
                stock_vector.append(val)
            
            # Yükselen hisselerin vektörleri
            rising_vectors = []
            rising_symbols = []
            
            for rs in rising_stocks_features:
                vector = []
                for f in feature_names:
                    val = rs.get(f, 0)
                    if pd.isna(val) or np.isinf(val):
                        val = 0
                    vector.append(val)
                rising_vectors.append(vector)
                rising_symbols.append(rs.get('symbol', 'UNKNOWN'))
            
            # Normalize et
            all_vectors = [stock_vector] + rising_vectors
            
            try:
                all_vectors_scaled = self.scaler.fit_transform(all_vectors)
            except:
                # Normalizasyon başarısız olursa ham vektörleri kullan
                all_vectors_scaled = np.array(all_vectors)
            
            stock_vector_scaled = all_vectors_scaled[0].reshape(1, -1)
            rising_vectors_scaled = all_vectors_scaled[1:]
            
            # Cosine similarity hesapla
            similarities = cosine_similarity(stock_vector_scaled, rising_vectors_scaled)[0]
            
            # Sonuçları düzenle
            similar_stocks = []
            for i, sim in enumerate(similarities):
                similar_stocks.append({
                    'symbol': rising_symbols[i],
                    'similarity': float(sim * 100)  # Yüzdeye çevir
                })
            
            # Sırala (en yüksek benzerlik önce)
            similar_stocks.sort(key=lambda x: x['similarity'], reverse=True)
            
            # İstatistikler
            avg_similarity = float(np.mean(similarities) * 100)
            max_similarity = float(np.max(similarities) * 100)
            
            return {
                'average_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'similar_stocks': similar_stocks[:5]  # En benzer 5 hisse
            }
        
        except Exception as e:
            logger.error(f"Benzerlik hesaplama hatası: {e}")
            return {
                'average_similarity': 0.0,
                'max_similarity': 0.0,
                'similar_stocks': []
            }
    
    def calculate_pattern_score(self, stock_features):
        """
        Bir hissenin ortak paternlere uygunluk skorunu hesapla
        
        Args:
            stock_features: dict, hisse özellikleri
        
        Returns:
            float: Pattern skoru (0-100)
        """
        try:
            if not self.common_patterns:
                logger.warning("Ortak pattern bulunamadı")
                return 0.0
            
            scores = []
            feature_scores = {}
            
            for feature, pattern in self.common_patterns.items():
                if feature not in stock_features:
                    continue
                
                value = stock_features[feature]
                
                # NaN kontrolü
                if pd.isna(value) or np.isinf(value):
                    continue
                
                # Hedef aralık
                target_min = pattern['range_min']
                target_max = pattern['range_max']
                mean = pattern['mean']
                std = pattern['std']
                
                # Skor hesapla
                if target_min <= value <= target_max:
                    # Aralık içinde - tam puan
                    score = 100.0
                else:
                    # Aralık dışında - uzaklığa göre ceza
                    if std > 0:
                        # Z-score bazlı
                        z_score = abs((value - mean) / std)
                        
                        if z_score <= 1:
                            score = 100.0 - (z_score * 20)
                        elif z_score <= 2:
                            score = 80.0 - ((z_score - 1) * 30)
                        elif z_score <= 3:
                            score = 50.0 - ((z_score - 2) * 25)
                        else:
                            score = max(0, 25.0 - ((z_score - 3) * 10))
                    else:
                        # Standart sapma 0 ise basit mesafe
                        distance = abs(value - mean)
                        range_size = target_max - target_min
                        
                        if range_size > 0:
                            penalty = min(distance / range_size, 1.0)
                            score = 100.0 * (1.0 - penalty)
                        else:
                            score = 100.0 if distance < 0.01 else 0.0
                
                scores.append(score)
                feature_scores[feature] = score
            
            # Ortalama skor
            if scores:
                pattern_score = np.mean(scores)
            else:
                pattern_score = 0.0
            
            return float(pattern_score)
        
        except Exception as e:
            logger.error(f"Pattern skoru hesaplama hatası: {e}")
            return 0.0
    
    def get_pattern_details(self, stock_features):
        """
        Detaylı pattern analizi
        
        Args:
            stock_features: dict, hisse özellikleri
        
        Returns:
            dict: Detaylı analiz
        """
        try:
            details = {
                'matching_features': [],
                'non_matching_features': [],
                'overall_score': 0.0
            }
            
            if not self.common_patterns:
                return details
            
            scores = []
            
            for feature, pattern in self.common_patterns.items():
                if feature not in stock_features:
                    continue
                
                value = stock_features[feature]
                
                if pd.isna(value) or np.isinf(value):
                    continue
                
                target_min = pattern['range_min']
                target_max = pattern['range_max']
                mean = pattern['mean']
                
                is_matching = target_min <= value <= target_max
                
                feature_detail = {
                    'feature': feature,
                    'current_value': float(value),
                    'target_min': float(target_min),
                    'target_max': float(target_max),
                    'target_mean': float(mean),
                    'is_matching': is_matching
                }
                
                if is_matching:
                    details['matching_features'].append(feature_detail)
                    scores.append(100.0)
                else:
                    details['non_matching_features'].append(feature_detail)
                    # Kısmi skor
                    if pattern['std'] > 0:
                        z_score = abs((value - mean) / pattern['std'])
                        score = max(0, 100.0 - (z_score * 30))
                        scores.append(score)
            
            if scores:
                details['overall_score'] = float(np.mean(scores))
            
            return details
        
        except Exception as e:
            logger.error(f"Pattern detay hatası: {e}")
            return {
                'matching_features': [],
                'non_matching_features': [],
                'overall_score': 0.0
            }
    
    def save_patterns(self, filepath):
        """Paternleri dosyaya kaydet"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.common_patterns, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Paternler kaydedildi: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Pattern kaydetme hatası: {e}")
            return False
    
    def load_patterns(self, filepath):
        """Paternleri dosyadan yükle"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.common_patterns = json.load(f)
            self.feature_names = list(self.common_patterns.keys())
            logger.info(f"✓ Paternler yüklendi: {filepath} ({len(self.feature_names)} özellik)")
            return True
        except Exception as e:
            logger.error(f"Pattern yükleme hatası: {e}")
            return False
