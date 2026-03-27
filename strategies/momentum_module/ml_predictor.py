"""
PATTERN-BASED TAHMİN MOTORU
Yükselen hisselerin ortak özelliklerini bulur ve benzerlik hesaplar
"""

import pandas as pd
import numpy as np
import logging
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from scipy import stats

logger = logging.getLogger(__name__)

class MLPredictor:
    """Pattern-based tahmin motoru"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        # Yükselen hisselerin profili
        self.rising_profile = {}  # Her özellik için: mean, std, min, max, median
        self.rising_features_scaled = None  # Normalize edilmiş yükselen hisse özellikleri
        self.rising_symbols = []
        
        # Clustering (farklı yükseliş tipleri için)
        self.n_clusters = 3
        self.kmeans = None
        self.cluster_profiles = {}
    
    def train(self, analyzed_stocks, min_rise_threshold=20.0):
        """
        Yükselen hisselerin ortak özelliklerini öğren
        
        Args:
            analyzed_stocks: Analiz edilmiş yükselen hisseler
            min_rise_threshold: Minimum yükseliş eşiği (kullanılmıyor ama uyumluluk için)
        
        Returns:
            dict: Eğitim sonuçları
        """
        try:
            logger.info("🎯 PATTERN-BASED MODEL EĞİTİLİYOR...")
            logger.info("📊 Amaç: Yükselen hisselerin ortak özelliklerini bulmak")
            
            if not analyzed_stocks:
                return {
                    'success': False,
                    'message': 'Eğitim verisi bulunamadı'
                }
            
            # DataFrame'e çevir
            df = pd.DataFrame(analyzed_stocks)
            
            # Özellikler
            exclude_cols = ['symbol', 'analysis_date', 'rise_pct', 'future_rise', 
                          'data_points', 'current_price', 'future_rise_pct', 'rise_duration']
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if not feature_cols:
                logger.error("Özellik bulunamadı")
                return {
                    'success': False,
                    'message': 'Özellik bulunamadı'
                }
            
            X = df[feature_cols].copy()
            
            # NaN ve inf temizle
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
            
            self.feature_names = feature_cols
            self.rising_symbols = df['symbol'].tolist() if 'symbol' in df.columns else []
            
            logger.info(f"✓ {len(X)} yükselen hissenin özellikleri analiz ediliyor...")
            logger.info(f"  Özellik sayısı: {len(feature_cols)}")
            
            # 1. HER ÖZELLİK İÇİN İSTATİSTİKLER
            logger.info("  📊 Özellik istatistikleri hesaplanıyor...")
            
            for col in feature_cols:
                values = X[col].values
                
                self.rising_profile[col] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75)),
                    'iqr': float(np.percentile(values, 75) - np.percentile(values, 25))
                }
            
            # 2. VERİYİ NORMALIZE ET
            logger.info("  🔄 Veri normalize ediliyor...")
            X_scaled = self.scaler.fit_transform(X)
            self.rising_features_scaled = X_scaled
            
            # 3. CLUSTERING (Farklı yükseliş tipleri)
            if len(X) >= self.n_clusters * 2:
                logger.info(f"  🎯 {self.n_clusters} farklı yükseliş tipi aranıyor...")
                
                self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                clusters = self.kmeans.fit_predict(X_scaled)
                
                # Her cluster için profil
                for cluster_id in range(self.n_clusters):
                    cluster_mask = clusters == cluster_id
                    cluster_data = X[cluster_mask]
                    
                    cluster_profile = {}
                    for col in feature_cols:
                        values = cluster_data[col].values
                        cluster_profile[col] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'count': int(np.sum(cluster_mask))
                        }
                    
                    self.cluster_profiles[cluster_id] = cluster_profile
                    
                    logger.info(f"    Cluster {cluster_id}: {np.sum(cluster_mask)} hisse")
            
            # 4. ÖNEMLİ ÖZELLİKLERİ BUL (Varyans analizi)
            logger.info("  🔍 En ayırt edici özellikler bulunuyor...")
            
            feature_importance = []
            for col in feature_cols:
                values = X[col].values
                
                # Coefficient of variation (CV) - ne kadar değişken?
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if mean_val != 0:
                    cv = abs(std_val / mean_val)
                else:
                    cv = 0
                
                # Range (normalized)
                range_val = (np.max(values) - np.min(values)) / (abs(mean_val) + 1e-10)
                
                # Importance score
                importance = cv + range_val
                
                feature_importance.append({
                    'feature': col,
                    'importance': importance,
                    'cv': cv,
                    'range': range_val
                })
            
            feature_importance_df = pd.DataFrame(feature_importance).sort_values('importance', ascending=False)
            top_features = feature_importance_df.head(10)
            
            logger.info("✓ PATTERN-BASED MODEL EĞİTİLDİ")
            logger.info(f"  Yükselen hisse sayısı: {len(X)}")
            logger.info(f"  Özellik sayısı: {len(feature_cols)}")
            if self.kmeans:
                logger.info(f"  Yükseliş tipi sayısı: {self.n_clusters}")
            logger.info("  En ayırt edici özellikler:")
            for idx, row in top_features.head(5).iterrows():
                profile = self.rising_profile[row['feature']]
                logger.info(f"    • {row['feature']}: {profile['mean']:.2f} ± {profile['std']:.2f} (önem: {row['importance']:.3f})")
            
            self.is_trained = True
            
            return {
                'success': True,
                'n_samples': len(X),
                'n_features': len(feature_cols),
                'n_clusters': self.n_clusters if self.kmeans else 0,
                'top_features': top_features.head(10).to_dict('records'),
                'message': 'Pattern-based model başarıyla eğitildi'
            }
        
        except Exception as e:
            logger.error(f"Model eğitim hatası: {e}", exc_info=True)
            return {
                'success': False,
                'message': f'Eğitim hatası: {str(e)}'
            }
    
    def predict(self, stock_features):
        """
        Bir hissenin yükselen hisselere benzerliğini hesapla
        
        Args:
            stock_features: dict, hisse özellikleri
        
        Returns:
            dict: Tahmin sonucu
        """
        try:
            if not self.is_trained:
                logger.error("Model henüz eğitilmedi")
                return {
                    'probability': 0.0,
                    'prediction': 0,
                    'confidence': 0.0
                }
            
            # Özellikleri hazırla
            feature_vector = []
            for feat in self.feature_names:
                value = stock_features.get(feat, 0)
                if pd.isna(value) or np.isinf(value):
                    value = 0
                feature_vector.append(float(value))
            
            # ❗ DataFrame olarak ver → sklearn feature-names uyarısı olmaz
            X_df = pd.DataFrame([feature_vector], columns=self.feature_names)
            X_scaled = self.scaler.transform(X_df)
            
            # 1. GENEL BENZERLİK (Tüm yükselen hisselerle)
            similarities = cosine_similarity(X_scaled, self.rising_features_scaled)[0]
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            top5_similarity = np.mean(np.sort(similarities)[-5:])  # En benzer 5'in ortalaması
            
            # 2. PATTERN MATCH SKORU (Her özellik için profil uyumu)
            pattern_scores = []
            
            for feat in self.feature_names:
                value = stock_features.get(feat, 0)
                if pd.isna(value) or np.isinf(value):
                    value = 0
                
                profile = self.rising_profile[feat]
                
                # Z-score bazlı benzerlik
                mean = profile['mean']
                std = profile['std']
                
                if std > 0:
                    z_score = abs((value - mean) / std)
                    
                    # Z-score'u benzerlik skoruna çevir (0-100)
                    if z_score <= 1:
                        score = 100  # Çok benzer
                    elif z_score <= 2:
                        score = 80 - (z_score - 1) * 30  # 80-50 arası
                    elif z_score <= 3:
                        score = 50 - (z_score - 2) * 25  # 50-25 arası
                    else:
                        score = max(0, 25 - (z_score - 3) * 10)  # 25-0 arası
                else:
                    # Std = 0 ise direkt karşılaştır
                    if abs(value - mean) < 0.01:
                        score = 100
                    else:
                        score = 0
                
                pattern_scores.append(score)
            
            pattern_match = np.mean(pattern_scores)
            
            # 3. CLUSTER BENZERLİĞİ (Eğer clustering yapıldıysa)
            cluster_match = 0
            best_cluster = -1
            
            if self.kmeans:
                # KMeans numpy ile eğitildi → numpy ver
                cluster_id = self.kmeans.predict(X_scaled)[0]
                best_cluster = int(cluster_id)
                
                # Bu cluster'a ne kadar benziyor?
                cluster_center = self.kmeans.cluster_centers_[cluster_id]
                distance = euclidean_distances(X_scaled, cluster_center.reshape(1, -1))[0][0]
                
                # Mesafeyi benzerliğe çevir (0-100)
                cluster_match = max(0, 100 - distance * 20)
            
            # 4. BİRLEŞİK SKOR
            # Tüm bileşenler 0-100 birimine getirildi
            avg_sim_pct = avg_similarity * 100
            
            if self.kmeans:
                # avg_sim %40 + pattern %35 + cluster %25
                probability = (
                    avg_sim_pct   * 0.40 +
                    pattern_match * 0.35 +
                    cluster_match * 0.25
                )
            else:
                # avg_sim %50 + pattern %50
                probability = (
                    avg_sim_pct   * 0.50 +
                    pattern_match * 0.50
                )
            
            # Normalize (0-100 arası)
            probability = np.clip(probability, 0, 100)
            
            # Tahmin
            prediction = 1 if probability >= 70 else 0
            
            # Güven skoru
            confidence = abs(probability - 50) * 2
            confidence = np.clip(confidence, 0, 100)
            
            return {
                'probability': float(probability),
                'prediction': int(prediction),
                'confidence': float(confidence),
                'avg_similarity': float(avg_similarity * 100),
                'max_similarity': float(max_similarity * 100),
                'top5_similarity': float(top5_similarity * 100),
                'pattern_match': float(pattern_match),
                'cluster_match': float(cluster_match),
                'best_cluster': best_cluster
            }
        
        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return {
                'probability': 0.0,
                'prediction': 0,
                'confidence': 0.0
            }
    
    def estimate_rise_range(self, probability, analyzed_rising_stocks):
        """
        Olasılığa göre tahmini yükseliş aralığını hesapla
        
        Args:
            probability: Tahmin olasılığı (0-100)
            analyzed_rising_stocks: Eğitim verisi
        
        Returns:
            tuple: (min_rise, max_rise)
        """
        try:
            rises = [s.get('rise_pct', 0) for s in analyzed_rising_stocks 
                    if s.get('rise_pct', 0) > 0]
            
            if not rises:
                return (20.0, 30.0)
            
            rises = np.array(rises)
            
            # Benzerlik oranına göre yükseliş tahmini
            if probability >= 85:
                min_rise = np.percentile(rises, 50)
                max_rise = np.percentile(rises, 90)
            elif probability >= 75:
                min_rise = np.percentile(rises, 40)
                max_rise = np.percentile(rises, 75)
            elif probability >= 65:
                min_rise = np.percentile(rises, 25)
                max_rise = np.percentile(rises, 60)
            else:
                min_rise = np.percentile(rises, 10)
                max_rise = np.percentile(rises, 40)
            
            return (float(min_rise), float(max_rise))
        
        except Exception as e:
            logger.error(f"Yükseliş aralığı hesaplama hatası: {e}")
            return (20.0, 30.0)
    
    def get_feature_comparison(self, stock_features):
        """
        Hissenin özelliklerini yükselen hisse profili ile karşılaştır
        
        Args:
            stock_features: dict, hisse özellikleri
        
        Returns:
            list: Özellik karşılaştırmaları
        """
        try:
            if not self.is_trained:
                return []
            
            comparisons = []
            
            for feat in self.feature_names:
                value = stock_features.get(feat, 0)
                if pd.isna(value) or np.isinf(value):
                    value = 0
                
                profile = self.rising_profile[feat]
                
                # Z-score
                mean = profile['mean']
                std = profile['std']
                
                if std > 0:
                    z_score = (value - mean) / std
                else:
                    z_score = 0
                
                # Profil içinde mi?
                in_range = profile['q25'] <= value <= profile['q75']
                
                comparisons.append({
                    'feature': feat,
                    'current_value': float(value),
                    'profile_mean': float(mean),
                    'profile_std': float(std),
                    'profile_median': float(profile['median']),
                    'profile_range': [float(profile['q25']), float(profile['q75'])],
                    'z_score': float(z_score),
                    'in_range': in_range,
                    'deviation': abs(z_score)
                })
            
            # Z-score'a göre sırala (en çok sapan özellikler)
            comparisons.sort(key=lambda x: x['deviation'], reverse=True)
            
            return comparisons
        
        except Exception as e:
            logger.error(f"Özellik karşılaştırma hatası: {e}")
            return []
    
    def save_model(self, filepath):
        """Modeli kaydet"""
        try:
            model_data = {
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'rising_profile': self.rising_profile,
                'rising_features_scaled': self.rising_features_scaled,
                'rising_symbols': self.rising_symbols,
                'kmeans': self.kmeans,
                'cluster_profiles': self.cluster_profiles,
                'n_clusters': self.n_clusters
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"✓ Model kaydedildi: {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {e}")
            return False
    
    def load_model(self, filepath):
        """Modeli yükle"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            self.rising_profile = model_data['rising_profile']
            self.rising_features_scaled = model_data['rising_features_scaled']
            self.rising_symbols = model_data['rising_symbols']
            self.kmeans = model_data.get('kmeans')
            self.cluster_profiles = model_data.get('cluster_profiles', {})
            self.n_clusters = model_data.get('n_clusters', 3)
            
            logger.info(f"✓ Model yüklendi: {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            return False
