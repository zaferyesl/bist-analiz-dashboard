"""
Pattern Analiz Motoru
Tavan olan hisselerin ortak özelliklerini çıkarır
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List , Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from advanced_indicators import AdvancedIndicators

logger = logging.getLogger(__name__)


class CeilingPatternAnalyzer:
    """Tavan olan hisselerin ortak özelliklerini çıkarır"""
    
    def __init__(self):
        self.pattern_database = {}
        self.load_patterns()
    
    def load_patterns(self):
        """Önceden bulunan paternleri yükle"""
        try:
            import os
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'pattern_database.json')
            with open(db_path, 'r', encoding='utf-8') as f:
                self.pattern_database = json.load(f)
            logger.info(f"Pattern veritabanı yüklendi: {len(self.pattern_database.get('common_features', {}))} özellik")
        except FileNotFoundError:
            logger.info("Pattern veritabanı bulunamadı, yeni oluşturulacak")
            self.pattern_database = {
                'common_features': {},
                'threshold_values': {},
                'success_rate': 0.0,
                'total_samples': 0,
                'last_update': None
            }
        except Exception as e:
            logger.error(f"Pattern veritabanı yükleme hatası: {e}")
            self.pattern_database = {
                'common_features': {},
                'threshold_values': {},
                'success_rate': 0.0,
                'total_samples': 0,
                'last_update': None
            }
    
    def save_patterns(self):
        """Paternleri kaydet"""
        try:
            os.makedirs('data', exist_ok=True)
            import os
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'pattern_database.json')
            with open(db_path, 'w', encoding='utf-8') as f:
                json.dump(self.pattern_database, f, indent=2, ensure_ascii=False)
            logger.info("Pattern veritabanı kaydedildi")
        except Exception as e:
            logger.error(f"Pattern veritabanı kaydetme hatası: {e}")
    
    def analyze_ceiling_stock(self, symbol: str, ceiling_date: str, ceiling_pct: float) -> Dict:
        """
        Tavan olan hissenin 1 gün önceki özelliklerini analiz et
        
        Args:
            symbol: Hisse sembolü (örn: THYAO)
            ceiling_date: Tavan yapma tarihi (YYYY-MM-DD)
            ceiling_pct: Tavan yüzde artışı
        
        Returns:
            İndikatör sözlüğü veya None
        """
        try:
            logger.info(f"Analiz ediliyor: {symbol} - {ceiling_date} (%{ceiling_pct})")
            
            # Tavan tarihinden 1 gün önce
            ceiling_dt = datetime.strptime(ceiling_date, '%Y-%m-%d')
            analysis_date = ceiling_dt - timedelta(days=1)
            
            # 1 yıllık veri çek (yfinance 1.0 uyumlu)
            start_date = analysis_date - timedelta(days=400)  # Biraz fazla al
            end_date = analysis_date + timedelta(days=1)  # Analysis date'i dahil et
            
            symbol_yahoo = symbol if symbol.endswith('.IS') else symbol + '.IS'
            
            logger.info(f"  Veri çekiliyor: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
            
            # yfinance 1.0 ile veri çekme
            ticker = yf.Ticker(symbol_yahoo)
            hist = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                auto_adjust=True,
                actions=False
            )
            
            # Multi-index kontrolü
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.droplevel(1)
            
            if hist is None or hist.empty:
                logger.warning(f"  {symbol}: Veri yok")
                return None
            
            if len(hist) < 100:
                logger.warning(f"  {symbol}: Yetersiz veri ({len(hist)} gün)")
                return None
            
            logger.info(f"  {len(hist)} günlük veri alındı")
            
            # Tüm indikatörleri hesapla
            indicators = AdvancedIndicators.calculate_all_indicators(hist)
            
            # Tavan bilgisini ekle
            indicators['ceiling_pct'] = ceiling_pct
            indicators['symbol'] = symbol
            indicators['date'] = ceiling_date
            
            logger.info(f"  ✓ {len(indicators)} indikatör hesaplandı")
            
            return indicators
            
        except Exception as e:
            logger.error(f"  {symbol} analiz hatası: {e}")
            return None
    
    def find_common_patterns(self, ceiling_features: List[Dict]) -> Dict:
        """
        Tavan olan hisselerin ortak özelliklerini bul
        
        Args:
            ceiling_features: Tavan olan hisselerin indikatör listesi
        
        Returns:
            Ortak özellikler ve istatistikler
        """
        if not ceiling_features:
            logger.warning("Ortak pattern bulunamadı: Veri yok")
            return {}
        
        logger.info(f"Ortak paternler aranıyor: {len(ceiling_features)} örnek")
        
        df = pd.DataFrame(ceiling_features)
        
        # Sayısal kolonları al
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        common_patterns = {}
        
        for col in numeric_cols:
            if col == 'ceiling_pct':
                continue
            
            values = df[col].dropna()
            
            if len(values) == 0:
                continue
            
            # İstatistikler
            common_patterns[col] = {
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'q25': float(values.quantile(0.25)),
                'q75': float(values.quantile(0.75))
            }
        
        # Veritabanını güncelle
        self.pattern_database['common_features'] = common_patterns
        self.pattern_database['total_samples'] = len(ceiling_features)
        self.pattern_database['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Başarı oranı hesapla
        if 'ceiling_pct' in df.columns:
            success_rate = (df['ceiling_pct'] >= 10).mean() * 100
            self.pattern_database['success_rate'] = float(success_rate)
        
        self.save_patterns()
        
        logger.info(f"✓ Ortak paternler bulundu: {len(common_patterns)} özellik")
        
        return common_patterns
    
    def calculate_similarity_score(self, current_indicators: Dict) -> float:
        """
        Mevcut hissenin tavan paternine benzerlik skoru (0-100)
        
        Args:
            current_indicators: Mevcut hissenin indikatörleri
        
        Returns:
            Benzerlik skoru (0-100)
        """
        if not self.pattern_database.get('common_features'):
            logger.warning("Pattern veritabanı boş, benzerlik hesaplanamıyor")
            return 0.0
        
        common_features = self.pattern_database['common_features']
        
        total_score = 0
        matched_features = 0
        
        for feature, stats in common_features.items():
            if feature not in current_indicators:
                continue
            
            current_value = current_indicators[feature]
            
            # NaN veya inf kontrolü
            if np.isnan(current_value) or np.isinf(current_value):
                continue
            
            mean = stats['mean']
            std = stats['std']
            
            if std > 0:
                # Z-score benzeri normalizasyon
                z_score = abs((current_value - mean) / std)
                
                # 0-2 std arası = iyi benzerlik
                if z_score < 0.5:
                    score = 100
                elif z_score < 1:
                    score = 80
                elif z_score < 1.5:
                    score = 60
                elif z_score < 2:
                    score = 40
                else:
                    score = 0
                
                total_score += score
                matched_features += 1
        
        if matched_features == 0:
            return 0.0
        
        avg_score = total_score / matched_features
        
        return float(avg_score)
    
    def get_pattern_summary(self) -> Dict:
        """Pattern veritabanı özeti"""
        return {
            'total_samples': self.pattern_database.get('total_samples', 0),
            'total_features': len(self.pattern_database.get('common_features', {})),
            'success_rate': self.pattern_database.get('success_rate', 0),
            'last_update': self.pattern_database.get('last_update', 'Hiç güncellenmedi')
        }
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, Dict]]:
        """En önemli özellikleri getir (std'ye göre)"""
        if not self.pattern_database.get('common_features'):
            return []
        
        features = self.pattern_database['common_features']
        
        # Std'ye göre sırala (yüksek std = daha ayırt edici)
        sorted_features = sorted(
            features.items(),
            key=lambda x: x[1].get('std', 0),
            reverse=True
        )
        
        return sorted_features[:n]
