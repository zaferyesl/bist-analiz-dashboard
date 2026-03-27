"""
ML Eğitim Motoru
Gradient Boosting ile tavan tahmin modeli
Class weight + Threshold optimization
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    accuracy_score, precision_recall_curve, f1_score,
    recall_score, precision_score
)
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class CeilingMLTrainer:
    """Tavan tahmin ML modeli eğitici"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = {}
        self.optimal_threshold = 0.5
    
    def prepare_training_data(self, ceiling_features: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Eğitim verisi hazırla
        
        Args:
            ceiling_features: Tavan olan hisselerin indikatör listesi
        
        Returns:
            X (features), y (target)
        """
        df = pd.DataFrame(ceiling_features)
        
        logger.info(f"Veri hazırlanıyor: {len(df)} örnek")
        
        # Hedef değişken: >= 9% tavan ise 1, değilse 0
        df['target'] = (df['ceiling_pct'] >= 9.0).astype(int)
        
        logger.info(f"  Pozitif örnekler (>=%9 tavan): {df['target'].sum()} (%{df['target'].mean()*100:.1f})")
        logger.info(f"  Negatif örnekler (<%9): {(1-df['target']).sum()} (%{(1-df['target']).mean()*100:.1f})")
        
        # Özellik seçimi: Sayısal kolonlar
        exclude_cols = ['symbol', 'date', 'ceiling_pct', 'target', 'gap_type', 
                       'ceiling_date', 'analysis_date', 'days_before_ceiling', 
                       'ceiling_rise_pct', 'is_negative_sample']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
        
        logger.info(f"  Toplam özellik: {len(feature_cols)}")
        
        X = df[feature_cols].fillna(0)
        y = df['target']
        
        # Sonsuz değerleri temizle
        X = X.replace([np.inf, -np.inf], 0)
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(self, ceiling_features: List[Dict]) -> Dict:
        """
        Model eğit
        
        Args:
            ceiling_features: Tavan olan hisselerin indikatör listesi
        
        Returns:
            Eğitim sonuçları
        """
        logger.info("=" * 80)
        logger.info("ML MODEL EĞİTİMİ BAŞLIYOR")
        logger.info("=" * 80)
        
        if len(ceiling_features) < 30:
            logger.warning(f"Yetersiz veri: {len(ceiling_features)} örnek (minimum 30 gerekli)")
            return {'error': 'Yetersiz veri', 'required': 30, 'available': len(ceiling_features)}
        
        # Veri hazırlama
        X, y = self.prepare_training_data(ceiling_features)
        
        # Train-test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # Stratify yapılamıyorsa (çok az örnek)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        logger.info(f"Train set: {len(X_train)} örnek (Pos: {y_train.sum()}, Neg: {(1-y_train).sum()})")
        logger.info(f"Test set: {len(X_test)} örnek (Pos: {y_test.sum()}, Neg: {(1-y_test).sum()})")
        
        # Class weight hesapla
        sample_weights = compute_sample_weight('balanced', y=y_train)
        
        logger.info(f"\nClass weight:")
        logger.info(f"  Pozitif (tavan): {sample_weights[y_train == 1].mean():.2f}x")
        logger.info(f"  Negatif: {sample_weights[y_train == 0].mean():.2f}x")
        
        # Ölçeklendirme
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Model: Gradient Boosting
        logger.info("\nModel eğitiliyor: Gradient Boosting Classifier")
        
        self.model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        
        # Class weight ile eğit
        self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        # Değerlendirme
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.5
        
        # Threshold optimization
        logger.info("\n" + "=" * 80)
        logger.info("THRESHOLD OPTİMİZASYONU")
        logger.info("=" * 80)
        
        threshold_results = self._optimize_threshold(y_test, y_pred_proba)
        
        # En iyi threshold'u bul (F1-Score'a göre)
        best_threshold = max(threshold_results.items(), key=lambda x: x[1]['f1'])[0]
        self.optimal_threshold = best_threshold
        
        logger.info(f"\n✅ OPTIMAL THRESHOLD: {best_threshold:.2f}")
        
        # Final tahmin (optimal threshold ile)
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Feature importance
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # En önemli 20 özellik
        top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✓ MODEL EĞİTİLDİ (OPTIMAL THRESHOLD: {best_threshold:.2f})")
        logger.info("=" * 80)
        logger.info(f"Accuracy: {accuracy:.2%}")
        logger.info(f"Recall: {recall:.2%}")
        logger.info(f"Precision: {precision:.2%}")
        logger.info(f"F1-Score: {f1:.2%}")
        logger.info(f"ROC-AUC: {roc_auc:.3f}")
        logger.info("=" * 80)
        
        logger.info("\nEn önemli özellikler:")
        for i, (feat, imp) in enumerate(top_features[:10], 1):
            logger.info(f"  {i}. {feat}: {imp:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info("\nConfusion Matrix (Optimal Threshold):")
        logger.info(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
        logger.info(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")
        
        # Threshold analizi kaydet
        self._save_threshold_analysis(threshold_results, best_threshold, y_test, y_pred_proba)
        
        return {
            'accuracy': float(accuracy),
            'recall': float(recall),
            'precision': float(precision),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'optimal_threshold': float(best_threshold),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'top_features': [(feat, float(imp)) for feat, imp in top_features],
            'confusion_matrix': cm.tolist(),
            'threshold_results': threshold_results
        }
    
    def _optimize_threshold(self, y_true, y_pred_proba):
        """Threshold optimizasyonu"""
        thresholds = np.arange(0.1, 0.95, 0.05)
        
        results = {}
        
        logger.info(f"\n{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1':<10} {'Accuracy':<10}")
        logger.info("-" * 60)
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            recall = recall_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            results[threshold] = {
                'recall': float(recall),
                'precision': float(precision),
                'f1': float(f1),
                'accuracy': float(accuracy),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            }
            
            logger.info(f"{threshold:<12.2f} {recall:<10.2%} {precision:<12.2%} {f1:<10.2%} {accuracy:<10.2%}")
        
        return results
    
    def _save_threshold_analysis(self, threshold_results, best_threshold, y_true, y_pred_proba):
        """Threshold analizi kaydet"""
        try:
            os.makedirs('data', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            import os
            filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', f'threshold_analysis_{timestamp}.txt')
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("THRESHOLD ANALİZİ - EĞİTİM SETİ\n")
                f.write("=" * 100 + "\n")
                f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Test Set: {len(y_true)} örnek\n")
                f.write(f"Gerçek Pozitif: {y_true.sum()}\n")
                f.write(f"Gerçek Negatif: {(1-y_true).sum()}\n")
                f.write(f"Optimal Threshold: {best_threshold:.2f}\n")
                f.write("\n")
                
                f.write("=" * 100 + "\n")
                f.write("THRESHOLD PERFORMANS TABLOSU\n")
                f.write("=" * 100 + "\n")
                f.write(f"{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1':<10} {'Accuracy':<12} "
                       f"{'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6}\n")
                f.write("-" * 100 + "\n")
                
                for threshold in sorted(threshold_results.keys()):
                    r = threshold_results[threshold]
                    marker = " ← OPTIMAL" if threshold == best_threshold else ""
                    f.write(f"{threshold:<12.2f} "
                           f"{r['recall']:<10.2%} "
                           f"{r['precision']:<12.2%} "
                           f"{r['f1']:<10.2%} "
                           f"{r['accuracy']:<12.2%} "
                           f"{r['tp']:<6} "
                           f"{r['fp']:<6} "
                           f"{r['tn']:<6} "
                           f"{r['fn']:<6}{marker}\n")
                
                f.write("\n")
                f.write("=" * 100 + "\n")
                f.write("AÇIKLAMALAR\n")
                f.write("=" * 100 + "\n")
                f.write("Recall:    Gerçek tavanların kaçını bulduk? (TP / (TP + FN))\n")
                f.write("Precision: Tavan dediğimizin kaçı gerçekten tavan? (TP / (TP + FP))\n")
                f.write("F1:        Recall ve Precision'ın harmonik ortalaması\n")
                f.write("Accuracy:  Toplam doğru tahmin oranı\n")
                f.write("\n")
                f.write("TP (True Positive):   Doğru tahmin - Tavan yaptı ✅\n")
                f.write("FP (False Positive):  Yanlış alarm - Tavan yapmadı ❌\n")
                f.write("TN (True Negative):   Doğru tahmin - Tavan yapmadı ✅\n")
                f.write("FN (False Negative):  Kaçırılan tavan ❌\n")
                f.write("\n")
            
            logger.info(f"✓ Threshold analizi kaydedildi: {filename}")
        
        except Exception as e:
            logger.error(f"Threshold analizi kaydetme hatası: {e}")
    
    def predict_ceiling_probability(self, indicators: Dict) -> float:
        """
        Tavan yapma olasılığını tahmin et
        
        Args:
            indicators: Hisse indikatörleri
        
        Returns:
            Tavan yapma olasılığı (0-1)
        """
        if self.model is None:
            logger.warning("Model yüklü değil, tahmin yapılamıyor")
            return 0.0
        
        # Feature vector oluştur
        feature_vector = []
        for feat in self.feature_names:
            value = indicators.get(feat, 0)
            
            # NaN veya inf kontrolü
            if np.isnan(value) or np.isinf(value):
                value = 0
            
            feature_vector.append(value)
        
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        probability = self.model.predict_proba(X_scaled)[0, 1]
        
        return float(probability)
    
    def save_model(self, filename=None):
        if filename is None:
            import os
            filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'ceiling_ml_model.pkl')
        """
        Modeli kaydet
        
        Args:
            filename: Kayıt dosyası yolu
        """
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'optimal_threshold': self.optimal_threshold
            }, filename)
            
            logger.info(f"✓ Model kaydedildi: {filename}")
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {e}")
    
    def load_model(self, filename=None):
        if filename is None:
            import os
            filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'ceiling_ml_model.pkl')
        """
        Modeli yükle
        
        Args:
            filename: Model dosyası yolu
        
        Returns:
            Başarılı ise True
        """
        try:
            if not os.path.exists(filename):
                logger.info(f"Model dosyası bulunamadı: {filename}")
                return False
            
            data = joblib.load(filename)
            
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.feature_importance = data['feature_importance']
            self.optimal_threshold = data.get('optimal_threshold', 0.5)
            
            logger.info(f"✓ Model yüklendi: {filename}")
            logger.info(f"  Toplam özellik: {len(self.feature_names)}")
            logger.info(f"  Optimal threshold: {self.optimal_threshold:.2f}")
            
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini getir"""
        if self.model is None:
            return {
                'loaded': False,
                'message': 'Model yüklü değil'
            }
        
        top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'loaded': True,
            'total_features': len(self.feature_names),
            'optimal_threshold': self.optimal_threshold,
            'top_features': top_features,
            'model_type': type(self.model).__name__
        }
