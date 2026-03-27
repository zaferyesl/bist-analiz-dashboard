"""
Tavan Tahmin Sistemi - Ana Motor
Tüm bileşenleri bir araya getirir + Self Analyzer + Negatif Örnek Toplama
"""

import logging
from datetime import datetime
from typing import List, Dict
import yfinance as yf
import pandas as pd

from ceiling_history_manager import CeilingHistoryManager
from pattern_analyzer import CeilingPatternAnalyzer
from ml_trainer import CeilingMLTrainer
from self_analyzer import SelfAnalyzer
from advanced_indicators import AdvancedIndicators
from config import CeilingConfig

logger = logging.getLogger(__name__)


class CeilingPredictor:
    """Tavan tahmin sistemi - Ana motor"""
    
    def __init__(self):
        self.history_manager = CeilingHistoryManager()
        self.pattern_analyzer = CeilingPatternAnalyzer()
        self.ml_trainer = CeilingMLTrainer()
        self.self_analyzer = SelfAnalyzer()
        
        # ML modeli varsa yükle
        self.ml_trainer.load_model()
        self.self_analyzer.load_patterns()
    
    def train_from_history(self) -> Dict:
        """
        Geçmişten öğren + Self Analyzer eğitimi + Negatif Örnek Toplama
        
        Returns:
            Eğitim sonuçları
        """
        logger.info("=" * 80)
        logger.info("TAVAN TAHMİN SİSTEMİ - EĞİTİM BAŞLIYOR")
        logger.info("=" * 80)
        
        # Tavan geçmişini al
        ceilings = self.history_manager.get_all_ceilings()
        
        if len(ceilings) == 0:
            logger.warning("Tavan geçmişi boş! Önce data/tavan_history.txt dosyasına tavan olan hisseleri ekleyin.")
            return {
                'error': 'Tavan geçmişi boş',
                'message': 'data/tavan_history.txt dosyasına tavan olan hisseleri ekleyin'
            }
        
        logger.info(f"Toplam {len(ceilings)} tavan kaydı bulundu")
        logger.info(f"İstatistikler: {self.history_manager.get_statistics()}")
        
        # Her tavan için 1 gün önceki özellikleri çıkar
        ceiling_features = []
        
        for i, ceiling in enumerate(ceilings):
            logger.info(f"\n[{i+1}/{len(ceilings)}] Analiz ediliyor: {ceiling['symbol']} - {ceiling['date']}")
            
            features = self.pattern_analyzer.analyze_ceiling_stock(
                ceiling['symbol'],
                ceiling['date'],
                ceiling['ceiling_pct']
            )
            
            if features:
                ceiling_features.append(features)
            else:
                logger.warning(f"  ✗ {ceiling['symbol']} analiz edilemedi")
        
        logger.info("\n" + "=" * 80)
        logger.info(f"TOPLAM {len(ceiling_features)}/{len(ceilings)} HİSSE BAŞARIYLA ANALİZ EDİLDİ")
        logger.info("=" * 80)
        
        if len(ceiling_features) < 10:
            logger.warning("Yetersiz veri! En az 10 tavan örneği gerekli.")
            return {
                'error': 'Yetersiz veri',
                'analyzed': len(ceiling_features),
                'required': 10
            }
        
        # Ortak paternleri bul
        logger.info("\n" + "=" * 80)
        logger.info("ORTAK PATERNLER BULUNUYOR")
        logger.info("=" * 80)
        
        common_patterns = self.pattern_analyzer.find_common_patterns(ceiling_features)
        
        # En önemli özellikleri göster
        top_features = self.pattern_analyzer.get_top_features(10)
        
        logger.info("\nTavan olan hisselerin ortak özellikleri (en değişken 10):")
        for i, (feature, stats) in enumerate(top_features, 1):
            logger.info(f"{i}. {feature}:")
            logger.info(f"   Ortalama: {stats['mean']:.2f}")
            logger.info(f"   Medyan: {stats['median']:.2f}")
            logger.info(f"   Std: {stats['std']:.2f}")
            logger.info(f"   Aralık: [{stats['min']:.2f}, {stats['max']:.2f}]")
        
        # ✅ YENİ: Negatif örnekler topla
        logger.info("\n" + "=" * 80)
        logger.info("NEGATİF ÖRNEKLER TOPLANIYOR (Tavan Yapmayan Günler)")
        logger.info("=" * 80)
        
        from negative_sampler import NegativeSampler
        
        negative_sampler = NegativeSampler(ceiling_threshold=9.0)
        
        # TXT'deki benzersiz hisseler
        unique_symbols = list(set([c['symbol'] for c in ceilings]))
        
        # Pozitif örnek sayısının %30'u kadar negatif örnek topla
        target_negative_count = max(int(len(ceiling_features) * 0.3), 100)
        
        logger.info(f"Hedef negatif örnek: {target_negative_count}")
        logger.info(f"Benzersiz hisse: {len(unique_symbols)}")
        
        negative_samples = negative_sampler.collect_negative_samples(
            symbols=unique_symbols,
            target_count=target_negative_count,
            lookback_days=365
        )
        
        logger.info(f"✓ Negatif örnekler toplandı: {len(negative_samples)}")
        
        # Pozitif + Negatif birleştir
        all_samples = ceiling_features + negative_samples
        
        logger.info("\n" + "=" * 80)
        logger.info("VERİ SETİ HAZIR")
        logger.info("=" * 80)
        logger.info(f"Pozitif örnekler (>=%9 tavan): {len(ceiling_features)}")
        logger.info(f"Negatif örnekler (<%9): {len(negative_samples)}")
        logger.info(f"Toplam örnek: {len(all_samples)}")
        logger.info(f"Pozitif/Negatif oranı: {len(ceiling_features)/len(negative_samples):.2f}" if negative_samples else "N/A")
        
        # ML modeli eğit (dengeli veri ile)
        logger.info("\n" + "=" * 80)
        logger.info("MACHINE LEARNING MODELİ EĞİTİLİYOR")
        logger.info("=" * 80)
        
        ml_results = self.ml_trainer.train(all_samples)
        
        if 'error' in ml_results:
            logger.error(f"ML eğitimi başarısız: {ml_results['error']}")
            return ml_results
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ MODEL BAŞARIYLA EĞİTİLDİ!")
        logger.info("=" * 80)
        logger.info(f"Accuracy: {ml_results['accuracy']:.2%}")
        logger.info(f"Recall: {ml_results['recall']:.2%}")
        logger.info(f"Precision: {ml_results['precision']:.2%}")
        logger.info(f"F1-Score: {ml_results['f1']:.2%}")
        logger.info(f"ROC-AUC: {ml_results['roc_auc']:.3f}")
        logger.info(f"Optimal Threshold: {ml_results['optimal_threshold']:.2f}")
        logger.info(f"Eğitim seti: {ml_results['train_size']} örnek")
        logger.info(f"Test seti: {ml_results['test_size']} örnek")
        
        logger.info("\nEn önemli özellikler (ML'e göre):")
        for i, (feat, imp) in enumerate(ml_results['top_features'][:15], 1):
            logger.info(f"{i}. {feat}: {imp:.4f}")
        
        # Modeli kaydet
        self.ml_trainer.save_model()
        
        # ✅ SELF ANALYZER EĞİTİMİ
        logger.info("\n" + "=" * 80)
        logger.info("KENDİ GEÇMİŞİ ANALİZİ BAŞLIYOR (SELF ANALYZER)")
        logger.info("=" * 80)
        
        logger.info(f"Toplam {len(unique_symbols)} farklı hisse analiz edilecek")
        
        self_analysis_results = {
            'total_symbols': len(unique_symbols),
            'analyzed_symbols': 0,
            'total_patterns': 0,
            'failed_symbols': []
        }
        
        for i, symbol in enumerate(unique_symbols, 1):
            logger.info(f"\n[{i}/{len(unique_symbols)}] {symbol} kendi geçmişi analiz ediliyor...")
            
            # Bu hissenin tavan tarihlerini al
            symbol_ceilings = [c['date'] for c in ceilings if c['symbol'] == symbol]
            
            if len(symbol_ceilings) < 2:
                logger.info(f"   {symbol}: Yetersiz tavan sayısı ({len(symbol_ceilings)}), atlanıyor")
                self_analysis_results['failed_symbols'].append({
                    'symbol': symbol,
                    'reason': f'Yetersiz tavan sayısı ({len(symbol_ceilings)})'
                })
                continue
            
            # Kendi geçmişini analiz et
            patterns = self.self_analyzer.analyze_stock_ceiling_history(symbol, symbol_ceilings)
            
            if patterns:
                logger.info(f"   ✓ {symbol}: {len(patterns)} pattern bulundu")
                self_analysis_results['analyzed_symbols'] += 1
                self_analysis_results['total_patterns'] += len(patterns)
            else:
                logger.warning(f"   ✗ {symbol}: Pattern bulunamadı")
                self_analysis_results['failed_symbols'].append({
                    'symbol': symbol,
                    'reason': 'Pattern bulunamadı'
                })
        
        # Self patterns'i kaydet
        self.self_analyzer.save_patterns()
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ KENDİ GEÇMİŞİ ANALİZİ TAMAMLANDI!")
        logger.info("=" * 80)
        logger.info(f"Analiz edilen hisse: {self_analysis_results['analyzed_symbols']}/{self_analysis_results['total_symbols']}")
        logger.info(f"Toplam pattern: {self_analysis_results['total_patterns']}")
        
        if self_analysis_results['failed_symbols']:
            logger.info(f"\nBaşarısız hisseler ({len(self_analysis_results['failed_symbols'])}):")
            for failed in self_analysis_results['failed_symbols'][:5]:
                logger.info(f"  • {failed['symbol']}: {failed['reason']}")
        
        logger.info("\n" + "=" * 80)
        logger.info("EĞİTİM TAMAMLANDI!")
        logger.info("=" * 80)
        
        return {
            'success': True,
            'total_ceilings': len(ceilings),
            'analyzed_ceilings': len(ceiling_features),
            'negative_samples': len(negative_samples),
            'total_samples': len(all_samples),
            'common_patterns': len(common_patterns),
            'ml_results': ml_results,
            'self_analysis_results': self_analysis_results
        }
    
    def predict_tomorrow_ceilings(self, symbols: List[str], min_probability: float = 0.6) -> List[Dict]:
        """
        Yarın tavan yapma olasılığı yüksek hisseleri bul + Self Analyzer
        
        Args:
            symbols: Taranacak hisse listesi
            min_probability: Minimum olasılık eşiği (0-1)
        
        Returns:
            Tavan adayları (kombine skora göre sıralı)
        """
        logger.info("=" * 80)
        logger.info("YARIN TAVAN YAPACAK HİSSELER TARANYOR (SELF ANALYZER AKTİF)")
        logger.info("=" * 80)
        logger.info(f"Taranacak hisse sayısı: {len(symbols)}")
        logger.info(f"Minimum olasılık eşiği: %{min_probability*100:.0f}")
        
        candidates = []
        
        for i, symbol in enumerate(symbols):
            logger.info(f"\n[{i+1}/{len(symbols)}] Taranıyor: {symbol}")
            
            try:
                # Bugünkü verileri çek (yfinance 1.0)
                symbol_yahoo = symbol if symbol.endswith('.IS') else symbol + '.IS'
                ticker = yf.Ticker(symbol_yahoo)
                
                hist = ticker.history(
                    period='18mo',
                    auto_adjust=True,
                    actions=False
                )
                
                # Multi-index kontrolü
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.droplevel(1)
                
                if hist is None or hist.empty or len(hist) < 100:
                    logger.warning(f"  {symbol}: Yetersiz veri")
                    continue
                
                logger.info(f"  {len(hist)} günlük veri alındı")
                
                # İndikatörleri hesapla
                indicators = AdvancedIndicators.calculate_all_indicators(hist)
                
                # Pattern benzerlik skoru
                pattern_score = self.pattern_analyzer.calculate_similarity_score(indicators)
                
                # ML tahmin
                ml_probability = self.ml_trainer.predict_ceiling_probability(indicators)
                
                # Self Analyzer
                symbol_clean = symbol.replace('.IS', '')
                self_analysis = self.self_analyzer.calculate_self_similarity(symbol_clean, indicators)
                self_similarity = self_analysis['similarity_score']
                
                # Kombine skor (ML %40, Pattern %30, Self %30)
                combined_score = (
                    (ml_probability * 100 * 0.4) + 
                    (pattern_score * 0.3) + 
                    (self_similarity * 0.3)
                )
                
                logger.info(f"  ML Olasılık: %{ml_probability*100:.1f}")
                logger.info(f"  Pattern Skor: {pattern_score:.1f}")
                logger.info(f"  Self Benzerlik: {self_similarity:.1f} {self_analysis['confidence_emoji']}")
                logger.info(f"  Kombine Skor: {combined_score:.1f}")
                
                # Eşik kontrolü
                if ml_probability >= min_probability or combined_score >= 60:
                    current_price = float(hist['Close'].iloc[-1])
                    
                    candidate = {
                        'symbol': symbol_clean,
                        'current_price': current_price,
                        'ml_probability': ml_probability * 100,
                        'pattern_score': pattern_score,
                        'self_similarity': self_similarity,
                        'self_confidence': self_analysis['confidence'],
                        'self_confidence_emoji': self_analysis['confidence_emoji'],
                        'self_matched_patterns': self_analysis['matched_patterns'],
                        'combined_score': combined_score,
                        'key_indicators': {
                            'rsi_14': indicators.get('rsi_14', 0),
                            'volume_breakout_score': indicators.get('volume_breakout_score', 0),
                            'pre_ceiling_squeeze': indicators.get('pre_ceiling_squeeze', 0),
                            'ceiling_breakout_prob': indicators.get('ceiling_breakout_prob', 0),
                            'distance_to_ceiling': indicators.get('distance_to_ceiling', 0),
                            'institutional_accumulation': indicators.get('institutional_accumulation', 0),
                            'retail_fomo': indicators.get('retail_fomo', 0),
                            'trend_alignment': indicators.get('trend_alignment', 0),
                            'compression_score': indicators.get('compression_score', 0),
                            'breakout_strength': indicators.get('breakout_strength', 0)
                        },
                        'self_analysis': self_analysis
                    }
                    
                    candidates.append(candidate)
                    
                    logger.info(f"  ✅ ADAY BULUNDU!")
                else:
                    logger.info(f"  ✗ Eşik altında")
                
            except Exception as e:
                logger.error(f"  {symbol} hata: {e}")
                continue
        
        # Kombine skora göre sırala
        candidates = sorted(candidates, key=lambda x: x['combined_score'], reverse=True)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"TARAMA TAMAMLANDI: {len(candidates)} ADAY BULUNDU")
        logger.info("=" * 80)
        
        if candidates:
            logger.info("\nEn güçlü 10 aday:")
            for i, c in enumerate(candidates[:10], 1):
                logger.info(f"{i}. {c['symbol']}: "
                          f"ML=%{c['ml_probability']:.1f}, "
                          f"Pattern={c['pattern_score']:.1f}, "
                          f"Self={c['self_similarity']:.1f} {c['self_confidence_emoji']}, "
                          f"Kombine={c['combined_score']:.1f}")
        
        return candidates
    
    def run_backtest(self, symbols: List[str], test_days: int = 10) -> Dict:
        """
        Backtest çalıştır
        
        Args:
            symbols: Test edilecek hisse listesi
            test_days: Kaç günlük test
        
        Returns:
            Backtest sonuçları
        """
        from backtester import CeilingBacktester
        
        backtester = CeilingBacktester(
            ml_trainer=self.ml_trainer,
            ceiling_threshold=9.0
        )
        
        results = backtester.run_backtest(
            symbols=symbols,
            test_days=test_days,
            thresholds=[i/100 for i in range(10, 95, 5)]
        )
        
        return results
    
    def generate_report(self, candidates: List[Dict]) -> str:
        """
        Detaylı rapor oluştur (Self Analyzer dahil)
        
        Args:
            candidates: Tavan adayları listesi
        
        Returns:
            Formatlanmış rapor metni
        """
        report = []
        
        report.append("=" * 80)
        report.append("YARIN TAVAN YAPMA OLASILIĞI YÜKSEK HİSSELER")
        report.append("=" * 80)
        report.append(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Toplam Aday: {len(candidates)}")
        report.append("")
        
        if not candidates:
            report.append("Hiç aday bulunamadı.")
            report.append("Minimum olasılık eşiğini düşürmeyi deneyin.")
        else:
            for i, candidate in enumerate(candidates, 1):
                report.append(f"\n{i}. {candidate['symbol']} - {candidate['current_price']:.2f} TL")
                report.append("-" * 80)
                report.append(f"   🎯 ML Tavan Olasılığı: %{candidate['ml_probability']:.1f}")
                report.append(f"   📊 Pattern Benzerlik: {candidate['pattern_score']:.1f}/100")
                report.append(f"   🔍 Self Benzerlik: {candidate['self_similarity']:.1f}/100 {candidate['self_confidence_emoji']}")
                report.append(f"   ⭐ Kombine Skor: {candidate['combined_score']:.1f}/100")
                
                # Self Analyzer detayları
                if candidate.get('self_matched_patterns', 0) > 0:
                    report.append(f"\n   📈 Kendi Geçmişi:")
                    report.append(f"      • Eşleşen Pattern: {candidate['self_matched_patterns']}")
                    report.append(f"      • Güven Seviyesi: {candidate['self_confidence']} {candidate['self_confidence_emoji']}")
                    
                    # En iyi eşleşme
                    if 'self_analysis' in candidate and 'best_match' in candidate['self_analysis']:
                        best = candidate['self_analysis']['best_match']
                        report.append(f"      • En İyi Eşleşme:")
                        report.append(f"        - Tavan Tarihi: {best['ceiling_date']}")
                        report.append(f"        - Analiz Günü: {best['analysis_date']}")
                        report.append(f"        - Benzerlik: %{best['similarity']:.1f}")
                        report.append(f"        - Tavana Kalan: {best['days_before_ceiling']} gün")
                        report.append(f"        - Tavan Yükselişi: %{best['ceiling_rise_pct']:.2f}")
                
                report.append("\n   Anahtar Göstergeler:")
                indicators = candidate['key_indicators']
                report.append(f"      • RSI(14): {indicators['rsi_14']:.1f}")
                report.append(f"      • Hacim Breakout: {indicators['volume_breakout_score']:.1f}")
                report.append(f"      • Tavan Öncesi Sıkışma: {indicators['pre_ceiling_squeeze']:.1f}")
                report.append(f"      • Tavan Kırılım Olasılığı: {indicators['ceiling_breakout_prob']:.1f}")
                report.append(f"      • Tavana Mesafe: %{indicators['distance_to_ceiling']:.1f}")
                report.append(f"      • Kurumsal Birikim: {indicators['institutional_accumulation']:.1f}")
                report.append(f"      • Retail FOMO: {indicators['retail_fomo']:.1f}")
                report.append(f"      • Trend Uyumu: {indicators['trend_alignment']:.1f}")
                report.append(f"      • Sıkışma Skoru: {indicators['compression_score']:.1f}")
                report.append(f"      • Breakout Gücü: {indicators['breakout_strength']:.1f}")
                
                # Yorum
                if candidate['combined_score'] >= CeilingConfig.EXCELLENT_SCORE:
                    report.append("\n   💎 ÇOK GÜÇLÜ ADAY - Yakından takip edin!")
                elif candidate['combined_score'] >= CeilingConfig.GOOD_SCORE:
                    report.append("\n   ✅ GÜÇLÜ ADAY - İzlemeye değer")
                elif candidate['combined_score'] >= CeilingConfig.FAIR_SCORE:
                    report.append("\n   ⚠️ ORTA ADAY - Dikkatli olun")
        
        report.append("\n" + "=" * 80)
        report.append("NOT: Bu tahminler geçmiş verilere dayanır. Garanti değildir!")
        report.append("Yatırım kararlarınızı kendi analizinizle destekleyin.")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def get_system_status(self) -> Dict:
        """Sistem durumu (Self Analyzer dahil)"""
        history_stats = self.history_manager.get_statistics()
        pattern_summary = self.pattern_analyzer.get_pattern_summary()
        model_info = self.ml_trainer.get_model_info()
        
        # Self Analyzer durumu
        self_analyzer_status = {
            'loaded': len(self.self_analyzer.self_patterns) > 0,
            'total_symbols': len(self.self_analyzer.self_patterns),
            'total_patterns': sum(len(patterns) for patterns in self.self_analyzer.self_patterns.values())
        }
        
        return {
            'history': history_stats,
            'patterns': pattern_summary,
            'model': model_info,
            'self_analyzer': self_analyzer_status
        }
