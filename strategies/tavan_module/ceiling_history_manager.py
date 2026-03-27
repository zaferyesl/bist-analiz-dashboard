"""
Tavan Geçmişi Yöneticisi
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict

logger = logging.getLogger(__name__)


class CeilingHistoryManager:
    """Tavan geçmişini yönetir"""
    
    def __init__(self, filename=None):
        if filename is None:
            import os
            filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'tavan_history.txt')
        self.filename = filename
        self.history = []
        self.load_history()
    
    def load_history(self):
        """Geçmişi yükle"""
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Format: TARIH|SEMBOL|YUZDE
                    parts = line.split('|')
                    if len(parts) == 3:
                        try:
                            self.history.append({
                                'date': parts[0].strip(),
                                'symbol': parts[1].strip(),
                                'ceiling_pct': float(parts[2].strip())
                            })
                        except ValueError:
                            logger.warning(f"Geçersiz satır atlandı: {line}")
                            continue
            
            logger.info(f"Tavan geçmişi yüklendi: {len(self.history)} kayıt")
        except FileNotFoundError:
            logger.info("Tavan geçmişi bulunamadı, yeni oluşturulacak")
            self.history = []
            self._create_example_file()
    
    def _create_example_file(self):
        """Örnek dosya oluştur"""
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("# TAVAN GEÇMİŞİ\n")
            f.write("# Format: TARIH|SEMBOL|YUZDE\n")
            f.write("# Örnek: 2024-01-15|THYAO|12.5\n")
            f.write("#\n")
            f.write("# KULLANIM:\n")
            f.write("# 1. Her gün tavan olan hisseleri buraya ekleyin\n")
            f.write("# 2. Tarih formatı: YYYY-MM-DD (örn: 2024-01-15)\n")
            f.write("# 3. Sembol: BIST kodu (örn: THYAO, AKBNK)\n")
            f.write("# 4. Yüzde: Tavan artış yüzdesi (örn: 12.5)\n")
            f.write("#\n")
            f.write("# ÖRNEK KAYITLAR:\n")
            f.write("# 2024-01-15|THYAO|12.5\n")
            f.write("# 2024-01-16|AKBNK|10.2\n")
            f.write("# 2024-01-17|GARAN|11.8\n")
            f.write("#\n")
        
        logger.info(f"Örnek dosya oluşturuldu: {self.filename}")
    
    def add_ceiling(self, date: str, symbol: str, ceiling_pct: float):
        """Yeni tavan ekle"""
        # Tarih formatı kontrolü
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Geçersiz tarih formatı: {date}. Format: YYYY-MM-DD olmalı")
            return False
        
        # Sembol kontrolü
        symbol = symbol.upper().strip()
        if not symbol:
            logger.error("Sembol boş olamaz")
            return False
        
        # Yüzde kontrolü
        if ceiling_pct <= 0 or ceiling_pct > 100:
            logger.error(f"Geçersiz yüzde: {ceiling_pct}. 0-100 arası olmalı")
            return False
        
        # Ekle
        self.history.append({
            'date': date,
            'symbol': symbol,
            'ceiling_pct': ceiling_pct
        })
        
        # Dosyaya ekle
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(f"{date}|{symbol}|{ceiling_pct}\n")
        
        logger.info(f"Tavan eklendi: {symbol} - {date} (%{ceiling_pct})")
        return True
    
    def get_recent_ceilings(self, days: int = 30) -> List[Dict]:
        """Son N günün tavanlarını getir"""
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        recent = [h for h in self.history if h['date'] >= cutoff_date]
        return recent
    
    def get_all_ceilings(self) -> List[Dict]:
        """Tüm tavanları getir"""
        return self.history
    
    def get_statistics(self) -> Dict:
        """İstatistikler"""
        if not self.history:
            return {
                'total': 0,
                'avg_pct': 0,
                'max_pct': 0,
                'min_pct': 0,
                'unique_symbols': 0
            }
        
        pcts = [h['ceiling_pct'] for h in self.history]
        
        return {
            'total': len(self.history),
            'avg_pct': sum(pcts) / len(pcts),
            'max_pct': max(pcts),
            'min_pct': min(pcts),
            'unique_symbols': len(set(h['symbol'] for h in self.history))
        }
    
    def remove_ceiling(self, date: str, symbol: str):
        """Tavan kaydını sil"""
        self.history = [h for h in self.history 
                       if not (h['date'] == date and h['symbol'] == symbol)]
        
        # Dosyayı yeniden yaz
        self._rewrite_file()
        
        logger.info(f"Tavan silindi: {symbol} - {date}")
    
    def _rewrite_file(self):
        """Dosyayı yeniden yaz"""
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("# TAVAN GEÇMİŞİ\n")
            f.write("# Format: TARIH|SEMBOL|YUZDE\n")
            f.write("#\n")
            
            for h in self.history:
                f.write(f"{h['date']}|{h['symbol']}|{h['ceiling_pct']}\n")
