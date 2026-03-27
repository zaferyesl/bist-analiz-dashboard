from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os
from datetime import datetime

app = FastAPI(title="BIST Analiz Web")

# Klasör yolu ayarları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Küresel analiz durumu
analysis_status = {
    "is_running": False,
    "progress": 0,
    "message": "Hazır."
}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/status")
async def get_status():
    return analysis_status

@app.post("/api/analyze")
async def trigger_analysis(background_tasks: BackgroundTasks):
    global analysis_status
    if analysis_status["is_running"]:
        return {"status": "error", "message": "Zaten bir analiz çalışıyor."}
    
    analysis_status["is_running"] = True
    analysis_status["progress"] = 0
    analysis_status["message"] = "Veriler indiriliyor..."
    
    background_tasks.add_task(run_all_strategies)
    return {"status": "success", "message": "Analiz başlatıldı."}

@app.get("/api/results")
async def get_results(date: str = None):
    # Eğer tarih belirtilmediyse, en son güncel sonuç dosyasını bul (veya bugünün)
    if not date:
        date = datetime.now().strftime("%Y%m%d")
        
    filename = f"results_{date}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # En son dosyayı bul
        files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("results_") and f.endswith(".json")]
        if files:
            files.sort(reverse=True) # En yenisi en başta (isme göre düz sıralama)
            with open(os.path.join(RESULTS_DIR, files[0]), "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "error", "message": "Kayıtlı sonuç bulunamadı."}

@app.get("/api/performance")
async def get_performance_stats(refresh: bool = False):
    """Geçmiş JSON'lardan strateji performans istatistikleri döndür."""
    from performance import get_performance
    return get_performance(force_refresh=refresh)

STRATEGIES = [
    "strat1_birlesik",
    "strat2_gelismis_yesil",
    "strat3_minervini",
    "strat4_mum",
    "strat5_sikismis",
    "strat6_tepki1",
    "strat7_tepki2",
    "strat8_yesil",
    "strat9_momentum",
    "strat10_tavan",
    "strat11_spek",
    "common"
]

def run_all_strategies():
    global analysis_status
    try:
        # Importlar burada yapılıyor
        import data_fetcher
        from strategies.runner import run_strategy
        
        # 1. Veri çek
        analysis_status["message"] = "Borsa verileri (Yahoo Finance) toplu olarak indiriliyor..."
        stock_data_dict = data_fetcher.fetch_all_data()
        
        if not stock_data_dict:
            analysis_status["is_running"] = False
            analysis_status["message"] = "Veri çekme işleminde hata oluştu."
            return
            
        analysis_status["progress"] = 20
        analysis_status["message"] = f"{len(stock_data_dict)} hisse verisi çekildi. Stratejiler çalıştırılıyor..."
        
        results = {}
        
        # 2. Stratejileri tek tek çalıştır
        total_strategies = len(STRATEGIES) - 1 # 'common' stratejisi hariç
        
        for i, strategy_name in enumerate(STRATEGIES):
            if strategy_name == 'common':
                continue
                
            analysis_status["message"] = f"'{strategy_name}' stratejisi çalıştırılıyor..."
            
            try:
                # Orijinal stratejiyi çalıştır
                res = run_strategy(strategy_name, stock_data_dict)
                results[strategy_name] = res
            except Exception as e:
                print(f"Strateji çalışırken hata ({strategy_name}): {e}")
                results[strategy_name] = []
            
            # İlerlemeyi güncelle
            analysis_status["progress"] = 20 + int(70 * ((i + 1) / total_strategies))
            import time
            time.sleep(0.5) 
            
        # 3. Ortak Hisseler analizi
        analysis_status["message"] = "Ortak hisseler hesaplanıyor..."
        
        common_counts = {}
        for strategy_name in STRATEGIES:
            if strategy_name == 'common': continue
            for item in results.get(strategy_name, []):
                sym = item.get('symbol')
                if sym:
                    if sym not in common_counts:
                        common_counts[sym] = {"count": 0, "strategies": []}
                    common_counts[sym]["count"] += 1
                    common_counts[sym]["strategies"].append(strategy_name.replace("strat", "").replace("_", " ").title().strip())
                
        common_list = []
        for sym, d in common_counts.items():
            if d["count"] > 1:
                common_list.append({
                    "symbol": sym, 
                    "count": d["count"], 
                    "found_in": d["strategies"]
                })
        
        # Kesisenleri puana göre sırala
        common_list = sorted(common_list, key=lambda x: x["count"], reverse=True)

        # Tüm sonuçları içeren nihai sözlük
        all_results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategies": results,
            "common_stocks": common_list
        }
        
        analysis_status["progress"] = 95
        
        # 4. JSON Olarak kaydet
        date_str = datetime.now().strftime("%Y%m%d")
        filepath = os.path.join(RESULTS_DIR, f"results_{date_str}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
            
        analysis_status["progress"] = 100
        analysis_status["message"] = "Analiz başarıyla tamamlandı!"
        analysis_status["is_running"] = False
        
    except Exception as e:
        analysis_status["is_running"] = False
        analysis_status["message"] = f"Hata: {str(e)}"
        print(f"Analiz sırasında hata: {e}")

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    from threading import Timer
    
    # 1.5 saniye sonra otomatik olarak tarayıcıyı aç
    def open_browser():
        webbrowser.open("http://localhost:8000")
        
    Timer(1.5, open_browser).start()
    
    # Uvicorn sunucusunu başlat
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
