document.addEventListener('DOMContentLoaded', () => {
    // Tab switching logic
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active class from all
            navItems.forEach(nav => nav.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // Add active class to clicked
            item.classList.add('active');
            const targetId = item.getAttribute('data-tab');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Ana analiz butonu
    const analyzeBtn = document.getElementById('analyze-btn');
    const progressContainer = document.getElementById('progress-container');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const progressText = document.getElementById('progress-text');
    const progressPercent = document.getElementById('progress-percent');
    
    let pollInterval = null;

    analyzeBtn.addEventListener('click', async () => {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fa-solid fa-rotate sync-icon"></i> Analiz Ediliyor...';
        
        progressContainer.classList.remove('hidden');
        progressBarFill.style.width = '0%';
        progressPercent.textContent = '0%';
        
        try {
            const res = await fetch('/api/analyze', { method: 'POST' });
            const data = await res.json();
            
            if (data.status === 'success') {
                // Başarılı başladıysa polling başlat (1 sn aralık)
                pollInterval = setInterval(pollStatus, 1000);
            } else {
                alert(data.message);
                resetButton();
            }
        } catch (error) {
            alert('Sunucu hatası! Analiz başlatılamadı.');
            resetButton();
        }
    });

    async function pollStatus() {
        try {
            const res = await fetch('/api/status');
            const status = await res.json();
            
            progressBarFill.style.width = status.progress + '%';
            progressPercent.textContent = status.progress + '%';
            progressText.textContent = status.message;
            
            if (!status.is_running) {
                // Analiz bitti (başarılı ya da hatayla)
                clearInterval(pollInterval);
                setTimeout(() => {
                    progressContainer.classList.add('hidden');
                    resetButton();
                    loadResults(); // Her durumda sonuçları çekmeye çalış
                }, 1500);
            }
        } catch (e) {
            console.error(e);
        }
    }

    function resetButton() {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fa-solid fa-play"></i> Analizi Başlat';
    }

    // Sayfa yüklendiğinde var olan son sonuçları çek
    loadResults();

    async function loadResults() {
        try {
            const res = await fetch('/api/results');
            const data = await res.json();
            
            if (data.status === 'error') {
                document.getElementById('last-update').textContent = "Geçmiş sonuç bulunmuyor.";
                return;
            }
            
            document.getElementById('last-update').textContent = "Son Güncelleme: " + data.date;
            
            // Ortak hisseleri doldur
            const commonTable = document.querySelector('#table-common tbody');
            commonTable.innerHTML = '';
            
            if(data.common_stocks && data.common_stocks.length > 0) {
                document.getElementById('common-count').textContent = data.common_stocks.length;
                
                data.common_stocks.forEach(stock => {
                    const stratPills = stock.found_in.map(s => `<span class="pill info" style="margin-right:4px">${s}</span>`).join('');
                    
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><strong>${stock.symbol}</strong></td>
                        <td><span class="pill warning">${stock.count} Onay</span></td>
                        <td>${stratPills}</td>
                    `;
                    commonTable.appendChild(tr);
                });
            } else {
                document.getElementById('common-count').textContent = "0";
                commonTable.innerHTML = '<tr><td colspan="3">Listeye giren ortak hisse bulunamadı.</td></tr>';
            }
            
            // Diğer stratejilerin kartlarını doldur
            if (data.strategies) {
                for (const [strat, items] of Object.entries(data.strategies)) {
                    if (strat === 'common') continue; 
                    
                    // strat1_birlesik -> cards-birlesik  (strat\d+_ kısmını kaldır)
                    const stratClean = strat.replace(/strat\d+_/, '');
                    const containerId = 'cards-' + stratClean;
                    const container = document.getElementById(containerId);
                    if (!container) continue;
                    
                    container.innerHTML = '';
                    
                    if (items.length === 0) {
                        container.innerHTML = `
                            <div style="text-align:center; padding: 3rem; color: var(--text-muted); grid-column: 1/-1;">
                                <div style="font-size:2.5rem; margin-bottom:0.75rem;">🔍</div>
                                <strong style="font-size:1.1rem;">Bugün kriterleri karşılayan hisse bulunamadı.</strong><br><br>
                                <small>Bu strateji analiz edildi ancak hiçbir hisse senedi gerekli koşulları sağlamadı.<br>Piyasa koşulları değiştiğinde sonuçlar burada görünecek.</small>
                            </div>`;
                        continue;
                    }
                    
                    items.forEach(item => {
                        const entry = item.entry_price || 0;
                        const stop  = item.stop_loss   || 0;
                        const target= item.target_price|| 0;
                        const rr    = item.risk_reward || 0;
                        const chg   = item.change_1d   || 0;
                        const chgColor = chg >= 0 ? '#22c55e' : '#ef4444';
                        const chgSign  = chg >= 0 ? '+' : '';
                        const stopPct  = entry > 0 ? ((stop - entry) / entry * 100).toFixed(1) : '0';
                        const tgtPct   = entry > 0 ? ((target - entry) / entry * 100).toFixed(1) : '0';

                        const card = document.createElement('div');
                        card.className = 'stock-card';
                        card.innerHTML = `
                            <div class="stock-card-header">
                                <span class="stock-symbol">${item.symbol.replace('.IS','')}</span>
                                <span style="color:${chgColor}; font-weight:700; font-size:0.95rem;">${chgSign}${chg}%</span>
                            </div>
                            <div class="price-row">
                                <div class="price-box entry">
                                    <div class="price-label">📌 Giriş</div>
                                    <div class="price-value">₺${entry.toFixed(2)}</div>
                                </div>
                                <div class="price-box stop">
                                    <div class="price-label">🛑 Stop</div>
                                    <div class="price-value">₺${stop.toFixed(2)}</div>
                                    <div class="price-pct">${stopPct}%</div>
                                </div>
                                <div class="price-box target">
                                    <div class="price-label">🎯 Hedef</div>
                                    <div class="price-value">₺${target.toFixed(2)}</div>
                                    <div class="price-pct">+${tgtPct}%</div>
                                </div>
                            </div>
                            <div class="card-footer">
                                <span class="pill ${rr >= 2 ? 'success' : 'warning'}">R/R: ${rr}</span>
                                <span class="card-details" title="${item.details}">${item.details || ''}</span>
                            </div>`;
                        container.appendChild(card);
                    });
                }
            }
            
        } catch (e) {
            console.error("Sonuçlar yüklenirken hata:", e);
        }
    }
});
