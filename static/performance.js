// performance.js — Genişletilmiş Strateji Performans Analizi

(function () {
    let perfData = null;
    let currentView = 'overview';   // 'overview' | 'thresholds' | 'horizon'
    let selectedHorizon = 5;        // T+5 varsayılan

    const horizonLabels = {
        1: 'T+1 (1 Gün)', 2: 'T+2', 3: 'T+3', 5: 'T+5 (1 Hafta)',
        10: 'T+10 (2 Hafta)', 22: 'T+22 (1 Ay)'
    };
    const thresholdColors = {
        3: '#60a5fa', 5: '#34d399', 8: '#fbbf24', 10: '#f87171'
    };

    // ── Tab açıldığında yükle ──────────────────────────────
    document.querySelectorAll('.nav-item').forEach(item => {
        if (item.getAttribute('data-tab') === 'performance') {
            item.addEventListener('click', () => {
                if (!perfData) loadPerformance(false);
            });
        }
    });

    // ── Yenile butonu ─────────────────────────────────────
    document.addEventListener('click', e => {
        if (e.target.closest('#perf-refresh-btn')) loadPerformance(true);
    });

    // ── Görünüm sekmeleri (overview / thresholds / horizon) ─
    document.addEventListener('click', e => {
        const vBtn = e.target.closest('[data-perf-view]');
        if (!vBtn) return;
        document.querySelectorAll('[data-perf-view]').forEach(b => b.classList.remove('active'));
        vBtn.classList.add('active');
        currentView = vBtn.getAttribute('data-perf-view');
        if (perfData) renderView();
    });

    // ── Horizon seçici ────────────────────────────────────
    document.addEventListener('change', e => {
        if (e.target.id === 'perf-horizon-select') {
            selectedHorizon = parseInt(e.target.value);
            if (perfData) renderView();
        }
    });

    // ─────────────────────────────────────────────────────
    async function loadPerformance(force) {
        setLoading(true);
        try {
            const url = '/api/performance' + (force ? '?refresh=true' : '');
            const res = await fetch(url);
            const data = await res.json();
            setLoading(false);
            if (data.error) { showEmpty(data.error); return; }
            perfData = data;
            renderAll();
        } catch (e) {
            setLoading(false);
            showEmpty('Sunucu hatası: ' + e.message);
        }
    }

    function renderAll() {
        renderHeader();
        renderViewTabs();
        renderView();
    }

    // ── Üst başlık ve özet ────────────────────────────────
    function renderHeader() {
        const el = document.getElementById('perf-summary');
        if (!el) return;
        const d = perfData;
        const strats = Object.values(d.strategies || {});
        const withData = strats.filter(s => s.signal_count > 0);
        const totalSigs = strats.reduce((a, s) => a + (s.signal_count || 0), 0);

        // En iyi strateji: en yüksek EV (5 günlük)
        const best = strats
            .filter(s => s.expected_value_5d != null)
            .sort((a, b) => b.expected_value_5d - a.expected_value_5d)[0];

        el.innerHTML = '';
        mkSumCard('📅', 'Analiz Dönemi', `${d.date_range?.start || '?'} → ${d.date_range?.end || '?'}`, el);
        mkSumCard('📊', 'Toplam Kayıt / Sinyal', `${d.record_count} gün / ${totalSigs} sinyal`, el);
        mkSumCard('✅', 'Aktif Strateji', `${withData.length} strateji`, el);
        if (best) {
            const ev = best.expected_value_5d?.toFixed(2);
            mkSumCard('🏆', 'En İyi EV (T+5)', `<b>${best.label}</b><br><span style="color:#34d399">EV: +${ev}%</span>`, el);
        }
    }

    // ── Görünüm tab bar ───────────────────────────────────
    function renderViewTabs() {
        const el = document.getElementById('perf-view-tabs');
        if (!el || el.children.length > 0) return;   // Zaten render edildi
        el.innerHTML = `
            <button class="perf-filter-btn active" data-perf-view="overview">📊 Genel Sıralama</button>
            <button class="perf-filter-btn" data-perf-view="thresholds">🎯 Hedef Başarısı</button>
            <button class="perf-filter-btn" data-perf-view="horizon">📈 Getiri Analizi</button>
            <select id="perf-horizon-select" class="perf-horizon-select">
                ${Object.entries(horizonLabels).map(([k,v]) =>
                    `<option value="${k}" ${k=='5'?'selected':''}>${v}</option>`
                ).join('')}
            </select>
        `;
    }

    function renderView() {
        const grid = document.getElementById('perf-strategy-grid');
        if (!grid) return;
        grid.innerHTML = '';
        if (currentView === 'overview')    renderOverview(grid);
        else if (currentView === 'thresholds') renderThresholds(grid);
        else if (currentView === 'horizon')    renderHorizon(grid);
    }

    // ── GENEL SIRALAMA ────────────────────────────────────
    function renderOverview(grid) {
        const ranking = perfData.strategy_ranking || Object.keys(perfData.strategies || {});
        ranking.forEach((sid, rank) => {
            const s = perfData.strategies[sid];
            if (!s || s.signal_count === 0) return;

            const h5 = s.horizons?.[5];
            const mg = s.max_gain;
            const ev = s.expected_value_5d;

            const card = document.createElement('div');
            card.className = 'perf-strat-card';
            card.innerHTML = `
                <div class="perf-rank">#${rank + 1}</div>
                <div class="perf-strat-name">${s.label}</div>
                <div style="font-size:0.75rem;color:var(--text-muted)">${s.signal_count} sinyal</div>

                <div class="perf-metrics">
                    ${metricBox('T+5 Ort. Getiri', h5 ? retSpan(h5.avg_return) : '—')}
                    ${metricBox('İsabet Oranı', h5 ? `${h5.win_rate}%` : '—')}
                    ${metricBox('EV (5g)', ev != null ? retSpan(ev) : '—')}
                    ${metricBox('Max Getiri', mg?.avg != null ? retSpan(mg.avg) : '—')}
                    ${metricBox('Optimal Çıkış', mg?.avg_optimal_day != null ? `${mg.avg_optimal_day} gün` : '—')}
                    ${metricBox('En İyi Kazanç', mg?.best != null ? retSpan(mg.best) : '—')}
                </div>

                ${renderTopStocks(s.top_stocks)}
            `;
            grid.appendChild(card);
        });
    }

    // ── HEDEF BAŞARISI ────────────────────────────────────
    function renderThresholds(grid) {
        const strats = Object.entries(perfData.strategies || {})
            .filter(([, s]) => s.signal_count > 0)
            .sort((a, b) => {
                const ta = a[1].thresholds?.[5]?.hit_rate ?? 0;
                const tb = b[1].thresholds?.[5]?.hit_rate ?? 0;
                return tb - ta;
            });

        strats.forEach(([sid, s], rank) => {
            const thrs = s.thresholds || {};
            const thKeys = perfData.thresholds_available || [3, 5, 8, 10];

            const thrRows = thKeys.map(t => {
                const data = thrs[t] || {};
                const hr = data.hit_rate;
                const ad = data.avg_days;
                const bar = hr != null
                    ? `<div class="perf-thr-bar"><div class="perf-thr-fill" style="width:${hr}%;background:${thresholdColors[t] || '#888'}"></div></div>`
                    : '';
                return `
                    <div class="perf-thr-row">
                        <span class="perf-thr-label" style="color:${thresholdColors[t] || '#aaa'}">%${t} hedef</span>
                        ${bar}
                        <span class="perf-thr-val">${hr != null ? hr.toFixed(1)+'%' : '—'}</span>
                        <span class="perf-thr-days">${ad != null ? `~${ad}g` : ''}</span>
                    </div>
                `;
            }).join('');

            const card = document.createElement('div');
            card.className = 'perf-strat-card';
            card.innerHTML = `
                <div class="perf-rank">#${rank + 1}</div>
                <div class="perf-strat-name">${s.label}</div>
                <div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:6px">${s.signal_count} sinyal • Ort. max kazanç: ${retSpan(s.max_gain?.avg)}</div>
                <div class="perf-thr-table">${thrRows}</div>
                ${renderTopStocks(s.top_stocks)}
            `;
            grid.appendChild(card);
        });
    }

    // ── GETİRİ ANALİZİ (seçilen horizon) ─────────────────
    function renderHorizon(grid) {
        const h = selectedHorizon;
        const strats = Object.entries(perfData.strategies || {})
            .filter(([, s]) => s.signal_count > 0 && s.horizons?.[h])
            .sort((a, b) => (b[1].horizons[h]?.avg_return ?? -999) - (a[1].horizons[h]?.avg_return ?? -999));

        if (strats.length === 0) {
            grid.innerHTML = `<div class="perf-loading">${horizonLabels[h]} için yeterli veri yok.</div>`;
            return;
        }

        strats.forEach(([sid, s], rank) => {
            const hd = s.horizons[h];
            const card = document.createElement('div');
            card.className = 'perf-strat-card';
            card.innerHTML = `
                <div class="perf-rank">#${rank + 1} — ${horizonLabels[h]}</div>
                <div class="perf-strat-name">${s.label}</div>
                <div class="perf-metrics">
                    ${metricBox('Ort. Getiri', retSpan(hd.avg_return))}
                    ${metricBox('İsabet Oranı', `${hd.win_rate}%`)}
                    ${metricBox('Örneklem', `${hd.sample} sinyal`)}
                    ${metricBox('Beklenen Değer', retSpan(hd.ev))}
                </div>
                ${renderTopStocks(s.top_stocks)}
            `;
            grid.appendChild(card);
        });
    }

    // ── Yardımcı render fonksiyonları ─────────────────────
    function renderTopStocks(tops) {
        if (!tops || tops.length === 0) return '';
        const pills = tops.map(t =>
            `<span class="perf-top-stock">${t.symbol} <span style="color:${t.avg_max_gain >= 0 ? '#34d399' : '#f87171'}">+${t.avg_max_gain}%</span> <span style="color:var(--text-muted);font-size:0.7rem">(${t.count})</span></span>`
        ).join('');
        return `<div class="perf-top-stocks"><b style="margin-right:6px">🏅 En iyi hisseler:</b>${pills}</div>`;
    }

    function metricBox(label, value) {
        return `<div class="perf-metric"><div class="perf-metric-label">${label}</div><div class="perf-metric-value">${value}</div></div>`;
    }

    function retSpan(val) {
        if (val == null) return '—';
        const v = parseFloat(val);
        const color = v >= 0 ? '#34d399' : '#f87171';
        return `<span style="color:${color};font-weight:700">${v >= 0 ? '+' : ''}${v.toFixed(2)}%</span>`;
    }

    function mkSumCard(icon, label, value, container) {
        const div = document.createElement('div');
        div.className = 'perf-sum-card';
        div.innerHTML = `<div class="perf-sum-icon">${icon}</div><div class="perf-sum-label">${label}</div><div class="perf-sum-value">${value}</div>`;
        container.appendChild(div);
    }

    function setLoading(on) {
        const el = document.getElementById('perf-loading');
        if (el) el.classList.toggle('hidden', !on);
        document.getElementById('perf-empty')?.classList.add('hidden');
    }
    function showEmpty(msg) {
        const el = document.getElementById('perf-empty');
        if (el) { el.textContent = msg; el.classList.remove('hidden'); }
    }
})();
