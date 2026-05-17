// Lightweight script: handle prediction form and update simple UI
let predictionCount = 0;

function byId(id){return document.getElementById(id)}

document.addEventListener('DOMContentLoaded', () => {
    const form = byId('predictionForm');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const fd = new FormData(form);
            const commodity = fd.get('commodity') || 'Onion';
            // collect available inputs
            const payload = { commodity };
            const date = fd.get('Date');
            if (date) payload.Date = date;
            const market = fd.get('Market Name'); if (market) payload['Market Name'] = market;
            ['temp_mean','temp_min','temp_max','rainfall_mm','humidity','wind_speed','rainfall_mm_7d_sum','arrival_qty','price_lag_1d','price_lag_3d','price_lag_7d','price_lag_30d'].forEach(k=>{
                const v = fd.get(k);
                if (v !== null && v !== '') payload[k] = Number(v);
            });

            // derive week_of_year/week_sin/week_cos if date provided
            if (payload.Date) {
                const d = new Date(payload.Date);
                const onejan = new Date(d.getFullYear(),0,1);
                const week = Math.ceil((((d - onejan) / 86400000) + onejan.getDay()+1)/7);
                payload.week_of_year = week;
                payload.week_sin = Math.sin(2*Math.PI*week/52);
                payload.week_cos = Math.cos(2*Math.PI*week/52);
                payload.month = d.getMonth()+1;
                payload.day_of_year = Math.ceil((d - new Date(d.getFullYear(),0,0))/86400000);
            }

            // approximate momentum if possible
            if (payload.price_lag_1d && payload.price_lag_7d) {
                payload.price_momentum_7d = (payload.price_lag_1d - payload.price_lag_7d) / Math.max(1, payload.price_lag_7d);
            }

            const resultEl = byId('result');
            const loading = byId('loadingSpinner');
            if (loading) loading.style.display = 'block';
            if (resultEl) resultEl.style.display = 'none';

            try {
                const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
                const json = await res.json();
                if (!json) throw new Error('No result');
                const priceKg = json.price_per_kg !== undefined ? Number(json.price_per_kg) : (json.price ? Number(json.price)/100.0 : null);
                if (priceKg === null || isNaN(priceKg)) throw new Error(json && json.error ? json.error : 'No price returned');
                byId('resultCommodity').textContent = (commodity === 'Onion' ? '🧅 ' : '🍅 ') + commodity;
                byId('resultPrice').textContent = priceKg.toFixed(2);
                byId('resultConfidence').textContent = json.confidence ? Math.round(json.confidence*100) : '—';
                byId('resultMAE').textContent = json.mae ? Number(json.mae).toFixed(2) : '—';
                byId('resultR2').textContent = json.rmse ? Number(json.rmse).toFixed(3) : '—';
                byId('insightText').textContent = json.drivers ? Object.keys(json.drivers).slice(0,3).join(', ') : 'Model returned a forecast.';
                predictionCount++;
                const total = byId('totalPredictions'); if (total) total.textContent = predictionCount;
                if (resultEl) resultEl.style.display = 'block';
            } catch (err) {
                console.error(err);
                const errorMsg = byId('errorMsg'); if (errorMsg) errorMsg.textContent = 'Error: ' + (err.message || 'Unknown');
                const errorBox = byId('error'); if (errorBox) errorBox.style.display = 'block';
            } finally {
                if (loading) loading.style.display = 'none';
            }
        });
    }

    // fill small tiles on home
    const miniOnion = byId('miniOnion'); if (miniOnion) miniOnion.textContent = '32.40';
    const miniTomato = byId('miniTomato'); if (miniTomato) miniTomato.textContent = '28.10';
    byId('year') && (byId('year').textContent = new Date().getFullYear());
});
