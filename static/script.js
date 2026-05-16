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
            const temp = Number(fd.get('temp_mean') || 25);
            const rainfall = Number(fd.get('rainfall_mm') || 0);
            const price7 = Number(fd.get('price_lag_7d') || 30);

            // keep payload small and simple
            const payload = { commodity, temp_mean: temp, rainfall_mm: rainfall, price_lag_7d: price7 };

            const resultEl = byId('result');
            const loading = byId('loadingSpinner');
            if (loading) loading.style.display = 'block';
            if (resultEl) resultEl.style.display = 'none';

            try {
                const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
                const json = await res.json();
                if (!json || !json.price) throw new Error(json && json.error ? json.error : 'No result');
                byId('resultCommodity').textContent = (commodity === 'Onion' ? '🧅 ' : '🍅 ') + commodity;
                byId('resultPrice').textContent = Number(json.price).toFixed(2);
                byId('resultConfidence').textContent = json.confidence ? Math.round(json.confidence*100) : '—';
                byId('resultMAE').textContent = json.mae ? Number(json.mae).toFixed(2) : '—';
                byId('resultR2').textContent = json.r2 ? Number(json.r2).toFixed(3) : '—';
                byId('insightText').textContent = json.insight || 'Model returned a forecast.';
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
