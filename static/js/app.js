(function(){
  const FEATURES = ['temp_mean','rainfall_mm','humidity','wind_speed','price_lag_7d','price_lag_30d','rainfall_mm_7d_sum','month','day_of_year','drought_indicator','flood_indicator'];

  const $ = id => document.getElementById(id);
  const commodity = $('commodity');
  const temp = $('temp'); const tempVal = $('tempVal');
  const rain = $('rain'); const rainVal = $('rainVal');
  const hum = $('hum'); const humVal = $('humVal');
  const predictBtn = $('predictBtn');
  const emptyState = $('emptyState');
  const resultState = $('resultState');
  const priceDisplayQuintal = $('priceDisplayQuintal');
  const priceDisplayKg = $('priceDisplayKg');
  const maeBadge = $('maeBadge');
  const rmseBadge = $('rmseBadge');
  const priceChartCtx = document.getElementById('priceChart').getContext('2d');
  const advancedToggle = $('advancedToggle');
  const advanced = $('advanced');

  temp.addEventListener('input', ()=> tempVal.textContent = temp.value);
  rain.addEventListener('input', ()=> rainVal.textContent = rain.value);
  hum.addEventListener('input', ()=> humVal.textContent = hum.value);

  advancedToggle.addEventListener('click', ()=>{
    advanced.classList.toggle('hidden');
    advancedToggle.textContent = advanced.classList.contains('hidden') ? 'Show' : 'Hide';
  });

  let chart = null;
  let driversChart = null;
  function drawChart(source){
    // source can be a number (center) or an array of numeric values
    let labels = [];
    let data = [];
    if(Array.isArray(source)){
      data = source.map(v => v === null || v === undefined ? 0 : Number(v));
      for(let i=0;i<data.length;i++){
        const d = new Date(); d.setDate(d.getDate()+i);
        labels.push(d.toLocaleDateString(undefined,{month:'short',day:'numeric'}));
      }
    } else {
      const center = Number(source) || 0;
      for(let i=6;i>=0;i--){
        const d = new Date(); d.setDate(d.getDate()-i);
        labels.push(d.toLocaleDateString(undefined,{month:'short',day:'numeric'}));
        const jitter = (Math.random()-0.5)*(center*0.03);
        data.push(Math.max(0, Number((center + jitter).toFixed(2))));
      }
    }

    if(chart) chart.destroy();
    const dataset = {
      label: 'Price (₹)',
      data: data,
      fill: true,
      backgroundColor: 'rgba(14,165,164,0.08)',
      borderColor: '#0ea5a4',
      tension: 0.3,
      pointRadius: 3,
      pointBackgroundColor: '#0ea5a4'
    };

    const cfg = {
      type: 'line',
      data: { labels: labels, datasets: [dataset] },
      options: {
        plugins: { legend: { display: false } },
        scales: { x: { grid: { display: false } }, y: { grid: { color: '#f3f4f6' } } }
      }
    };

    chart = new Chart(priceChartCtx, cfg);
  }

  function drawDrivers(drivers){
    const ctx = document.getElementById('driversChart').getContext('2d');
    const labels = Object.keys(drivers);
    const data = labels.map(k => drivers[k]);
    if(driversChart) driversChart.destroy();
    driversChart = new Chart(ctx,{
      type: 'bar',
      data: { labels: labels.map(l => l.replace(/_/g,' ')), datasets: [{label:'Driver weight', data, backgroundColor: '#34c759'}] },
      options: { indexAxis: 'y', plugins:{legend:{display:false}}, scales:{x:{display:false}} }
    });
  }

  async function doPredict(){
    const payload = {};
    payload.commodity = commodity.value;
    // basic inputs
    payload.temp_mean = parseFloat(temp.value);
    payload.rainfall_mm = parseFloat(rain.value);
    payload.humidity = parseFloat(hum.value);
    payload.wind_speed = 2.0;
    payload.price_lag_7d = parseFloat($('lag7').value || 0);
    payload.price_lag_30d = parseFloat($('lag30').value || 0);
    payload.rainfall_mm_7d_sum = parseFloat($('rain7').value || (parseFloat(rain.value)*7));
    payload.month = parseInt($('month').value || (new Date().getMonth()+1));
    payload.day_of_year = parseInt($('day').value || (Math.ceil((new Date()- new Date(new Date().getFullYear(),0,1))/(1000*60*60*24))));
    payload.drought_indicator = $('drought').checked ? 1 : 0;
    payload.flood_indicator = $('flood').checked ? 1 : 0;

    // show loading state
    predictBtn.disabled = true; predictBtn.textContent = 'Predicting...'; predictBtn.classList.add('opacity-80');

    try{
      const res = await fetch('/predict',{
        method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)
      });
      const json = await res.json();
      if(res.ok && json.status === 'Success'){
        emptyState.classList.add('hidden'); resultState.classList.remove('hidden');
        // backend now returns both quintal and kg prices
        const pq = json.price_per_quintal !== undefined ? Number(json.price_per_quintal) : (json.price !== undefined ? Number(json.price) : 0);
        const pk = json.price_per_kg !== undefined ? Number(json.price_per_kg) : +(pq/100).toFixed(2);
        priceDisplayQuintal.textContent = '₹'+pq.toFixed(2) + ' / quintal';
        priceDisplayKg.textContent = '₹'+pk.toFixed(2) + ' / kg';
        maeBadge.textContent = json.mae !== null ? json.mae : '-';
        rmseBadge.textContent = json.rmse !== null ? json.rmse : '-';
        // price chart: use forecast array if available
        if(json.forecast_quintal && Array.isArray(json.forecast_quintal)){
          drawChart(json.forecast_quintal);
        } else if(json.forecast && Array.isArray(json.forecast)){
          drawChart(json.forecast);
        } else {
          drawChart(pq);
        }

        // drivers
        if(json.drivers){ drawDrivers(json.drivers); }

        // swap images
        if(json.image) document.getElementById('assetImage').src = json.image;
      } else {
        alert(json.message || json.error || 'Prediction failed');
      }
    } catch(err){
      alert('Network error: '+err.message);
    } finally{
      predictBtn.disabled = false; predictBtn.textContent = 'Predict Price'; predictBtn.classList.remove('opacity-80');
    }
  }

  predictBtn.addEventListener('click', doPredict);

  // theme toggle
  const themeToggle = $('themeToggle');
  themeToggle.addEventListener('click', ()=>{
    document.documentElement.classList.toggle('dark');
  });

  // ambience toggle initial state
  const ambienceToggle = document.getElementById('ambienceToggle');
  if(ambienceToggle){
    const setBtn = (on)=>{ ambienceToggle.textContent = on ? '💫' : '⛔'; };
    try{
      const stored = localStorage.getItem('ambienceEnabled');
      const on = stored === null ? true : stored === '1';
      setBtn(on);
    }catch(e){ setBtn(true); }
    ambienceToggle.addEventListener('click', ()=>{
      try{
        const cur = localStorage.getItem('ambienceEnabled');
        const next = cur === '1' ? '0' : '1';
        localStorage.setItem('ambienceEnabled', next);
        setBtn(next === '1');
        // trigger a resize to let three.js pick new state
        window.dispatchEvent(new Event('resize'));
      }catch(e){}
    });
  }

  // Fetch analytics on load and render charts
  async function loadAnalytics(){
    try{
      const res = await fetch('/analytics');
      const j = await res.json();
      if(res.ok && !j.error){
        // render price chart using series (take Onion by default)
        const onionSeries = j.series.Onion.map(v=>v===null?0:v);
        if(onionSeries && onionSeries.length){
          drawChart(onionSeries);
        }
        // render drivers for Onion
        if(j.drivers && j.drivers.Onion){ drawDrivers(j.drivers.Onion); }
        // set KPI badges
        if(j.kpis && j.kpis.onion){
          // show latest onion price somewhere if needed
        }
      }
    }catch(e){ console.warn('analytics load failed', e); }
  }

  // load analytics immediately
  loadAnalytics();
})();
