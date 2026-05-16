// Lightweight frontend interactions, charts and prediction handling
document.addEventListener('DOMContentLoaded', () => {
  // DOM refs
  const yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  const navToggle = document.getElementById('navToggle');
  const navLinks = document.getElementById('nav-links');
  navToggle && navToggle.addEventListener('click', () => {
    if (navLinks.style.display === 'flex') navLinks.style.display = 'none';
    else navLinks.style.display = 'flex';
  });

  // Hero sparkline
  const heroSpark = document.getElementById('heroSpark');
  if (heroSpark) {
    const ctx = heroSpark.getContext('2d');
    const data = Array.from({length:28}, ()=>32 + (Math.random()-0.5)*2);
    drawSparkline(ctx, data, {stroke:'#22C55E'});
  }

  // Line chart
  const lineCanvas = document.getElementById('lineChart');
  if (lineCanvas) {
    const ctx = lineCanvas.getContext('2d');
    const labels = Array.from({length:14}, (_,i)=>`D${i+1}`);
    const onion = generateSeries(32,1.2,14);
    const tomato = generateSeries(28,1.8,14);
    drawLineChart(ctx, labels, [onion,tomato], {colors:['#22C55E','#F59E0B']});
  }

  // Prediction form
  const form = document.getElementById('predictionForm');
  form && form.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const btn = form.querySelector('button[type="submit"]');
    btn.disabled = true; btn.textContent = 'Predicting...';
    const payload = {
      commodity: form.commodity.value,
      temp_mean: parseFloat(form.temp_mean.value || 25),
      rainfall_mm: parseFloat(form.rainfall_mm.value || 0),
      price_lag_7d: parseFloat(form.price_lag_7d.value || 0)
    };
    showLoading(true);
    try{
      const res = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      const json = await res.json();
      if(json && json.success){
        showResult(json);
      } else {
        showError(json.error || 'Prediction failed');
      }
    }catch(err){
      showError(err.message || 'Network error');
    }
    btn.disabled = false; btn.textContent = 'Predict Price';
    showLoading(false);
  });

  // helpers
  function showLoading(on){
    document.getElementById('loading').style.display = on ? 'block' : 'none';
  }

  function showResult(json){
    const resEl = document.getElementById('result');
    document.getElementById('resultCommodity').textContent = (json.commodity || '—');
    document.getElementById('resultPrice').textContent = (json.price !== undefined) ? Number(json.price).toFixed(2) : '--';
    document.getElementById('resultConfidence').textContent = json.accuracy || '—';
    document.getElementById('resultMAE').textContent = json.mae || '—';
    document.getElementById('resultR2').textContent = json.r2_score || '—';
    document.getElementById('insightText').textContent = json.insight || 'Model indicates short-term movement based on recent patterns.';
    resEl.classList.remove('hidden');
    resEl.animate([{opacity:0, transform:'translateY(8px)'},{opacity:1, transform:'translateY(0)'}],{duration:420,easing:'cubic-bezier(.2,.9,.2,1)'});
  }

  function showError(msg){
    alert(msg);
  }

});

/* Utility: draw small sparkline */
function drawSparkline(ctx,data,opts={}){
  const w = ctx.canvas.width, h = ctx.canvas.height; ctx.clearRect(0,0,w,h);
  const pad = 6; const min = Math.min(...data), max = Math.max(...data);
  ctx.beginPath();
  data.forEach((v,i)=>{
    const x = pad + (i/(data.length-1))*(w-pad*2);
    const y = h - pad - ((v-min)/(max-min || 1))*(h-pad*2);
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.strokeStyle = opts.stroke || '#22C55E'; ctx.lineWidth = 2; ctx.stroke();
  // area
  ctx.lineTo(w-pad, h-pad); ctx.lineTo(pad,h-pad); ctx.closePath();
  const grad = ctx.createLinearGradient(0,0,0,h); grad.addColorStop(0, 'rgba(34,197,94,0.12)'); grad.addColorStop(1,'rgba(34,197,94,0)');
  ctx.fillStyle = grad; ctx.fill();
}

/* generate smooth series */
function generateSeries(mean,vol,len){
  const arr=[]; let v=mean; for(let i=0;i<len;i++){ v += (Math.random()-0.5)*vol; arr.push(Math.max(0.1, +v.toFixed(2))); } return arr;
}

/* Simple animated line chart without libraries */
function drawLineChart(ctx, labels, seriesArr, opts={colors:['#22C55E','#F59E0B']}){
  const w = ctx.canvas.width, h = ctx.canvas.height; ctx.clearRect(0,0,w,h);
  // find global min/max
  const all = seriesArr.flat(); const min = Math.min(...all), max = Math.max(...all);
  // draw grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.03)'; ctx.lineWidth = 1; ctx.beginPath();
  for(let i=0;i<5;i++){ const y = 20 + i*(h-40)/4; ctx.moveTo(40,y); ctx.lineTo(w-20,y);} ctx.stroke();

  seriesArr.forEach((series,idx)=>{
    const color = opts.colors[idx] || '#999';
    ctx.beginPath();
    series.forEach((v,i)=>{
      const x = 40 + (i/(series.length-1))*(w-60);
      const y = h-20 - ((v-min)/(max-min || 1))*(h-60);
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    // stroke
    ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.stroke();
    // fill under curve
    ctx.lineTo(w-20,h-20); ctx.lineTo(40,h-20); ctx.closePath();
    const g = ctx.createLinearGradient(0,0,0,h); g.addColorStop(0, hexToRgba(color,0.14)); g.addColorStop(1, hexToRgba(color,0));
    ctx.fillStyle = g; ctx.fill();
  });
}

function hexToRgba(hex, a){
  const h = hex.replace('#',''); const r=parseInt(h.substring(0,2),16), g=parseInt(h.substring(2,4),16), b=parseInt(h.substring(4,6),16);
  return `rgba(${r},${g},${b},${a})`;
}

/* Parallax background + glow cursor */
(() => {
  const blobsLayer = document.querySelector('.bg-blobs');
  // create blobs if missing (index.html uses .bg-blobs)
  if (blobsLayer && blobsLayer.children.length === 0) {
    ['b1','b2','b3'].forEach((c)=>{
      const d = document.createElement('div'); d.className = `blob ${c} parallax-layer`; blobsLayer.appendChild(d);
    });
  }

  // add global glow overlay
  if (!document.querySelector('.bg-glow')){
    const g = document.createElement('div'); g.className = 'bg-glow'; document.body.appendChild(g);
  }

  const layers = Array.from(document.querySelectorAll('.parallax-layer'));
  const glow = document.querySelector('.bg-glow');

  let mouse = {x: window.innerWidth/2, y: window.innerHeight/2};
  let pos = {x: mouse.x, y: mouse.y};

  window.addEventListener('mousemove', (e)=>{
    mouse.x = e.clientX; mouse.y = e.clientY;
    // update CSS vars for background glow
    const gx = Math.round((mouse.x / window.innerWidth) * 100);
    const gy = Math.round((mouse.y / window.innerHeight) * 100);
    document.documentElement.style.setProperty('--mx', gx + '%');
    document.documentElement.style.setProperty('--my', gy + '%');
    document.documentElement.style.setProperty('--glow', '0.16');
  });

  window.addEventListener('mouseleave', ()=>{ document.documentElement.style.setProperty('--glow','0'); });

  function lerp(a,b,t){return a + (b-a)*t}

  function frame(){
    pos.x = lerp(pos.x, mouse.x, 0.12);
    pos.y = lerp(pos.y, mouse.y, 0.12);
    const cx = (pos.x - window.innerWidth/2) / (window.innerWidth/2);
    const cy = (pos.y - window.innerHeight/2) / (window.innerHeight/2);

    // transform blobs at different depths
    layers.forEach((el, i) => {
      const depth = (i+1) * 18; // px multiplier
      const tx = (cx * depth).toFixed(2);
      const ty = (cy * depth * -1).toFixed(2);
      const rot = (cx * depth * (i%2?1:-1)).toFixed(2);
      el.style.transform = `translate3d(${tx}px, ${ty}px, 0) rotate(${rot}deg)`;
      el.style.opacity = 0.9 - i*0.12;
    });

    // subtle rotate/tilt for entire blob layer for 3D effect
    if (blobsLayer) blobsLayer.style.transform = `perspective(800px) rotateX(${cy*6}deg) rotateY(${cx*6}deg)`;

    // update glow position smoothly
    if (glow) {
      const gx = (pos.x / window.innerWidth) * 100;
      const gy = (pos.y / window.innerHeight) * 100;
      glow.style.setProperty('--mx', gx + '%');
      glow.style.setProperty('--my', gy + '%');
      // shift background-position by setting pseudo-element via CSS var on :root
      document.documentElement.style.setProperty('--mx', gx + '%');
      document.documentElement.style.setProperty('--my', gy + '%');
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
})();
