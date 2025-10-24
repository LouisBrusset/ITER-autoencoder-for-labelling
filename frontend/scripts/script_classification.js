(function(){
const API_BASE = window.API_BASE || "http://localhost:8000";

window.classificationChart = null;
window.classificationData = null;
window.overlayCanvas = null;
window.overlayCtx = null;
window.isDrawing = false;
window.polygon = [];
window._pendingLabelChanges = false;

// Helper: produce HTML summary with colored badges for each label
window._buildLabelSummaryHTML = function(counts, colorMap){
    const total = Object.values(counts).reduce((s,v)=>s+v,0);
    const keys = Object.keys(counts).sort((a,b)=>parseInt(a)-parseInt(b));
    let html = `<div style="font-size:14px; margin-bottom:6px;"><strong>Total:</strong> ${total}</div>`;
    html += `<div style="display:flex; flex-wrap:wrap; gap:8px; align-items:center;">`;
    keys.forEach(k=>{
        const color = (colorMap && colorMap[k]) ? colorMap[k] : 'rgba(128,128,128,0.6)';
        const count = counts[k];
        html += `<span style="display:inline-flex;align-items:center;padding:4px 8px;background:rgba(0,0,0,0.03);border-radius:12px;border:1px solid rgba(0,0,0,0.04);">
            <span style=\"width:12px;height:12px;background:${color};display:inline-block;margin-right:8px;border-radius:2px;\"></span>
            <span style=\"font-weight:600;margin-right:6px;\">Label ${k}</span>
            <span style=\"font-size:12px;opacity:0.85;\">(n=${count})</span>
        </span>`;
    });
    html += `</div>`;
    return html;
};

window.initClassificationChart = function() {
    const ctx = document.getElementById('classificationChart').getContext('2d');
    // destroy previous chart instance if present to avoid duplicate listeners/state
    if (window.classificationChart) {
        try { window.classificationChart.destroy(); } catch (e) { /* ignore */ }
        window.classificationChart = null;
    }
    window.classificationChart = new Chart(ctx, {
        type: 'scatter',
        data: { datasets: [{ label: 'Points', data: [], pointRadius: 4 }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { title: { display: true, text: 'UMAP 1' } }, y: { title: { display: true, text: 'UMAP 2' } } }, plugins: { tooltip: { callbacks: { label: function(context){ const p = context.raw; return `Index: ${p.original_index}, Label: ${p.label}`; } } } } }
    });

    // setup overlay canvas for drawing selection polygon
    overlayCanvas = document.getElementById('classificationOverlay');
    overlayCanvas.style.pointerEvents = 'auto';
    overlayCtx = overlayCanvas.getContext('2d');
    // ensure overlay matches the chart immediately
    syncOverlayToChart();
    // clear any previous drawings
    if (overlayCtx) overlayCtx.clearRect(0,0,overlayCanvas.width, overlayCanvas.height);
    window.addEventListener('resize', () => { if (classificationChart) syncOverlayToChart(); });

    // mouse events for drawing polygon
    overlayCanvas.addEventListener('mousedown', (e)=>{
        e.preventDefault();
        isDrawing = true; polygon = [];
        const r = overlayCanvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const x = (e.clientX - r.left) * (overlayCanvas.width / (r.width * dpr));
        const y = (e.clientY - r.top) * (overlayCanvas.height / (r.height * dpr));
        polygon.push([x,y]); overlayCtx.clearRect(0,0,overlayCanvas.width, overlayCanvas.height);
        overlayCtx.beginPath(); overlayCtx.moveTo(x,y);
    });
    overlayCanvas.addEventListener('mousemove', (e)=>{
        if(!isDrawing) return; e.preventDefault(); const r = overlayCanvas.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1; const x = (e.clientX - r.left) * (overlayCanvas.width / (r.width * dpr)); const y = (e.clientY - r.top) * (overlayCanvas.height / (r.height * dpr)); polygon.push([x,y]); overlayCtx.lineTo(x,y); overlayCtx.strokeStyle='rgba(0,0,0,0.6)'; overlayCtx.lineWidth=2; overlayCtx.stroke();
    });
    overlayCanvas.addEventListener('mouseup', async (e)=>{
        if(!isDrawing) return; isDrawing=false; overlayCtx.closePath();
        const selected = [];
        const xScale = classificationChart.scales['x']; const yScale = classificationChart.scales['y'];
        if(!xScale || !yScale) return;
        // convert polygon points to chart pixel coordinates
        const r = overlayCanvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const polyPixels = polygon.map(pt => {
            const px = pt[0] * (r.width * dpr) / overlayCanvas.width + r.left * dpr;
            const py = pt[1] * (r.height * dpr) / overlayCanvas.height + r.top * dpr;
            return [px/dpr - r.left, py/dpr - r.top];
        });
        for(let i=0;i<classificationData.points.length;i++){
            const p = classificationData.points[i];
            const px = xScale.getPixelForValue(p.x); const py = yScale.getPixelForValue(p.y);
            if(pointInPolygon([px,py], polyPixels)) selected.push(i);
        }
        if(selected.length===0){ document.getElementById('classificationInfo').textContent='No points selected.'; return; }
        const newLabel = prompt(`Assign new integer label to ${selected.length} selected points:`, '1');
        if(newLabel===null){ document.getElementById('classificationInfo').textContent='Selection cancelled.'; return; }
        const lab = parseInt(newLabel); if(isNaN(lab)){ alert('Label must be integer'); return; }
        selected.forEach(idx => classificationData.points[idx].label = lab);
        // mark pending changes and enable apply button
        window._pendingLabelChanges = true;
        const applyBtn = document.getElementById('applyLabelsBtn'); if (applyBtn) applyBtn.disabled = false;
        document.getElementById('classificationInfo').textContent = `Assigned label ${lab} to ${selected.length} points (pending apply).`;
        setTimeout(()=>{ overlayCtx.clearRect(0,0,overlayCanvas.width, overlayCanvas.height); }, 500);
    });
};

window.pointInPolygon = function(point, vs){
    const x = point[0], y = point[1]; let inside=false; for(let i=0,j=vs.length-1;i<vs.length;j=i++){
        const xi=vs[i][0], yi=vs[i][1], xj=vs[j][0], yj=vs[j][1]; const intersect = ((yi>y)!=(yj>y)) && (x < (xj-xi)*(y-yi)/(yj-yi)+xi); if(intersect) inside=!inside;
    } return inside;
};

window.syncOverlayToChart = function(){
    if(!classificationChart || !overlayCanvas) return;
    const chartCanvas = classificationChart.canvas;
    const chartRect = chartCanvas.getBoundingClientRect();
    overlayCanvas.style.left = chartCanvas.offsetLeft + 'px';
    overlayCanvas.style.top = chartCanvas.offsetTop + 'px';
    overlayCanvas.style.width = chartRect.width + 'px';
    overlayCanvas.style.height = chartRect.height + 'px';
    const dpr = window.devicePixelRatio || 1;
    overlayCanvas.width = Math.round(chartRect.width * dpr);
    overlayCanvas.height = Math.round(chartRect.height * dpr);
    overlayCtx = overlayCanvas.getContext('2d');
    overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    overlayCtx.clearRect(0,0,overlayCanvas.width, overlayCanvas.height);
    overlayCanvas.style.pointerEvents = 'auto';
};

window.generateClassificationLatentSpace = async function(){
    try{
        const subset = document.querySelector('input[name="classSubset"]:checked')?.value || 'validation';
        const resp = await fetch(`${API_BASE}/latent-space?subset=${encodeURIComponent(subset)}`);
        const data = await resp.json();
        classificationData = { points: data.points.map(p=>({x:p.x,y:p.y,label:p.label,original_index:p.original_index})) };
        initClassificationChart(); renderClassificationPoints();
        document.getElementById('classificationDatasetInfo').innerHTML = document.getElementById('visualizationDatasetInfo').innerHTML;
        document.getElementById('classificationModelInfo').innerHTML = document.getElementById('visualizationModelInfo').innerHTML;
        // reset pending flag and disable apply button
        window._pendingLabelChanges = false;
        // enable save button by default (user requested it's not hidden)
        const saveBtn = document.getElementById('saveLabelsBtn'); if (saveBtn) saveBtn.disabled = false;
        // if there are existing labels (non-zero), apply per-label coloring immediately so they display
        const hasLabels = classificationData.points && classificationData.points.some(pt => typeof pt.label !== 'undefined' && pt.label !== null && Number(pt.label) !== 0);
        if(hasLabels){
            // apply colors which will also enable the save button
            applyClassificationColors();
        }
    }catch(err){ console.error(err); alert('Error loading classification latent space'); }
};

window.renderClassificationPoints = function(){
    if(!classificationChart || !classificationData) return;
    const palette=['rgba(54,162,235,0.8)','rgba(255,99,132,0.8)','rgba(75,192,192,0.8)','rgba(255,205,86,0.8)','rgba(153,102,255,0.8)','rgba(201,203,207,0.8)'];
    const colorMap={}; let mapCount=0;
    // If chart currently has more than one dataset (per-label view), update datasets in-place
    if(classificationChart.data.datasets && classificationChart.data.datasets.length > 1){
        // build groups by label
        const groups = {};
        classificationData.points.forEach(p=>{ const lab = p.label||0; if(!groups[lab]) groups[lab]=[]; groups[lab].push(p); });
        // update each existing dataset by extracting the label number from its label text
        classificationChart.data.datasets.forEach(ds => {
            const labelText = ds.label || '';
            const m = labelText.match(/Label\s+(\d+)/);
            if(m){ const key = m[1]; ds.data = (groups[key]||[]).map(p=>({x:p.x,y:p.y,original_index:p.original_index,label:p.label})); }
        });
        classificationChart.update();
    } else {
        const ds = classificationData.points.map(p=>{ const lab = p.label||0; if(!(lab in colorMap)){ colorMap[lab]=palette[mapCount%palette.length]; mapCount++; } return {x:p.x,y:p.y,original_index:p.original_index,label:lab, backgroundColor: colorMap[lab]}; });
        classificationChart.data.datasets[0].data = ds;
        classificationChart.update();
        // show counts per label (unless there are pending changes)
        if(!window._pendingLabelChanges){
            const counts = {};
            classificationData.points.forEach(p=>{ const lab = p.label||0; counts[lab] = (counts[lab]||0)+1; });
            const infoEl = document.getElementById('classificationInfo');
            if(infoEl){ infoEl.innerHTML = window._buildLabelSummaryHTML(counts, colorMap); }
        }
    }
};

window.applyClassificationColors = function(){
    if(!classificationChart || !classificationData) return;
    // build groups by label
    const groups = {};
    classificationData.points.forEach(p => {
        const lab = p.label || 0;
        if(!groups[lab]) groups[lab] = [];
        groups[lab].push(p);
    });
    // create datasets per label with colors
    // palette of 15 different colors
    const palette = [
        'rgba(54,162,235,0.8)',
        'rgba(255,99,132,0.8)',
        'rgba(75,192,192,0.8)',
        'rgba(255,205,86,0.8)',
        'rgba(153,102,255,0.8)',
        'rgba(201,203,207,0.8)',
        'rgba(255,159,64,0.8)',
        'rgba(99,255,132,0.8)', 
        'rgba(132,99,255,0.8)',
        'rgba(255,99,245,0.8)', 
        'rgba(255,99,64,0.8)', 
        'rgba(255,99,128,0.8)', 
        'rgba(255,99,191,0.8)'
    ];
    const datasets = [];
    const colorMap = {};
    let i = 0;
    Object.keys(groups).forEach(key => {
        const color = palette[i % palette.length];
        colorMap[key] = color;
        const count = groups[key].length;
        datasets.push({ label: `Label ${key} (n=${count})`, data: groups[key].map(p=>({x:p.x,y:p.y,original_index:p.original_index,label:p.label})), backgroundColor: color, pointRadius: 4 });
        i++;
    });
    classificationChart.data.datasets = datasets;
    classificationChart.update();
    // build summary string with counts per label
    const counts = {};
    Object.keys(groups).forEach(k=> counts[k] = groups[k].length);
    const infoEl = document.getElementById('classificationInfo');
    if(infoEl){ infoEl.innerHTML = `<div style="margin-bottom:6px;"><strong>Applied ${Object.keys(groups).length} label(s)</strong></div>` + window._buildLabelSummaryHTML(counts, colorMap); }
    // reset pending flag and disable apply button
    window._pendingLabelChanges = false;
    const applyBtn = document.getElementById('applyLabelsBtn'); if (applyBtn) applyBtn.disabled = true;
    // enable save button after applying labels
    const saveBtn = document.getElementById('saveLabelsBtn'); if (saveBtn) saveBtn.disabled = false;
};

// Send labels to backend to persist them
window.saveLabels = async function(){
    if(!classificationData || !classificationData.points) return alert('No classification data to save');
    // Build pairs and sort by original_index so saved arrays are in dataset order
    const pairs = classificationData.points.map(p => ({ index: (p.original_index != null ? p.original_index : null), label: (p.label != null ? p.label : 0) }));
    pairs.sort((a,b) => (a.index === null ? 1 : (b.index === null ? -1 : a.index - b.index)));
    const indices = pairs.map(p => p.index);
    const labels = pairs.map(p => p.label);
    try{
    const resp = await fetch(`${API_BASE}/save-labels`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({labels, indices}) });
        if(!resp.ok){ const txt = await resp.text(); throw new Error(txt||'Failed to save labels'); }
        const data = await resp.json();
        const infoEl = document.getElementById('classificationInfo');
        if(infoEl){
            if(data.file){
                infoEl.innerHTML = `<div>Labels saved (<strong>${data.n_labels}</strong>)</div><div style="font-size:12px;opacity:0.85;">Saved to: ${data.file}</div>`;
            }else{
                infoEl.textContent = `Labels saved (${data.n_labels}).`;
            }
        }
        const saveBtn = document.getElementById('saveLabelsBtn'); if (saveBtn) saveBtn.disabled = true;
    }catch(err){ console.error(err); alert('Error saving labels: '+err.message); }
};

// export all labels grouped by train/validation and download as JSON
window.extractAllLabels = async function(){
    try{
        const resp = await fetch(`${API_BASE}/export-all-labels`);
        if(!resp.ok){ const txt = await resp.text(); throw new Error(txt||'Failed to export labels'); }
        const payload = await resp.json();
        const filename = payload.file ? payload.file.split('/').pop() : `exported_labels_${Date.now()}.json`;
        const data = payload.data || {};

        // create blob and download
        const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
        URL.revokeObjectURL(url);

        const infoEl = document.getElementById('classificationInfo');
        if(infoEl) infoEl.textContent = `Exported labels to ${filename}`;
    }catch(err){ console.error(err); alert('Error exporting labels: '+err.message); }
};

})();