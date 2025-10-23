(function(){
const API_BASE = window.API_BASE || "http://localhost:8000";

window.classificationChart = null;
window.classificationData = null;
window.overlayCanvas = null;
window.overlayCtx = null;
window.isDrawing = false;
window.polygon = [];

window.initClassificationChart = function() {
    const ctx = document.getElementById('classificationChart').getContext('2d');
    window.classificationChart = new Chart(ctx, {
        type: 'scatter',
        data: { datasets: [{ label: 'Points', data: [], pointRadius: 4 }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { title: { display: true, text: 'UMAP 1' } }, y: { title: { display: true, text: 'UMAP 2' } } }, plugins: { tooltip: { callbacks: { label: function(context){ const p = context.raw; return `Index: ${p.original_index}, Label: ${p.label}`; } } } } }
    });

    // setup overlay canvas for drawing selection polygon
    overlayCanvas = document.getElementById('classificationOverlay');
    overlayCanvas.style.pointerEvents = 'auto';
    overlayCtx = overlayCanvas.getContext('2d');
    syncOverlayToChart();
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
        renderClassificationPoints();
        document.getElementById('classificationInfo').textContent = `Assigned label ${lab} to ${selected.length} points.`;
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
        const resp = await fetch(`${API_BASE}/latent-space`);
        const data = await resp.json();
    classificationData = { points: data.points.map(p=>({x:p.x,y:p.y,label:p.label,original_index:p.original_index})) };
        initClassificationChart(); renderClassificationPoints();
        document.getElementById('classificationDatasetInfo').innerHTML = document.getElementById('visualizationDatasetInfo').innerHTML;
        document.getElementById('classificationModelInfo').innerHTML = document.getElementById('visualizationModelInfo').innerHTML;
    }catch(err){ console.error(err); alert('Error loading classification latent space'); }
};

window.renderClassificationPoints = function(){
    if(!classificationChart || !classificationData) return; const palette=['rgba(54,162,235,0.8)','rgba(255,99,132,0.8)','rgba(75,192,192,0.8)','rgba(255,205,86,0.8)','rgba(153,102,255,0.8)','rgba(201,203,207,0.8)']; const colorMap={}; let mapCount=0;
    const ds = classificationData.points.map(p=>{ const lab = p.label||0; if(!(lab in colorMap)){ colorMap[lab]=palette[mapCount%palette.length]; mapCount++; } return {x:p.x,y:p.y,original_index:p.original_index,label:lab, backgroundColor: colorMap[lab]}; });
    classificationChart.data.datasets[0].data = ds; classificationChart.update();
};

})();