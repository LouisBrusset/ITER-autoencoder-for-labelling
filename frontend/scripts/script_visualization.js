(function(){
const API_BASE = window.API_BASE || "http://localhost:8000";
window.latentSpaceChart = null;
window.reconstructionChart = null;

window.initLatentSpaceChart = function() {
    const ctx = document.getElementById('latentSpaceChart').getContext('2d');
    window.latentSpaceChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{ label: 'Normal Points', data: [], backgroundColor: 'rgba(54, 162, 235, 0.6)', borderColor: 'rgba(54, 162, 235, 1)', borderWidth: 1, pointRadius: 5 }, { label: 'Anomalies', data: [], backgroundColor: 'rgba(255, 99, 132, 0.6)', borderColor: 'rgba(255, 99, 132, 1)', borderWidth: 1, pointRadius: 5 }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { x: { title: { display: true, text: 'UMAP 1' } }, y: { title: { display: true, text: 'UMAP 2' } } },
            plugins: { tooltip: { callbacks: { label: function(context){ const point = context.raw; return `Index: ${point.original_index}, Label: ${point.label}`; } } } }
        }
    });
};

window.initReconstructionChart = function() {
    const ctx = document.getElementById('reconstructionChart').getContext('2d');
    window.reconstructionChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [ { label: 'Original', data: [], borderColor: 'rgba(54,162,235,1)', fill: false }, { label: 'Reconstruction', data: [], borderColor: 'rgba(255,99,132,1)', fill: false } ] },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { title: { display: true, text: 'Feature index' } }, y: { title: { display: true, text: 'Value' } } } }
    });
};

window.generateLatentSpace = async function() {
    try {
        // update inference status badge on visualization page
        try { if (window.checkInferenceStatus) await window.checkInferenceStatus(); } catch(e) { /* ignore */ }
        const subset = document.querySelector('input[name="visSubset"]:checked')?.value || 'validation';
        const response = await fetch(`${API_BASE}/latent-space?subset=${encodeURIComponent(subset)}`);
        const data = await response.json();
        // Group points dynamically by label so we can support an unknown number of labels
        const groups = {};
        data.points.forEach(p => {
            const lab = (typeof p.label === 'undefined' || p.label === null) ? 0 : String(p.label);
            if (!groups[lab]) groups[lab] = [];
            groups[lab].push(p);
        });

        // color palette and map (generate extra colors when needed)
        const palette = [
            'rgba(54,162,235,0.8)','rgba(255,99,132,0.8)','rgba(75,192,192,0.8)','rgba(255,205,86,0.8)',
            'rgba(153,102,255,0.8)','rgba(201,203,207,0.8)','rgba(255,159,64,0.8)','rgba(99,255,132,0.8)',
            'rgba(132,99,255,0.8)','rgba(255,99,245,0.8)','rgba(255,159,199,0.8)','rgba(99,132,255,0.8)'
        ];
        const colorMap = {};
        let pi = 0;
        const datasets = [];
        Object.keys(groups).sort((a,b)=>parseInt(a)-parseInt(b)).forEach(key => {
            let color;
            if (pi < palette.length) { color = palette[pi]; }
            else {
                // generate a new color using HSL to avoid repeats
                const hue = (pi * 47) % 360;
                color = `hsla(${hue},70%,50%,0.8)`;
            }
            colorMap[key] = color;
            datasets.push({ label: `Label ${key} (n=${groups[key].length})`, data: groups[key], backgroundColor: color, borderColor: color, pointRadius: 4 });
            pi++;
        });

        latentSpaceChart.data.datasets = datasets;
        latentSpaceChart.update();

        // Build a compact HTML summary with colored badges and counts
        const total = data.points.length;
        const parts = Object.keys(groups).sort((a,b)=>parseInt(a)-parseInt(b)).map(k => {
            const c = colorMap[k] || 'rgba(128,128,128,0.6)';
            return `<span style="display:inline-flex;align-items:center;padding:4px 8px;margin-right:8px;border-radius:12px;background:#f6f6f6;border:1px solid #eee;">
                <span style=\"width:12px;height:12px;background:${c};display:inline-block;margin-right:8px;border-radius:2px;\"></span>
                <strong style=\"margin-right:6px;\">Label ${k}</strong>
                <span style=\"font-size:12px;opacity:0.85;\">(n=${groups[k].length})</span>
            </span>`;
        });
        document.getElementById('pointInfo').innerHTML = `<div style="margin-bottom:6px;"><strong>Total:</strong> ${total}</div><div style="display:flex;flex-wrap:wrap;gap:6px;">${parts.join('')}</div>`;

        // attach mousemove listener to show dynamic coordinates and nearest sample
        const canvas = document.getElementById('latentSpaceChart');
        canvas.onmousemove = function(evt) {
            const rect = canvas.getBoundingClientRect();
            const x = evt.clientX - rect.left;
            const y = evt.clientY - rect.top;

            const chartArea = latentSpaceChart.chartArea;
            if (!chartArea) return;

            // convert pixel coordinates to chart (data) coordinates
            const xScale = latentSpaceChart.scales['x'];
            const yScale = latentSpaceChart.scales['y'];
            if (!xScale || !yScale) return;

            const px = xScale.getValueForPixel(x);
            const py = yScale.getValueForPixel(y);

            // find nearest in projected (UMAP) space and show nearest sample index + latent if present
            let nearestLocal = -1;
            let nearestDistSq = Infinity;
            const allPoints = data.points || [];
            for (let i = 0; i < allPoints.length; i++) {
                const dx = allPoints[i].x - px;
                const dy = allPoints[i].y - py;
                const dSq = dx*dx + dy*dy;
                if (dSq < nearestDistSq) {
                    nearestDistSq = dSq;
                    nearestLocal = i;
                }
            }
            let msg = `UMAP: (${px.toFixed(3)}, ${py.toFixed(3)})`;
            if (nearestLocal >= 0) {
                const p = allPoints[nearestLocal];
                msg += ` | Nearest sample: ${p.original_index}`;
                if (p.label !== undefined && p.label !== null) msg += ` | Label: ${p.label}`;
                if (p.latent) {
                    // show first 6 components of latent vector
                    try {
                        const latent = Array.isArray(p.latent) ? p.latent : [];
                        const show = latent.slice(0,6).map(v => Number(v).toFixed(3)).join(', ');
                        msg += ` | Latent[:6]=[${show}${latent.length>6? ', ...':''}]`;
                    } catch(e) {}
                }
            }
            document.getElementById('pointInfo').innerHTML = msg;
        };

        // click handler: find nearest point and request reconstruction
        canvas.onclick = async function(evt) {
            try {
                const rect = canvas.getBoundingClientRect();
                const x = evt.clientX - rect.left;
                const y = evt.clientY - rect.top;


                const xScale = latentSpaceChart.scales['x'];
                const yScale = latentSpaceChart.scales['y'];
                if (!xScale || !yScale) return;

                // convert the click pixel coordinates into chart (data) coordinates
                const px = xScale.getValueForPixel(x);
                const py = yScale.getValueForPixel(y);

                // find nearest in projected (UMAP) space using Euclidean distance
                let nearestLocal = -1;
                let nearestDistSq = Infinity;
                const allPoints = data.points || [];
                for (let i = 0; i < allPoints.length; i++) {
                    const dx = allPoints[i].x - px;
                    const dy = allPoints[i].y - py;
                    const dSq = dx*dx + dy*dy;
                    if (dSq < nearestDistSq) {
                        nearestDistSq = dSq;
                        nearestLocal = i;
                    }
                }
                const nearestIdx = (nearestLocal >= 0) ? allPoints[nearestLocal].original_index : -1;
                const nearestDist = (nearestDistSq === Infinity) ? Infinity : Math.sqrt(nearestDistSq);

                if (nearestIdx < 0) {
                    document.getElementById('reconInfo').textContent = 'No sample found near click';
                    return;
                }

                // fetch reconstruction for the selected sample (backend will prefer saved reconstructions if available)
                const recResp = await fetch(`${API_BASE}/reconstruct?index=${nearestIdx}`);
                if (!recResp.ok) {
                    const err = await recResp.json().catch(() => ({}));
                    document.getElementById('reconInfo').textContent = `Error: ${err.detail || recResp.statusText}`;
                    return;
                }
                const recData = await recResp.json();
                const original = recData.original || [];
                const reconstruction = recData.reconstruction || [];

                // prepare labels
                const labels = original.map((_, i) => i);

                // initialize reconstruction chart if needed
                if (!reconstructionChart) initReconstructionChart();

                reconstructionChart.data.labels = labels;
                reconstructionChart.data.datasets[0].data = original;
                reconstructionChart.data.datasets[1].data = reconstruction;
                reconstructionChart.update();

                // compute RMSE between original and reconstruction
                let rmse = NaN;
                if (Array.isArray(original) && Array.isArray(reconstruction) && original.length === reconstruction.length && original.length > 0) {
                    let s = 0.0;
                    for (let i = 0; i < original.length; i++) {
                        const d = original[i] - reconstruction[i];
                        s += d * d;
                    }
                    rmse = Math.sqrt(s / original.length);
                }
                document.getElementById('reconInfo').textContent = `Index: ${recData.index} | RMSE: ${Number.isFinite(rmse) ? rmse.toFixed(4) : 'N/A'}`;

            } catch (err) {
                console.error('Reconstruction error', err);
                document.getElementById('reconInfo').textContent = 'Error fetching reconstruction';
            }
        };

    } catch (error) {
        console.error('Error:', error);
        alert('Error generating latent space');
    }
};

// Visualization-specific inference status display: reuse training checkInferenceStatus if present,
// then render into #inferenceStatusViz
window.checkInferenceStatusViz = async function() {
    const el = document.getElementById('inferenceStatusViz');
    if (!el) return;
    try {
        // call the shared function to refresh server-side info/UI state (it is debounced)
        if (window.checkInferenceStatus) await window.checkInferenceStatus();
    } catch (e) {
        // ignore
    }
    try {
        // reuse cached current-inference result when available to avoid extra fetches
        const cur = (window._inferenceStatus && window._inferenceStatus.lastCurrent) ? window._inferenceStatus.lastCurrent : null;
        if (cur) {
            if (cur.loaded) {
                const parts = [];
                if (cur.projection2d && cur.projection2d.dataset) parts.push(cur.projection2d.dataset);
                if (cur.projection2d && cur.projection2d.timestamp) parts.push(new Date(cur.projection2d.timestamp*1000).toLocaleString());
                el.innerHTML = `<div class="success">Inference loaded — ${parts.join(' • ')}</div>`;
            } else {
                el.innerHTML = '<div class="warning">Inference not loaded</div>';
            }
        } else {
            // fallback message when no cached info is present (shared check already attempted)
            el.innerHTML = '<div class="warning">Inference not loaded</div>';
        }
    } catch (err) {
        console.error(err);
        el.innerHTML = '<div class="error">Error checking inference</div>';
    }
};

// ensure visualization inference status check on load
document.addEventListener('DOMContentLoaded', function() { if (window.checkInferenceStatusViz) window.checkInferenceStatusViz(); });

})();