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
        const response = await fetch(`${API_BASE}/latent-space`);
        const data = await response.json();
        
        const normalPoints = data.points.filter(p => p.label === 0);
        const anomalyPoints = data.points.filter(p => p.label === 1);
        
        latentSpaceChart.data.datasets[0].data = normalPoints;
        latentSpaceChart.data.datasets[1].data = anomalyPoints;
        latentSpaceChart.update();

        // we no longer store full latent vectors client-side; show summary info
        document.getElementById('pointInfo').innerHTML =
            `${data.points.length} points projected (${anomalyPoints.length} anomalies detected)`;

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

            // find nearest in projected (UMAP) space and show nearest sample index
            let nearestIdx = -1;
            let nearestDistSq = Infinity;
            const allPoints = data.points || [];
            for (let i = 0; i < allPoints.length; i++) {
                const dx = allPoints[i].x - px;
                const dy = allPoints[i].y - py;
                const dSq = dx*dx + dy*dy;
                if (dSq < nearestDistSq) {
                    nearestDistSq = dSq;
                    nearestIdx = allPoints[i].original_index;
                }
            }

            document.getElementById('pointInfo').innerHTML = 
                `UMAP: (${px.toFixed(3)}, ${py.toFixed(3)})` +
                (nearestIdx>=0 ? ` | Nearest sample: ${nearestIdx}` : '');
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
                let nearestIdx = -1;
                let nearestDistSq = Infinity;
                const allPoints = data.points || [];
                for (let i = 0; i < allPoints.length; i++) {
                    const dx = allPoints[i].x - px;
                    const dy = allPoints[i].y - py;
                    const dSq = dx*dx + dy*dy;
                    if (dSq < nearestDistSq) {
                        nearestDistSq = dSq;
                        nearestIdx = allPoints[i].original_index;
                    }
                }
                const nearestDist = (nearestDistSq === Infinity) ? Infinity : Math.sqrt(nearestDistSq);

                if (nearestIdx < 0) {
                    document.getElementById('reconInfo').textContent = 'No sample found near click';
                    return;
                }

                // fetch reconstruction for the selected sample
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

})();