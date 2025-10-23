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

        // store latent vectors for hover interactions
        window._latent_vectors = data.latent_vectors || [];

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

            // find nearest latent vector in original latent space (not UMAP)
            let nearestIdx = -1;
            let nearestDist = Infinity;
            if (window._latent_vectors && window._latent_vectors.length) {
                // we need to find the inverse mapping from UMAP to latent - not trivial.
                // instead, we find nearest in UMAP projection space using original projected points (data.points)
                const allPoints = data.points;
                for (let i = 0; i < allPoints.length; i++) {
                    const dx = allPoints[i].x - px;
                    const dy = allPoints[i].y - py;
                    const d = dx*dx + dy*dy;
                    if (d < nearestDist) {
                        nearestDist = d;
                        nearestIdx = allPoints[i].original_index;
                    }
                }
            }

            const latentVec = (window._latent_vectors && nearestIdx >= 0) ? window._latent_vectors[nearestIdx] : null;

            document.getElementById('pointInfo').innerHTML = 
                `UMAP: (${px.toFixed(3)}, ${py.toFixed(3)})` +
                (latentVec ? ` | Latent: [${latentVec.map(v => v.toFixed(4)).slice(0,8).join(', ')}${latentVec.length>8? ', ...':''}]` : '') +
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

                const px = xScale.getPixelForValue(x);
                const py = yScale.getPixelForValue(y);

                // find nearest in projected space
                let nearestIdx = -1;
                let nearestDist = Infinity;
                const allPoints = data.points || [];
                for (let i = 0; i < allPoints.length; i++) {
                    const dx = allPoints[i].x - px;
                    const dy = allPoints[i].y - py;
                    const d = dx*dx + dy*dy;
                    if (d < nearestDist) {
                        nearestDist = d;
                        nearestIdx = allPoints[i].original_index;
                    }
                }

                if (nearestIdx < 0) {
                    document.getElementById('reconInfo').textContent = 'No sample found near click';
                    return;
                }

                // fetch reconstruction
                const resp = await fetch(`${API_BASE}/reconstruct?index=${nearestIdx}`);
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({}));
                    document.getElementById('reconInfo').textContent = `Error: ${err.detail || resp.statusText}`;
                    return;
                }
                const recData = await resp.json();

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

                document.getElementById('reconInfo').textContent = `Index: ${recData.index} | Distance (proj): ${nearestDist.toFixed(4)}`;

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