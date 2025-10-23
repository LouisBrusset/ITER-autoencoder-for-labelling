(function(){
const API_BASE = window.API_BASE || "http://localhost:8000";

window.showRandomSample = async function() {
    try {
        const resp = await fetch(`${API_BASE}/current-dataset`);
        const info = await resp.json();
        if (!info.loaded) {
            alert('No dataset loaded');
            return;
        }
        const max = info.shape[0] - 1;
        const idx = Math.floor(Math.random() * (max + 1));
        document.getElementById('checkSampleIndex').value = idx;
        await showSamplePlot(idx);
    } catch (err) {
        console.error('Error selecting random sample', err);
    }
};

window.showSamplePlot = async function(index) {
    try {
        const idx = parseInt(index);
        if (isNaN(idx)) {
            alert('Invalid index');
            return;
        }

        const resp = await fetch(`${API_BASE}/check-sample-plot?index=${idx}`);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            alert(`Error: ${err.detail || resp.statusText}`);
            return;
        }

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const img = document.getElementById('checkPlot');
        img.src = url;
        img.onload = () => { URL.revokeObjectURL(url); };
    } catch (err) {
        console.error('Error fetching sample plot', err);
        alert('Error fetching sample plot');
    }
};

})();