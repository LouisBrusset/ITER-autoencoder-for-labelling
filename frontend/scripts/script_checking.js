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

        // Determine selected subset
        let subset = 'all';
        try {
            const radios = document.getElementsByName('checkSubset');
            for (let r of radios) {
                if (r.checked) {
                    subset = r.value;
                    break;
                }
            }
        } catch (e) {
            // ignore
        }

        let idx = 0;
        if (subset === 'train' && info.is_train_present) {
            const nTrain = info.n_train || 0;
            if (nTrain <= 0) {
                alert('No training samples available');
                return;
            }
            idx = Math.floor(Math.random() * nTrain);
        } else {
            // global index across dataset
            const max = info.shape[0] - 1;
            idx = Math.floor(Math.random() * (max + 1));
        }

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

        // determine subset selection from radio buttons (train/all)
        let subsetParam = '';
        try {
            const radios = document.getElementsByName('checkSubset');
            for (let r of radios) {
                if (r.checked) {
                    if (r.value === 'train') subsetParam = '&subset=train';
                    break;
                }
            }
        } catch (e) {
            // ignore if radios not present
        }

        const resp = await fetch(`${API_BASE}/check-sample-plot?index=${idx}${subsetParam}`);
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