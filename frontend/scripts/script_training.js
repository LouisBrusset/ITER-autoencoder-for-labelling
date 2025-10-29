(function(){
const API_BASE = window.API_BASE || "http://localhost:8000";
window.trainingInterval = null;
window.trainingChart = null;

window.initTrainingChart = function() {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    window.trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
            {
                label: 'Training Loss',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                borderWidth: 2,
                pointRadius: 3,
                fill: true,
                parsing: false
            },
            {
                label: 'Validation Loss',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.08)',
                borderWidth: 2,
                pointRadius: 3,
                fill: true,
                parsing: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { x: { type: 'linear', title: { display: true, text: 'Epoch' }, ticks: { precision:0 } }, y: { title: { display: true, text: 'Loss' }, beginAtZero: true } }
        }
    });
};

// Dynamically render encoder/decoder layer size inputs based on counts
window.renderLayerInputs = function() {
    const encCount = parseInt(document.getElementById('encoderLayersCount')?.value || 0, 10);
    const decCount = parseInt(document.getElementById('decoderLayersCount')?.value || 0, 10);

    const encContainer = document.getElementById('encoderLayersContainer');
    const decContainer = document.getElementById('decoderLayersContainer');
    if (encContainer) {
        let html = '';
        for (let i=0;i<encCount;i++) {
            const id = `encoder_layer_${i}`;
            html += `<div class="small">Layer ${i+1} neurons: <input type="number" id="${id}" value="${Math.max(8, Math.floor(32/Math.pow(2,i)))}" min="1" step="1"></div>`;
        }
        encContainer.innerHTML = html;
    }
    if (decContainer) {
        let html = '';
        for (let i=0;i<decCount;i++) {
            const id = `decoder_layer_${i}`;
            html += `<div class="small">Layer ${i+1} neurons: <input type="number" id="${id}" value="${Math.max(8, Math.floor(32/Math.pow(2,i)))}" min="1" step="1"></div>`;
        }
        decContainer.innerHTML = html;
    }
};

// Attach listeners once DOM is ready
window.addEventListener('load', function(){
    try {
        const encEl = document.getElementById('encoderLayersCount');
        const decEl = document.getElementById('decoderLayersCount');
        if (encEl) encEl.addEventListener('change', window.renderLayerInputs);
        if (decEl) decEl.addEventListener('change', window.renderLayerInputs);
        // initial render
        window.renderLayerInputs();
    } catch (e) {
        // ignore
    }
});

window.loadModelOptions = async function() {
    try {
        const response = await fetch(`${API_BASE}/model-options`);
        const data = await response.json();
        
        let html = '';
        data.saved_models.forEach(model => {
            html += `<div class="file-item">
                <input type="radio" name="modelFile" value="${model}">
                ${model}
                <button onclick="deleteModel('${model}')" class="btn-small">Delete</button>
            </div>`;
        });
        
        document.getElementById('modelOptions').innerHTML = html || 'No saved models available.';

    } catch (error) {
        console.error('Error:', error);
    }
};


window.deleteModel = async function(modelFilename) {
    if (!confirm(`Are you sure you want to delete model ${modelFilename}?`)) {
        return;
    }
    try {
        const response = await fetch(
            `${API_BASE}/delete-model?model_filename=${modelFilename}`,
            { method: 'DELETE' }
        );
        const result = await response.json();
        alert(`Model deleted: ${result.model_filename || modelFilename}`);
        loadModelOptions();
        checkCurrentModel();
    } catch (error) {
        console.error('Error:', error);
        alert('Error deleting model');
    }
};

window.checkCurrentModel = async function() {
    try {
        const res = await fetch(`${API_BASE}/current-model`);
        const info = await res.json();
        
        const currentEl = document.getElementById('currentModelStatus');
        const vizEl = document.getElementById('visualizationModelInfo');
        const classEl = document.getElementById('classificationModelInfo');
        if (info.error) {
            if (currentEl) currentEl.innerHTML = `<div class="error">Error loading model: ${info.error}</div>`;
            if (vizEl) vizEl.innerHTML = `<div class="error">Error loading model: ${info.error}</div>`;
            if (classEl) classEl.innerHTML = `<div class="error">Error loading model: ${info.error}</div>`;
        } else if (info.loaded) {
            if (currentEl) currentEl.innerHTML = `<div class="success">Model loaded${info.encoding_dim? (': encoding_dim ' + info.encoding_dim) : ''}${info.input_dim? (', input_dim ' + info.input_dim) : ''}</div>`;
            if (vizEl) vizEl.innerHTML = `<div class="success">Ready for visualization${info.encoding_dim? (': encoding_dim ' + info.encoding_dim) : ''}</div>`;
            if (classEl) classEl.innerHTML = `<div class="success">Ready for classification${info.encoding_dim? (': encoding_dim ' + info.encoding_dim) : ''}</div>`;
        } else {
            if (currentEl) currentEl.innerHTML = '<div class="warning">No model loaded</div>';
            if (vizEl) vizEl.innerHTML = '<div class="warning">No model loaded for visualization</div>';
            if (classEl) classEl.innerHTML = '<div class="warning">No model loaded for classification</div>';
        }
        // update inference status for the current dataset/model
        try {
            if (window.checkInferenceStatus) window.checkInferenceStatus();
        } catch (e) {
            // ignore
        }
    } catch (error) {
        console.error('Error:', error);
        const currentEl = document.getElementById('currentModelStatus');
        const vizEl = document.getElementById('visualizationModelInfo');
        const classEl = document.getElementById('classificationModelInfo');
        if (currentEl) currentEl.innerHTML = '<div class="error">Error checking model</div>';
        if (vizEl) vizEl.innerHTML = '<div class="error">Error checking model</div>';
        if (classEl) classEl.innerHTML = '<div class="error">Error checking model</div>';
    }
};


window.startTraining = async function() {
    const epochs = document.getElementById('trainingEpochs').value;
    const learningRate = document.getElementById('learningRate').value;
    const encodingDim = document.getElementById('encodingDim').value;
    // gather encoder/decoder layer sizes from dynamic inputs
    const encCount = parseInt(document.getElementById('encoderLayersCount')?.value || 0, 10);
    const decCount = parseInt(document.getElementById('decoderLayersCount')?.value || 0, 10);
    const encoder_layer_sizes = [];
    const decoder_layer_sizes = [];
    for (let i=0;i<encCount;i++) {
        const el = document.getElementById(`encoder_layer_${i}`);
        if (el) encoder_layer_sizes.push(parseInt(el.value));
    }
    for (let i=0;i<decCount;i++) {
        const el = document.getElementById(`decoder_layer_${i}`);
        if (el) decoder_layer_sizes.push(parseInt(el.value));
    }

    // convolutional options (encoder only; decoder will mirror encoder config)
    const convLayers = parseInt(document.getElementById('convLayersCount')?.value || 0, 10);
    const convFilterSize = parseInt(document.getElementById('convFilterSize')?.value || 3, 10);
    const betaValue = parseFloat(document.getElementById('betaValue')?.value || 0.01);

    const payload = {
        epochs: parseInt(epochs,10),
        learning_rate: parseFloat(learningRate),
        encoding_dim: parseInt(encodingDim,10),
        encoder_layer_sizes: encoder_layer_sizes,
        decoder_layer_sizes: decoder_layer_sizes,
        conv_layers: convLayers,
        conv_filter_size: convFilterSize,
        beta: betaValue
    };

    try {
        const response = await fetch(
            `${API_BASE}/start-training`,
            { method: 'POST', headers: { 'Content-Type':'application/json' }, body: JSON.stringify(payload) }
        );
        const result = await response.json();

        document.getElementById('statusText').textContent = 'Training in progress...';
        document.getElementById('totalEpochs').textContent = result.total_epochs;
        
        startTrainingMonitoring();
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error starting training');
    }
};

window.startTrainingMonitoring = function() {
    window.trainingInterval = setInterval(window.updateTrainingStatus, 500);
};

window.updateTrainingStatus = async function() {
    try {
        const response = await fetch(`${API_BASE}/training-status`);
        const data = await response.json();
        
        document.getElementById('epochText').textContent = data.current_epoch;
        // display current train/val loss if available
        const trainLoss = data.current_train_loss !== undefined ? data.current_train_loss : null;
        const valLoss = data.current_val_loss !== undefined ? data.current_val_loss : null;
        document.getElementById('lossText').textContent = (trainLoss !== null ? trainLoss.toFixed(4) : 'N/A') + (valLoss !== null ? ` / ${valLoss.toFixed(4)}` : '');

        // Update chart: use epochs_data and train_loss_history / val_loss_history
        if (data.epochs_data) {
            // Build xy points to ensure epochs are used as numeric x values and ordered correctly
            const epochs = data.epochs_data || [];
            const trainLoss = data.train_loss_history || [];
            const valLoss = data.val_loss_history || [];
            const trainPoints = [];
            const valPoints = [];
            for (let i=0;i<epochs.length;i++) {
                const e = Number(epochs[i]);
                if (!Number.isFinite(e)) continue;
                if (i < trainLoss.length) trainPoints.push({x: e, y: Number(trainLoss[i])});
                if (i < valLoss.length) valPoints.push({x: e, y: Number(valLoss[i])});
            }
            trainingChart.data.datasets[0].data = trainPoints;
            trainingChart.data.datasets[1].data = valPoints;
            trainingChart.update();
        }
        
        if (!data.is_training && data.current_epoch > 0) {
            document.getElementById('statusText').textContent = 'Training complete';
            clearInterval(trainingInterval);
            loadModelOptions(); // Reload model list
        }
        
    } catch (error) {
        console.error('Error:', error);
    }
};

window.resetTraining = async function() {
    try {
        await fetch(`${API_BASE}/reset-training`, { method: 'POST' });
        document.getElementById('statusText').textContent = 'Waiting...';
        document.getElementById('epochText').textContent = '0';
        document.getElementById('lossText').textContent = '0.0000';
        trainingChart.data.labels = [];
        if (trainingChart.data.datasets && trainingChart.data.datasets[0]) trainingChart.data.datasets[0].data = [];
        if (trainingChart.data.datasets && trainingChart.data.datasets[1]) trainingChart.data.datasets[1].data = [];
        trainingChart.update();
        clearInterval(trainingInterval);
    } catch (error) {
        console.error('Error:', error);
    }
};

window.loadSelectedModel = async function() {
    const selected = document.querySelector('input[name="modelFile"]:checked');
    if (!selected) {
        alert('Please select a model');
        return;
    }
    
    try {
        const response = await fetch(
            `${API_BASE}/load-model?model_filename=${selected.value}`,
            { method: 'POST' }
        );
        const result = await response.json();
        alert(`Model loaded: ${result.model_loaded}`);
        // update UI state
        checkCurrentModel();
        loadModelOptions();
    } catch (error) {
        console.error('Error:', error);
        alert('Error loading model');
    }
};

})();


window.checkInferenceStatus = async function() {
    const trainingEl = document.getElementById('inferenceStatus');
    const actionEl = document.getElementById('inferenceActionStatus');
    const vizEl = document.getElementById('inferenceStatusViz');
    const classEl = document.getElementById('classificationInferenceStatus');

    // simple debounce / coalescing: avoid firing multiple fetches in parallel or too frequently
    if (!window._inferenceStatus) window._inferenceStatus = { inFlight: false, lastTs: 0, ttl: 1500, lastOptions: null, lastCurrent: null };
    const now = Date.now();
    if (window._inferenceStatus.inFlight) {
        // another caller is already refreshing; reuse last known values if available
        if (window._inferenceStatus.lastOptions) {
            const data = window._inferenceStatus.lastOptions;
            const nLat = (data.latents && data.latents.length) ? data.latents.length : 0;
            const nProj = (data.projections && data.projections.length) ? data.projections.length : 0;
            const nRec = (data.reconstructions && data.reconstructions.length) ? data.reconstructions.length : 0;
            if (nLat + nProj + nRec > 0) {
                if (trainingEl) trainingEl.innerHTML = `<div class="success">Inference run — latents: ${nLat}, projections: ${nProj}, reconstructions: ${nRec}</div>`;
            } else {
                if (trainingEl) trainingEl.innerHTML = '<div class="warning">Inference not run</div>';
            }
        }
        return;
    }
    if (now - window._inferenceStatus.lastTs < window._inferenceStatus.ttl) {
        // recent check already done; populate UI from cache and skip network
        const data = window._inferenceStatus.lastOptions;
        if (data && trainingEl) {
            const nLat = (data.latents && data.latents.length) ? data.latents.length : 0;
            const nProj = (data.projections && data.projections.length) ? data.projections.length : 0;
            const nRec = (data.reconstructions && data.reconstructions.length) ? data.reconstructions.length : 0;
            if (nLat + nProj + nRec > 0) trainingEl.innerHTML = `<div class="success">Inference run — latents: ${nLat}, projections: ${nProj}, reconstructions: ${nRec}</div>`;
            else trainingEl.innerHTML = '<div class="warning">Inference not run</div>';
        }
        // Also apply last current into other UI pieces
        const cur = window._inferenceStatus.lastCurrent;
        if (cur) {
            const s = cur.loaded ? `<div class="success">Inference loaded — ${cur.projection2d?.dataset || cur.latents?.dataset || cur.reconstructions?.dataset || ''} • ts ${cur.projection2d?.timestamp || cur.latents?.timestamp || cur.reconstructions?.timestamp || ''}</div>` : '<div class="warning">No inference loaded</div>';
            if (actionEl) actionEl.innerHTML = s;
            if (vizEl) vizEl.innerHTML = s;
            if (classEl) classEl.innerHTML = s;
        }
        return;
    }
    window._inferenceStatus.inFlight = true;

    // 1) Check whether an inference has been run (saved artifacts exist)
    try {
        const resp = await fetch(`${API_BASE}/inference-options`);
        if (!resp.ok) {
            if (trainingEl) trainingEl.innerHTML = '<div class="error">Error checking inference</div>';
        } else {
            const data = await resp.json();
            // cache options
            window._inferenceStatus.lastOptions = data;
            window._inferenceStatus.lastTs = Date.now();
            const nLat = (data.latents && data.latents.length) ? data.latents.length : 0;
            const nProj = (data.projections && data.projections.length) ? data.projections.length : 0;
            const nRec = (data.reconstructions && data.reconstructions.length) ? data.reconstructions.length : 0;
            if (nLat + nProj + nRec > 0) {
                if (trainingEl) trainingEl.innerHTML = `<div class="success">Inference run — latents: ${nLat}, projections: ${nProj}, reconstructions: ${nRec}</div>`;
            } else {
                if (trainingEl) trainingEl.innerHTML = '<div class="warning">Inference not run</div>';
            }
        }
    } catch (err) {
        console.error('Error fetching inference-options', err);
        if (trainingEl) trainingEl.innerHTML = '<div class="error">No response checking inference</div>';
    }

    // 2) Check which inference (if any) is currently loaded and update Available/Visualization/Classification
    try {
        const curResp = await fetch(`${API_BASE}/current-inference`);
        if (!curResp.ok) {
            const msg = '<div class="error">Unable to check current inference</div>';
            if (actionEl) actionEl.innerHTML = msg;
            if (vizEl) vizEl.innerHTML = msg;
            if (classEl) classEl.innerHTML = msg;
            window._inferenceStatus.lastCurrent = null;
            return;
        }
        const cur = await curResp.json();
        // cache current
        window._inferenceStatus.lastCurrent = cur;
        if (cur.loaded) {
            const meta = cur.projection2d || cur.latents || cur.reconstructions || null;
            const parts = [];
            if (meta && meta.dataset) parts.push(`dataset ${meta.dataset}`);
            if (meta && meta.timestamp) parts.push(`ts ${new Date(meta.timestamp*1000).toLocaleString()}`);
            const s = `<div class="success">Inference loaded — ${parts.join(' • ')}</div>`;
            if (actionEl) actionEl.innerHTML = s;
            if (vizEl) vizEl.innerHTML = s;
            if (classEl) classEl.innerHTML = s;
        } else {
            const w = '<div class="warning">No inference loaded</div>';
            if (actionEl) actionEl.innerHTML = w;
            if (vizEl) vizEl.innerHTML = w;
            if (classEl) classEl.innerHTML = w;
        }
    } catch (err) {
        console.error('Error fetching current-inference', err);
        const msg = '<div class="error">Error checking current inference</div>';
        const actionEl2 = document.getElementById('inferenceActionStatus');
        if (actionEl2) actionEl2.innerHTML = msg;
        if (vizEl) vizEl.innerHTML = msg;
        if (classEl) classEl.innerHTML = msg;
        window._inferenceStatus.lastCurrent = null;
    } finally {
        window._inferenceStatus.inFlight = false;
    }
};


window.runInference = async function() {
    const btn = document.getElementById('runInferenceBtn');
    const statusEl = document.getElementById('inferenceStatus');
    if (!btn || !statusEl) return;
    try {
        btn.disabled = true;
        const det = document.getElementById('inferenceDeterministic')?.checked ? 'true' : 'false';
        statusEl.innerHTML = '<div class="status">Running inference...</div>';
        const resp = await fetch(`${API_BASE}/run-inference?deterministic=${det}`, { method: 'POST' });
        if (!resp.ok) {
            const err = await resp.json().catch(()=>({}));
            statusEl.innerHTML = `<div class="error">Inference failed: ${err.detail || resp.statusText}</div>`;
            btn.disabled = false;
            return;
        }
        const payload = await resp.json();
        statusEl.innerHTML = `<div class="success">Inference completed. Files saved.</div>`;
        // refresh status to show counts
    await window.checkInferenceStatus();
    // update the available inference list now that new artifacts exist
    if (window.loadInferenceOptions) await window.loadInferenceOptions();
    } catch (err) {
        console.error(err);
        statusEl.innerHTML = `<div class="error">Error running inference</div>`;
    } finally {
        if (btn) btn.disabled = false;
    }
};


window.refreshInferenceOptions = async function() {
    // alias
    if (window.loadInferenceOptions) return loadInferenceOptions();
};

window.loadInferenceOptions = async function() {
    try {
        const resp = await fetch(`${API_BASE}/inference-options`);
        if (!resp.ok) {
            document.getElementById('inferenceOptions').innerHTML = '<div class="warning">Unable to load inference list</div>';
            return;
        }
        const data = await resp.json();

        // Group artifacts by (timestamp + dataset) to form triplets
        const groups = {}; // key -> { timestamp, dataset, latents, projection, reconstructions }

        function addToGroup(item, type) {
            if (!item || !item.timestamp) return;
            const key = `${item.timestamp}__${item.dataset || ''}`;
            if (!groups[key]) groups[key] = { timestamp: item.timestamp, dataset: item.dataset || '', latents: null, projection: null, reconstructions: null };
            if (type === 'latents') groups[key].latents = item.file;
            if (type === 'projection') groups[key].projection = item.file;
            if (type === 'reconstructions') groups[key].reconstructions = item.file;
        }

        (data.latents || []).forEach(i => addToGroup(i, 'latents'));
        (data.projections || []).forEach(i => addToGroup(i, 'projection'));
        (data.reconstructions || []).forEach(i => addToGroup(i, 'reconstructions'));

        // Build html: one radio per triplet
        let html = '';
        if (Object.keys(groups).length === 0) {
            html = '<div class="small muted">No inference artifacts available</div>';
        } else {
            html = '<div class="inference-triplets">';
            // sort by timestamp desc
            const sorted = Object.values(groups).sort((a,b) => b.timestamp - a.timestamp);
            sorted.forEach(g => {
                const label = `${g.timestamp} — ${g.dataset || 'unknown dataset'}`;
                const payload = encodeURIComponent(JSON.stringify({ latents: g.latents, projection: g.projection, reconstructions: g.reconstructions, dataset: g.dataset, timestamp: g.timestamp }));
                html += `<div class="file-item">
                    <label style="display:flex; gap:8px; align-items:center;">
                        <input type="radio" name="inferenceTriplet" value="${payload}">
                        <strong>${label}</strong>
                    </label>
                    <div class="small muted">`;
                html += `Latents: ${g.latents ? g.latents.split('/').pop() : '<em>missing</em>'} • `;
                html += `Projection2D: ${g.projection ? g.projection.split('/').pop() : '<em>missing</em>'} • `;
                html += `Reconstructions: ${g.reconstructions ? g.reconstructions.split('/').pop() : '<em>missing</em>'}`;
                html += `</div>`;
                html += `<div style="margin-top:6px; display:flex; gap:8px;">
                    ${g.latents ? `<button onclick="deleteInference('${g.latents.replace(/"/g,'&quot;')}')" class="btn-small">Delete latents</button>` : ''}
                    ${g.projection ? `<button onclick="deleteInference('${g.projection.replace(/"/g,'&quot;')}')" class="btn-small">Delete projection</button>` : ''}
                    ${g.reconstructions ? `<button onclick="deleteInference('${g.reconstructions.replace(/"/g,'&quot;')}')" class="btn-small">Delete reconstructions</button>` : ''}
                </div>`;
                html += `</div>`;
            });
            html += '</div>';
        }

        document.getElementById('inferenceOptions').innerHTML = html;
        if (window.checkInferenceStatus) window.checkInferenceStatus();
    } catch (err) {
        console.error(err);
        document.getElementById('inferenceOptions').innerHTML = '<div class="error">Error loading inference options</div>';
    }
};

window.loadSelectedInference = async function() {
    const sel = document.querySelector('input[name="inferenceTriplet"]:checked');
    if (!sel) {
        alert('Please select an inference triplet to load');
        return;
    }
    try {
        const obj = JSON.parse(decodeURIComponent(sel.value));
        const params = new URLSearchParams();
        if (obj.latents) params.append('latents_file', obj.latents);
        if (obj.projection) params.append('projection_file', obj.projection);
        if (obj.reconstructions) params.append('reconstructions_file', obj.reconstructions);

        const resp = await fetch(`${API_BASE}/load-inference?${params.toString()}`, { method: 'POST' });
        if (!resp.ok) {
            const err = await resp.json().catch(()=>({}));
            alert('Failed to load inference: ' + (err.detail || resp.statusText));
            return;
        }
        document.getElementById('inferenceActionStatus').innerHTML = `<div class="success">Inference loaded</div>`;
        if (window.checkInferenceStatus) window.checkInferenceStatus();
        if (window.loadInferenceOptions) window.loadInferenceOptions();
        // notify visualization/classification that new current_* files exist
        if (window.generateLatentSpace) {
            try { /* no-op: modules should check for current files when run */ } catch(e){}
        }
    } catch (err) {
        console.error(err);
        alert('Error loading inference');
    }
};

window.deleteSelectedInference = async function() {
    const sel = document.querySelector('input[name="inferenceTriplet"]:checked');
    if (!sel) {
        alert('Please select an inference triplet to delete');
        return;
    }
    if (!confirm('Delete the selected inference triplet (this will remove all three files)?')) return;
    try {
        const obj = JSON.parse(decodeURIComponent(sel.value));
        const files = [];
        if (obj.latents) files.push(obj.latents);
        if (obj.projection) files.push(obj.projection);
        if (obj.reconstructions) files.push(obj.reconstructions);

        for (const f of files) {
            const url = `${API_BASE}/delete-inference?file_path=${encodeURIComponent(f)}`;
            const resp = await fetch(url, { method: 'DELETE' });
            if (!resp.ok) {
                const err = await resp.json().catch(()=>({}));
                alert('Failed to delete ' + (f.split('/').pop()) + ': ' + (err.detail || resp.statusText));
                // continue trying to delete others
            }
        }
        alert('Selected inference triplet deleted (where possible)');
        if (window.loadInferenceOptions) window.loadInferenceOptions();
        if (window.checkInferenceStatus) window.checkInferenceStatus();
    } catch (err) {
        console.error(err);
        alert('Error deleting inference triplet');
    }
};

window.deleteInference = async function(filePath) {
    if (!confirm(`Are you sure you want to delete ${filePath.split('/').pop()}?`)) return;
    try {
        const url = `${API_BASE}/delete-inference?file_path=${encodeURIComponent(filePath)}`;
        const resp = await fetch(url, { method: 'DELETE' });
        if (!resp.ok) {
            const err = await resp.json().catch(()=>({}));
            alert('Failed to delete inference: ' + (err.detail || resp.statusText));
            return;
        }
        const r = await resp.json();
        alert(`Deleted: ${r.file || filePath}`);
        if (window.loadInferenceOptions) window.loadInferenceOptions();
        if (window.checkInferenceStatus) window.checkInferenceStatus();
    } catch (err) {
        console.error(err);
        alert('Error deleting inference');
    }
};