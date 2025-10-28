(function(){
const API_BASE = window.API_BASE || "http://localhost:8000";
window.trainingInterval = null;
window.trainingChart = null;

window.initTrainingChart = function() {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    window.trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
            {
                label: 'Training Loss',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                borderWidth: 2,
                pointRadius: 3,
                fill: true
            },
            {
                label: 'Validation Loss',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.08)',
                borderWidth: 2,
                pointRadius: 3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { x: { title: { display: true, text: 'Epoch' } }, y: { title: { display: true, text: 'Loss' }, beginAtZero: true } }
        }
    });
};

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
    
    try {
        const response = await fetch(
            `${API_BASE}/start-training?epochs=${epochs}&learning_rate=${learningRate}&encoding_dim=${encodingDim}`,
            { method: 'POST' }
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
            trainingChart.data.labels = data.epochs_data;
            // training dataset
            if (data.train_loss_history) trainingChart.data.datasets[0].data = data.train_loss_history;
            // validation dataset
            if (data.val_loss_history) trainingChart.data.datasets[1].data = data.val_loss_history;
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
    const statusEl = document.getElementById('inferenceStatus');
    if (!statusEl) return;
    try {
        const resp = await fetch(`${API_BASE}/inference-options`);
        if (!resp.ok) {
            statusEl.innerHTML = '<div class="warning">Unable to check inference</div>';
            return;
        }
        const data = await resp.json();
        const nLat = (data.latents && data.latents.length) ? data.latents.length : 0;
        const nProj = (data.projections && data.projections.length) ? data.projections.length : 0;
        const nRec = (data.reconstructions && data.reconstructions.length) ? data.reconstructions.length : 0;
        if (nLat > 0 && nProj > 0 && nRec > 0) {
            statusEl.innerHTML = `<div class="success">Inference available — latents: ${nLat}, projections: ${nProj}, reconstructions: ${nRec}</div>`;
        } else if (nLat+ nProj + nRec > 0) {
            statusEl.innerHTML = `<div class="warning">Partial inference artifacts present — latents: ${nLat}, projections: ${nProj}, reconstructions: ${nRec}</div>`;
        } else {
            statusEl.innerHTML = '<div class="warning">Inference not run</div>';
        }
    } catch (err) {
        console.error(err);
        statusEl.innerHTML = '<div class="error">Error checking inference</div>';
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
    } catch (err) {
        console.error(err);
        statusEl.innerHTML = `<div class="error">Error running inference</div>`;
    } finally {
        if (btn) btn.disabled = false;
    }
};