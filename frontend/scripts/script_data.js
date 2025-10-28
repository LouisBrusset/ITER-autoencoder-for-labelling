(function(){
const API_BASE = window.API_BASE || "http://localhost:8000";

window.loadDataOptions = async function() {
    try {
        const response = await fetch(`${API_BASE}/data-options`);
        const data = await response.json();
        
        let html = '<h4>Synthetic files:</h4>';
        html += '<ul class="round-list">';
        data.synthetic_files.forEach(file => {
            html += `<li class="file-item">` +
                `<span class="file-name">${file}</span>` +
                `<button onclick="loadDataset('${file}', 'synthetic')" class="btn-small">Load</button>` +
                `<button onclick="deleteDataset('${file}', 'synthetic')" class="btn-small">Delete</button>` +
            `</li>`;
        });
        html += '</ul>';

        html += '<h4>Uploaded files:</h4>';
        html += '<ul class="round-list">';
        data.uploaded_files.forEach(file => {
            html += `<li class="file-item">` +
                `<span class="file-name">${file}</span>` +
                `<button onclick="loadDataset('${file}', 'uploaded')" class="btn-small">Load</button>` +
                `<button onclick="deleteDataset('${file}', 'uploaded')" class="btn-small">Delete</button>` +
            `</li>`;
        });
        html += '</ul>';
        
        document.getElementById('dataOptions').innerHTML = html;
    } catch (error) {
        console.error('Error:', error);
    }
};

window.loadDataset = async function(filename, dataType) {
    try {
        const response = await fetch(
            `${API_BASE}/load-dataset?filename=${filename}&data_type=${dataType}`,
            { method: 'POST' }
        );
        const result = await response.json();
        checkCurrentDataset();
        alert(`Dataset loaded: ${result.filename}`);
    } catch (error) {
        console.error('Error:', error);
        alert('Error loading dataset');
    }
};

window.deleteDataset = async function(filename, dataType) {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) {
        return;
    }
    try {
        const response = await fetch(
            `${API_BASE}/delete-dataset?filename=${filename}&data_type=${dataType}`,
            { method: 'DELETE' }
        );
        const result = await response.json();
        alert(`Dataset deleted: ${result.filename}`);
        loadDataOptions();
        checkCurrentDataset();
    } catch (error) {
        console.error('Error:', error);
        alert('Error deleting dataset');
    }
};

window.checkCurrentDataset = async function() {
    try {
        const response = await fetch(`${API_BASE}/current-dataset`);
        const data = await response.json();
        
        const currentEl = document.getElementById('currentDataset');
        const checkEl = document.getElementById('checkingDatasetStatus');
        const trainEl = document.getElementById('trainingDatasetInfo');
        const vizEl = document.getElementById('visualizationDatasetInfo');
        const classEl = document.getElementById('classificationDatasetInfo');
        if (data.loaded) {
            if (currentEl) currentEl.innerHTML = `<div class="success">Dataset loaded: ${data.filename} (${data.shape[0]} samples, ${data.shape[1]} features)</div>`;
            if (checkEl) checkEl.innerHTML = `<div class="success">Dataset loaded: ${data.filename} (${data.shape[0]} samples, ${data.shape[1]} features)</div>`;
            if (trainEl) trainEl.innerHTML = `<div class="success">Dataset loaded: ${data.filename} (${data.shape[0]} samples, ${data.shape[1]} features)</div>`;
            if (vizEl) vizEl.innerHTML = `<div class="success">Dataset loaded: ${data.filename} (${data.shape[0]} samples, ${data.shape[1]} features)</div>`;
            if (classEl) classEl.innerHTML = `<div class="success">Dataset loaded: ${data.filename} (${data.shape[0]} samples, ${data.shape[1]} features)</div>`;
        } else if (data.error) {
            if (currentEl) currentEl.innerHTML = `<div class="error">Error loading dataset: ${data.error}</div>`;
            if (checkEl) checkEl.innerHTML = `<div class="error">Error loading dataset: ${data.error}</div>`;
            if (trainEl) trainEl.innerHTML = `<div class="error">Error loading dataset: ${data.error}</div>`;
            if (vizEl) vizEl.innerHTML = `<div class="error">Error loading dataset: ${data.error}</div>`;
            if (classEl) classEl.innerHTML = `<div class="error">Error loading dataset: ${data.error}</div>`;
        } else {
            if (currentEl) currentEl.innerHTML = `<div class="warning">No dataset loaded</div>`;
            if (checkEl) checkEl.innerHTML = `<div class="warning">No dataset loaded for checking</div>`;
            if (trainEl) trainEl.innerHTML = `<div class="warning">No dataset loaded for training</div>`;
            if (vizEl) vizEl.innerHTML = `<div class="warning">No dataset loaded for visualization</div>`;
            if (classEl) classEl.innerHTML = `<div class="warning">No dataset loaded for classification</div>`;
        }
    } catch (error) {
        console.error('Error:', error);
        const currentEl = document.getElementById('currentDataset');
        const checkEl = document.getElementById('checkingDatasetStatus');
        const trainEl = document.getElementById('trainingDatasetInfo');
        const vizEl = document.getElementById('visualizationDatasetInfo');
        const classEl = document.getElementById('classificationDatasetInfo');
        if (currentEl) currentEl.innerHTML = '<div class="error">Error checking dataset</div>';
        if (checkEl) checkEl.innerHTML = '<div class="error">Error checking dataset</div>';
        if (trainEl) trainEl.innerHTML = '<div class="error">Error checking dataset</div>';
        if (vizEl) vizEl.innerHTML = '<div class="error">Error checking dataset</div>';
        if (classEl) classEl.innerHTML = '<div class="error">Error checking dataset</div>';
    }
}

window.createSyntheticData = async function() {
    const nSamples = document.getElementById('nSamples').value;
    const inputDim = document.getElementById('inputDim').value;
    const nAnomalies = document.getElementById('nAnomalies').value;
    const valRatio = document.getElementById('valRatio') ? document.getElementById('valRatio').value : 0.2;
    
    try {
        const response = await fetch(
            `${API_BASE}/create-synthetic-data?n_samples=${nSamples}&input_dim=${inputDim}&n_anomalies=${nAnomalies}&val_ratio=${valRatio}`,
            { method: 'POST' }
        );
        const result = await response.json();
        alert(`Created data: ${result.filename}`);
        loadDataOptions();
    } catch (error) {
        console.error('Error:', error);
        alert('Error creating data');
    }
};

window.uploadData = async function() {
    const fileInput = document.getElementById('fileUpload');
    if (!fileInput.files[0]) {
        alert('Please select a file to upload');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        const response = await fetch(`${API_BASE}/upload-data`, {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        document.getElementById('uploadStatus').innerHTML = 
            `<div class="success">Uploaded file: ${result.filename}</div>`;
        loadDataOptions();

    } catch (error) {
        console.error('Error:', error);
        document.getElementById('uploadStatus').innerHTML = 
            `<div class="error">Error uploading file</div>`;
    }
};

})();