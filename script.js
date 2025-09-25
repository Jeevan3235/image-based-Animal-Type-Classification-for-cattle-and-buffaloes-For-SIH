// Upload page functionality
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const classifyBtn = document.getElementById('classifyBtn');
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsContent = document.getElementById('resultsContent');

    if (uploadArea) {
        // Drag and drop functionality
        uploadArea.addEventListener('click', () => imageInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.background = '#f8f9fa';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.background = '';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.background = '';
            if (e.dataTransfer.files.length) {
                handleImageUpload(e.dataTransfer.files[0]);
            }
        });
        
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleImageUpload(e.target.files[0]);
            }
        });
    }

    if (classifyBtn) {
        classifyBtn.addEventListener('click', classifyImage);
    }

    function handleImageUpload(file) {
        if (!file.type.match('image.*')) {
            alert('Please upload an image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewContainer.style.display = 'block';
            resultsContainer.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    function classifyImage() {
        const file = imageInput.files[0];
        if (!file) {
            alert('Please select an image first');
            return;
        }

        classifyBtn.innerHTML = '<div class="loading"></div> Classifying...';
        classifyBtn.disabled = true;

        const formData = new FormData();
        formData.append('image', file);

        fetch('/classify', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayResults(data);
            } else {
                throw new Error(data.error || 'Classification failed');
            }
        })
        .catch(error => {
            alert('Error: ' + error.message);
        })
        .finally(() => {
            classifyBtn.innerHTML = 'Classify Animal';
            classifyBtn.disabled = false;
        });
    }

    function displayResults(data) {
        resultsContainer.style.display = 'block';
        
        const traits = data.traits;
        const classification = data.classification;
        
        resultsContent.innerHTML = `
            <div class="classification-result">
                <h4>Animal Type: ${classification.type}</h4>
                <p>Confidence: ${(classification.confidence * 100).toFixed(2)}%</p>
            </div>
            
            <h4>Extracted Physical Traits:</h4>
            <table class="trait-table">
                <tr>
                    <th>Trait</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Body Length</td>
                    <td>${traits.body_length.toFixed(2)} px</td>
                </tr>
                <tr>
                    <td>Height at Withers</td>
                    <td>${traits.withers_height.toFixed(2)} px</td>
                </tr>
                <tr>
                    <td>Chest Width</td>
                    <td>${traits.chest_width.toFixed(2)} px</td>
                </tr>
                <tr>
                    <td>Rump Angle</td>
                    <td>${traits.rump_angle.toFixed(2)}°</td>
                </tr>
                <tr>
                    <td>Body Solidarity</td>
                    <td>${traits.solidity.toFixed(3)}</td>
                </tr>
            </table>
            
            <div style="margin-top: 20px;">
                <button onclick="location.reload()" class="btn-primary">Classify Another Animal</button>
                <a href="/history" class="btn-primary" style="margin-left: 10px;">View History</a>
            </div>
        `;
    }
});
