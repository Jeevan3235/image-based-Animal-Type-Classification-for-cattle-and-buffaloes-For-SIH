class MLClassifier {
    constructor() {
        this.modelLoaded = false;
        this.apiEndpoint = 'http://localhost:5000/api/classify';
    }

    async classifyImage(imageElement) {
        if (!this.modelLoaded) {
            this.modelLoaded = true; // Skip metadata loading for now
        }

        // Convert image to base64
        const canvas = document.createElement('canvas');
        canvas.width = imageElement.naturalWidth || imageElement.width;
        canvas.height = imageElement.naturalHeight || imageElement.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(imageElement, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        // Call classification API
        const response = await fetch(this.apiEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData }),
        });

        if (!response.ok) throw new Error(`Classification API error: ${response.status}`);

        return await response.json();
    }
}

async function measureImage(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = imageElement.naturalWidth || imageElement.width;
    canvas.height = imageElement.naturalHeight || imageElement.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0);

    const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, 'image/jpeg', 0.8)
    );

    const formData = new FormData();
    formData.append('file', blob, 'animal.jpg');

    const response = await fetch('http://localhost:8001/measure', {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) throw new Error(`Measurement API error: ${response.status}`);

    return await response.json();
}

async function fullAnimalAnalysis(imageElement) {
    const mlClassifier = new MLClassifier();

    try {
        const classification = await mlClassifier.classifyImage(imageElement);
        const measurement = await measureImage(imageElement);

        return { classification, measurement };
    } catch (error) {
        console.error('Error during full analysis:', error);
        return null;
    }
}

// Example: Hook this to your file input
document.getElementById('imageInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
        const result = await fullAnimalAnalysis(img);
        console.log('Full Analysis Result:', result);

        // Example: show in a <pre id="results"></pre> element
        const resultsEl = document.getElementById('results');
        if (resultsEl) {
            resultsEl.textContent = JSON.stringify(result, null, 2);
        }
    };
});
