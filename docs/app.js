// Dashcam AI Analyzer - app.js

const ModelPath = './yolov8n.onnx';
let session = null;
let isRunning = false;
let isInferring = false;
let animationId = null;
let lastFrameTime = 0;
let frameCount = 0;
let fps = 0;

// UI Elements
const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const cameraSelect = document.getElementById('cameraSelect');
const statusBadge = document.getElementById('aiStatus');
const fpsValue = document.getElementById('fpsValue');
const alertBanner = document.getElementById('alertBanner');

// Constants
const VEHICLE_CLASSES = [2, 3, 5, 7]; // COCO: car, motorcycle, bus, truck
const INPUT_DIM = 640;

// Init ONNX Session
async function init() {
    try {
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/';
        session = await ort.InferenceSession.create(ModelPath, { executionProviders: ['wasm'] });
        console.log("Model loaded.");
        statusBadge.textContent = "AI Ready";
        statusBadge.className = "status-badge ready";
        
        await initCameras();
        
        startBtn.disabled = false;
    } catch (e) {
        console.error("Failed to load model:", e);
        statusBadge.textContent = "Error: AI Load Failed";
        statusBadge.style.color = "red";
    }
}

async function initCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');
        
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
        });
    } catch (err) {
        console.error("Camera access error:", err);
    }
}

async function startCamera() {
    if (isRunning) return;
    const deviceId = cameraSelect.value;
    const constraints = {
        video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "environment" },
        audio: false
    };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        
        // Wait for video to be ready
        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();
                overlay.width = video.videoWidth;
                overlay.height = video.videoHeight;
                resolve();
            }
        });

        isRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusBadge.textContent = "AI Active";
        statusBadge.className = "status-badge active";
        
        lastFrameTime = performance.now();
        loop();
    } catch (e) {
        alert("Cannot access camera: " + e.message);
    }
}

function stopCamera() {
    isRunning = false;
    if (animationId) cancelAnimationFrame(animationId);
    
    const stream = video.srcObject;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    video.srcObject = null;
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusBadge.textContent = "AI Ready";
    statusBadge.className = "status-badge ready";
    
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    alertBanner.classList.add('hidden');
}

// Main Loop
async function loop() {
    if (!isRunning) return;

    // Calculate FPS
    const now = performance.now();
    frameCount++;
    if (now - lastFrameTime >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastFrameTime = now;
        fpsValue.textContent = fps;
    }

    if (!isInferring && video.readyState === 4) {
        isInferring = true;
        await processFrame();
        isInferring = false;
    }
    animationId = requestAnimationFrame(loop);
}

// Convert RGB to HSV
function rgbToHsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, s, v = max;
    const d = max - min;
    s = max === 0 ? 0 : d / max;
    if (max === min) {
        h = 0; // achromatic
    } else {
        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }
    return [h * 180, s * 255, v * 255]; // Map to OpenCV ranges: H=0-180, S=0-255, V=0-255
}

// Check for emergency lights in a specific bounding box zone
function detectEmergencyLight(x1, y1, x2, y2) {
    // We only care about the top 25% of the vehicle (roof) + expanded search space above
    const height = y2 - y1;
    let extendY = Math.max(0, parseInt(y1 - height * 0.25));
    let cropHeight = parseInt(height * 0.35); // Check from slightly above to top 10%
    if (extendY + cropHeight > overlay.height) cropHeight = overlay.height - extendY;
    if (cropHeight <= 0) return false;
    
    const w = x2 - x1;
    if (w <= 0 || extendY < 0 || extendY + cropHeight > overlay.height || x1 < 0 || x2 > overlay.width) return false;

    // We get image data from the actual video resolution (not 640x640)
    // Create a temporary canvas matching video dimensions to read pixels
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = overlay.width;
    tempCanvas.height = overlay.height;
    const tCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
    tCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    
    // Safety clamp
    extendY = Math.max(0, Math.min(extendY, tempCanvas.height - 1));
    let safeX1 = Math.max(0, Math.min(x1, tempCanvas.width - 1));
    let safeW = Math.max(1, Math.min(w, tempCanvas.width - safeX1));
    let safeH = Math.max(1, Math.min(cropHeight, tempCanvas.height - extendY));

    const imgData = tCtx.getImageData(safeX1, extendY, safeW, safeH).data;
    let redCount = 0;
    
    for (let i = 0; i < imgData.length; i += 4) {
        const r = imgData[i];
        const g = imgData[i + 1];
        const b = imgData[i + 2];
        const [h, s, v] = rgbToHsv(r, g, b);

        // Lower: (0, 70, 70) to (10, 255, 255)
        // Upper: (160, 70, 70) to (180, 255, 255)
        if (s >= 70 && v >= 70) {
            if ((h >= 0 && h <= 10) || (h >= 160 && h <= 180)) {
                redCount++;
            }
        }
    }

    const totalPixels = safeW * safeH;
    const redRatio = totalPixels > 0 ? redCount / totalPixels : 0;
    
    // Use the same heuristic logic as Python
    if (redCount >= 5 && redRatio > 0.0005 && redRatio < 0.12) {
        return { isEmergency: true, roi: {x1: safeX1, y: extendY, w: safeW, h: safeH} };
    }
    return { isEmergency: false, roi: null };
}

// Inference mapping
async function processFrame() {
    // 1. Prepare image onnx input (resize to 640x640)
    const preprocessCanvas = document.createElement('canvas');
    preprocessCanvas.width = INPUT_DIM;
    preprocessCanvas.height = INPUT_DIM;
    const pCtx = preprocessCanvas.getContext('2d', { willReadFrequently: true });
    pCtx.drawImage(video, 0, 0, INPUT_DIM, INPUT_DIM);
    const imgData = pCtx.getImageData(0, 0, INPUT_DIM, INPUT_DIM).data;

    // Convert to Float32Array [1, 3, 640, 640] and normalize RGB / 255.0
    const float32Data = new Float32Array(3 * INPUT_DIM * INPUT_DIM);
    for (let i = 0; i < imgData.length / 4; i++) {
        float32Data[i] = imgData[i * 4] / 255.0; // R
        float32Data[i + INPUT_DIM * INPUT_DIM] = imgData[i * 4 + 1] / 255.0; // G
        float32Data[i + 2 * INPUT_DIM * INPUT_DIM] = imgData[i * 4 + 2] / 255.0; // B
    }

    const tensor = new ort.Tensor('float32', float32Data, [1, 3, INPUT_DIM, INPUT_DIM]);

    // 2. Run Inference
    const results = await session.run({ images: tensor });
    const output = results.output0.data; // Float32Array of size 84 * 8400

    // Output shape: [1, 84, 8400]
    // Indices:
    // 0-3: cx, cy, w, h
    // 4-83: class scores
    const numBoxes = 8400;
    const numClasses = 80;

    let boxes = [];
    const confThreshold = 0.3;

    // Scale back to original video size
    const scaleX = video.videoWidth / INPUT_DIM;
    const scaleY = video.videoHeight / INPUT_DIM;

    for (let i = 0; i < numBoxes; i++) {
        let maxClassScore = 0;
        let classId = -1;

        // Find best class
        for (let c = 0; c < numClasses; c++) {
            const score = output[(4 + c) * numBoxes + i];
            if (score > maxClassScore) {
                maxClassScore = score;
                classId = c;
            }
        }

        if (maxClassScore > confThreshold && VEHICLE_CLASSES.includes(classId)) {
            const cx = output[0 * numBoxes + i];
            const cy = output[1 * numBoxes + i];
            const w = output[2 * numBoxes + i];
            const h = output[3 * numBoxes + i];

            const x1 = Math.max(0, (cx - w / 2) * scaleX);
            const y1 = Math.max(0, (cy - h / 2) * scaleY);
            const x2 = Math.min(video.videoWidth, (cx + w / 2) * scaleX);
            const y2 = Math.min(video.videoHeight, (cy + h / 2) * scaleY);

            boxes.push({ x1, y1, x2, y2, score: maxClassScore, classId });
        }
    }

    // Apply Non-Maximum Suppression (NMS)
    boxes.sort((a, b) => b.score - a.score);
    const nmsBoxes = [];
    const iouThreshold = 0.45;

    while (boxes.length > 0) {
        const best = boxes.shift();
        nmsBoxes.push(best);
        boxes = boxes.filter(box => {
            const interX1 = Math.max(best.x1, box.x1);
            const interY1 = Math.max(best.y1, box.y1);
            const interX2 = Math.min(best.x2, box.x2);
            const interY2 = Math.min(best.y2, box.y2);
            
            const interW = Math.max(0, interX2 - interX1);
            const interH = Math.max(0, interY2 - interY1);
            const interArea = interW * interH;
            
            const bestArea = (best.x2 - best.x1) * (best.y2 - best.y1);
            const boxArea = (box.x2 - box.x1) * (box.y2 - box.y1);
            const unionArea = bestArea + boxArea - interArea;
            
            const iou = interArea / unionArea;
            return iou < iouThreshold;
        });
    }

    // 3. Draw and Detect
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    let emergencyDetected = false;

    for (const box of nmsBoxes) {
        // Red Light Detection Phase
        const { isEmergency, roi } = detectEmergencyLight(box.x1, box.y1, box.x2, box.y2);

        if (isEmergency) {
            emergencyDetected = true;
            ctx.strokeStyle = '#ff4b4b';
            ctx.lineWidth = 4;
            ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
            
            // Draw ROI debug area
            if (roi) {
                ctx.fillStyle = 'rgba(255, 75, 75, 0.4)';
                ctx.fillRect(roi.x1, roi.y, roi.w, roi.h);
            }

            ctx.fillStyle = '#ff4b4b';
            ctx.font = 'bold 20px Inter, sans-serif';
            ctx.fillText("POLICE " + (box.classId === 3 ? "MOTO" : "CAR"), box.x1, Math.max(20, box.y1 - 10));
        } else {
            ctx.strokeStyle = '#3fb950';
            ctx.lineWidth = 2;
            ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
            
            ctx.fillStyle = '#3fb950';
            ctx.font = '14px Inter, sans-serif';
            ctx.fillText(box.classId === 3 ? "Motorcycle" : "Vehicle", box.x1, Math.max(14, box.y1 - 5));
        }
    }

    if (emergencyDetected) {
        alertBanner.classList.remove('hidden');
    } else {
        alertBanner.classList.add('hidden');
    }
}

// Events
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
cameraSelect.addEventListener('change', () => {
    if (isRunning) {
        stopCamera();
        startCamera();
    }
});

// Boot
window.onload = init;
