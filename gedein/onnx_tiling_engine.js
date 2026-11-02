/**
 * Gedein App - Advanced AI Edge Engine (Client-Side)
 * Fitur: Tiling Strategy, Padding, Asynchronous Queue, dan Progress Callback
 * Mencegah WebGL/WebGPU Out of Memory (OOM) saat memproses gambar besar.
 */

let aiSession = null;
let isModelLoaded = false;
const SCALE_FACTOR = 4; // Ubah sesuai model AI Anda (misal Real-ESRGAN x4)

// 1. INISIALISASI SESSION (Sama seperti Tahap 2)
async function loadAIModel() {
    try {
        console.log("Memulai inisialisasi ONNX Runtime Web...");
        const options = { executionProviders: ['webgpu', 'wasm'], graphOptimizationLevel: 'all' };
        
        // HAPUS BARIS LAMA:
        // const modelPath = './models/real-esrgan-x4-int8.onnx'; 
        
        // GANTI DENGAN LINK RESOLVE HUGGINGFACE INI (Perhatikan kata /resolve/ di tengah URL):
        const modelPath = 'https://huggingface.co/AXERA-TECH/Real-ESRGAN/resolve/main/onnx/realesrgan-x4.onnx'; 
        
        aiSession = await ort.InferenceSession.create(modelPath, options);
        isModelLoaded = true;
        console.log("Model AI berhasil dimuat!");
        return true;
    } catch (error) {
        console.error("Gagal memuat model:", error);
        return false;
    }
}

// 2. HELPER: PRE-PROCESSING DAN INFERENCE UNTUK 1 KOTAK KECIL (TILE)
async function processSingleTileWithAI(tileCanvasElement) {
    // A. Pre-processing: Canvas -> Tensor (NCHW)
    const ctx = tileCanvasElement.getContext('2d');
    const width = tileCanvasElement.width;
    const height = tileCanvasElement.height;
    const imageData = ctx.getImageData(0, 0, width, height).data;
    
    const float32Data = new Float32Array(3 * width * height);
    for (let i = 0; i < width * height; i++) {
        float32Data[i] = imageData[i * 4] / 255.0;                         // R
        float32Data[i + width * height] = imageData[i * 4 + 1] / 255.0;    // G
        float32Data[i + 2 * width * height] = imageData[i * 4 + 2] / 255.0;// B
    }
    
    const tensor = new ort.Tensor('float32', float32Data, [1, 3, height, width]);
    const feeds = {};
    feeds[aiSession.inputNames[0]] = tensor;

    // B. Inference
    const results = await aiSession.run(feeds);
    const outputTensor = results[aiSession.outputNames[0]];
    
    // C. Post-processing: Tensor -> Canvas
    const outChannels = outputTensor.dims[1];
    const outHeight = outputTensor.dims[2];
    const outWidth = outputTensor.dims[3];
    const outData = outputTensor.data;

    const outCanvas = document.createElement('canvas');
    outCanvas.width = outWidth;
    outCanvas.height = outHeight;
    const outCtx = outCanvas.getContext('2d');
    const outImgData = outCtx.createImageData(outWidth, outHeight);

    for (let i = 0; i < outWidth * outHeight; i++) {
        outImgData.data[i * 4] = Math.max(0, Math.min(255, outData[i] * 255.0));
        outImgData.data[i * 4 + 1] = Math.max(0, Math.min(255, outData[i + outWidth * outHeight] * 255.0));
        outImgData.data[i * 4 + 2] = Math.max(0, Math.min(255, outData[i + 2 * outWidth * outHeight] * 255.0));
        outImgData.data[i * 4 + 3] = 255;
    }
    outCtx.putImageData(outImgData, 0, 0);
    return outCanvas;
}

// 3. FUNGSI UTAMA: TILING & STITCHING
/**
 * Memotong gambar menjadi kecil-kecil, memprosesnya, dan menjahitnya kembali.
 * @param {HTMLImageElement} sourceImage - Gambar sumber.
 * @param {HTMLCanvasElement} outputCanvas - Canvas tujuan hasil render akhir.
 * @param {Function} onProgress - Callback untuk update UI (current, total, percentage).
 */
async function processImageWithTiling(sourceImage, outputCanvas, onProgress) {
    if (!isModelLoaded) await loadAIModel();

    // Konfigurasi Pemotongan (Sesuaikan dengan kekuatan hardware target)
    const TILE_SIZE = 256; // Maksimal ukuran 1 sisi kotak (256x256 px)
    const PADDING = 16;    // Area overlap agar tidak ada garis jahitan kasar

    const imgW = sourceImage.naturalWidth || sourceImage.width;
    const imgH = sourceImage.naturalHeight || sourceImage.height;

    // Siapkan kanvas raksasa untuk hasil akhir
    outputCanvas.width = imgW * SCALE_FACTOR;
    outputCanvas.height = imgH * SCALE_FACTOR;
    const finalCtx = outputCanvas.getContext('2d');

    const cols = Math.ceil(imgW / TILE_SIZE);
    const rows = Math.ceil(imgH / TILE_SIZE);
    const totalTiles = cols * rows;
    let processedTiles = 0;

    console.log(`Memulai Tiling: Memotong gambar menjadi ${cols}x${rows} grid (${totalTiles} kotak).`);

    // Proses kotak satu per satu (Queueing) untuk mencegah VRAM meledak
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            
            // 3.1 Hitung koordinat kotak asli
            let startX = c * TILE_SIZE;
            let startY = r * TILE_SIZE;
            let tWidth = Math.min(TILE_SIZE, imgW - startX);
            let tHeight = Math.min(TILE_SIZE, imgH - startY);

            // 3.2 Hitung area aman (Padding/Overlap) agar hasil jahitan mulus
            let padLeft = Math.min(PADDING, startX);
            let padTop = Math.min(PADDING, startY);
            let padRight = Math.min(PADDING, imgW - (startX + tWidth));
            let padBottom = Math.min(PADDING, imgH - (startY + tHeight));

            let extractX = startX - padLeft;
            let extractY = startY - padTop;
            let extractW = tWidth + padLeft + padRight;
            let extractH = tHeight + padTop + padBottom;

            // 3.3 Ekstrak gambar (kotak + padding) menggunakan temporary canvas
            const tileCanvas = document.createElement('canvas');
            tileCanvas.width = extractW;
            tileCanvas.height = extractH;
            const tileCtx = tileCanvas.getContext('2d');
            tileCtx.drawImage(
                sourceImage, 
                extractX, extractY, extractW, extractH, // Koordinat di sumber asal
                0, 0, extractW, extractH                // Koordinat di kanvas sementara
            );

            // 3.4 JALANKAN AI UNTUK KOTAK KECIL INI SAJA
            console.log(`Memproses Tile ${processedTiles + 1}/${totalTiles}...`);
            const processedTileCanvas = await processSingleTileWithAI(tileCanvas);

            // 3.5 Buang (Crop) area padding yang ikut diperbesar, lalu jahit ke kanvas utama
            const cropX = padLeft * SCALE_FACTOR;
            const cropY = padTop * SCALE_FACTOR;
            const cropW = tWidth * SCALE_FACTOR;
            const cropH = tHeight * SCALE_FACTOR;

            finalCtx.drawImage(
                processedTileCanvas,
                cropX, cropY, cropW, cropH,                                   // Crop area yang bukan padding
                startX * SCALE_FACTOR, startY * SCALE_FACTOR, cropW, cropH    // Tempel sesuai koordinat aslinya (x4)
            );

            // 3.6 Update Progress ke UI
            processedTiles++;
            if (typeof onProgress === 'function') {
                const percentage = Math.round((processedTiles / totalTiles) * 100);
                onProgress(processedTiles, totalTiles, percentage);
            }
            
            // Beri waktu browser bernapas (mencegah UI Freeze) menggunakan trik setTimeout
            await new Promise(resolve => setTimeout(resolve, 10));
        }
    }

    console.log("Stitching Selesai! Mengembalikan hasil akhir...");
    return outputCanvas.toDataURL('image/png');
}
