import * as THREE from 'three';

/**
 * ============================================================================
 * LAYER 2: PROCEDURAL PBR ASSET FACTORY (ZERO-DOWNLOAD + ZERO-VRAM-LEAK CACHE)
 * Upgraded for Three.js r170 Color Management (`SRGBColorSpace` on Albedo,
 * Linear `NoColorSpace` on Bump/Roughness) + Deterministic Texture Caching.
 * ============================================================================
 */
const _textureCache = new Map();

function markAlbedoSRGB(tex, anisotropy = 4) {
  if ('colorSpace' in tex && THREE.SRGBColorSpace) {
    tex.colorSpace = THREE.SRGBColorSpace;
  }
  tex.anisotropy = anisotropy;
  return tex;
}

export class ProceduralAssetFactory {
  static createEnvironmentCubeMap(weather = "MORNING") {
    const cacheKey = `envCube_${weather}`;
    if (_textureCache.has(cacheKey)) return _textureCache.get(cacheKey);

    const palettes = {
      MORNING: { top: '#0b2447', mid: '#38bdf8', hor: '#bae6fd', gnd: '#1e293b', sun: '#fffbeb' },
      NOON:    { top: '#0284c7', mid: '#7dd3fc', hor: '#e0f2fe', gnd: '#334155', sun: '#ffffff' },
      SUNSET:  { top: '#1e1b4b', mid: '#f97316', hor: '#fde047', gnd: '#1c1917', sun: '#fef08a' },
      STORM:   { top: '#020617', mid: '#0f172a', hor: '#1e293b', gnd: '#090d16', sun: '#38bdf8' }
    };
    const pal = palettes[weather] || palettes.MORNING;
    const canvases = [];

    for (let f = 0; f < 6; f++) {
      const c = document.createElement('canvas');
      c.width = 128;
      c.height = 128;
      const ctx = c.getContext('2d');

      if (f === 2) {
        ctx.fillStyle = pal.top;
        ctx.fillRect(0, 0, 128, 128);
      } else if (f === 3) {
        ctx.fillStyle = pal.gnd;
        ctx.fillRect(0, 0, 128, 128);
      } else {
        const grad = ctx.createLinearGradient(0, 0, 0, 128);
        grad.addColorStop(0.0, pal.top);
        grad.addColorStop(0.45, pal.mid);
        grad.addColorStop(0.58, pal.hor);
        grad.addColorStop(0.62, '#334155');
        grad.addColorStop(1.0, pal.gnd);
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 128, 128);

        ctx.fillStyle = 'rgba(30, 41, 59, 0.55)';
        ctx.beginPath();
        ctx.moveTo(0, 78);
        for (let x = 0; x <= 128; x += 16) {
          ctx.lineTo(x, 66 + Math.sin(x * 0.12 + f * 1.7) * 8);
        }
        ctx.lineTo(128, 78);
        ctx.fill();

        if (f === 0 || f === 4) {
          const sunG = ctx.createRadialGradient(64, 46, 2, 64, 46, 38);
          sunG.addColorStop(0, pal.sun);
          sunG.addColorStop(1, 'rgba(255,255,255,0)');
          ctx.fillStyle = sunG;
          ctx.fillRect(0, 0, 128, 128);
        }
      }
      canvases.push(c);
    }

    const cubeTex = new THREE.CubeTexture(canvases);
    markAlbedoSRGB(cubeTex, 1);
    cubeTex.needsUpdate = true;
    _textureCache.set(cacheKey, cubeTex);
    return cubeTex;
  }

  static createMicroBumpTexture() {
    if (_textureCache.has('microBump')) return _textureCache.get('microBump');
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, 512, 512);

    for (let i = 0; i < 32000; i++) {
      const x = Math.random() * 512;
      const y = Math.random() * 512;
      const v = 64 + Math.floor(Math.random() * 128);
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(x, y, 2.5, 2.5);
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    // Bump map stays linear (NoColorSpace)
    _textureCache.set('microBump', tex);
    return tex;
  }

  static createBallastAndSleeperTexture() {
    if (_textureCache.has('ballast')) return _textureCache.get('ballast');
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#2b3038';
    ctx.fillRect(0, 0, 512, 512);

    // High-contrast crushed basalt/granite railway ballast aggregate
    for (let i = 0; i < 24000; i++) {
      const x = Math.random() * 512;
      const y = Math.random() * 512;
      const r = 1.0 + Math.random() * 2.8;
      const c = 34 + Math.floor(Math.random() * 76);
      ctx.fillStyle = `rgb(${c},${c + 2},${c + 5})`;
      ctx.fillRect(x, y, r, r);
    }

    // Center inter-track drainage cess & concrete inspection strip
    ctx.fillStyle = '#1b222c';
    ctx.fillRect(232, 0, 48, 512);
    ctx.fillStyle = '#64748b';
    ctx.fillRect(230, 0, 3, 512);
    ctx.fillRect(279, 0, 3, 512);

    // Sub-sleeper compacted ballast shadows & Pandrol rail-seat grease weathering bands
    [92, 166, 344, 418].forEach((rx) => {
      ctx.fillStyle = 'rgba(59, 40, 30, 0.34)';
      ctx.fillRect(rx - 12, 0, 24, 512);
    });

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    markAlbedoSRGB(tex, 8);
    _textureCache.set('ballast', tex);
    return tex;
  }

  static createTerrainDetailTexture() {
    if (_textureCache.has('terrainDetail')) return _textureCache.get('terrainDetail');
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#b8c4b8';
    ctx.fillRect(0, 0, 512, 512);

    for (let i = 0; i < 28000; i++) {
      const x = Math.random() * 512;
      const y = Math.random() * 512;
      const v = 145 + Math.floor(Math.random() * 95);
      ctx.fillStyle = `rgb(${v - 8},${v},${v - 10})`;
      ctx.fillRect(x, y, 2, 3);
    }

    for (let s = 0; s < 45; s++) {
      ctx.strokeStyle = 'rgba(80, 90, 80, 0.14)';
      ctx.lineWidth = 2 + Math.random() * 4;
      ctx.beginPath();
      const sy = Math.random() * 512;
      ctx.moveTo(0, sy);
      ctx.bezierCurveTo(170, sy + 14, 340, sy - 14, 512, sy);
      ctx.stroke();
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    markAlbedoSRGB(tex, 4);
    _textureCache.set('terrainDetail', tex);
    return tex;
  }

  static createWaterRippleTexture() {
    if (_textureCache.has('waterRipple')) return _textureCache.get('waterRipple');
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0369a1';
    ctx.fillRect(0, 0, 256, 256);

    for (let i = 0; i < 900; i++) {
      const x = Math.random() * 256;
      const y = Math.random() * 256;
      const w = 8 + Math.random() * 22;
      ctx.fillStyle = Math.random() > 0.35 ? 'rgba(56, 189, 248, 0.28)' : 'rgba(224, 242, 254, 0.22)';
      ctx.fillRect(x, y, w, 2);
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    markAlbedoSRGB(tex, 4);
    _textureCache.set('waterRipple', tex);
    return tex;
  }

  static createBulletNoseTexture(liveryCfg, isTail = false) {
    const cacheKey = `bulletNose_${liveryCfg.name}_${isTail ? 'T' : 'F'}`;
    if (_textureCache.has(cacheKey)) return _textureCache.get(cacheKey);

    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');

    const hexBody = '#' + liveryCfg.bodyColor.toString(16).padStart(6, '0');
    const hexS1 = '#' + liveryCfg.stripe1.toString(16).padStart(6, '0');
    const hexS2 = '#' + liveryCfg.stripe2.toString(16).padStart(6, '0');

    const baseGrad = ctx.createLinearGradient(0, 0, 0, 1024);
    baseGrad.addColorStop(0.0, '#ffffff');
    baseGrad.addColorStop(0.08, '#f8fafc');
    baseGrad.addColorStop(0.22, hexBody);
    baseGrad.addColorStop(0.76, hexBody);
    baseGrad.addColorStop(0.90, '#cbd5e1');
    baseGrad.addColorStop(1.0, '#0f172a');
    ctx.fillStyle = baseGrad;
    ctx.fillRect(0, 0, 1024, 1024);

    ctx.fillStyle = '#1e293b';
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(420, 0);
    ctx.quadraticCurveTo(260, 68, 0, 84);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = hexS1;
    ctx.beginPath();
    ctx.moveTo(0, 84);
    ctx.quadraticCurveTo(280, 76, 520, 0);
    ctx.lineTo(660, 0);
    ctx.quadraticCurveTo(340, 112, 0, 124);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = '#090d16';
    ctx.beginPath();
    ctx.moveTo(0, 896);
    ctx.quadraticCurveTo(520, 880, 1024, 790);
    ctx.lineTo(1024, 1024);
    ctx.lineTo(0, 1024);
    ctx.closePath();
    ctx.fill();

    const s1Grad = ctx.createLinearGradient(0, 500, 0, 720);
    s1Grad.addColorStop(0.0, hexS1);
    s1Grad.addColorStop(0.5, '#ff1e42');
    s1Grad.addColorStop(1.0, hexS1);
    ctx.fillStyle = s1Grad;
    ctx.beginPath();
    ctx.moveTo(0, 532);
    ctx.bezierCurveTo(320, 532, 620, 480, 965, 620);
    ctx.lineTo(995, 710);
    ctx.bezierCurveTo(640, 620, 340, 684, 0, 684);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = hexS1;
    ctx.beginPath();
    ctx.moveTo(140, 380);
    ctx.quadraticCurveTo(460, 355, 810, 525);
    ctx.lineTo(745, 560);
    ctx.quadraticCurveTo(430, 425, 110, 430);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = '#f8fafc';
    ctx.beginPath();
    ctx.moveTo(0, 684);
    ctx.quadraticCurveTo(460, 680, 985, 712);
    ctx.lineTo(985, 726);
    ctx.quadraticCurveTo(460, 694, 0, 696);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = hexS2;
    ctx.beginPath();
    ctx.moveTo(0, 696);
    ctx.quadraticCurveTo(460, 694, 985, 726);
    ctx.lineTo(995, 774);
    ctx.quadraticCurveTo(460, 756, 0, 756);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = '#020617';
    ctx.beginPath();
    ctx.moveTo(128, 36);
    ctx.lineTo(445, 36);
    ctx.quadraticCurveTo(535, 180, 548, 365);
    ctx.quadraticCurveTo(340, 415, 138, 392);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 8;
    ctx.stroke();

    const glassGrad = ctx.createLinearGradient(150, 60, 520, 375);
    glassGrad.addColorStop(0.0, '#020617');
    glassGrad.addColorStop(0.30, '#0c4a6e');
    glassGrad.addColorStop(0.68, '#0284c7');
    glassGrad.addColorStop(0.92, '#7dd3fc');
    glassGrad.addColorStop(1.0, '#0f172a');
    ctx.fillStyle = glassGrad;
    ctx.beginPath();
    ctx.moveTo(148, 52);
    ctx.lineTo(432, 52);
    ctx.quadraticCurveTo(512, 185, 524, 348);
    ctx.quadraticCurveTo(335, 392, 156, 372);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = '#020617';
    ctx.lineWidth = 22;
    ctx.beginPath();
    ctx.moveTo(265, 42);
    ctx.lineTo(295, 388);
    ctx.stroke();

    ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.beginPath();
    ctx.moveTo(330, 58);
    ctx.lineTo(395, 58);
    ctx.lineTo(475, 345);
    ctx.lineTo(415, 355);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 7;
    ctx.beginPath();
    ctx.moveTo(505, 330);
    ctx.lineTo(375, 195);
    ctx.moveTo(485, 348);
    ctx.lineTo(355, 235);
    ctx.stroke();

    ctx.fillStyle = '#020617';
    ctx.beginPath();
    ctx.moveTo(635, 555);
    ctx.quadraticCurveTo(735, 545, 835, 605);
    ctx.quadraticCurveTo(745, 675, 625, 645);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 5;
    ctx.stroke();

    ctx.fillStyle = isTail ? '#f43f5e' : '#ffffff';
    ctx.beginPath();
    ctx.ellipse(705, 602, 34, 22, 0.15, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = isTail ? '#fb7185' : '#38bdf8';
    ctx.beginPath();
    ctx.ellipse(775, 616, 24, 16, 0.15, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = 'rgba(15, 23, 42, 0.45)';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(915, 350);
    ctx.quadraticCurveTo(885, 580, 915, 820);
    ctx.stroke();

    const tex = new THREE.CanvasTexture(canvas);
    markAlbedoSRGB(tex, 8);
    _textureCache.set(cacheKey, tex);
    return tex;
  }

  static createRollingStockTexture(liveryCfg, isLocomotive = false) {
    const cacheKey = `rollingStock_${liveryCfg.name}_${isLocomotive ? 'L' : 'C'}`;
    if (_textureCache.has(cacheKey)) return _textureCache.get(cacheKey);

    const canvas = document.createElement('canvas');
    canvas.width = 2048;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');

    const hexBody = '#' + liveryCfg.bodyColor.toString(16).padStart(6, '0');
    const hexS1 = '#' + liveryCfg.stripe1.toString(16).padStart(6, '0');
    const hexS2 = '#' + liveryCfg.stripe2.toString(16).padStart(6, '0');
    const isBullet = (liveryCfg.noseType === "BULLET");

    const bodyGrad = ctx.createLinearGradient(0, 0, 0, 512);
    bodyGrad.addColorStop(0.0, '#ffffff');
    bodyGrad.addColorStop(0.08, '#f8fafc');
    bodyGrad.addColorStop(0.22, hexBody);
    bodyGrad.addColorStop(0.76, hexBody);
    bodyGrad.addColorStop(0.90, '#cbd5e1');
    bodyGrad.addColorStop(1.0, '#64748b');
    ctx.fillStyle = bodyGrad;
    ctx.fillRect(0, 0, 2048, 512);

    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 0, 2048, 42);
    ctx.fillStyle = '#334155';
    for (let rx = 0; rx < 2048; rx += 16) {
      ctx.fillRect(rx, 8, 10, 26);
    }
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 448, 2048, 64);

    ctx.strokeStyle = 'rgba(15, 23, 42, 0.14)';
    ctx.lineWidth = 2;
    for (let px = 128; px < 2048; px += 128) {
      ctx.beginPath();
      ctx.moveTo(px, 42);
      ctx.lineTo(px, 448);
      ctx.stroke();
    }

    const s1Grad = ctx.createLinearGradient(0, 266, 0, 344);
    s1Grad.addColorStop(0.0, hexS1);
    s1Grad.addColorStop(0.5, '#ff1e42');
    s1Grad.addColorStop(1.0, hexS1);
    ctx.fillStyle = isBullet ? s1Grad : hexS1;
    ctx.fillRect(0, 266, 2048, 76);

    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 342, 2048, 6);
    ctx.fillStyle = hexS2;
    ctx.fillRect(0, 348, 2048, 30);

    if (isBullet) {
      ctx.save();
      ctx.fillStyle = 'rgba(255, 255, 255, 0.13)';
      for (let kx = 40; kx < 2000; kx += 36) {
        for (let ky = 274; ky < 332; ky += 18) {
          ctx.beginPath();
          ctx.moveTo(kx, ky + 14);
          ctx.lineTo(kx + 14, ky);
          ctx.lineTo(kx + 28, ky + 14);
          ctx.closePath();
          ctx.fill();
        }
      }
      ctx.restore();

      const beltGrad = ctx.createLinearGradient(0, 88, 0, 224);
      beltGrad.addColorStop(0.0, '#020617');
      beltGrad.addColorStop(0.5, '#090f1d');
      beltGrad.addColorStop(1.0, '#020617');
      ctx.fillStyle = beltGrad;
      ctx.fillRect(36, 88, 1976, 136);

      ctx.fillStyle = hexS1;
      ctx.fillRect(0, 44, 2048, 18);
    }

    if (!isLocomotive || isBullet) {
      [72, 1848].forEach((dx) => {
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(dx - 4, 82, 128, 358);
        ctx.fillStyle = isBullet ? '#e2e8f0' : hexBody;
        ctx.fillRect(dx, 86, 120, 350);
        ctx.fillStyle = hexS1;
        ctx.fillRect(dx, 266, 120, 76);
        ctx.fillStyle = hexS2;
        ctx.fillRect(dx, 348, 120, 30);

        ctx.fillStyle = '#020617';
        ctx.fillRect(dx + 22, 108, 76, 106);
        const dWinG = ctx.createLinearGradient(dx + 26, 112, dx + 26, 210);
        dWinG.addColorStop(0, '#0f172a');
        dWinG.addColorStop(0.6, '#0284c7');
        dWinG.addColorStop(1, '#bae6fd');
        ctx.fillStyle = dWinG;
        ctx.fillRect(dx + 26, 112, 68, 98);

        ctx.fillStyle = '#10b981';
        ctx.beginPath();
        ctx.arc(dx + 106, 246, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#facc15';
        ctx.fillRect(dx + 6, 428, 108, 8);
      });

      for (let x = 236; x < 1760; x += 156) {
        ctx.fillStyle = '#020617';
        ctx.fillRect(x - 8, 96, 124, 120);
        ctx.strokeStyle = '#475569';
        ctx.lineWidth = 3;
        ctx.strokeRect(x - 6, 98, 120, 116);

        const winGrad = ctx.createLinearGradient(x, 102, x, 208);
        winGrad.addColorStop(0.0, '#fef08a');
        winGrad.addColorStop(0.14, '#0f172a');
        winGrad.addColorStop(0.52, '#0369a1');
        winGrad.addColorStop(0.88, '#38bdf8');
        winGrad.addColorStop(1.0, '#e0f2fe');
        ctx.fillStyle = winGrad;
        ctx.fillRect(x, 102, 108, 106);

        ctx.fillStyle = '#1e293b';
        ctx.fillRect(x + 14, 152, 34, 56);
        ctx.fillRect(x + 60, 152, 34, 56);
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(x + 18, 146, 26, 12);
        ctx.fillRect(x + 64, 146, 26, 12);

        ctx.fillStyle = 'rgba(255, 255, 255, 0.18)';
        ctx.beginPath();
        ctx.moveTo(x + 18, 102);
        ctx.lineTo(x + 46, 102);
        ctx.lineTo(x + 14, 208);
        ctx.lineTo(x + 2, 208);
        ctx.closePath();
        ctx.fill();
      }

      ctx.fillStyle = '#020617';
      ctx.fillRect(228, 232, 340, 28);
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 2;
      ctx.strokeRect(228, 232, 340, 28);
      ctx.fillStyle = '#f59e0b';
      ctx.font = 'bold 17px JetBrains Mono, monospace';
      const destLabel = isBullet
        ? 'G1024 WHOOSH • HALIM - TEGALLUAR'
        : (liveryCfg.consistType === 'KAI_STAINLESS' ? 'KA GAJAYANA • EKSEKUTIF NEW GEN' : 'KA ARGO PARAHYANGAN • GAMBIR - BD');
      ctx.fillText(destLabel, 238, 252);

      if (liveryCfg.consistType === 'KAI_STAINLESS') {
        ctx.fillStyle = 'rgba(15, 23, 42, 0.16)';
        for (let ry = 50; ry < 86; ry += 6) ctx.fillRect(0, ry, 2048, 2);
        for (let ry = 382; ry < 440; ry += 6) ctx.fillRect(0, ry, 2048, 2);
      }

      ctx.fillStyle = '#ffffff';
      ctx.font = '900 44px Inter, sans-serif';
      ctx.fillText(isBullet ? 'WHOOSH' : 'KAI • KERETA API INDONESIA', 760, 318);
      if (isBullet) {
        ctx.font = 'bold 22px JetBrains Mono, monospace';
        ctx.fillStyle = '#fde047';
        ctx.fillText('KCIC CR400AF • 350 KM/H', 1120, 314);
      }
    } else {
      for (let x = 220; x < 1780; x += 210) {
        ctx.fillStyle = '#090d16';
        ctx.fillRect(x, 96, 164, 124);
        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, 96, 164, 124);

        ctx.fillStyle = '#334155';
        for (let ly = 106; ly < 212; ly += 12) {
          ctx.fillRect(x + 10, ly, 144, 6);
        }

        ctx.strokeStyle = 'rgba(15, 23, 42, 0.35)';
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 8, 226, 70, 180);
        ctx.strokeRect(x + 86, 226, 70, 180);
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(x + 66, 300, 6, 22);
        ctx.fillRect(x + 92, 300, 6, 22);
      }

      ctx.fillStyle = '#facc15';
      ctx.fillRect(160, 286, 36, 36);
      ctx.fillRect(1840, 286, 36, 36);

      ctx.fillStyle = '#ffffff';
      ctx.font = '900 46px JetBrains Mono, monospace';
      ctx.fillText(liveryCfg.name, 690, 320);
    }

    for (let ux = 140; ux < 1920; ux += 180) {
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(ux, 456, 110, 44);
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 2;
      ctx.strokeRect(ux, 456, 110, 44);
      ctx.fillStyle = '#ef4444';
      ctx.fillRect(ux + 45, 450, 20, 5);
    }

    const tex = new THREE.CanvasTexture(canvas);
    markAlbedoSRGB(tex, 8);
    _textureCache.set(cacheKey, tex);
    return tex;
  }

  static createPlatformTopTexture() {
    if (_textureCache.has('platformTop')) return _textureCache.get('platformTop');
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#475569';
    ctx.fillRect(0, 0, 256, 256);

    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 2;
    for (let i = 0; i <= 256; i += 32) {
      ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 256); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(256, i); ctx.stroke();
    }

    ctx.fillStyle = '#facc15';
    ctx.fillRect(0, 0, 22, 256);
    ctx.fillRect(234, 0, 22, 256);
    ctx.fillStyle = '#ca8a04';
    for (let y = 4; y < 256; y += 12) {
      ctx.fillRect(4, y, 14, 6);
      ctx.fillRect(238, y, 14, 6);
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    markAlbedoSRGB(tex, 4);
    _textureCache.set('platformTop', tex);
    return tex;
  }

  static createStationPlatformTexture() {
    if (_textureCache.has('platformWall')) return _textureCache.get('platformWall');
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#334155';
    ctx.fillRect(0, 0, 256, 256);

    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 3;
    for (let x = 0; x <= 256; x += 64) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 256); ctx.stroke();
    }
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 210, 256, 46);

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    markAlbedoSRGB(tex, 4);
    _textureCache.set('platformWall', tex);
    return tex;
  }

  static createStationSignTexture(stationName, elevM = 712) {
    const cacheKey = `stSign_${stationName}_${elevM}`;
    if (_textureCache.has(cacheKey)) return _textureCache.get(cacheKey);

    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, 512, 128);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 6;
    ctx.strokeRect(4, 4, 504, 120);

    ctx.fillStyle = '#e11d48';
    ctx.fillRect(8, 8, 496, 26);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('KCIC WHOOSH HIGH-SPEED RAILWAY STATION', 256, 26);

    ctx.fillStyle = '#ffffff';
    ctx.font = '900 28px Inter, sans-serif';
    ctx.fillText(stationName, 256, 74);

    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 15px JetBrains Mono, monospace';
    ctx.fillText(`ELEVASI: +${elevM}M DPL • JALUR GANDA HSR 350 KM/H`, 256, 106);

    const tex = new THREE.CanvasTexture(canvas);
    markAlbedoSRGB(tex, 4);
    _textureCache.set(cacheKey, tex);
    return tex;
  }

  static createTunnelWallTexture() {
    if (_textureCache.has('tunnelWall')) return _textureCache.get('tunnelWall');
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 0, 256, 256);
    for (let y = 0; y < 256; y += 32) {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, y, 256, 4);
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    markAlbedoSRGB(tex, 4);
    _textureCache.set('tunnelWall', tex);
    return tex;
  }

  static createSkyscraperTexture() {
    if (_textureCache.has('skyscraper')) return _textureCache.get('skyscraper');
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, 256, 512);
    for (let y = 14; y < 496; y += 22) {
      for (let x = 14; x < 240; x += 26) {
        if (Math.random() > 0.32) {
          ctx.fillStyle = Math.random() > 0.25 ? '#fef08a' : '#38bdf8';
        } else {
          ctx.fillStyle = '#1e293b';
        }
        ctx.fillRect(x, y, 16, 13);
      }
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    markAlbedoSRGB(tex, 4);
    _textureCache.set('skyscraper', tex);
    return tex;
  }

  static createNPCFaceTexture(variant = 0) {
    const cacheKey = `npcFace_${variant}`;
    if (_textureCache.has(cacheKey)) return _textureCache.get(cacheKey);

    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');

    const skinTones = ['#f5d0b5', '#eab68c', '#d99b72', '#f8d9c0', '#c88a62', '#e5b895'];
    const skin = skinTones[variant % skinTones.length];
    ctx.fillStyle = skin;
    ctx.fillRect(0, 0, 256, 256);

    const shade = ctx.createRadialGradient(128, 126, 16, 128, 128, 132);
    shade.addColorStop(0, 'rgba(255, 255, 255, 0.14)');
    shade.addColorStop(0.55, 'rgba(244, 114, 182, 0.05)');
    shade.addColorStop(1, 'rgba(120, 53, 15, 0.24)');
    ctx.fillStyle = shade;
    ctx.fillRect(0, 0, 256, 256);

    ctx.fillStyle = '#1c1917';
    [-1, 1].forEach((dir) => {
      ctx.beginPath();
      const bx = 128 + dir * 21;
      ctx.moveTo(bx - dir * 12, 99);
      ctx.quadraticCurveTo(bx, 93, bx + dir * 12, 98);
      ctx.quadraticCurveTo(bx, 97, bx - dir * 12, 102);
      ctx.fill();
    });

    [107, 149].forEach((ex) => {
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.ellipse(ex, 114, 10, 6.5, 0, 0, Math.PI * 2);
      ctx.fill();

      const irisG = ctx.createRadialGradient(ex, 114, 1, ex, 114, 5.5);
      irisG.addColorStop(0, '#090d16');
      irisG.addColorStop(0.55, '#451a03');
      irisG.addColorStop(1, '#1c1917');
      ctx.fillStyle = irisG;
      ctx.beginPath();
      ctx.arc(ex, 114, 5.2, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(ex - 2.0, 112.0, 1.8, 0, Math.PI * 2);
      ctx.arc(ex + 1.5, 115.5, 0.9, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(ex, 115, 10, Math.PI * 1.12, Math.PI * 1.88);
      ctx.stroke();
    });

    if (variant % 3 === 1) {
      ctx.strokeStyle = variant % 2 === 0 ? '#0f172a' : '#b45309';
      ctx.lineWidth = 3.2;
      ctx.strokeRect(93, 104, 28, 19);
      ctx.strokeRect(135, 104, 28, 19);
      ctx.beginPath();
      ctx.moveTo(121, 112);
      ctx.lineTo(135, 112);
      ctx.stroke();
    }

    ctx.strokeStyle = 'rgba(120, 53, 15, 0.42)';
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    ctx.moveTo(126, 110);
    ctx.lineTo(124, 134);
    ctx.quadraticCurveTo(128, 139, 134, 135);
    ctx.stroke();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.22)';
    ctx.beginPath();
    ctx.arc(128, 132, 3, 0, Math.PI * 2);
    ctx.fill();

    if (variant % 7 === 4) {
      ctx.fillStyle = '#f8fafc';
      ctx.beginPath();
      ctx.moveTo(90, 132);
      ctx.quadraticCurveTo(128, 124, 166, 132);
      ctx.lineTo(156, 174);
      ctx.quadraticCurveTo(128, 182, 100, 174);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = '#cbd5e1';
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      ctx.fillStyle = '#be123c';
      ctx.beginPath();
      ctx.ellipse(128, 150, 12, 4.5, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#881337';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(128, 146, 13, 0.15 * Math.PI, 0.85 * Math.PI);
      ctx.stroke();
    }

    const tex = new THREE.CanvasTexture(canvas);
    markAlbedoSRGB(tex, 2);
    _textureCache.set(cacheKey, tex);
    return tex;
  }

  static createNPCTorsoTexture(primaryHex, secondaryHex, style = 0) {
    const cacheKey = `npcTorso_${primaryHex}_${secondaryHex}_${style}`;
    if (_textureCache.has(cacheKey)) return _textureCache.get(cacheKey);

    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = primaryHex;
    ctx.fillRect(0, 0, 256, 256);

    const grad = ctx.createLinearGradient(0, 0, 0, 256);
    grad.addColorStop(0, 'rgba(255,255,255,0.14)');
    grad.addColorStop(0.5, 'rgba(0,0,0,0)');
    grad.addColorStop(1, 'rgba(0,0,0,0.32)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 256, 256);

    if (style === 0) {
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.moveTo(94, 0);
      ctx.lineTo(162, 0);
      ctx.lineTo(138, 152);
      ctx.lineTo(118, 152);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = 'rgba(0,0,0,0.38)';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(92, 0); ctx.lineTo(118, 152);
      ctx.moveTo(164, 0); ctx.lineTo(138, 152);
      ctx.stroke();

      ctx.fillStyle = secondaryHex;
      ctx.beginPath();
      ctx.moveTo(122, 14);
      ctx.lineTo(134, 14);
      ctx.lineTo(137, 132);
      ctx.lineTo(128, 148);
      ctx.lineTo(119, 132);
      ctx.closePath();
      ctx.fill();

      ctx.fillStyle = '#38bdf8';
      ctx.fillRect(152, 74, 24, 32);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(155, 80, 18, 22);
    } else if (style === 1) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.14)';
      for (let y = 16; y < 228; y += 28) {
        for (let x = 16; x < 240; x += 28) {
          ctx.beginPath();
          ctx.arc(x, y, 8, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillRect(x - 3, y - 3, 6, 6);
        }
      }
      ctx.fillStyle = secondaryHex;
      ctx.fillRect(122, 0, 12, 234);
      ctx.strokeStyle = '#e11d48';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(104, 0); ctx.lineTo(128, 108); ctx.lineTo(152, 0);
      ctx.stroke();
      ctx.fillStyle = '#f8fafc';
      ctx.fillRect(115, 106, 26, 36);
    } else if (style === 2) {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, 256, 256);
      ctx.fillStyle = '#eab308';
      ctx.fillRect(24, 0, 68, 256);
      ctx.fillRect(164, 0, 68, 256);
      ctx.fillStyle = '#e2e8f0';
      ctx.fillRect(0, 136, 256, 24);
      ctx.fillRect(0, 192, 256, 24);
      ctx.fillStyle = '#020617';
      ctx.fillRect(168, 42, 20, 38);
      ctx.fillRect(175, 18, 4, 24);
    }

    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 232, 256, 24);
    ctx.fillStyle = '#e2e8f0';
    ctx.fillRect(114, 234, 28, 20);

    const tex = new THREE.CanvasTexture(canvas);
    markAlbedoSRGB(tex, 2);
    _textureCache.set(cacheKey, tex);
    return tex;
  }

  static createStationFacadeTexture(stationName) {
    const cacheKey = `stFacade_${stationName}`;
    if (_textureCache.has(cacheKey)) return _textureCache.get(cacheKey);

    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');

    const bg = ctx.createLinearGradient(0, 0, 0, 256);
    bg.addColorStop(0, '#090d16');
    bg.addColorStop(0.35, '#0f172a');
    bg.addColorStop(1, '#1e293b');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, 1024, 256);

    for (let row = 0; row < 2; row++) {
      for (let x = 28; x < 996; x += 64) {
        const wGrad = ctx.createLinearGradient(x, 92 + row * 76, x, 156 + row * 76);
        wGrad.addColorStop(0, '#fef08a');
        wGrad.addColorStop(0.25, '#38bdf8');
        wGrad.addColorStop(1, '#0284c7');
        ctx.fillStyle = wGrad;
        ctx.fillRect(x, 92 + row * 76, 52, 62);
      }
    }

    ctx.fillStyle = '#dc2626';
    ctx.fillRect(0, 0, 1024, 76);
    ctx.fillStyle = '#facc15';
    ctx.fillRect(0, 76, 1024, 6);

    ctx.fillStyle = '#ffffff';
    ctx.font = '900 38px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`WHOOSH • ${stationName}`, 512, 48);

    ctx.fillStyle = '#fde047';
    ctx.font = 'bold 15px JetBrains Mono, monospace';
    ctx.fillText('KERETA CEPAT INDONESIA CHINA • HIGH-SPEED RAILWAY TERMINAL', 512, 69);

    const tex = new THREE.CanvasTexture(canvas);
    markAlbedoSRGB(tex, 4);
    _textureCache.set(cacheKey, tex);
    return tex;
  }
}
