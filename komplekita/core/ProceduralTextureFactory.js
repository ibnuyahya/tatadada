import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (True-Color Photo-Realistic Edition)
 * ./core/ProceduralTextureFactory.js
 *
 * High-Resolution Procedural PBR Material, 2048x2048 Typography & Signage Atlas,
 * Equirectangular Depok Afternoon Sky Dome + PMREM Environment Map,
 * World-Space Multi-Scale FBM Macro-Variation & Tropical Weathering Shader,
 * and SSoST Clay/Zebra Reflection Line Inspection Engine.
 */
export class ProceduralTextureFactory {
  static createSharedShaderUniforms() {
    return {
      uInspectMode: { value: 0 }, // 0: True-Color Photorealistic PBR, 1: Clay Matte, 2: SSoST Zebra Stripes
      uTime: { value: 0.0 },
      uDehazeStrength: { value: 0.18 }
    };
  }

  /**
   * Injects World-Space Multi-Scale FBM Macro Variation, Tropical Rain-Streak Weathering,
   * Ground Contact AO, Leaf-Cluster Normal Perturbation (for foliage), Filmic True-Color Dehaze,
   * and Clay/Zebra SSoST Inspection into MeshStandardMaterial.
   */
  static injectOmniScaleSurfaceShader(material, sharedUniforms, opts = {}) {
    if (!material || material.userData.nexusShaderInjected) return material;
    material.userData.nexusShaderInjected = true;
    const isFoliage = !!opts.isFoliage;
    const isTerrain = !!opts.isTerrain;
    const isMasonry = !!opts.isMasonry;
    const isRoof = !!opts.isRoof;

    material.onBeforeCompile = (shader) => {
      shader.uniforms.uInspectMode = sharedUniforms.uInspectMode;
      shader.uniforms.uTime = sharedUniforms.uTime;
      shader.uniforms.uDehazeStrength = sharedUniforms.uDehazeStrength;

      shader.vertexShader = shader.vertexShader
        .replace(
          '#include <common>',
          `#include <common>
          varying vec3 vNexusWorldPos;
          varying vec3 vNexusWorldNorm;`
        )
        .replace(
          '#include <worldpos_vertex>',
          `#include <worldpos_vertex>
          vec4 nxWp = vec4(transformed, 1.0);
          #ifdef USE_INSTANCING
            nxWp = instanceMatrix * nxWp;
          #endif
          nxWp = modelMatrix * nxWp;
          vNexusWorldPos = nxWp.xyz;
          vec3 nxWn = objectNormal;
          #ifdef USE_INSTANCING
            nxWn = mat3(instanceMatrix) * nxWn;
          #endif
          vNexusWorldNorm = normalize(mat3(modelMatrix) * nxWn);`
        );

      shader.fragmentShader = shader.fragmentShader
        .replace(
          '#include <common>',
          `#include <common>
          uniform int uInspectMode;
          uniform float uDehazeStrength;
          varying vec3 vNexusWorldPos;
          varying vec3 vNexusWorldNorm;

          float nxHash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
          }
          float nxNoise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            vec2 u = f * f * (3.0 - 2.0 * f);
            return mix(
              mix(nxHash(i + vec2(0.0, 0.0)), nxHash(i + vec2(1.0, 0.0)), u.x),
              mix(nxHash(i + vec2(0.0, 1.0)), nxHash(i + vec2(1.0, 1.0)), u.x),
              u.y
            );
          }
          float nxFbm(vec2 p) {
            float v = 0.0;
            v += 0.5000 * nxNoise(p); p *= 2.03;
            v += 0.2500 * nxNoise(p); p *= 2.01;
            v += 0.1250 * nxNoise(p);
            return v;
          }`
        )
        .replace(
          '#include <dithering_fragment>',
          `
          if (uInspectMode == 0) {
            vec3 wNorm = normalize(vNexusWorldNorm);
            vec3 sunDir = normalize(vec3(-0.41, 0.85, 0.32));
            float camDist = length(cameraPosition - vNexusWorldPos);
            // Distance-Adaptive Multi-LOD Close-Up Micro-Detail Factor (active at FPV Runner eye-level < 38m)
            float closeUpLod = 1.0 - smoothstep(6.0, 38.0, camDist);
            float microGrain = (nxNoise(vNexusWorldPos.xz * 18.5 + vNexusWorldPos.xy * 14.0) - 0.5) * 0.11 * closeUpLod;

            ${isTerrain ? `
              // World-space macro FBM variation + close-up grass/soil blade micro-detail
              float macro1 = nxFbm(vNexusWorldPos.xz * 0.11);
              float macro2 = nxFbm(vNexusWorldPos.xz * 0.37 + vec2(19.3, 7.1));
              float bladeMicro = (nxFbm(vNexusWorldPos.xz * 9.5) - 0.45) * 0.16 * closeUpLod;
              vec3 dryTint = vec3(1.12, 0.98, 0.78);
              vec3 lushTint = vec3(0.86, 1.04, 0.84);
              gl_FragColor.rgb *= mix(lushTint, dryTint, smoothstep(0.28, 0.74, macro1));
              gl_FragColor.rgb *= (0.90 + 0.20 * macro2 + bladeMicro);
            ` : ''}

            ${isRoof ? `
              // Natural tropical ceramic roof tile weathering, course shadow & close-up glaze sheen
              float roofPatina = nxFbm(vNexusWorldPos.xz * 0.24);
              float mossPatch = smoothstep(0.58, 0.82, nxFbm(vNexusWorldPos.xz * 0.45 + vec2(4.2, 9.8)));
              float tileCourseWave = sin((vNexusWorldPos.y * 14.0 + vNexusWorldPos.x * 3.0 + vNexusWorldPos.z * 3.0)) * 0.045 * closeUpLod;
              gl_FragColor.rgb *= (0.88 + 0.22 * roofPatina + tileCourseWave);
              gl_FragColor.rgb = mix(gl_FragColor.rgb, gl_FragColor.rgb * vec3(0.76, 0.82, 0.68), mossPatch * 0.28);
            ` : ''}

            ${isMasonry ? `
              // Subtle vertical tropical rain-streak patina, plinth base moisture & close-up stucco/stone pore relief
              float streak = nxFbm(vec2(vNexusWorldPos.x * 0.6 + vNexusWorldPos.z * 0.6, vNexusWorldPos.y * 0.22));
              float baseDamp = 1.0 - 0.18 * (1.0 - smoothstep(0.0, 1.8, max(0.0, vNexusWorldPos.y)));
              gl_FragColor.rgb *= (0.93 + 0.12 * streak + microGrain) * baseDamp;
            ` : `
              gl_FragColor.rgb *= (1.0 + microGrain * 0.65);
            `}

            ${isFoliage ? `
              // Procedural leaf-cluster micro-relief & warm afternoon sun subsurface scatter
              float leafCluster = nxFbm(vNexusWorldPos.xz * 1.45 + vec2(vNexusWorldPos.y * 1.2));
              float leafMicro = nxNoise(vNexusWorldPos.xy * 4.2 + vNexusWorldPos.z * 3.7);
              float sunScatter = pow(clamp(dot(wNorm, sunDir) * 0.5 + 0.5, 0.0, 1.0), 1.6);
              vec3 sunLeafGlow = vec3(1.14, 1.18, 0.82);
              gl_FragColor.rgb *= (0.76 + 0.36 * leafCluster) * (0.88 + 0.24 * leafMicro);
              gl_FragColor.rgb *= mix(vec3(0.86, 0.92, 0.96), sunLeafGlow, sunScatter * 0.55);
              float canopyHeightAO = smoothstep(0.8, 5.5, vNexusWorldPos.y);
              gl_FragColor.rgb *= (0.78 + 0.22 * canopyHeightAO);
            ` : `
              // Ground-plane Contact AO (natural grounding shadow where walls/curbs meet terrain)
              float groundProx = clamp((vNexusWorldPos.y + 0.12) / 2.2, 0.0, 1.0);
              float contactAO = mix(0.80, 1.0, smoothstep(0.0, 1.0, groundProx));
              gl_FragColor.rgb *= contactAO;
            `}

            // True-Color Filmic Contrast & Atmospheric Dehaze
            vec3 dehazed = pow(max(gl_FragColor.rgb, vec3(0.0)), vec3(1.05)) * 1.03;
            gl_FragColor.rgb = mix(gl_FragColor.rgb, dehazed, uDehazeStrength);
          } else if (uInspectMode == 1) {
            vec3 nrm = normalize(vNexusWorldNorm);
            vec3 sunDir = normalize(vec3(-0.42, 0.86, 0.28));
            float ndl = clamp(dot(nrm, sunDir), 0.0, 1.0);
            float hemi = 0.5 + 0.5 * nrm.y;
            vec3 clay = vec3(0.80, 0.79, 0.77) * (0.42 + 0.52 * ndl + 0.18 * hemi);
            gl_FragColor.rgb = clay;
          } else if (uInspectMode == 2) {
            vec3 nrm = normalize(vNexusWorldNorm);
            vec3 viewDir = normalize(cameraPosition - vNexusWorldPos);
            vec3 refl = reflect(-viewDir, nrm);
            float wave = sin(refl.y * 34.0 + refl.x * 14.0 + refl.z * 10.0);
            float stripe = smoothstep(-0.12, 0.12, wave);
            vec3 base = mix(vec3(0.10, 0.13, 0.18), vec3(0.95, 0.97, 1.00), stripe);
            float rim = pow(1.0 - max(0.0, dot(viewDir, nrm)), 2.0);
            gl_FragColor.rgb = mix(base, vec3(0.22, 0.68, 0.98), rim * 0.35);
          }
          #include <dithering_fragment>
          `
        );
    };

    return material;
  }

  /**
   * Builds a photorealistic 14:00 WIB UI Depok Tropical Afternoon Sky Texture (with soft cumulus clouds & horizon haze)
   * plus a prefiltered PMREM Environment Map for realistic glass/metal reflections and GI skylight.
   */
  static createAfternoonEnvironmentMap(renderer) {
    const skyCanvas = this._makeCanvas(1024, 512);
    const ctx = skyCanvas.getContext('2d');

    // Equirectangular sky + ground gradient (top half = sky, bottom half = Depok foliage/earth bounce)
    const skyGrad = ctx.createLinearGradient(0, 0, 0, 262);
    skyGrad.addColorStop(0.00, '#4f86c6'); // zenith tropical blue
    skyGrad.addColorStop(0.45, '#7aaae0'); // mid sky
    skyGrad.addColorStop(0.82, '#bfd6ee'); // warm afternoon sky
    skyGrad.addColorStop(0.98, '#e6eef7'); // Depok humid horizon mist
    skyGrad.addColorStop(1.00, '#dde6ef');
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, 1024, 262);

    // Soft tropical cumulus cloud bands near horizon & mid-altitude
    for (let i = 0; i < 65; i++) {
      const cx = (i * 173) % 1024;
      const cy = 115 + ((i * 47) % 130);
      const rx = 48 + (i % 60);
      const ry = 14 + (i % 24);
      const g = ctx.createRadialGradient(cx, cy, 2, cx, cy, rx);
      const alpha = 0.08 + (i % 5) * 0.025;
      g.addColorStop(0, `rgba(255, 253, 248, ${alpha})`);
      g.addColorStop(0.6, `rgba(242, 247, 252, ${alpha * 0.55})`);
      g.addColorStop(1, 'rgba(225, 236, 248, 0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // Bottom hemisphere for PMREM reflection bounce: lush campus tree canopy & warm terracotta earth bounce
    const gndGrad = ctx.createLinearGradient(0, 256, 0, 512);
    gndGrad.addColorStop(0.00, '#c8d4c2');
    gndGrad.addColorStop(0.12, '#5a6e4c');
    gndGrad.addColorStop(0.50, '#445638');
    gndGrad.addColorStop(1.00, '#2e3b27');
    ctx.fillStyle = gndGrad;
    ctx.fillRect(0, 260, 1024, 252);

    const pmremSourceTex = new THREE.CanvasTexture(skyCanvas);
    pmremSourceTex.mapping = THREE.EquirectangularReflectionMapping;
    pmremSourceTex.colorSpace = THREE.SRGBColorSpace;
    pmremSourceTex.needsUpdate = true;

    const pmrem = new THREE.PMREMGenerator(renderer);
    pmrem.compileEquirectangularShader();
    const envRT = pmrem.fromEquirectangular(pmremSourceTex);
    pmremSourceTex.dispose();
    pmrem.dispose();

    // Now paint clean atmospheric aerial-perspective mist in the lower hemisphere of the background sky dome
    // so Oblique 45° and Nadir 90° views have a natural horizon sky backdrop around the island plinth!
    const bgCanvas = this._makeCanvas(1024, 512);
    const bctx = bgCanvas.getContext('2d');
    bctx.drawImage(skyCanvas, 0, 0);
    const mistGrad = bctx.createLinearGradient(0, 252, 0, 512);
    mistGrad.addColorStop(0.00, '#dde6ef');
    mistGrad.addColorStop(0.18, '#cedbe8');
    mistGrad.addColorStop(0.55, '#b8cadc');
    mistGrad.addColorStop(1.00, '#9fb6cd');
    bctx.fillStyle = mistGrad;
    bctx.fillRect(0, 254, 1024, 258);

    const skyTex = new THREE.CanvasTexture(bgCanvas);
    skyTex.mapping = THREE.EquirectangularReflectionMapping;
    skyTex.colorSpace = THREE.SRGBColorSpace;
    skyTex.needsUpdate = true;

    // Attach skyTex so WeatherOpticsEngine can use it directly as scene.background!
    envRT.texture.userData = { skyBackgroundTexture: skyTex };
    return envRT.texture;
  }

  static _makeCanvas(w, h) {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    return c;
  }

  static _toTexture(canvas, isColor = true, repeatX = 1, repeatY = 1) {
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(repeatX, repeatY);
    tex.anisotropy = 8;
    if (isColor) {
      tex.colorSpace = THREE.SRGBColorSpace;
    }
    tex.needsUpdate = true;
    return tex;
  }

  /**
   * Draws an authentic Golden Universitas Indonesia Makara Crest onto a 2D Canvas context.
   */
  static drawUIMakaraEmblem(ctx, cx, cy, radius, goldColor = '#eab308', strokeDark = '#713f12') {
    ctx.save();
    ctx.translate(cx, cy);

    const rays = 15;
    ctx.fillStyle = goldColor;
    ctx.strokeStyle = strokeDark;
    ctx.lineWidth = Math.max(1.2, radius * 0.045);

    ctx.beginPath();
    for (let i = 0; i <= rays * 2; i++) {
      const t = i / (rays * 2);
      const rad = -Math.PI * 0.5 + (t - 0.5) * Math.PI * 1.68;
      const rPeak = i % 2 === 0 ? radius * (1.0 - 0.14 * Math.abs(t - 0.5)) : radius * 0.72;
      const x = Math.sin(rad) * rPeak;
      const y = -Math.cos(rad) * rPeak;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#fde047';
    ctx.beginPath();
    ctx.arc(0, -radius * 0.06, radius * 0.52, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.strokeStyle = strokeDark;
    ctx.lineWidth = Math.max(1.6, radius * 0.075);
    ctx.beginPath();
    ctx.moveTo(0, radius * 0.42);
    ctx.lineTo(0, -radius * 0.42);
    ctx.moveTo(0, -radius * 0.08);
    ctx.quadraticCurveTo(-radius * 0.28, -radius * 0.22, -radius * 0.34, -radius * 0.04);
    ctx.moveTo(0, -radius * 0.08);
    ctx.quadraticCurveTo(radius * 0.28, -radius * 0.22, radius * 0.34, -radius * 0.04);
    ctx.moveTo(0, -radius * 0.24);
    ctx.quadraticCurveTo(-radius * 0.20, -radius * 0.36, -radius * 0.24, -radius * 0.20);
    ctx.moveTo(0, -radius * 0.24);
    ctx.quadraticCurveTo(radius * 0.20, -radius * 0.36, radius * 0.24, -radius * 0.20);
    ctx.stroke();

    ctx.fillStyle = '#fef9c3';
    ctx.beginPath();
    ctx.moveTo(0, radius * 0.44);
    ctx.lineTo(-radius * 0.38, radius * 0.30);
    ctx.lineTo(-radius * 0.34, radius * 0.52);
    ctx.lineTo(0, radius * 0.62);
    ctx.lineTo(radius * 0.34, radius * 0.52);
    ctx.lineTo(radius * 0.38, radius * 0.30);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.restore();
  }

  /**
   * Creates the complete shared True-Color PBR Material Bank (<= 21 Draw Calls total).
   */
  static createComplexMaterialBank(sharedUniforms) {
    const inject = (mat, opts) => ProceduralTextureFactory.injectOmniScaleSurfaceShader(mat, sharedUniforms, opts);

    // 1. Authentic Glazed Dark Brownish-Terracotta Ceramic S-Profile Roof Tiles (ref_084, ref_184, ref_174)
    const roofColC = this._makeCanvas(512, 512);
    const roofBmpC = this._makeCanvas(512, 512);
    {
      const cx = roofColC.getContext('2d');
      const bx = roofBmpC.getContext('2d');
      cx.fillStyle = '#6c3428';
      cx.fillRect(0, 0, 512, 512);
      bx.fillStyle = '#808080';
      bx.fillRect(0, 0, 512, 512);

      const cols = 16; // Tighter, authentic ceramic tile wave profile
      const rows = 12;
      const cw = 512 / cols;
      const rh = 512 / rows;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const x0 = c * cw;
          const y0 = r * rh;
          const grad = cx.createLinearGradient(x0, y0, x0 + cw, y0);
          const jitter = ((r * 11 + c * 17) % 13) - 6;
          grad.addColorStop(0.00, `rgb(${84 + jitter}, ${38 + Math.round(jitter * 0.5)}, ${29 + Math.round(jitter * 0.4)})`);
          grad.addColorStop(0.38, `rgb(${136 + jitter}, ${63 + Math.round(jitter * 0.6)}, ${47 + Math.round(jitter * 0.5)})`);
          grad.addColorStop(0.72, `rgb(${114 + jitter}, ${51 + Math.round(jitter * 0.5)}, ${38 + Math.round(jitter * 0.4)})`);
          grad.addColorStop(1.00, `rgb(${72 + jitter}, ${31 + Math.round(jitter * 0.4)}, ${24 + Math.round(jitter * 0.3)})`);
          cx.fillStyle = grad;
          cx.fillRect(x0, y0, cw, rh - 2);
          // Overlapping bottom tile lip shadow
          cx.fillStyle = 'rgba(28, 11, 8, 0.52)';
          cx.fillRect(x0, y0 + rh - 3, cw, 3);

          const bGrad = bx.createLinearGradient(x0, y0, x0 + cw, y0);
          bGrad.addColorStop(0.0, '#303030');
          bGrad.addColorStop(0.42, '#eaeaea');
          bGrad.addColorStop(1.0, '#282828');
          bx.fillStyle = bGrad;
          bx.fillRect(x0, y0, cw, rh - 2);
          bx.fillStyle = '#101010';
          bx.fillRect(x0, y0 + rh - 3, cw, 3);
        }
      }
    }
    const terracottaRoof = inject(new THREE.MeshStandardMaterial({
      name: 'TerracottaRoof',
      map: this._toTexture(roofColC, true, 1, 1),
      bumpMap: this._toTexture(roofBmpC, false, 1, 1),
      bumpScale: 0.055,
      roughness: 0.44, // Glazed ceramic tile sheen under 14:00 WIB sun
      metalness: 0.06
    }), { isRoof: true });

    // 2. Warm Cream Stucco (DPPD Gable Tympanum, Columns, Loggia Walls — ref_084, ref_184)
    const creamC = this._makeCanvas(512, 512);
    const creamB = this._makeCanvas(512, 512);
    {
      const cx = creamC.getContext('2d');
      const bx = creamB.getContext('2d');
      cx.fillStyle = '#ece5d5';
      cx.fillRect(0, 0, 512, 512);
      bx.fillStyle = '#808080';
      bx.fillRect(0, 0, 512, 512);
      for (let i = 0; i < 4200; i++) {
        const x = (i * 73) % 512;
        const y = (i * 137) % 512;
        const v = 224 + (i % 18);
        cx.fillStyle = `rgba(${v},${v - 6},${v - 16},0.32)`;
        cx.fillRect(x, y, 2, 2);
        bx.fillStyle = i % 2 === 0 ? '#929292' : '#6e6e6e';
        bx.fillRect(x, y, 2, 2);
      }
    }
    const creamStucco = inject(new THREE.MeshStandardMaterial({
      name: 'CreamStucco',
      map: this._toTexture(creamC, true, 1, 1),
      bumpMap: this._toTexture(creamB, false, 1, 1),
      bumpScale: 0.012,
      roughness: 0.82,
      metalness: 0.02
    }), { isMasonry: true });

    // 3. Cool Architectural Light-Grey / White ACP & Concrete (STP-UI L4-L8, ILRC Cornice & Porticos, ITTC Grid — ref_111, ref_164)
    const whiteC = this._makeCanvas(512, 512);
    {
      const cx = whiteC.getContext('2d');
      cx.fillStyle = '#e8ecf1';
      cx.fillRect(0, 0, 512, 512);
      for (let i = 0; i < 2400; i++) {
        const x = (i * 97) % 512;
        const y = (i * 151) % 512;
        cx.fillStyle = i % 2 === 0 ? 'rgba(255,255,255,0.22)' : 'rgba(203,213,225,0.22)';
        cx.fillRect(x, y, 3, 3);
      }
      // Subtle ACP cladding hairlines
      cx.strokeStyle = 'rgba(148, 163, 184, 0.28)';
      cx.lineWidth = 1.5;
      for (let x = 0; x <= 512; x += 128) {
        cx.beginPath(); cx.moveTo(x, 0); cx.lineTo(x, 512); cx.stroke();
      }
      for (let y = 0; y <= 512; y += 128) {
        cx.beginPath(); cx.moveTo(0, y); cx.lineTo(512, y); cx.stroke();
      }
    }
    const architecturalWhite = inject(new THREE.MeshStandardMaterial({
      name: 'ArchitecturalWhite',
      map: this._toTexture(whiteC, true, 1, 1),
      roughness: 0.62,
      metalness: 0.08
    }), { isMasonry: true });

    // 4. True-Color Weathered Burnt-Terracotta Plaster / Fine Masonry (ILRC Atrium Flanks, 6-Story Tower, DPPD NW Gable & Stairs — ref_029, ref_076, ref_084)
    const brickC = this._makeCanvas(512, 512);
    const brickB = this._makeCanvas(512, 512);
    {
      const cx = brickC.getContext('2d');
      const bx = brickB.getContext('2d');
      cx.fillStyle = '#9e4f42';
      cx.fillRect(0, 0, 512, 512);
      bx.fillStyle = '#808080';
      bx.fillRect(0, 0, 512, 512);

      // Fine micro-brick & terracotta plaster grain (no oversized LEGO blocks!)
      const rows = 32;
      const cols = 16;
      const rh = 512 / rows;
      const cw = 512 / cols;
      for (let r = 0; r < rows; r++) {
        const off = (r % 2) * (cw * 0.5);
        for (let c = -1; c <= cols; c++) {
          const j = ((r * 13 + c * 29) & 15) - 7;
          cx.fillStyle = `rgb(${158 + j}, ${79 + Math.round(j * 0.55)}, ${66 + Math.round(j * 0.45)})`;
          cx.fillRect(c * cw + off + 0.8, r * rh + 0.8, cw - 1.4, rh - 1.4);
          bx.fillStyle = '#9a9a9a';
          bx.fillRect(c * cw + off + 0.8, r * rh + 0.8, cw - 1.4, rh - 1.4);
        }
      }
      // Subtle plaster wash overlay so from medium distance it reads as smooth weathered terracotta architectural wall (ref_076)
      cx.fillStyle = 'rgba(156, 78, 65, 0.52)';
      cx.fillRect(0, 0, 512, 512);
    }
    const terracottaBrick = inject(new THREE.MeshStandardMaterial({
      name: 'TerracottaBrick',
      map: this._toTexture(brickC, true, 1, 1),
      bumpMap: this._toTexture(brickB, false, 1, 1),
      bumpScale: 0.016,
      roughness: 0.78,
      metalness: 0.03
    }), { isMasonry: true });

    // 5. Metallic Brushed Copper-Bronze ACP (STP-UI L2-L3 Trapezoids, Corner Blades, Louver Fins & Crown — ref_111, ref_164)
    const copC = this._makeCanvas(512, 512);
    {
      const cx = copC.getContext('2d');
      const g = cx.createLinearGradient(0, 0, 512, 512);
      g.addColorStop(0.00, '#a86d49');
      g.addColorStop(0.35, '#c48860');
      g.addColorStop(0.68, '#b37750');
      g.addColorStop(1.00, '#8f5a38');
      cx.fillStyle = g;
      cx.fillRect(0, 0, 512, 512);
      cx.strokeStyle = 'rgba(52, 28, 15, 0.26)';
      cx.lineWidth = 2;
      for (let y = 0; y < 512; y += 128) {
        cx.beginPath(); cx.moveTo(0, y); cx.lineTo(512, y); cx.stroke();
      }
    }
    const copperACP = inject(new THREE.MeshStandardMaterial({
      name: 'CopperBronzeACP',
      map: this._toTexture(copC, true, 1, 1),
      roughness: 0.34,
      metalness: 0.62,
      envMapIntensity: 1.25
    }));

    // 6. Anodized Silver-Grey Horizontal Sun-Louvers (ILRC East & South L2-L5 Louver Screens — ref_009, ref_029)
    const louvC = this._makeCanvas(512, 512);
    const louvB = this._makeCanvas(512, 512);
    {
      const cx = louvC.getContext('2d');
      const bx = louvB.getContext('2d');
      cx.fillStyle = '#47505a';
      cx.fillRect(0, 0, 512, 512);
      bx.fillStyle = '#383838';
      bx.fillRect(0, 0, 512, 512);
      const slats = 36;
      const sh = 512 / slats;
      for (let i = 0; i < slats; i++) {
        const y = i * sh;
        const g = cx.createLinearGradient(0, y, 0, y + sh);
        g.addColorStop(0.00, '#d8e0e8');
        g.addColorStop(0.50, '#adb7c2');
        g.addColorStop(0.82, '#49525e');
        g.addColorStop(1.00, '#252b33');
        cx.fillStyle = g;
        cx.fillRect(0, y, 512, sh);
        bx.fillStyle = '#e5e5e5';
        bx.fillRect(0, y, 512, sh * 0.62);
      }
      // Vertical mullion sub-frames
      cx.fillStyle = '#dfe5ec';
      cx.fillRect(0, 0, 7, 512);
      cx.fillRect(168, 0, 5, 512);
      cx.fillRect(338, 0, 5, 512);
    }
    const silverLouver = inject(new THREE.MeshStandardMaterial({
      name: 'SilverHorizontalLouver',
      map: this._toTexture(louvC, true, 1, 1),
      bumpMap: this._toTexture(louvB, false, 1, 1),
      bumpScale: 0.038,
      roughness: 0.38,
      metalness: 0.56
    }));

    // 7. Reflective Architectural Window Glass + True #000000 Passive Night Emissive Mask (Hukum Emas 16)
    const winColC = this._makeCanvas(512, 512);
    const winEmiC = this._makeCanvas(512, 512);
    {
      const cx = winColC.getContext('2d');
      const ex = winEmiC.getContext('2d');
      ex.fillStyle = '#000000';
      ex.fillRect(0, 0, 512, 512);

      cx.fillStyle = '#182635';
      cx.fillRect(0, 0, 512, 512);

      const cols = 4;
      const rows = 2;
      const cw = 512 / cols;
      const rh = 512 / rows;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const x = c * cw + 6;
          const y = r * rh + 6;
          const w = cw - 12;
          const h = rh - 12;
          const g = cx.createLinearGradient(x, y, x + w * 0.6, y + h);
          const hasBlind = (r + c * 3) % 4 === 0;
          if (hasBlind) {
            g.addColorStop(0.0, '#6c7d8c');
            g.addColorStop(0.5, '#4b5d6b');
            g.addColorStop(1.0, '#2e3d49');
          } else {
            g.addColorStop(0.0, '#2e4c63');
            g.addColorStop(0.45, '#1d3346');
            g.addColorStop(1.0, '#12202d');
          }
          cx.fillStyle = g;
          cx.fillRect(x, y, w, h);

          // Subtle interior Venetian blind slats on upper half of some panes
          if (hasBlind) {
            cx.fillStyle = 'rgba(218, 228, 238, 0.25)';
            for (let by = y + 6; by < y + h * 0.55; by += 8) {
              cx.fillRect(x + 4, by, w - 8, 3);
            }
          }

          // Horizontal awning transom bar at 68% height
          cx.fillStyle = '#131a22';
          cx.fillRect(x, y + h * 0.68, w, 6);

          if ((c + r * 2) % 3 !== 1) {
            ex.fillStyle = (c + r) % 2 === 0 ? '#ffdfa8' : '#d8ecff';
            ex.fillRect(x + 4, y + 4, w - 8, h * 0.62);
          }
        }
      }
      cx.strokeStyle = '#141b22';
      cx.lineWidth = 10;
      for (let c = 0; c <= cols; c++) {
        cx.beginPath(); cx.moveTo(c * cw, 0); cx.lineTo(c * cw, 512); cx.stroke();
      }
      for (let r = 0; r <= rows; r++) {
        cx.beginPath(); cx.moveTo(0, r * rh); cx.lineTo(512, r * rh); cx.stroke();
      }
    }
    const windowGlass = inject(new THREE.MeshStandardMaterial({
      name: 'ArchitecturalWindowGlass',
      map: this._toTexture(winColC, true, 1, 1),
      emissive: new THREE.Color(0xffffff),
      emissiveMap: this._toTexture(winEmiC, true, 1, 1),
      emissiveIntensity: 0.0,
      roughness: 0.11,
      metalness: 0.78,
      envMapIntensity: 1.55
    }));

    // 8. Weathered Batu-Kali Rubble Stone Retaining Wall (DPPD Toe Wall, Cheek Walls, & Perimeter Ditch — ref_084, ref_184)
    const stoneC = this._makeCanvas(512, 512);
    const stoneB = this._makeCanvas(512, 512);
    {
      const cx = stoneC.getContext('2d');
      const bx = stoneB.getContext('2d');
      cx.fillStyle = '#7c7e78';
      cx.fillRect(0, 0, 512, 512);
      bx.fillStyle = '#383838';
      bx.fillRect(0, 0, 512, 512);

      for (let i = 0; i < 220; i++) {
        const px = ((i * 67) % 480) + 16;
        const py = ((i * 109) % 480) + 16;
        const rx = 16 + (i % 16);
        const ry = 12 + ((i * 5) % 12);
        const shade = 78 + ((i * 23) % 52);
        cx.fillStyle = `rgb(${shade + 3}, ${shade + 2}, ${shade - 4})`;
        cx.beginPath();
        cx.ellipse(px, py, rx, ry, (i % 7) * 0.35, 0, Math.PI * 2);
        cx.fill();
        cx.strokeStyle = '#9fa39e';
        cx.lineWidth = 2.8;
        cx.stroke();

        bx.fillStyle = '#d8d8d8';
        bx.beginPath();
        bx.ellipse(px, py, rx - 2, ry - 2, (i % 7) * 0.35, 0, Math.PI * 2);
        bx.fill();
      }
    }
    const stoneRetaining = inject(new THREE.MeshStandardMaterial({
      name: 'StoneRetainingBatuKali',
      map: this._toTexture(stoneC, true, 1, 1),
      bumpMap: this._toTexture(stoneB, false, 1, 1),
      bumpScale: 0.055,
      roughness: 0.88,
      metalness: 0.02
    }), { isMasonry: true });

    // 9. Koral Sikat Washed-Pebble Bridge Ramp with Concrete Grid Bands (ILRC Atrium Entrance — ref_029)
    const koralC = this._makeCanvas(512, 512);
    {
      const cx = koralC.getContext('2d');
      cx.fillStyle = '#8f634f';
      cx.fillRect(0, 0, 512, 512);
      for (let i = 0; i < 4200; i++) {
        const x = (i * 61) % 512;
        const y = (i * 113) % 512;
        cx.fillStyle = i % 3 === 0 ? 'rgba(195,152,126,0.48)' : (i % 3 === 1 ? 'rgba(88,56,42,0.48)' : 'rgba(145,104,82,0.45)');
        cx.fillRect(x, y, 3, 3);
      }
      cx.strokeStyle = '#cbd2d9';
      cx.lineWidth = 22;
      cx.strokeRect(0, 0, 512, 256);
      cx.strokeRect(0, 256, 512, 256);
    }
    const koralSikat = inject(new THREE.MeshStandardMaterial({
      name: 'KoralSikatRamp',
      map: this._toTexture(koralC, true, 1, 1),
      roughness: 0.86,
      metalness: 0.02
    }), { isTerrain: true });

    // 10. Interlocking Grey Concrete Paving Block & Weathered Concrete
    const pavC = this._makeCanvas(512, 512);
    {
      const cx = pavC.getContext('2d');
      cx.fillStyle = '#969ca3';
      cx.fillRect(0, 0, 512, 512);
      for (let i = 0; i < 2000; i++) {
        const x = (i * 53) % 512;
        const y = (i * 97) % 512;
        cx.fillStyle = i % 2 === 0 ? 'rgba(120,126,134,0.35)' : 'rgba(175,180,186,0.35)';
        cx.fillRect(x, y, 3, 3);
      }
      cx.strokeStyle = '#68707a';
      cx.lineWidth = 2.5;
      for (let y = 0; y < 512; y += 32) {
        const off = ((y / 32) % 2) * 32;
        for (let x = -32; x < 512; x += 64) {
          cx.strokeRect(x + off, y, 64, 32);
        }
      }
    }
    const pavingBlock = inject(new THREE.MeshStandardMaterial({
      name: 'GreyPavingBlock',
      map: this._toTexture(pavC, true, 1, 1),
      roughness: 0.84,
      metalness: 0.02
    }), { isMasonry: true });

    // 11. True-Color Weathered Tropical Asphalt Road (ref_144, ref_076, ref_203 — warm medium-dark asphalt with subtle wheel tracks)
    const aspC = this._makeCanvas(512, 512);
    const aspB = this._makeCanvas(512, 512);
    {
      const cx = aspC.getContext('2d');
      const bx = aspB.getContext('2d');
      cx.fillStyle = '#474b51';
      cx.fillRect(0, 0, 512, 512);
      bx.fillStyle = '#808080';
      bx.fillRect(0, 0, 512, 512);
      for (let i = 0; i < 6500; i++) {
        const x = (i * 47) % 512;
        const y = (i * 89) % 512;
        const g = 58 + (i % 26);
        cx.fillStyle = `rgb(${g + 2},${g + 2},${g})`;
        cx.fillRect(x, y, 2, 2);
        bx.fillStyle = i % 2 === 0 ? '#a0a0a0' : '#606060';
        bx.fillRect(x, y, 2, 2);
      }
    }
    const asphaltRoad = inject(new THREE.MeshStandardMaterial({
      name: 'AsphaltRoad',
      map: this._toTexture(aspC, true, 1, 1),
      bumpMap: this._toTexture(aspB, false, 1, 1),
      bumpScale: 0.015,
      roughness: 0.86,
      metalness: 0.02
    }));

    // 12. Black-and-White Striped Roundabout Concrete Curb (ref_195, ref_144)
    const curbC = this._makeCanvas(256, 64);
    {
      const cx = curbC.getContext('2d');
      cx.fillStyle = '#e8ecf0';
      cx.fillRect(0, 0, 256, 64);
      cx.fillStyle = '#1e2228';
      cx.fillRect(0, 0, 128, 64);
    }
    const stripedCurb = inject(new THREE.MeshStandardMaterial({
      name: 'BlackWhiteCurb',
      map: this._toTexture(curbC, true, 1, 1),
      roughness: 0.78,
      metalness: 0.02
    }));

    // 13. Natural UI Depok Tropical Grass Turf (modulated by World-Space FBM in shader for olive/dry-grass realism — ref_084, ref_195)
    const grassC = this._makeCanvas(512, 512);
    {
      const cx = grassC.getContext('2d');
      cx.fillStyle = '#4f6d33';
      cx.fillRect(0, 0, 512, 512);
      for (let i = 0; i < 7000; i++) {
        const x = (i * 59) % 512;
        const y = (i * 101) % 512;
        cx.fillStyle = i % 4 === 0 ? '#5c7c3c' : (i % 4 === 1 ? '#415b29' : (i % 4 === 2 ? '#6b783e' : '#556e35'));
        cx.fillRect(x, y, 2, 3);
      }
    }
    const lushGrass = inject(new THREE.MeshStandardMaterial({
      name: 'TropicalGrassTurf',
      map: this._toTexture(grassC, true, 1, 1),
      roughness: 0.92,
      metalness: 0.0
    }), { isTerrain: true });

    // 14. South Dry Brown Leaf-Litter Mound & Tree Bark (ref_029, ref_033)
    const leafC = this._makeCanvas(512, 512);
    {
      const cx = leafC.getContext('2d');
      cx.fillStyle = '#65442e';
      cx.fillRect(0, 0, 512, 512);
      for (let i = 0; i < 5500; i++) {
        const x = (i * 43) % 512;
        const y = (i * 97) % 512;
        cx.fillStyle = i % 3 === 0 ? '#7e553a' : (i % 3 === 1 ? '#4c3220' : '#735c36');
        cx.fillRect(x, y, 3, 2);
      }
    }
    const dryLeafSoil = inject(new THREE.MeshStandardMaterial({
      name: 'DryLeafLitterSoil',
      map: this._toTexture(leafC, true, 1, 1),
      roughness: 0.94,
      metalness: 0.0
    }), { isTerrain: true });

    // 15. Dark Bronze/Charcoal Metal (Bargeboard Trim, Window Frames, Open Iron Gates, Lamp Posts)
    const darkMetal = inject(new THREE.MeshStandardMaterial({
      name: 'DarkBronzeMetal',
      color: new THREE.Color(0x282524),
      roughness: 0.42,
      metalness: 0.62
    }));

    // 16. UI Gold/Brass & Yellow Road Marking Material
    const goldAccent = inject(new THREE.MeshStandardMaterial({
      name: 'UIGoldBrassAndRoadYellow',
      color: new THREE.Color(0xe2a818),
      roughness: 0.38,
      metalness: 0.45
    }));

    // 17. Tropical Foliage Canopy Material (with leaf-cluster normal & sun-scatter shader)
    const foliageC = this._makeCanvas(256, 256);
    {
      const cx = foliageC.getContext('2d');
      cx.fillStyle = '#ffffff';
      cx.fillRect(0, 0, 256, 256);
      for (let i = 0; i < 3200; i++) {
        const x = (i * 37) % 256;
        const y = (i * 83) % 256;
        cx.fillStyle = i % 3 === 0 ? 'rgba(215,235,195,0.45)' : (i % 3 === 1 ? 'rgba(155,180,145,0.45)' : 'rgba(245,252,225,0.4)');
        cx.beginPath();
        cx.arc(x, y, 2.5, 0, Math.PI * 2);
        cx.fill();
      }
    }
    const foliageCanopy = inject(new THREE.MeshStandardMaterial({
      name: 'TropicalFoliageCanopy',
      map: this._toTexture(foliageC, true, 2, 2),
      color: new THREE.Color(0xffffff),
      roughness: 0.78,
      metalness: 0.01
    }), { isFoliage: true });

    // 18. Single Unified 2048x2048 Campus Signage, ADA Stall, Solar PV & Totem Atlas (1 Single Draw Call!)
    const signAtlasMat = inject(this.createUnifiedSignageAtlasMaterial());

    return {
      terracottaRoof,
      creamStucco,
      architecturalWhite,
      terracottaBrick,
      copperACP,
      silverLouver,
      windowGlass,
      stoneRetaining,
      terracottaPlaza: terracottaBrick,
      koralSikat,
      pavingBlock,
      asphaltRoad,
      stripedCurb,
      yellowLine: goldAccent,
      adaBlueBay: signAtlasMat,
      lushGrass,
      dryLeafSoil,
      darkMetal,
      warmTaupeSoffit: copperACP,
      goldAccent,
      redHydrant: terracottaRoof,
      bambooWood: dryLeafSoil,
      palmTrunk: stoneRetaining,
      foliageCanopy,
      solarPV: signAtlasMat,
      plinthEarthSkirt: asphaltRoad,
      signAtlasMat
    };
  }

  /**
   * Packs ALL campus signage, 3D typography plaques, Makara logos, KAN accreditation, totems, gate signs,
   * Blue ADA Wheelchair Parking Stalls, Rooftop Solar PV arrays, Roundabout Traffic Signs, and Hanging Banners
   * into ONE crisp 2048x2048 UV Atlas (1024x1024 logical coordinates scaled 2x) -> 1 SINGLE DRAW CALL!
   */
  static createUnifiedSignageAtlasMaterial() {
    const c = this._makeCanvas(2048, 2048);
    const ctx = c.getContext('2d');
    ctx.scale(2, 2); // Logical coordinate space remains 0..1024 so all existing UV coordinates work seamlessly at 2x retina sharpness!

    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, 1024, 1024);

    // Tile 0 (y: 0..160): STP-UI Top L8 Light-Grey Louvered Frieze + 3D Gold UI Makara + 2-Row White 3D Channel Letters (ref_111 — NO BLACK BOX!)
    const stpFriezeGrad = ctx.createLinearGradient(0, 0, 0, 160);
    stpFriezeGrad.addColorStop(0.00, '#abb4bf'); // natural shadow right under the copper roof overhang
    stpFriezeGrad.addColorStop(0.25, '#cfd6df'); // architectural light grey facade
    stpFriezeGrad.addColorStop(1.00, '#dfe5ec');
    ctx.fillStyle = stpFriezeGrad;
    ctx.fillRect(0, 0, 1024, 160);
    // Subtle horizontal louver/cladding lines behind the 3D letters (ref_111)
    ctx.strokeStyle = 'rgba(100, 112, 126, 0.35)';
    ctx.lineWidth = 2;
    for (let y = 12; y < 156; y += 10) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(1024, y); ctx.stroke();
    }
    for (let x = 0; x < 1024; x += 128) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 160); ctx.stroke();
    }
    this.drawUIMakaraEmblem(ctx, 92, 82, 56, '#f59e0b', '#451a03');
    // 3D Extruded Drop-Shadow + Crisp White Channel Letters
    ctx.textAlign = 'left';
    ctx.font = '900 52px "Inter", "Segoe UI", Arial, sans-serif';
    ctx.fillStyle = '#334155';
    ctx.fillText('SCIENCE TECHNO PARK', 177, 75);
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 2.5;
    ctx.strokeText('SCIENCE TECHNO PARK', 174, 72);
    ctx.fillText('SCIENCE TECHNO PARK', 174, 72);

    ctx.font = '800 47px "Inter", "Segoe UI", Arial, sans-serif';
    ctx.fillStyle = '#334155';
    ctx.fillText('UNIVERSITAS INDONESIA', 177, 135);
    ctx.fillStyle = '#ffffff';
    ctx.strokeText('UNIVERSITAS INDONESIA', 174, 132);
    ctx.fillText('UNIVERSITAS INDONESIA', 174, 132);

    // Tile 1 (y: 160..272): STP-UI L1 Silver Portico Fascia (ref_094, ref_164)
    const portGrad = ctx.createLinearGradient(0, 160, 0, 272);
    portGrad.addColorStop(0, '#eef2f6');
    portGrad.addColorStop(1, '#d5dce4');
    ctx.fillStyle = portGrad;
    ctx.fillRect(0, 160, 1024, 112);
    ctx.strokeStyle = '#b8c2ce';
    ctx.lineWidth = 3;
    ctx.strokeRect(2, 162, 1020, 108);
    ctx.fillStyle = '#1e293b';
    ctx.font = '800 35px "Inter", "Segoe UI", Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('SCIENCE TECHNO PARK UNIVERSITAS INDONESIA', 512, 228);

    // Tile 2 (y: 272..416): ILRC Atrium White Portal Beam (UI Makara + KAN + 2-line title, ref_024, ref_029)
    ctx.fillStyle = '#f1f4f8';
    ctx.fillRect(0, 272, 1024, 144);
    this.drawUIMakaraEmblem(ctx, 370, 310, 26, '#eab308', '#713f12');
    ctx.fillStyle = '#1d4ed8';
    ctx.font = '900 24px Arial, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('KAN', 605, 314);
    ctx.fillStyle = '#334155';
    ctx.font = '600 10px Arial, sans-serif';
    ctx.fillText('Komite Akreditasi Nasional', 580, 328);

    ctx.fillStyle = '#0f172a';
    ctx.textAlign = 'center';
    ctx.font = '800 29px "Inter", "Segoe UI", Arial, sans-serif';
    ctx.fillText('INTEGRATED LABORATORY AND RESEARCH CENTER', 512, 368);
    ctx.font = '700 26px "Inter", "Segoe UI", Arial, sans-serif';
    ctx.fillText('UNIVERSITAS INDONESIA', 512, 402);

    // Tile 3 (y: 416..528): DPPD L1 Entrance Cream Wall Lettering ("GEDUNG / KSDP UI" + DPPD plaque, ref_084, ref_184)
    ctx.fillStyle = '#ece5d5';
    ctx.fillRect(0, 416, 1024, 112);
    ctx.fillStyle = '#292524';
    ctx.textAlign = 'center';
    ctx.font = '800 38px "Inter", Arial, sans-serif';
    ctx.fillText('GEDUNG', 256, 464);
    ctx.fillText('KSDP UI', 256, 508);
    ctx.fillStyle = '#5c1d18';
    ctx.fillRect(530, 428, 474, 88);
    ctx.strokeStyle = '#eab308';
    ctx.lineWidth = 3;
    ctx.strokeRect(534, 432, 466, 80);
    this.drawUIMakaraEmblem(ctx, 578, 472, 24, '#facc15', '#451a03');
    ctx.fillStyle = '#fef08a';
    ctx.textAlign = 'left';
    ctx.font = '800 22px Arial, sans-serif';
    ctx.fillText('GEDUNG DPPD UI', 618, 466);
    ctx.fillStyle = '#f8fafc';
    ctx.font = '600 15px Arial, sans-serif';
    ctx.fillText('Direktorat Pengembangan Pendidikan Digital', 618, 494);

    // Tile 4 (x: 0..340, y: 528..1024): STP-UI Black/Yellow Directory Totem (ref_105, ref_156)
    ctx.fillStyle = '#181b20';
    ctx.fillRect(0, 528, 340, 496);
    this.drawUIMakaraEmblem(ctx, 64, 586, 26, '#facc15', '#451a03');
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'left';
    ctx.font = '800 34px Arial, sans-serif';
    ctx.fillText('STP UI', 108, 596);
    ctx.font = '600 14px Arial, sans-serif';
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText('SCIENCE TECHNO PARK', 108, 616);
    for (let r = 0; r < 7; r++) {
      const py = 648 + r * 34;
      ctx.fillStyle = '#f1f5f9';
      ctx.fillRect(24, py, 138, 22);
      ctx.fillRect(178, py, 138, 22);
      ctx.fillStyle = '#334155';
      ctx.fillRect(32, py + 8, 96, 6);
      ctx.fillRect(186, py + 8, 104, 6);
    }
    ctx.fillStyle = '#eab308';
    ctx.beginPath();
    ctx.moveTo(0, 930);
    ctx.lineTo(170, 966);
    ctx.lineTo(340, 930);
    ctx.lineTo(340, 1024);
    ctx.lineTo(0, 1024);
    ctx.closePath();
    ctx.fill();

    // Tile 5 (x: 340..680, y: 528..1024): DPPD Brushed Gold/Brass UI-Huawei Training Center Totem (ref_184)
    const bgBrass = ctx.createLinearGradient(340, 528, 680, 1024);
    bgBrass.addColorStop(0, '#e5c98b');
    bgBrass.addColorStop(0.5, '#d4b26a');
    bgBrass.addColorStop(1, '#b89246');
    ctx.fillStyle = bgBrass;
    ctx.fillRect(340, 528, 340, 496);
    ctx.fillStyle = '#facc15';
    ctx.beginPath();
    ctx.moveTo(340, 528);
    ctx.lineTo(480, 528);
    ctx.lineTo(340, 610);
    ctx.closePath();
    ctx.fill();
    this.drawUIMakaraEmblem(ctx, 435, 620, 32, '#ca8a04', '#451a03');
    ctx.fillStyle = '#dc2626';
    for (let p = 0; p < 8; p++) {
      const a = -Math.PI * 0.88 + (p / 7) * Math.PI * 0.76;
      ctx.save();
      ctx.translate(585, 628);
      ctx.rotate(a + Math.PI * 0.5);
      ctx.beginPath();
      ctx.ellipse(0, -18, 5, 14, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    ctx.fillStyle = '#1c1917';
    ctx.textAlign = 'center';
    ctx.font = '800 18px Arial, sans-serif';
    ctx.fillText('HUAWEI', 585, 654);
    ctx.font = '800 22px Arial, sans-serif';
    ctx.fillText('UNIVERSITY OF INDONESIA', 510, 726);
    ctx.fillText('HUAWEI TECHNOLOGIES', 510, 762);
    ctx.fillText('TRAINING CENTER', 510, 798);
    ctx.fillStyle = '#27272a';
    ctx.fillRect(340, 960, 340, 64);

    // Tile 6A (x: 680..852, y: 528..776): DRPM "P PARKIR KHUSUS PIMPINAN" Sign (ref_012)
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(680, 528, 172, 248);
    ctx.fillStyle = '#1d4ed8';
    ctx.fillRect(692, 540, 148, 132);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;
    ctx.strokeRect(700, 548, 132, 116);
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.font = '900 84px Arial, sans-serif';
    ctx.fillText('P', 766, 636);
    ctx.fillStyle = '#0f172a';
    ctx.font = '800 16px Arial, sans-serif';
    ctx.fillText('PARKIR KHUSUS', 766, 710);
    ctx.fillText('PIMPINAN', 766, 734);

    // Tile 6B (x: 852..1024, y: 528..652): Planter Topiary "STP-UI" on Lush Hedge Green + Brass 3D Letters (ref_164)
    ctx.fillStyle = '#3f6229';
    ctx.fillRect(852, 528, 172, 124);
    ctx.fillStyle = '#d4a338';
    ctx.strokeStyle = '#fef08a';
    ctx.lineWidth = 2;
    ctx.font = '900 40px Arial, sans-serif';
    ctx.strokeText('STP-UI', 938, 604);
    ctx.fillText('STP-UI', 938, 604);

    // Tile 6C (x: 852..938, y: 652..776): Blue ADA Wheelchair Parking Stall w/ White Hatching & Symbol (ref_184, ref_029)
    ctx.fillStyle = '#2563eb';
    ctx.fillRect(852, 652, 86, 124);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.strokeRect(855, 655, 80, 118);
    // Top white diagonal hatching strip (matching ref_184!)
    ctx.strokeRect(855, 655, 80, 24);
    for (let hx = 858; hx < 932; hx += 10) {
      ctx.beginPath(); ctx.moveTo(hx, 679); ctx.lineTo(hx + 8, 655); ctx.stroke();
    }
    // Wheelchair icon in white
    ctx.beginPath();
    ctx.arc(892, 730, 14, 0.3, Math.PI * 1.65);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(895, 702, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(895, 708);
    ctx.lineTo(895, 726);
    ctx.lineTo(910, 726);
    ctx.lineTo(916, 742);
    ctx.stroke();

    // Tile 6D (x: 938..1024, y: 652..776): Rooftop Solar PV Silicon Cell Array (ref_174) & Blue "-> MASUK" Sign (ref_203)
    ctx.fillStyle = '#152c4a';
    ctx.fillRect(938, 652, 86, 64);
    ctx.strokeStyle = '#93c5fd';
    ctx.lineWidth = 2;
    ctx.strokeRect(940, 654, 82, 60);
    ctx.lineWidth = 1;
    for (let sx = 952; sx < 1020; sx += 14) {
      ctx.beginPath(); ctx.moveTo(sx, 654); ctx.lineTo(sx, 714); ctx.stroke();
    }
    for (let sy = 668; sy < 712; sy += 14) {
      ctx.beginPath(); ctx.moveTo(940, sy); ctx.lineTo(1022, sy); ctx.stroke();
    }
    // Bottom half of 6D (x: 938..1024, y: 716..776): Blue "-> MASUK" & Roundabout Sign (ref_203, ref_009)
    ctx.fillStyle = '#1d4ed8';
    ctx.fillRect(938, 716, 86, 60);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.strokeRect(941, 719, 80, 54);
    ctx.fillStyle = '#ffffff';
    ctx.font = '900 18px Arial, sans-serif';
    ctx.fillText('➔', 981, 742);
    ctx.font = '800 14px Arial, sans-serif';
    ctx.fillText('MASUK', 981, 764);

    // Tile 7 (x: 680..1024, y: 776..1024): DPPD Relief Campus Map Board (x: 680..864) + ILRC Vertical Black/Gold Banner (x: 864..960) + DPPD Hanging White/Blue Banner (x: 960..1024)
    ctx.fillStyle = '#c8a97e';
    ctx.fillRect(680, 776, 184, 248);
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 10;
    ctx.strokeRect(685, 781, 174, 238);
    ctx.strokeStyle = '#57534e';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.arc(772, 905, 52, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = '#9a3412';
    ctx.fillRect(756, 889, 32, 32);
    ctx.fillStyle = '#1c1917';
    ctx.font = '800 15px Arial, sans-serif';
    ctx.fillText('PETA KAMPUS UI', 772, 816);

    // ILRC Vertical Navy/Black Banner with Gold UI Makara (x: 864..960, y: 776..1024)
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(864, 776, 96, 248);
    this.drawUIMakaraEmblem(ctx, 912, 868, 34, '#facc15', '#713f12');
    ctx.fillStyle = '#facc15';
    ctx.font = '800 12px Arial, sans-serif';
    ctx.fillText('UNIVERSITAS', 912, 950);
    ctx.fillText('INDONESIA', 912, 968);

    // DPPD Hanging White/Blue/Yellow Cloth Banner on Right Ladder Frame (x: 960..1024, y: 776..1024 — ref_084)
    ctx.fillStyle = '#f1f5f9';
    ctx.fillRect(960, 776, 64, 248);
    ctx.fillStyle = '#1d4ed8';
    ctx.fillRect(966, 920, 52, 64);
    ctx.fillStyle = '#eab308';
    ctx.fillRect(966, 984, 52, 32);
    ctx.fillStyle = '#334155';
    ctx.font = '700 11px Arial, sans-serif';
    ctx.fillText('UI DEPOK', 992, 880);

    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    tex.needsUpdate = true;

    return new THREE.MeshStandardMaterial({
      name: 'UnifiedCampusSignageAtlas',
      map: tex,
      roughness: 0.45,
      metalness: 0.12
    });
  }

  /**
   * Helper returning a True-3D Volumetric Signage Plaque (BoxGeometry with physical thickness depthMeters = 0.048m)
   * pre-mapped to an exact logical sub-rectangle [xPx, yPx, wPx, hPx] of the Signage Atlas on front/back faces
   * and dark-bronze frame return edges on the 4 side faces (zero paper-thin 2D planes!).
   */
  static createAtlasSignPlane(widthMeters, heightMeters, xPx, yPx, wPx, hPx, depthMeters = 0.048) {
    const geo = new THREE.BoxGeometry(widthMeters, heightMeters, depthMeters);
    const u0 = (xPx + 1.5) / 1024;
    const u1 = (xPx + wPx - 1.5) / 1024;
    const vTop = 1.0 - (yPx + 1.5) / 1024;
    const vBot = 1.0 - (yPx + hPx - 1.5) / 1024;

    // Dark-slate/bronze frame edge UV region inside Tile 4 border (xPx: 8..24, yPx: 536..552)
    const eu0 = 10 / 1024;
    const eu1 = 22 / 1024;
    const evTop = 1.0 - 536 / 1024;
    const evBot = 1.0 - 552 / 1024;

    const uv = geo.attributes.uv;
    // Faces 0..3 (+X, -X, +Y, -Y return edges): dark frame border UV
    for (let f = 0; f < 4; f++) {
      const base = f * 4;
      uv.setXY(base, eu0, evTop);
      uv.setXY(base + 1, eu1, evTop);
      uv.setXY(base + 2, eu0, evBot);
      uv.setXY(base + 3, eu1, evBot);
    }
    // Face 4 (+Z front face) & Face 5 (-Z back face): crisp signage atlas rect
    for (const f of [4, 5]) {
      const base = f * 4;
      uv.setXY(base, u0, vTop);
      uv.setXY(base + 1, u1, vTop);
      uv.setXY(base + 2, u0, vBot);
      uv.setXY(base + 3, u1, vBot);
    }
    uv.needsUpdate = true;
    return geo;
  }

  /**
   * Helper returning a BoxGeometry whose top face (and all faces) map to a sub-rectangle [xPx, yPx, wPx, hPx] of the Signage Atlas.
   */
  static createAtlasBox(w, h, d, xPx, yPx, wPx, hPx) {
    const geo = new THREE.BoxGeometry(w, h, d);
    const u0 = (xPx + 2.0) / 1024;
    const u1 = (xPx + wPx - 2.0) / 1024;
    const vTop = 1.0 - (yPx + 2.0) / 1024;
    const vBot = 1.0 - (yPx + hPx - 2.0) / 1024;
    const uv = geo.attributes.uv;
    for (let i = 0; i < uv.count; i += 4) {
      uv.setXY(i, u0, vTop);
      uv.setXY(i + 1, u1, vTop);
      uv.setXY(i + 2, u0, vBot);
      uv.setXY(i + 3, u1, vBot);
    }
    uv.needsUpdate = true;
    return geo;
  }
}
