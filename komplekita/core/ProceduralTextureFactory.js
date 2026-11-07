import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./core/ProceduralTextureFactory.js
 *
 * High-Resolution Procedural PBR Material, Typography, Official UI Makara Crest,
 * PMREM Sky/Ground Environment Map, & SSoST Clay/Zebra Shader Injection Engine.
 * Strictly obeys Golden Laws 13, 14, 15, 16 (No double sRGB gamma, true #000000 passive emissive masks).
 */
export class ProceduralTextureFactory {
  static createSharedShaderUniforms() {
    return {
      uInspectMode: { value: 0 }, // 0: Realistic PBR, 1: Clay Matte, 2: SSoST Zebra Stripes
      uTime: { value: 0.0 },
      uDehazeStrength: { value: 0.14 }
    };
  }

  /**
   * Injects Contact AO, Filmic Dehaze, and Clay/Zebra SSoST Reflection Line Inspection into MeshStandardMaterial.
   */
  static injectOmniScaleSurfaceShader(material, sharedUniforms, opts = {}) {
    if (!material || material.userData.nexusShaderInjected) return material;
    material.userData.nexusShaderInjected = true;
    const isFoliage = !!opts.isFoliage;

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
          varying vec3 vNexusWorldNorm;`
        )
        .replace(
          '#include <dithering_fragment>',
          `
          // Ground-plane Contact AO (subtle grounding shadow where walls/curbs meet terrain)
          ${isFoliage ? '' : `
          float groundProx = clamp((vNexusWorldPos.y + 0.15) / 2.1, 0.0, 1.0);
          float contactAO = mix(0.82, 1.0, smoothstep(0.0, 1.0, groundProx));
          gl_FragColor.rgb *= contactAO;
          `}

          // Mild Filmic Dehaze (keeps tropical afternoon colors crisp without crushing midtones)
          gl_FragColor.rgb = mix(gl_FragColor.rgb, pow(max(gl_FragColor.rgb, vec3(0.0)), vec3(1.04)) * 1.02, uDehazeStrength);

          // SSoST Class-A Clay & Zebra Reflection Line Inspector
          if (uInspectMode == 1) {
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
            float fres = pow(1.0 - clamp( abs(dot(viewDir, nrm)), 0.0, 1.0 ), 2.0);
            vec3 zebraCol = mix(vec3(0.08, 0.11, 0.16), vec3(0.95, 0.98, 1.0), stripe);
            zebraCol += vec3(0.12, 0.36, 0.55) * fres;
            gl_FragColor.rgb = zebraCol;
          }
          #include <dithering_fragment>`
        );
    };
    return material;
  }

  /**
   * Builds a procedural 14:00 WIB UI Depok Sky + Tropical Ground Environment Map via PMREMGenerator
   * so glass windows and copper/aluminum louvers have realistic daylight reflections instead of black voids.
   */
  static createAfternoonEnvironmentMap(renderer) {
    const pmrem = new THREE.PMREMGenerator(renderer);
    pmrem.compileEquirectangularShader();

    const envScene = new THREE.Scene();
    const skyGeo = new THREE.SphereGeometry(60, 32, 20);
    const pos = skyGeo.attributes.position;
    const cols = new Float32Array(pos.count * 3);
    const cZenith = new THREE.Color(0x6da9e9);
    const cHorizon = new THREE.Color(0xeaf3fc);
    const cGround = new THREE.Color(0x4f5f43);
    const cSunGlow = new THREE.Color(0xfff7df);
    const sunDir = new THREE.Vector3(-62, 138, 32).normalize();
    const v = new THREE.Vector3();
    const tmp = new THREE.Color();

    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).normalize();
      if (v.y >= 0) {
        tmp.lerpColors(cHorizon, cZenith, Math.pow(v.y, 0.58));
        const sunDot = Math.max(0, v.dot(sunDir));
        tmp.lerp(cSunGlow, Math.pow(sunDot, 10.0) * 0.65);
      } else {
        tmp.lerpColors(cHorizon, cGround, Math.min(1.0, Math.pow(-v.y * 2.2, 0.65)));
      }
      cols[i * 3] = tmp.r;
      cols[i * 3 + 1] = tmp.g;
      cols[i * 3 + 2] = tmp.b;
    }
    skyGeo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    const skyMat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide });
    envScene.add(new THREE.Mesh(skyGeo, skyMat));

    const rt = pmrem.fromScene(envScene, 0.03);
    skyGeo.dispose();
    skyMat.dispose();
    pmrem.dispose();
    return rt.texture;
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
   * Draws an authentic stylized Golden Universitas Indonesia Makara Crest onto a 2D Canvas context.
   */
  static drawUIMakaraEmblem(ctx, cx, cy, radius, goldColor = '#eab308', strokeDark = '#713f12') {
    ctx.save();
    ctx.translate(cx, cy);

    // Outer sunburst / kala-makara flame crown
    const rays = 15;
    ctx.fillStyle = goldColor;
    ctx.strokeStyle = strokeDark;
    ctx.lineWidth = Math.max(1.2, radius * 0.045);

    ctx.beginPath();
    for (let i = 0; i <= rays * 2; i++) {
      const t = i / (rays * 2);
      const ang = -Math.PI * 0.88 + t * Math.PI * 1.76 - Math.PI * 0.5 + Math.PI * 0.5;
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

    // Inner tree-of-knowledge arch & water drop base
    ctx.fillStyle = '#fde047';
    ctx.beginPath();
    ctx.arc(0, -radius * 0.06, radius * 0.52, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Central tree trunk & 5 branches
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

    // Open book at base of Makara
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
   * Creates the complete shared Material Bank for the entire UI Depok Roundabout Complex (~20 shared materials -> <= 25 Draw Calls!).
   */
  static createComplexMaterialBank(sharedUniforms) {
    const inject = (mat, opts) => ProceduralTextureFactory.injectOmniScaleSurfaceShader(mat, sharedUniforms, opts);

    // 1. Terracotta Glazed S-Profile Roof Tiles (DPPD & ITTC Perpustakaan Lama)
    const roofColC = this._makeCanvas(256, 256);
    const roofBmpC = this._makeCanvas(256, 256);
    {
      const cx = roofColC.getContext('2d');
      const bx = roofBmpC.getContext('2d');
      cx.fillStyle = '#9e4731';
      cx.fillRect(0, 0, 256, 256);
      bx.fillStyle = '#808080';
      bx.fillRect(0, 0, 256, 256);

      const cols = 8;
      const rows = 6;
      const cw = 256 / cols;
      const rh = 256 / rows;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const x0 = c * cw;
          const y0 = r * rh;
          const grad = cx.createLinearGradient(x0, y0, x0 + cw, y0);
          const jitter = ((r * 7 + c * 13) % 9) - 4;
          grad.addColorStop(0.0, `rgb(${128 + jitter}, ${54 + jitter}, ${36 + jitter})`);
          grad.addColorStop(0.42, `rgb(${176 + jitter}, ${79 + jitter}, ${53 + jitter})`);
          grad.addColorStop(0.78, `rgb(${152 + jitter}, ${66 + jitter}, ${44 + jitter})`);
          grad.addColorStop(1.0, `rgb(${108 + jitter}, ${45 + jitter}, ${30 + jitter})`);
          cx.fillStyle = grad;
          cx.fillRect(x0, y0, cw, rh - 2);
          cx.fillStyle = 'rgba(42,16,10,0.42)';
          cx.fillRect(x0, y0 + rh - 2.5, cw, 2.5);

          const bGrad = bx.createLinearGradient(x0, y0, x0 + cw, y0);
          bGrad.addColorStop(0.0, '#454545');
          bGrad.addColorStop(0.45, '#d8d8d8');
          bGrad.addColorStop(1.0, '#383838');
          bx.fillStyle = bGrad;
          bx.fillRect(x0, y0, cw, rh - 2);
          bx.fillStyle = '#181818';
          bx.fillRect(x0, y0 + rh - 2.5, cw, 2.5);
        }
      }
    }
    const terracottaRoof = inject(new THREE.MeshStandardMaterial({
      name: 'TerracottaRoof',
      map: this._toTexture(roofColC, true, 1, 1),
      bumpMap: this._toTexture(roofBmpC, false, 1, 1),
      bumpScale: 0.045,
      roughness: 0.58,
      metalness: 0.04
    }));

    // 2. Warm Cream Stucco (DPPD Gable Tympanum, Columns, Parapets)
    const creamC = this._makeCanvas(256, 256);
    {
      const cx = creamC.getContext('2d');
      cx.fillStyle = '#efe9dc';
      cx.fillRect(0, 0, 256, 256);
      for (let i = 0; i < 1800; i++) {
        const x = (i * 73) % 256;
        const y = (i * 137) % 256;
        const v = 230 + (i % 16);
        cx.fillStyle = `rgba(${v},${v - 5},${v - 15},0.35)`;
        cx.fillRect(x, y, 2, 2);
      }
    }
    const creamStucco = inject(new THREE.MeshStandardMaterial({
      name: 'CreamStucco',
      map: this._toTexture(creamC, true, 1, 1),
      roughness: 0.84,
      metalness: 0.02
    }));

    // 3. Crisp Architectural Cool-White Concrete / ACP (STP-UI L4-L8, ILRC Sunshades & Porticos, ITTC Grid)
    const whiteC = this._makeCanvas(256, 256);
    {
      const cx = whiteC.getContext('2d');
      cx.fillStyle = '#f3f5f8';
      cx.fillRect(0, 0, 256, 256);
      cx.strokeStyle = 'rgba(180, 190, 202, 0.34)';
      cx.lineWidth = 1.5;
      for (let x = 0; x <= 256; x += 64) {
        cx.beginPath(); cx.moveTo(x, 0); cx.lineTo(x, 256); cx.stroke();
      }
      for (let y = 0; y <= 256; y += 64) {
        cx.beginPath(); cx.moveTo(0, y); cx.lineTo(256, y); cx.stroke();
      }
    }
    const architecturalWhite = inject(new THREE.MeshStandardMaterial({
      name: 'ArchitecturalWhite',
      map: this._toTexture(whiteC, true, 1, 1),
      roughness: 0.68,
      metalness: 0.06
    }));

    // 4. Salmon-Terracotta Brick / Burnt-Red Masonry (DPPD NW Gable End, ILRC Atrium Flanks & 6-Story South Tower)
    const brickC = this._makeCanvas(256, 256);
    const brickB = this._makeCanvas(256, 256);
    {
      const cx = brickC.getContext('2d');
      const bx = brickB.getContext('2d');
      cx.fillStyle = '#b65a46';
      cx.fillRect(0, 0, 256, 256);
      bx.fillStyle = '#909090';
      bx.fillRect(0, 0, 256, 256);
      const rows = 16;
      const cols = 8;
      const rh = 256 / rows;
      const cw = 256 / cols;
      for (let r = 0; r < rows; r++) {
        const off = (r % 2) * (cw * 0.5);
        for (let c = -1; c <= cols; c++) {
          const j = ((r * 11 + c * 17) & 15) - 8;
          cx.fillStyle = `rgb(${182 + j}, ${89 + Math.round(j * 0.6)}, ${69 + Math.round(j * 0.5)})`;
          cx.fillRect(c * cw + off + 1, r * rh + 1, cw - 2, rh - 2);
          bx.fillStyle = '#c8c8c8';
          bx.fillRect(c * cw + off + 1, r * rh + 1, cw - 2, rh - 2);
        }
      }
    }
    const terracottaBrick = inject(new THREE.MeshStandardMaterial({
      name: 'TerracottaBrick',
      map: this._toTexture(brickC, true, 1, 1),
      bumpMap: this._toTexture(brickB, false, 1, 1),
      bumpScale: 0.02,
      roughness: 0.78,
      metalness: 0.03
    }));

    // 5. Metallic Copper-Bronze ACP (STP-UI L2-L3 Trapezoids, End Corner Piers & Crown Parapet)
    const copC = this._makeCanvas(256, 256);
    {
      const cx = copC.getContext('2d');
      const g = cx.createLinearGradient(0, 0, 256, 256);
      g.addColorStop(0.0, '#b57852');
      g.addColorStop(0.5, '#c98d63');
      g.addColorStop(1.0, '#9e6440');
      cx.fillStyle = g;
      cx.fillRect(0, 0, 256, 256);
      cx.strokeStyle = 'rgba(62, 34, 18, 0.28)';
      cx.lineWidth = 2;
      for (let y = 0; y < 256; y += 64) {
        cx.beginPath(); cx.moveTo(0, y); cx.lineTo(256, y); cx.stroke();
      }
    }
    const copperACP = inject(new THREE.MeshStandardMaterial({
      name: 'CopperBronzeACP',
      map: this._toTexture(copC, true, 1, 1),
      roughness: 0.36,
      metalness: 0.58
    }));

    // 6. Anodized Silver-Grey Horizontal Sun-Louvers (ILRC East & South L2-L5 Louver Screens)
    const louvC = this._makeCanvas(256, 256);
    const louvB = this._makeCanvas(256, 256);
    {
      const cx = louvC.getContext('2d');
      const bx = louvB.getContext('2d');
      cx.fillStyle = '#525b66';
      cx.fillRect(0, 0, 256, 256);
      bx.fillStyle = '#404040';
      bx.fillRect(0, 0, 256, 256);
      const slats = 24;
      const sh = 256 / slats;
      for (let i = 0; i < slats; i++) {
        const y = i * sh;
        const g = cx.createLinearGradient(0, y, 0, y + sh);
        g.addColorStop(0.0, '#d8e0e8');
        g.addColorStop(0.55, '#aeb8c4');
        g.addColorStop(0.85, '#4c5561');
        g.addColorStop(1.0, '#2c333c');
        cx.fillStyle = g;
        cx.fillRect(0, y, 256, sh);

        bx.fillStyle = '#e5e5e5';
        bx.fillRect(0, y, 256, sh * 0.65);
      }
      // Subtle vertical mullion lines
      cx.fillStyle = '#e2e8f0';
      cx.fillRect(0, 0, 5, 256);
      cx.fillRect(126, 0, 4, 256);
    }
    const silverLouver = inject(new THREE.MeshStandardMaterial({
      name: 'SilverHorizontalLouver',
      map: this._toTexture(louvC, true, 1, 1),
      bumpMap: this._toTexture(louvB, false, 1, 1),
      bumpScale: 0.035,
      roughness: 0.42,
      metalness: 0.52
    }));

    // 7. Reflective Architectural Window Glass + True #000000 Passive Night Emissive Mask (Hukum Emas 16)
    const winColC = this._makeCanvas(256, 256);
    const winEmiC = this._makeCanvas(256, 256);
    {
      const cx = winColC.getContext('2d');
      const ex = winEmiC.getContext('2d');
      // Pure #000000 base on emissive mask so mullions & unlit rooms emit ZERO photons
      ex.fillStyle = '#000000';
      ex.fillRect(0, 0, 256, 256);

      cx.fillStyle = '#1e2f3f';
      cx.fillRect(0, 0, 256, 256);

      const cols = 4;
      const rows = 2;
      const cw = 256 / cols;
      const rh = 256 / rows;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const x = c * cw + 4;
          const y = r * rh + 4;
          const w = cw - 8;
          const h = rh - 8;
          const g = cx.createLinearGradient(x, y, x + w, y + h);
          const hasBlind = (r + c * 3) % 5 === 0;
          if (hasBlind) {
            g.addColorStop(0, '#5a6d7c');
            g.addColorStop(1, '#3b4c5b');
          } else {
            g.addColorStop(0, '#274156');
            g.addColorStop(0.5, '#1d3244');
            g.addColorStop(1, '#152534');
          }
          cx.fillStyle = g;
          cx.fillRect(x, y, w, h);

          // Horizontal transom bar at 30% from bottom
          cx.fillStyle = '#182028';
          cx.fillRect(x, y + h * 0.68, w, 4);

          // Selective warm night office rooms (strictly inside glass pane, #000000 elsewhere)
          if ((c + r * 2) % 3 !== 1) {
            ex.fillStyle = (c + r) % 2 === 0 ? '#ffdfa8' : '#d8ecff';
            ex.fillRect(x + 3, y + 3, w - 6, h * 0.64);
          }
        }
      }
      // Dark aluminum perimeter & mullion frames
      cx.strokeStyle = '#161d24';
      cx.lineWidth = 6;
      for (let c = 0; c <= cols; c++) {
        cx.beginPath(); cx.moveTo(c * cw, 0); cx.lineTo(c * cw, 256); cx.stroke();
      }
      for (let r = 0; r <= rows; r++) {
        cx.beginPath(); cx.moveTo(0, r * rh); cx.lineTo(256, r * rh); cx.stroke();
      }
    }
    const windowGlass = inject(new THREE.MeshStandardMaterial({
      name: 'ArchitecturalWindowGlass',
      map: this._toTexture(winColC, true, 1, 1),
      emissive: new THREE.Color(0xffffff),
      emissiveMap: this._toTexture(winEmiC, true, 1, 1),
      emissiveIntensity: 0.0, // 0.0 at 14:00 WIB daylight; raised smoothly in Night GI mode
      roughness: 0.14,
      metalness: 0.72,
      envMapIntensity: 1.35
    }));

    // 8. Rubble Stone Retaining Wall ("Batu Kali" — DPPD Toe Wall, Cheek Walls, & Perimeter Ditch)
    const stoneC = this._makeCanvas(256, 256);
    const stoneB = this._makeCanvas(256, 256);
    {
      const cx = stoneC.getContext('2d');
      const bx = stoneB.getContext('2d');
      cx.fillStyle = '#8b8d88';
      cx.fillRect(0, 0, 256, 256);
      bx.fillStyle = '#404040';
      bx.fillRect(0, 0, 256, 256);

      for (let i = 0; i < 95; i++) {
        const px = ((i * 53) % 240) + 8;
        const py = ((i * 97) % 240) + 8;
        const rx = 12 + (i % 11);
        const ry = 9 + ((i * 3) % 8);
        const shade = 92 + ((i * 19) % 48);
        cx.fillStyle = `rgb(${shade + 4}, ${shade + 2}, ${shade - 3})`;
        cx.beginPath();
        cx.ellipse(px, py, rx, ry, (i % 5) * 0.4, 0, Math.PI * 2);
        cx.fill();
        cx.strokeStyle = '#abaead';
        cx.lineWidth = 2.2;
        cx.stroke();

        bx.fillStyle = '#d4d4d4';
        bx.beginPath();
        bx.ellipse(px, py, rx - 1.5, ry - 1.5, (i % 5) * 0.4, 0, Math.PI * 2);
        bx.fill();
      }
    }
    const stoneRetaining = inject(new THREE.MeshStandardMaterial({
      name: 'StoneRetainingBatuKali',
      map: this._toTexture(stoneC, true, 1, 1),
      bumpMap: this._toTexture(stoneB, false, 1, 1),
      bumpScale: 0.05,
      roughness: 0.9,
      metalness: 0.02
    }));

    // 9. Terracotta Plaza & Stair Tiles (DPPD Forecourt, Stairs & ADA Ramp) + Koral Sikat (ILRC Bridge)
    const plazaC = this._makeCanvas(256, 256);
    {
      const cx = plazaC.getContext('2d');
      cx.fillStyle = '#b9553d';
      cx.fillRect(0, 0, 256, 256);
      cx.strokeStyle = '#8a3a27';
      cx.lineWidth = 2;
      for (let i = 0; i <= 256; i += 32) {
        cx.beginPath(); cx.moveTo(i, 0); cx.lineTo(i, 256); cx.stroke();
        cx.beginPath(); cx.moveTo(0, i); cx.lineTo(256, i); cx.stroke();
      }
    }
    const terracottaPlaza = inject(new THREE.MeshStandardMaterial({
      name: 'TerracottaPlazaTiles',
      map: this._toTexture(plazaC, true, 1, 1),
      roughness: 0.76,
      metalness: 0.02
    }));

    const koralC = this._makeCanvas(256, 256);
    {
      const cx = koralC.getContext('2d');
      cx.fillStyle = '#9c6e58';
      cx.fillRect(0, 0, 256, 256);
      for (let i = 0; i < 1600; i++) {
        const x = (i * 61) % 256;
        const y = (i * 113) % 256;
        cx.fillStyle = i % 2 === 0 ? 'rgba(195,152,126,0.45)' : 'rgba(98,64,48,0.45)';
        cx.fillRect(x, y, 2, 2);
      }
      // Crisp light-grey concrete grid frame bands (ref_023, ref_029)
      cx.strokeStyle = '#cbd2d9';
      cx.lineWidth = 14;
      cx.strokeRect(0, 0, 256, 128);
      cx.strokeRect(0, 128, 256, 128);
    }
    const koralSikat = inject(new THREE.MeshStandardMaterial({
      name: 'KoralSikatRamp',
      map: this._toTexture(koralC, true, 1, 1),
      roughness: 0.86,
      metalness: 0.02
    }));

    // 10. Interlocking Grey Concrete Paving Block & Weathered Concrete
    const pavC = this._makeCanvas(256, 256);
    {
      const cx = pavC.getContext('2d');
      cx.fillStyle = '#9ca3af';
      cx.fillRect(0, 0, 256, 256);
      cx.strokeStyle = '#6b7280';
      cx.lineWidth = 2;
      for (let y = 0; y < 256; y += 16) {
        const off = ((y / 16) % 2) * 16;
        for (let x = -16; x < 256; x += 32) {
          cx.strokeRect(x + off, y, 32, 16);
        }
      }
    }
    const pavingBlock = inject(new THREE.MeshStandardMaterial({
      name: 'GreyPavingBlock',
      map: this._toTexture(pavC, true, 1, 1),
      roughness: 0.85,
      metalness: 0.02
    }));

    // 11. Ring Road Asphalt with Annular Markings Baked Cleanly (Zero Z-Fighting, Zero Green Zebra!)
    const aspC = this._makeCanvas(256, 256);
    {
      const cx = aspC.getContext('2d');
      cx.fillStyle = '#31363d';
      cx.fillRect(0, 0, 256, 256);
      for (let i = 0; i < 2400; i++) {
        const x = (i * 47) % 256;
        const y = (i * 89) % 256;
        const g = 42 + (i % 20);
        cx.fillStyle = `rgb(${g},${g + 2},${g + 4})`;
        cx.fillRect(x, y, 2, 2);
      }
    }
    const asphaltRoad = inject(new THREE.MeshStandardMaterial({
      name: 'AsphaltRoad',
      map: this._toTexture(aspC, true, 1, 1),
      roughness: 0.86,
      metalness: 0.03
    }));

    // 12. Black-and-White Striped Roundabout Concrete Curb
    const curbC = this._makeCanvas(256, 64);
    {
      const cx = curbC.getContext('2d');
      cx.fillStyle = '#f1f5f9';
      cx.fillRect(0, 0, 256, 64);
      cx.fillStyle = '#1e242b';
      cx.fillRect(0, 0, 128, 64);
    }
    const stripedCurb = inject(new THREE.MeshStandardMaterial({
      name: 'BlackWhiteCurb',
      map: this._toTexture(curbC, true, 1, 1),
      roughness: 0.78,
      metalness: 0.02
    }));

    // 13. Road Markings + ADA Blue Bay
    const yellowLine = inject(new THREE.MeshStandardMaterial({
      name: 'RoadYellowLine',
      color: new THREE.Color(0xeab308),
      roughness: 0.65,
      metalness: 0.02
    }));
    const adaBlueBay = inject(new THREE.MeshStandardMaterial({
      name: 'ADABlueParkingBay',
      color: new THREE.Color(0x1d6fb8),
      roughness: 0.72,
      metalness: 0.02
    }));

    // 14. Tropical Turf Grass & South Dry Brown Leaf-Litter Mound (Zero Magenta Lens-Flare Bias!)
    const grassC = this._makeCanvas(256, 256);
    {
      const cx = grassC.getContext('2d');
      cx.fillStyle = '#4f7938';
      cx.fillRect(0, 0, 256, 256);
      for (let i = 0; i < 2600; i++) {
        const x = (i * 59) % 256;
        const y = (i * 101) % 256;
        cx.fillStyle = i % 3 === 0 ? '#5f8d44' : (i % 3 === 1 ? '#42682d' : '#6b964c');
        cx.fillRect(x, y, 2, 3);
      }
    }
    const lushGrass = inject(new THREE.MeshStandardMaterial({
      name: 'TropicalGrassTurf',
      map: this._toTexture(grassC, true, 1, 1),
      roughness: 0.92,
      metalness: 0.0
    }));

    const leafC = this._makeCanvas(256, 256);
    {
      const cx = leafC.getContext('2d');
      cx.fillStyle = '#6b4730';
      cx.fillRect(0, 0, 256, 256);
      for (let i = 0; i < 2000; i++) {
        const x = (i * 43) % 256;
        const y = (i * 97) % 256;
        cx.fillStyle = i % 3 === 0 ? '#85583c' : (i % 3 === 1 ? '#543623' : '#7a6338');
        cx.fillRect(x, y, 3, 2);
      }
    }
    const dryLeafSoil = inject(new THREE.MeshStandardMaterial({
      name: 'DryLeafLitterSoil',
      map: this._toTexture(leafC, true, 1, 1),
      roughness: 0.95,
      metalness: 0.0
    }));

    // 15. Dark Metal / Fascia / Gates, Warm Taupe Soffit, Bamboo Stakes, Solar PV, Gold Accent, & Plinth Skirt
    const darkMetal = inject(new THREE.MeshStandardMaterial({
      name: 'DarkBronzeMetal',
      color: new THREE.Color(0x232830),
      roughness: 0.44,
      metalness: 0.62
    }));

    const warmTaupeSoffit = inject(new THREE.MeshStandardMaterial({
      name: 'WarmTaupeTerracottaSoffit',
      color: new THREE.Color(0xc59b88),
      roughness: 0.78,
      metalness: 0.03
    }));

    const goldAccent = inject(new THREE.MeshStandardMaterial({
      name: 'UIGoldBrass',
      color: new THREE.Color(0xd99b1e),
      roughness: 0.32,
      metalness: 0.68
    }));

    const redHydrant = inject(new THREE.MeshStandardMaterial({
      name: 'FireHydrantRed',
      color: new THREE.Color(0xb92b27),
      roughness: 0.45,
      metalness: 0.35
    }));

    const bambooWood = inject(new THREE.MeshStandardMaterial({
      name: 'BambooAndBark',
      color: new THREE.Color(0x6e583f),
      roughness: 0.88,
      metalness: 0.02
    }));

    const palmTrunk = inject(new THREE.MeshStandardMaterial({
      name: 'RoyalPalmGreyTrunk',
      color: new THREE.Color(0x8c8f88),
      roughness: 0.85,
      metalness: 0.02
    }));

    const foliageCanopy = inject(new THREE.MeshStandardMaterial({
      name: 'TropicalFoliageCanopy',
      color: new THREE.Color(0xffffff), // modulated per-instance via InstancedMesh.setColorAt!
      roughness: 0.82,
      metalness: 0.01
    }), { isFoliage: true });

    const pvC = this._makeCanvas(128, 128);
    {
      const cx = pvC.getContext('2d');
      cx.fillStyle = '#132642';
      cx.fillRect(0, 0, 128, 128);
      cx.strokeStyle = '#7fa1c3';
      cx.lineWidth = 2;
      cx.strokeRect(2, 2, 124, 124);
      cx.lineWidth = 1;
      for (let i = 16; i < 128; i += 16) {
        cx.beginPath(); cx.moveTo(i, 2); cx.lineTo(i, 126); cx.stroke();
        cx.beginPath(); cx.moveTo(2, i); cx.lineTo(126, i); cx.stroke();
      }
    }
    const solarPV = inject(new THREE.MeshStandardMaterial({
      name: 'RooftopSolarPV',
      map: this._toTexture(pvC, true, 1, 1),
      roughness: 0.22,
      metalness: 0.65
    }));

    const plinthEarthSkirt = inject(new THREE.MeshStandardMaterial({
      name: 'FoundationSkirtEarth',
      color: new THREE.Color(0x2a312b),
      roughness: 0.96,
      metalness: 0.0
    }));

    // 16. Single Unified Signage Atlas (1024x1024) — Packs ALL 8 Building Signs, Logos & Totems into 1 Single Draw Call!
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
      adaBlueBay: windowGlass,
      lushGrass,
      dryLeafSoil,
      darkMetal,
      warmTaupeSoffit: copperACP,
      goldAccent,
      redHydrant: terracottaRoof,
      bambooWood: dryLeafSoil,
      palmTrunk: stoneRetaining,
      foliageCanopy,
      solarPV: windowGlass,
      plinthEarthSkirt: asphaltRoad,
      signAtlasMat
    };
  }

  /**
   * Packs ALL campus signage, 3D typography plaques, Makara logos, KAN accreditation, totems, and gate signs
   * into ONE crisp 1024x1024 UV Atlas so all signage across all 4 buildings batches into 1 SINGLE DRAW CALL!
   *
   * Atlas UV Tiles (u0, v0, u1, v1 in Three.js UV space where v=1 is top of canvas, v=0 is bottom):
   * - Tile 0 (yPx 0..160):    STP-UI Top L8 Frieze ("SCIENCE TECHNO PARK / UNIVERSITAS INDONESIA" + Gold UI Makara)
   * - Tile 1 (yPx 160..272):  STP-UI L1 Portico Fascia ("SCIENCE TECHNO PARK UNIVERSITAS INDONESIA")
   * - Tile 2 (yPx 272..416):  ILRC Main Atrium Portal (UI Makara + KAN + "INTEGRATED LABORATORY AND RESEARCH CENTER / UNIVERSITAS INDONESIA")
   * - Tile 3 (yPx 416..528):  DPPD L1 Wall Sign ("GEDUNG / KSDP UI" + "DIREKTORAT PENGEMBANGAN PENDIDIKAN DIGITAL")
   * - Tile 4 (xPx 0..340,   yPx 528..1024): STP-UI Black/Yellow Directory Totem
   * - Tile 5 (xPx 340..680, yPx 528..1024): DPPD UI-Huawei Brass Training Center Totem
   * - Tile 6 (xPx 680..1024,yPx 528..776):  DRPM Gate Sign ("P / PARKIR KHUSUS PIMPINAN") & "STP-UI" Planter Badge
   * - Tile 7 (xPx 680..1024,yPx 776..1024): DPPD Campus Relief Map Board & ILRC Vertical Black/Gold Makara Banner
   */
  static createUnifiedSignageAtlasMaterial() {
    const c = this._makeCanvas(1024, 1024);
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, 1024, 1024);

    // Tile 0 (y: 0..160): STP-UI Top L8 Dark Frieze + Gold UI Makara + 2-Row White 3D Channel Letters (ref_111)
    ctx.fillStyle = '#262b33';
    ctx.fillRect(0, 0, 1024, 160);
    this.drawUIMakaraEmblem(ctx, 92, 82, 56, '#f59e0b', '#451a03');
    ctx.fillStyle = '#f8fafc';
    ctx.shadowColor = 'rgba(0,0,0,0.65)';
    ctx.shadowBlur = 6;
    ctx.shadowOffsetY = 3;
    ctx.font = '800 52px "Inter", "Segoe UI", Arial, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('SCIENCE TECHNO PARK', 174, 72);
    ctx.font = '700 47px "Inter", "Segoe UI", Arial, sans-serif';
    ctx.fillText('UNIVERSITAS INDONESIA', 174, 132);
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;

    // Tile 1 (y: 160..272): STP-UI L1 Silver Portico Fascia (ref_094, ref_164)
    ctx.fillStyle = '#e5e9ef';
    ctx.fillRect(0, 160, 1024, 112);
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 3;
    ctx.strokeRect(2, 162, 1020, 108);
    ctx.fillStyle = '#1e293b';
    ctx.font = '800 35px "Inter", "Segoe UI", Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('SCIENCE TECHNO PARK UNIVERSITAS INDONESIA', 512, 228);

    // Tile 2 (y: 272..416): ILRC Atrium White Portal Beam (UI Makara + KAN + 2-line title, ref_024, ref_029)
    ctx.fillStyle = '#f4f6f9';
    ctx.fillRect(0, 272, 1024, 144);
    this.drawUIMakaraEmblem(ctx, 370, 310, 26, '#eab308', '#713f12');
    // Stylized KAN Logo on upper right-center
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
    ctx.fillStyle = '#efe9dc';
    ctx.fillRect(0, 416, 1024, 112);
    // Left half: GEDUNG KSDP UI
    ctx.fillStyle = '#292524';
    ctx.textAlign = 'center';
    ctx.font = '800 38px "Inter", Arial, sans-serif';
    ctx.fillText('GEDUNG', 256, 464);
    ctx.fillText('KSDP UI', 256, 508);
    // Right half: Maroon/Gold DPPD UI Nameplate + "STP-UI" Planter Text
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
    // Two columns of white tenant directory plates
    for (let r = 0; r < 7; r++) {
      const py = 648 + r * 34;
      ctx.fillStyle = '#f1f5f9';
      ctx.fillRect(24, py, 138, 22);
      ctx.fillRect(178, py, 138, 22);
      ctx.fillStyle = '#334155';
      ctx.fillRect(32, py + 8, 96, 6);
      ctx.fillRect(186, py + 8, 104, 6);
    }
    // Yellow chevron base
    ctx.fillStyle = '#eab308';
    ctx.beginPath();
    ctx.moveTo(0, 930);
    ctx.lineTo(170, 966);
    ctx.lineTo(340, 930);
    ctx.lineTo(340, 1024);
    ctx.lineTo(0, 1024);
    ctx.closePath();
    ctx.fill();

    // Tile 5 (x: 340..680, y: 528..1024): DPPD Brushed Gold/Brass UI-Huawei Training Center Totem (ref_184, ref_185)
    const bgBrass = ctx.createLinearGradient(340, 528, 680, 1024);
    bgBrass.addColorStop(0, '#e5c98b');
    bgBrass.addColorStop(0.5, '#d4b26a');
    bgBrass.addColorStop(1, '#b89246');
    ctx.fillStyle = bgBrass;
    ctx.fillRect(340, 528, 340, 496);
    // Folded yellow top-left cap accent
    ctx.fillStyle = '#facc15';
    ctx.beginPath();
    ctx.moveTo(340, 528);
    ctx.lineTo(480, 528);
    ctx.lineTo(340, 610);
    ctx.closePath();
    ctx.fill();
    this.drawUIMakaraEmblem(ctx, 435, 620, 32, '#ca8a04', '#451a03');
    // Stylized Red Huawei Flower mark
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

    // Tile 6 (x: 680..1024, y: 528..776): DRPM "P PARKIR KHUSUS PIMPINAN" Sign (left) + Topiary "STP-UI" Turf Sign (right)
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

    // Right sub-tile (x: 852..1024, y: 528..776): Planter Topiary "STP-UI"
    ctx.fillStyle = '#3f6a2c';
    ctx.fillRect(852, 528, 172, 248);
    ctx.fillStyle = '#bef264';
    ctx.font = '900 42px Arial, sans-serif';
    ctx.fillText('STP-UI', 938, 666);

    // Tile 7 (x: 680..1024, y: 776..1024): DPPD Relief Campus Map Board (x: 680..864) + ILRC Vertical Black/Gold Banner (x: 864..1024)
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

    // ILRC Vertical Navy/Black Banner with Gold UI Makara (ref_012, ref_053)
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(864, 776, 160, 248);
    this.drawUIMakaraEmblem(ctx, 944, 868, 42, '#facc15', '#713f12');
    ctx.fillStyle = '#facc15';
    ctx.font = '800 16px Arial, sans-serif';
    ctx.fillText('UNIVERSITAS', 944, 952);
    ctx.fillText('INDONESIA', 944, 974);

    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    tex.needsUpdate = true;

    return new THREE.MeshStandardMaterial({
      name: 'UnifiedCampusSignageAtlas',
      map: tex,
      roughness: 0.42,
      metalness: 0.18
    });
  }

  /**
   * Helper returning PlaneGeometry pre-mapped to an exact pixel sub-rectangle [x0, y0, w, h] of the 1024x1024 Signage Atlas.
   */
  static createAtlasSignPlane(widthMeters, heightMeters, xPx, yPx, wPx, hPx) {
    const geo = new THREE.PlaneGeometry(widthMeters, heightMeters, 1, 1);
    const u0 = (xPx + 1.5) / 1024;
    const u1 = (xPx + wPx - 1.5) / 1024;
    const vTop = 1.0 - (yPx + 1.5) / 1024;
    const vBot = 1.0 - (yPx + hPx - 1.5) / 1024;
    const uv = geo.attributes.uv;
    // PlaneGeometry vertex order: (0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right)
    uv.setXY(0, u0, vTop);
    uv.setXY(1, u1, vTop);
    uv.setXY(2, u0, vBot);
    uv.setXY(3, u1, vBot);
    uv.needsUpdate = true;
    return geo;
  }
}
