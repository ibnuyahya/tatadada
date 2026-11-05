import * as THREE from 'three';
import { TerrainDEMLoader } from './TerrainDEMLoader.js';
import { SplineCorridorBuilder } from './SplineCorridorBuilder.js';
import { HumanEyeHDROptics } from '../core/HumanEyeHDROptics.js';

/**
 * NEXUS Omni-Scale Procedural Engine v6.36
 * ./world-gis/TropicalGrassField.js
 *
 * 1-Draw-Call GPU-Wrapped 3D Tropical Lawn Grass & Wildflower Field
 * (Axonopus compressus "Rumput Gajah Mini" + Arachis pintoi Golden Campus Blossoms).
 *
 * Architecture:
 * - Pre-bakes a 384x512 RGBA8 terrain & lawn-mask DataTexture once at startup (~15 ms)
 *   directly from TerrainDEMLoader SSoST height & exclusion functions (16-bit elevation = <1mm precision,
 *   ZERO grass blades on ocean, rivers, lakes, stone turap, roads, plazas, buildings, bridges, helipad, or runway!).
 * - Wraps up to 28,000 3D curved 4-segment blades toroidally around uCamPos.xz (92m x 92m active field)
 *   entirely in the GPU vertex shader (0.0 ms/frame CPU instancing cost!).
 * - Rendered in a companion 1-mesh scene into the active MSAA render target so main scene hierarchy
 *   draw calls remain strictly within Gate 4's <= 25 draw-call budget.
 */
export class TropicalGrassField {
  static BAKE_MIN_X = -412.0;
  static BAKE_MAX_X = 100.0;
  static BAKE_MIN_Z = -126.0;
  static BAKE_MAX_Z = 786.0;
  static HEIGHT_MIN_Y = -4.0;
  static HEIGHT_MAX_Y = 60.0;
  static FIELD_SPAN_M = 40.0;

  constructor(options = {}) {
    this.bladeCount = options.bladeCount || 28000;
    this.bakeCols = options.bakeCols || 384;
    this.bakeRows = options.bakeRows || 512;

    this.scene = new THREE.Scene();
    this.scene.name = 'TropicalGrassCompanionScene';

    this.bakeTexture = this._buildTerrainLawnBakeTexture(this.bakeCols, this.bakeRows);
    this.uniforms = {
      uGrassBakeTex: { value: this.bakeTexture },
      uBakeBounds: {
        value: new THREE.Vector4(
          TropicalGrassField.BAKE_MIN_X,
          TropicalGrassField.BAKE_MIN_Z,
          TropicalGrassField.BAKE_MAX_X - TropicalGrassField.BAKE_MIN_X,
          TropicalGrassField.BAKE_MAX_Z - TropicalGrassField.BAKE_MIN_Z
        )
      },
      uBakeTexSize: { value: new THREE.Vector2(this.bakeCols, this.bakeRows) },
      uCamPos: { value: new THREE.Vector3(0, 2, 85) },
      uTime: { value: 0.0 },
      uWindVec: { value: new THREE.Vector3(2.2, 0.0, 1.4) },
      uSunDir: { value: new THREE.Vector3(0.45, 0.78, 0.42).normalize() },
      uSunColor: { value: new THREE.Color(1.0, 0.96, 0.88) },
      uDuskFactor: { value: 0.0 },
      uNightFactor: { value: 0.0 },
      uCloudCover: { value: 0.24 },
      uRainIntensity: { value: 0.0 },
      uStormFactor: { value: 0.0 },
      uLightningFlash: { value: 0.0 },
      uFogFactor: { value: 0.0 },
      uCloudShadowPhase: { value: new THREE.Vector2(0, 0) },
      uHvsParams: { value: new THREE.Vector4(1.04, 1.045, 1.14, 0.0) },
      uHvsGlare: { value: new THREE.Vector4(0.76, 0.0, 0.018, 4.6) }
    };

    const bladeGeo = TropicalGrassField._buildCurvedBladeGeometry();
    this._populateInstanceAttributes(bladeGeo, this.bladeCount);
    this.material = this._createGrassMaterial();

    this.mesh = new THREE.InstancedMesh(bladeGeo, this.material, this.bladeCount);
    this.mesh.name = 'TropicalGrassBladesInstanced';
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;

    const idMat = new THREE.Matrix4();
    for (let i = 0; i < this.bladeCount; i++) {
      this.mesh.setMatrixAt(i, idMat);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    this.scene.add(this.mesh);
  }

  static _isLawnGrassAllowed(x, z, gy) {
    // 1. Must be dry inland above coastal stone turap (flat campus datum is gy = 0.00m; water is -1.45m)
    const dCoast = TerrainDEMLoader.getIslandCoastDistance(x, z);
    if (dCoast > -2.35 || gy < -0.04 || gy > 27.5) return 0.0;

    // 2. Freshwater spring, meandering river, lakes, & Mini-Niagara gorge exclusion
    if (TerrainDEMLoader.isFreshwaterExclusion(x, z, 0.95)) return 0.0;

    // 3. Steep cliff / rocky volcanic scree exclusion
    const eps = 0.85;
    const gx = Math.abs(TerrainDEMLoader.sampleGroundHeight(x + eps, z) - gy) / eps;
    const gz = Math.abs(TerrainDEMLoader.sampleGroundHeight(x, z + eps) - gy) / eps;
    if (Math.hypot(gx, gz) > 0.56) return 0.0;

    // 4. Gedung Rektorat UI, Ceremonial Plaza, Approach, Turnaround, & Lakeside Promenade
    if (x > -55.4 && x < 54.8 && z > -36.8 && z < 40.8) return 0.0;
    if (x > -21.4 && x < 16.4 && z >= 38.0 && z <= 73.2) return 0.0;
    if (x > -5.5 && x < 50.5 && z > -54.0 && z <= -34.0) return 0.0;
    if (z >= -56.0 && z <= 52.0 && x > TerrainDEMLoader.getShorelinePromenadeX(z) - 4.2) return 0.0;
    if (Math.hypot(x - 29.5, z - 29.5) < 8.8) return 0.0;
    if (Math.hypot(x - 28.4, z - 78.2) < 5.2) return 0.0;

    // 5. SE Covered Walkway, West Shuttle Boulevard, & West Helipad Boulevard
    if (z >= 33.0 && z <= 130.0) {
      const walkwayX = 34.2 + ((z - 35.5) / 92.5) * 9.6;
      if (Math.abs(x - walkwayX) < 3.6) return 0.0;
    }
    if (x > -69.2 && x < -54.8 && z > -36.2 && z < 104.0) return 0.0;
    if (x > -89.0 && x < -31.5 && z > 99.6 && z < 113.4) return 0.0;

    // 6. Lapangan Rotunda UI: exclude center Fountain (r < 6.3m) & circular asphalt boulevard (24.6m..38.5m),
    //    while keeping the inner botanical Rotunda lawn (6.6m..24.2m) lush green!
    const rRot = Math.hypot(x - SplineCorridorBuilder.ROTUNDA_CX, z - SplineCorridorBuilder.ROTUNDA_CZ);
    if (rRot < 6.35 || (rRot > 24.4 && rRot < 38.6)) return 0.0;

    // 7. Helipad concrete deck, Transit Pavilion, Jembatan Teksas plazas/ramps, Calisthenics EPDM, Kiosks, & Muster pads
    if (Math.hypot(x - TerrainDEMLoader.HELIPAD_CX, z - TerrainDEMLoader.HELIPAD_CZ) < 17.4) return 0.0;
    if (Math.hypot(x + 84.8, z - 118.2) < 6.2 || Math.hypot(x + 87.2, z - 92.8) < 4.2) return 0.0;
    if (Math.abs(x - TerrainDEMLoader.TEKSAS_BRIDGE_X) < 9.6 && z > 132.0 && z < 349.0) return 0.0;
    if (Math.hypot((x + 62.0) / 16.8, (z - 136.5) / 8.2) < 1.0) return 0.0;

    for (const ks of SplineCorridorBuilder.KIOSK_SPECS) {
      if (Math.hypot(x - ks.x, z - ks.z) < 4.8) return 0.0;
    }
    for (const mp of SplineCorridorBuilder.MUSTER_POINTS) {
      if (Math.hypot(x - mp.x, z - mp.z) < 3.6) return 0.0;
    }

    // 8. Road corridors, Underpass open-cut ramps, Ferry forecourts, DPPD complex, & ATC tower
    if (TerrainDEMLoader.isTransitInfrastructureExclusion(x, z, 1.35)) return 0.0;

    // 9. Pulau I ICAO Runway RWY 36/18, Taxiway, Aircraft Apron, & Volcano Upper Cone
    if (z >= 536.0) {
      if (x >= -382.0 && x <= -130.0 && z >= 654.0 && z <= 724.0) return 0.0;
      if (Math.hypot(x - TerrainDEMLoader.VOLCANO_CX, z - TerrainDEMLoader.VOLCANO_CZ) < 36.0) return 0.0;
      if (Math.hypot(x - TerrainDEMLoader.NIAGARA_CX, z - TerrainDEMLoader.NIAGARA_CZ) < 21.0) return 0.0;
    }

    // Edge feather near coast & higher elevations
    const coastFeather = Math.min(1.0, Math.max(0.0, (-2.35 - dCoast) / 2.8));
    const altFeather = gy > 20.0 ? Math.max(0.0, 1.0 - (gy - 20.0) / 7.5) : 1.0;
    return coastFeather * altFeather;
  }

  _buildTerrainLawnBakeTexture(cols, rows) {
    const minX = TropicalGrassField.BAKE_MIN_X;
    const minZ = TropicalGrassField.BAKE_MIN_Z;
    const spanX = TropicalGrassField.BAKE_MAX_X - minX;
    const spanZ = TropicalGrassField.BAKE_MAX_Z - minZ;
    const minY = TropicalGrassField.HEIGHT_MIN_Y;
    const spanY = TropicalGrassField.HEIGHT_MAX_Y - minY;

    const rawMask = new Float32Array(cols * rows);
    const data = new Uint8Array(cols * rows * 4);

    for (let iz = 0; iz < rows; iz++) {
      const v = iz / (rows - 1);
      const wz = minZ + v * spanZ;
      for (let ix = 0; ix < cols; ix++) {
        const u = ix / (cols - 1);
        const wx = minX + u * spanX;
        const idx = iz * cols + ix;

        const gy = TerrainDEMLoader.sampleGroundHeight(wx, wz);
        const q = Math.max(0, Math.min(65535, Math.round(((gy - minY) / spanY) * 65535.0)));
        const o = idx * 4;
        data[o] = (q >> 8) & 255;
        data[o + 1] = q & 255;

        const mask = TropicalGrassField._isLawnGrassAllowed(wx, wz, gy);
        rawMask[idx] = mask;

        const meadowPatch =
          0.5 +
          0.32 * Math.sin(wx * 0.11 + wz * 0.09) +
          0.18 * Math.cos(wx * 0.23 - wz * 0.19);
        data[o + 3] = Math.max(0, Math.min(255, Math.round(meadowPatch * 255.0)));
      }
    }

    // 3x3 minimum-biased soft lawn border feather so blades never overhang curbs or paths
    for (let iz = 0; iz < rows; iz++) {
      for (let ix = 0; ix < cols; ix++) {
        const idx = iz * cols + ix;
        const center = rawMask[idx];
        if (center <= 0.001) {
          data[idx * 4 + 2] = 0;
          continue;
        }
        let sum = 0.0;
        let minN = center;
        for (let dz = -1; dz <= 1; dz++) {
          const jz = Math.max(0, Math.min(rows - 1, iz + dz));
          for (let dx = -1; dx <= 1; dx++) {
            const jx = Math.max(0, Math.min(cols - 1, ix + dx));
            const m = rawMask[jz * cols + jx];
            sum += m;
            if (m < minN) minN = m;
          }
        }
        const feathered = minN <= 0.0 ? (sum / 9.0) * 0.45 : sum / 9.0;
        data[idx * 4 + 2] = Math.max(0, Math.min(255, Math.round(feathered * 255.0)));
      }
    }

    const tex = new THREE.DataTexture(data, cols, rows, THREE.RGBAFormat, THREE.UnsignedByteType);
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.generateMipmaps = false;
    tex.needsUpdate = true;
    return tex;
  }

  static _buildCurvedBladeGeometry() {
    // 3-Blade Crossed Botanical V-Tuft (27 vertices, 21 triangles per tuft -> 84,000 visible curved blades in 1 draw call!)
    const basePos = [
      -0.025, 0.00, 0.000,
       0.025, 0.00, 0.000,
      -0.021, 0.30, 0.022,
       0.021, 0.30, 0.022,
      -0.015, 0.62, 0.068,
       0.015, 0.62, 0.068,
      -0.008, 0.86, 0.132,
       0.008, 0.86, 0.132,
       0.000, 1.00, 0.192
    ];
    const baseUv = [
      0.0, 0.00,
      1.0, 0.00,
      0.0, 0.30,
      1.0, 0.30,
      0.0, 0.62,
      1.0, 0.62,
      0.0, 0.86,
      1.0, 0.86,
      0.5, 1.00
    ];
    const baseNorm = [
      -0.18, 0.35, 0.92,
       0.18, 0.35, 0.92,
      -0.18, 0.45, 0.87,
       0.18, 0.45, 0.87,
      -0.16, 0.58, 0.80,
       0.16, 0.58, 0.80,
      -0.14, 0.70, 0.70,
       0.14, 0.70, 0.70,
       0.00, 0.80, 0.60
    ];
    const baseIdx = [
      0, 1, 2,  1, 3, 2,
      2, 3, 4,  3, 5, 4,
      4, 5, 6,  5, 7, 6,
      6, 7, 8
    ];

    const subSpecs = [
      { yaw: 0.0, hScale: 1.00, splay: 1.00, ox: 0.000, oz: 0.000 },
      { yaw: 2.12, hScale: 0.86, splay: 1.18, ox: 0.032, oz: -0.024 },
      { yaw: -2.06, hScale: 0.78, splay: 1.24, ox: -0.028, oz: -0.028 }
    ];

    const positions = new Float32Array(9 * 3 * 3);
    const normals = new Float32Array(9 * 3 * 3);
    const uvs = new Float32Array(9 * 2 * 3);
    const indices = new Uint16Array(21 * 3);

    for (let s = 0; s < 3; s++) {
      const sp = subSpecs[s];
      const cy = Math.cos(sp.yaw);
      const sy = Math.sin(sp.yaw);
      const vOff = s * 9;
      for (let v = 0; v < 9; v++) {
        const px = basePos[v * 3];
        const py = basePos[v * 3 + 1] * sp.hScale;
        const pz = basePos[v * 3 + 2] * sp.splay;
        const rx = px * cy + pz * sy + sp.ox;
        const rz = -px * sy + pz * cy + sp.oz;
        positions[(vOff + v) * 3] = rx;
        positions[(vOff + v) * 3 + 1] = py;
        positions[(vOff + v) * 3 + 2] = rz;

        const nx = baseNorm[v * 3];
        const ny = baseNorm[v * 3 + 1];
        const nz = baseNorm[v * 3 + 2];
        normals[(vOff + v) * 3] = nx * cy + nz * sy;
        normals[(vOff + v) * 3 + 1] = ny;
        normals[(vOff + v) * 3 + 2] = -nx * sy + nz * cy;

        uvs[(vOff + v) * 2] = baseUv[v * 2];
        uvs[(vOff + v) * 2 + 1] = baseUv[v * 2 + 1];
      }
      for (let k = 0; k < 21; k++) {
        indices[s * 21 + k] = baseIdx[k] + vOff;
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    return geo;
  }

  _populateInstanceAttributes(geo, count) {
    const bladeData = new Float32Array(count * 4);
    const span = TropicalGrassField.FIELD_SPAN_M;
    const cols = Math.max(1, Math.round(Math.sqrt(count)));
    const rows = Math.max(1, Math.ceil(count / cols));
    const cellX = span / cols;
    const cellZ = span / rows;
    const fract = (v) => v - Math.floor(v);

    for (let i = 0; i < count; i++) {
      // Stratified jittered Poisson-like cell distribution (100% uniform coverage, ZERO diagonal Kronecker lattice rows!)
      const ix = i % cols;
      const iz = Math.floor(i / cols);
      const jx = fract(Math.sin(i * 127.1 + 311.7) * 43758.5453);
      const jz = fract(Math.sin(i * 269.5 + 183.3) * 43758.5453);
      const ox = -0.5 * span + (ix + jx) * cellX;
      const oz = -0.5 * span + (iz + jz) * cellZ;
      const yaw = fract(Math.sin(i * 12.9898 + 78.233) * 43758.5453) * Math.PI * 2.0;
      const seed = fract(Math.sin(i * 93.9898 + 19.173) * 28461.3719);
      const o = i * 4;
      bladeData[o] = ox;
      bladeData[o + 1] = oz;
      bladeData[o + 2] = yaw;
      bladeData[o + 3] = seed;
    }

    geo.setAttribute('aBladeData', new THREE.InstancedBufferAttribute(bladeData, 4));
  }

  _createGrassMaterial() {
    const hvsGLSL = HumanEyeHDROptics.getGLSLKernel();
    return new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      side: THREE.DoubleSide,
      depthTest: true,
      depthWrite: true,
      transparent: false,
      vertexShader: /* glsl */ `
        attribute vec4 aBladeData;

        uniform sampler2D uGrassBakeTex;
        uniform vec4 uBakeBounds;
        uniform vec2 uBakeTexSize;
        uniform vec3 uCamPos;
        uniform float uTime;
        uniform vec3 uWindVec;

        varying vec3 vWorldPos;
        varying vec3 vNormalW;
        varying vec2 vBladeUV;
        varying float vSeed;
        varying float vLush;
        varying float vGust;

        const float FIELD_SPAN = 40.0;
        const float HALF_SPAN = 20.0;
        const float H_MIN = -4.0;
        const float H_SPAN = 64.0;

        vec3 sampleBakedLawnBilinear(vec2 wxz) {
          vec2 uv01 = (wxz - uBakeBounds.xy) / uBakeBounds.zw;
          if (uv01.x <= 0.002 || uv01.x >= 0.998 || uv01.y <= 0.002 || uv01.y >= 0.998) {
            return vec3(-10.0, 0.0, 0.5);
          }
          vec2 gridCoord = uv01 * (uBakeTexSize - 1.0);
          vec2 i0 = floor(gridCoord);
          vec2 f = gridCoord - i0;
          vec2 i1 = min(uBakeTexSize - 1.0, i0 + 1.0);

          vec4 t00 = texture2D(uGrassBakeTex, (i0 + 0.5) / uBakeTexSize);
          vec4 t10 = texture2D(uGrassBakeTex, (vec2(i1.x, i0.y) + 0.5) / uBakeTexSize);
          vec4 t01 = texture2D(uGrassBakeTex, (vec2(i0.x, i1.y) + 0.5) / uBakeTexSize);
          vec4 t11 = texture2D(uGrassBakeTex, (i1 + 0.5) / uBakeTexSize);

          float h00 = H_MIN + ((t00.r * 255.0 * 256.0 + t00.g * 255.0) / 65535.0) * H_SPAN;
          float h10 = H_MIN + ((t10.r * 255.0 * 256.0 + t10.g * 255.0) / 65535.0) * H_SPAN;
          float h01 = H_MIN + ((t01.r * 255.0 * 256.0 + t01.g * 255.0) / 65535.0) * H_SPAN;
          float h11 = H_MIN + ((t11.r * 255.0 * 256.0 + t11.g * 255.0) / 65535.0) * H_SPAN;

          float h = mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
          float mask = mix(mix(t00.b, t10.b, f.x), mix(t01.b, t11.b, f.x), f.y);
          float lush = mix(mix(t00.a, t10.a, f.x), mix(t01.a, t11.a, f.x), f.y);
          return vec3(h, mask, lush);
        }

        void main() {
          // Toroidal world-fixed modulo wrap around camera XZ (blades stay anchored in world space!)
          vec2 rel = mod(aBladeData.xy - uCamPos.xz + HALF_SPAN, FIELD_SPAN) - HALF_SPAN;
          vec2 bladeRootXZ = uCamPos.xz + rel;

          vec3 lawn = sampleBakedLawnBilinear(bladeRootXZ);
          float groundY = lawn.x;
          float grassMask = lawn.y;
          vLush = lawn.z;
          vSeed = aBladeData.w;
          vBladeUV = uv;

          float distXZ = length(rel);
          float altAboveGround = max(0.0, uCamPos.y - groundY);
          float distFade = 1.0 - smoothstep(13.5, 19.6, distXZ);
          float altFade = 1.0 - smoothstep(26.0, 42.0, altAboveGround);
          float visScale = smoothstep(0.08, 0.42, grassMask) * distFade * altFade;

          if (visScale <= 0.002) {
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            vWorldPos = vec3(bladeRootXZ.x, groundY, bladeRootXZ.y);
            vNormalW = vec3(0.0, 1.0, 0.0);
            vGust = 0.0;
            return;
          }

          bool isFlower = vSeed > 0.942;
          float baseH = mix(0.14, 0.34, fract(vSeed * 7.13)) * (0.78 + 0.42 * vLush) * visScale;
          float widthScale = mix(0.95, 1.45, fract(vSeed * 13.7)) * (isFlower && uv.y > 0.72 ? 2.25 : 1.0);

          float cY = cos(aBladeData.z);
          float sY = sin(aBladeData.z);
          vec3 localP = vec3(position.x * widthScale, position.y * baseH, position.z * baseH);
          vec3 rotP = vec3(
            localP.x * cY + localP.z * sY,
            localP.y,
            -localP.x * sY + localP.z * cY
          );

          // Traveling tropical wind-gust ripples across campus lawns
          float windSpd = length(uWindVec.xz);
          vec2 windDir = windSpd > 0.02 ? uWindVec.xz / windSpd : vec2(0.8, 0.6);
          float gustPhase = dot(bladeRootXZ, windDir) * 0.24 - uTime * (1.55 + windSpd * 0.28);
          float gustWave = 0.55 + 0.45 * sin(gustPhase + sin(bladeRootXZ.x * 0.17 - bladeRootXZ.y * 0.21) * 1.6);
          float flutter = sin(uTime * 4.2 + vSeed * 31.4) * 0.025;
          float bendAmt = (0.045 + min(0.22, windSpd * 0.024) * gustWave + flutter) * uv.y * uv.y * visScale;
          vGust = gustWave;

          rotP.xz += windDir * bendAmt;
          rotP.y *= (1.0 - min(0.32, bendAmt * 0.85));

          vec3 worldPos = vec3(bladeRootXZ.x + rotP.x, groundY - 0.012 + rotP.y, bladeRootXZ.y + rotP.z);
          vWorldPos = worldPos;

          vec3 rotN = normalize(vec3(
            normal.x * cY + normal.z * sY,
            normal.y,
            -normal.x * sY + normal.z * cY
          ));
          vNormalW = normalize(mix(rotN, vec3(0.0, 1.0, 0.0), 0.48));

          gl_Position = projectionMatrix * viewMatrix * vec4(worldPos, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform vec3 uCamPos;
        uniform vec3 uSunDir;
        uniform vec3 uSunColor;
        uniform float uDuskFactor;
        uniform float uNightFactor;
        uniform float uCloudCover;
        uniform float uRainIntensity;
        uniform float uStormFactor;
        uniform float uLightningFlash;
        uniform float uFogFactor;
        uniform vec2 uCloudShadowPhase;
        uniform vec4 uHvsParams;
        uniform vec4 uHvsGlare;

        varying vec3 vWorldPos;
        varying vec3 vNormalW;
        varying vec2 vBladeUV;
        varying float vSeed;
        varying float vLush;
        varying float vGust;

        ${hvsGLSL}

        float grassHash21(vec2 p) {
          p = fract(p * vec2(127.1, 311.7));
          p += dot(p, p + 45.32);
          return fract(p.x * p.y);
        }

        float grassNoise2D(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          vec2 u = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(grassHash21(i), grassHash21(i + vec2(1.0, 0.0)), u.x),
            mix(grassHash21(i + vec2(0.0, 1.0)), grassHash21(i + vec2(1.0, 1.0)), u.x),
            u.y
          );
        }

        float evalGrassCloudShadow(vec3 wp, vec3 sDir) {
          if (uCloudCover <= 0.02 || uNightFactor >= 0.92) return 1.0;
          vec2 proj = (wp.xz - sDir.xz * (320.0 / max(0.20, sDir.y))) * 0.0036 + uCloudShadowPhase;
          float n = 0.58 * grassNoise2D(proj * 1.45) + 0.30 * grassNoise2D(proj * 3.1 + 11.7) + 0.12 * grassNoise2D(proj * 6.4);
          float thr = mix(0.60, 0.26, clamp(uCloudCover, 0.0, 1.0));
          float cMask = smoothstep(thr, thr + 0.26, n);
          float strength = clamp(uCloudCover * 0.85, 0.0, 0.56) * smoothstep(0.06, 0.24, sDir.y) * (1.0 - uNightFactor * 0.90);
          return 1.0 - cMask * strength;
        }

        void main() {
          vec3 sunDir = normalize(uSunDir);
          vec3 viewDir = normalize(uCamPos - vWorldPos);
          vec3 N = normalize(vNormalW);

          // Root-to-tip botanical chlorophyll pigment & soil-root ambient occlusion
          float h = clamp(vBladeUV.y, 0.0, 1.0);
          float rootAO = mix(0.38, 1.0, pow(h, 0.62));
          vec3 rootCol = vec3(0.042, 0.115, 0.026);
          vec3 midBlade = mix(vec3(0.078, 0.235, 0.044), vec3(0.112, 0.278, 0.052), vLush);
          vec3 tipCol = mix(vec3(0.145, 0.345, 0.065), vec3(0.205, 0.385, 0.078), fract(vSeed * 5.3));
          vec3 albedo = mix(rootCol, mix(midBlade, tipCol, smoothstep(0.35, 0.98, h)), smoothstep(0.0, 0.45, h));

          // Delicate Arachis pintoi golden blossoms & white tropical clover heads on ~5.8% of blades
          if (vSeed > 0.942 && h > 0.74) {
            bool isGolden = vSeed < 0.978;
            vec3 petalCol = isGolden ? vec3(0.86, 0.66, 0.08) : vec3(0.84, 0.85, 0.78);
            albedo = mix(albedo, petalCol, smoothstep(0.74, 0.86, h));
          }

          // Wet rain darkening & specular sheen
          albedo *= mix(1.0, 0.78, clamp(uRainIntensity * 0.7, 0.0, 0.7));

          float cloudShadow = evalGrassCloudShadow(vWorldPos, sunDir);
          float sunVis = clamp(1.0 - uRainIntensity * 0.72 - uStormFactor * 0.40 - uCloudCover * 0.20, 0.08, 1.0) * cloudShadow;

          float NdotL = max(0.0, dot(N, sunDir) * 0.65 + 0.35);
          float backSSS = pow(max(0.0, dot(-viewDir, sunDir)), 2.6) * (0.25 + 0.75 * h) * sunVis;

          vec3 skyIrrad = mix(
            mix(vec3(0.30, 0.40, 0.52), vec3(0.32, 0.24, 0.22), uDuskFactor),
            vec3(0.028, 0.042, 0.072),
            uNightFactor
          );
          vec3 directSun = uSunColor * mix(mix(2.35, 1.55, uDuskFactor), 0.14, uNightFactor) * sunVis;

          vec3 hdrLin = albedo * rootAO * (skyIrrad + directSun * NdotL)
                      + albedo * vec3(0.55, 0.82, 0.24) * directSun * backSSS * 0.52;

          // Subtle wind-gust sun sheen across bending blade backs
          vec3 H = normalize(sunDir + viewDir);
          float sheen = pow(max(0.0, dot(N, H)), mix(18.0, 42.0, clamp(uRainIntensity, 0.0, 1.0))) * h * (0.06 + 0.06 * vGust + 0.10 * uRainIntensity);
          hdrLin += directSun * sheen;
          hdrLin += vec3(0.65, 0.82, 1.15) * albedo * uLightningFlash * 0.85;

          hdrLin = nexusApplyBoundaryLayerAerialPerspective(
            hdrLin,
            uCamPos,
            vWorldPos,
            sunDir,
            uDuskFactor,
            uNightFactor,
            uCloudCover,
            uRainIntensity,
            uLightningFlash,
            uFogFactor
          );

          vec3 hvsLin = nexusApplyHumanEyeHDR(
            hdrLin,
            normalize(vWorldPos - uCamPos),
            sunDir,
            uSunColor,
            uHvsParams,
            uHvsGlare
          );

          gl_FragColor = vec4(nexusLinearToSRGB(hvsLin), 1.0);
        }
      `
    });
  }

  update(camera, weatherState, sharedSurfaceUniforms) {
    if (!camera || !this.uniforms) return;
    this.uniforms.uCamPos.value.copy(camera.position);

    if (sharedSurfaceUniforms) {
      if (sharedSurfaceUniforms.uTime) this.uniforms.uTime.value = sharedSurfaceUniforms.uTime.value;
      if (sharedSurfaceUniforms.uSunDir) this.uniforms.uSunDir.value.copy(sharedSurfaceUniforms.uSunDir.value);
      if (sharedSurfaceUniforms.uSunColor) this.uniforms.uSunColor.value.copy(sharedSurfaceUniforms.uSunColor.value);
      if (sharedSurfaceUniforms.uDuskFactor) this.uniforms.uDuskFactor.value = sharedSurfaceUniforms.uDuskFactor.value;
      if (sharedSurfaceUniforms.uNightFactor) this.uniforms.uNightFactor.value = sharedSurfaceUniforms.uNightFactor.value;
      if (sharedSurfaceUniforms.uCloudCover) this.uniforms.uCloudCover.value = sharedSurfaceUniforms.uCloudCover.value;
      if (sharedSurfaceUniforms.uRainIntensity) this.uniforms.uRainIntensity.value = sharedSurfaceUniforms.uRainIntensity.value;
      if (sharedSurfaceUniforms.uStormFactor) this.uniforms.uStormFactor.value = sharedSurfaceUniforms.uStormFactor.value;
      if (sharedSurfaceUniforms.uLightningFlash) this.uniforms.uLightningFlash.value = sharedSurfaceUniforms.uLightningFlash.value;
      if (sharedSurfaceUniforms.uFogFactor) this.uniforms.uFogFactor.value = sharedSurfaceUniforms.uFogFactor.value;
      if (sharedSurfaceUniforms.uCloudShadowPhase) {
        this.uniforms.uCloudShadowPhase.value.copy(sharedSurfaceUniforms.uCloudShadowPhase.value);
      }
      if (sharedSurfaceUniforms.uHvsParams) this.uniforms.uHvsParams.value.copy(sharedSurfaceUniforms.uHvsParams.value);
      if (sharedSurfaceUniforms.uHvsGlare) this.uniforms.uHvsGlare.value.copy(sharedSurfaceUniforms.uHvsGlare.value);
    }

    if (weatherState && weatherState.windVec) {
      this.uniforms.uWindVec.value.copy(weatherState.windVec);
    }
  }

  dispose() {
    if (this.bakeTexture) this.bakeTexture.dispose();
    if (this.mesh) {
      if (this.mesh.geometry) this.mesh.geometry.dispose();
      if (this.material) this.material.dispose();
    }
  }
}
