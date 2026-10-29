import * as THREE from 'three';
import { RainCollisionBake } from './RainCollisionBake.js';
import { HumanEyeHDROptics } from './HumanEyeHDROptics.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.5 (100x Architectural & World Surface Detail Edition)
 * ./core/ProceduralTextureFactory.js
 *
 * Implements Hukum Emas 3, 10, 13, & 14 at 100x Sub-Millimeter Fidelity:
 * 1. Analytical Multi-Octave Screen-Space Surface-Gradient Micro-Normal Perturbation (Mikkelsen dFdx/dFdy)
 *    directly inside `#include <normal_fragment_maps>` — ZERO tangent buffers, ZERO extra Draw Calls,
 *    infinite close-up resolution (1.5m inspection) with automatic distance anti-aliasing!
 * 2. 11 Domain-Specific Architectural & Natural Surface Profiles (`SURFACE_PROFILE`):
 *    - PRECAST_CONCRETE: Formwork expansion joints, tie-rod pin dimples, bush-hammered mineral aggregate pores
 *    - JAVANESE_CLAY_TILE: Interlocking Genteng Kodok overlap lips, drainage flutes, kiln-fired terracotta grain
 *    - UI_DIAMOND_PLAZA: Split-face Andesite pavers, honed White-Diamond (◇◇◇) inlays, recessed mortar bevels
 *    - RIBBON_INTERIOR_GLASS: True 3D Ray-Box Parallax Interior Office Rooms (suspended ceiling grid, glowing
 *      LED troffers, partition walls, teak bookshelves, Venetian blinds, & Fresnel sky reflections!)
 *    - ATRIUM_DIAMOND_GLASS: 3D Multi-Story Mezzanine Atrium Void Parallax behind X-diamond glass
 *    - WEATHERED_FLAT_ROOF, TERRACOTTA_MASONRY, MAKARA_STONE_BRONZE, TERRAIN_AND_TURAP (Voronoi Batu Kali
 *      river-stone masonry on steep turap + tropical grass blade tufts on lawns), ASPHALT_TEAK_BARK, & FOLIAGE_CANOPY.
 * 3. Photometrically calibrated 2048px Procedural Canvas Textures + Physical Wet-Surface PBR, Roof-Sheltered
 *    Standing Puddle Ripples, Gravity Runoff Rivulets, & Strict #000000 Night Emissive Masking.
 */

export const SURFACE_PROFILE = {
  DEFAULT: 0.0,
  PRECAST_CONCRETE: 1.0,
  JAVANESE_CLAY_TILE: 2.0,
  UI_DIAMOND_PLAZA: 3.0,
  RIBBON_INTERIOR_GLASS: 4.0,
  ATRIUM_DIAMOND_GLASS: 5.0,
  WEATHERED_FLAT_ROOF: 6.0,
  TERRACOTTA_MASONRY: 7.0,
  MAKARA_STONE_BRONZE: 8.0,
  TERRAIN_AND_TURAP: 9.0,
  ASPHALT_TEAK_BARK: 10.0,
  FOLIAGE_CANOPY: 11.0
};

export function createSharedSurfaceUniforms() {
  const hvsUniforms = HumanEyeHDROptics.createUniforms();
  return {
    uIsNight: { value: 0.0 },
    uNightFactor: { value: 0.0 },
    uDuskFactor: { value: 0.0 },
    uCloudCover: { value: 0.22 },
    uSunDir: { value: new THREE.Vector3(0.52, 0.72, 0.45).normalize() },
    uSunColor: { value: new THREE.Color(0xfff5e4) },
    uShaderQuality: { value: 1.0 },
    uZebraInspectMode: { value: 0.0 }, // 0 = PBR Ultra, 1 = Studio Zebra, 2 = Studio Clay
    uTime: { value: 0.0 },
    uWetness: { value: 0.0 },
    uRainIntensity: { value: 0.0 },
    uLightningFlash: { value: 0.0 },
    uWindVec: { value: new THREE.Vector3(0.65, 0.0, 0.35) },
    uCollisionMap: { value: RainCollisionBake.createFallbackTexture() },
    uCollisionReady: { value: 0.0 },
    uHvsParams: hvsUniforms.uHvsParams,
    uHvsGlare: hvsUniforms.uHvsGlare
  };
}

/**
 * NEXUS v2.6 ULTRA-LEVEL PHYSICAL SURFACE SHADER:
 * - Seamless 3D Isotropic Band-Limited Micro-Normals (0 aliasing, 0 seams, soft-clamped <= 6.3 deg)
 * - Fujii Oren-Nayar Microfacet Rough Diffuse Retro-Reflection + Cavity Specular Occlusion
 * - Beer-Lambert Botanical Leaf Subsurface Scattering (Translucency) & Waxy Cuticle Fresnel
 * - True 3D Ray-Box Parallax Interior Office & Atrium Rooms + Physical Fresnel Sky/Sun Glass Reflection
 * - Lekner-Dorf Wet Porous Darkening + Dielectric Water Clearcoat (IOR 1.333) + Puddle Ripples & Rivulets
 * - Directional Rayleigh + Henyey-Greenstein Mie Aerial Perspective in True Linear HDR (pre-AgX ToneMapping!)
 */
export function injectOmniScaleSurfaceShader(
  material,
  {
    ySegmentStart = 5.65,
    segmentStepH = 3.25,
    segmentCount = 6.0,
    enableWindSway = false,
    surfaceProfile = SURFACE_PROFILE.DEFAULT,
    bumpScale = 1.0,
    sharedUniforms = createSharedSurfaceUniforms()
  } = {}
) {
  if (!material || material.userData.omniShaderInjected) return material;
  material.userData.omniShaderInjected = true;
  const profId = Math.max(0, Math.min(11, Math.round(Number(surfaceProfile) || 0)));
  const hvsFallback = HumanEyeHDROptics.createUniforms();
  material.customProgramCacheKey = () =>
    `nexus-omni-v36-hvs-p${profId}-${enableWindSway ? 'sway' : 'static'}`;

  material.onBeforeCompile = (shader) => {
    shader.uniforms.uYSegStart = { value: ySegmentStart };
    shader.uniforms.uSegStepH = { value: segmentStepH };
    shader.uniforms.uSegCount = { value: segmentCount };
    shader.uniforms.uWindSwayAmp = { value: enableWindSway ? 1.0 : 0.0 };
    shader.uniforms.uSurfaceProfile = { value: surfaceProfile };
    shader.uniforms.uBumpScale = { value: bumpScale };
    shader.uniforms.uIsNight = sharedUniforms.uIsNight || { value: 0.0 };
    shader.uniforms.uNightFactor = sharedUniforms.uNightFactor || { value: 0.0 };
    shader.uniforms.uDuskFactor = sharedUniforms.uDuskFactor || { value: 0.0 };
    shader.uniforms.uCloudCover = sharedUniforms.uCloudCover || { value: 0.22 };
    shader.uniforms.uSunDir =
      sharedUniforms.uSunDir || { value: new THREE.Vector3(0.52, 0.72, 0.45).normalize() };
    shader.uniforms.uSunColor = sharedUniforms.uSunColor || { value: new THREE.Color(0xfff5e4) };
    shader.uniforms.uShaderQuality = sharedUniforms.uShaderQuality || { value: 1.0 };
    shader.uniforms.uZebraInspectMode = sharedUniforms.uZebraInspectMode || { value: 0.0 };
    shader.uniforms.uTime = sharedUniforms.uTime || { value: 0.0 };
    shader.uniforms.uWetness = sharedUniforms.uWetness || { value: 0.0 };
    shader.uniforms.uRainIntensity = sharedUniforms.uRainIntensity || { value: 0.0 };
    shader.uniforms.uLightningFlash = sharedUniforms.uLightningFlash || { value: 0.0 };
    shader.uniforms.uWindVec =
      sharedUniforms.uWindVec || { value: new THREE.Vector3(0.65, 0.0, 0.35) };
    shader.uniforms.uCollisionMap =
      sharedUniforms.uCollisionMap || { value: RainCollisionBake.createFallbackTexture() };
    shader.uniforms.uCollisionReady = sharedUniforms.uCollisionReady || { value: 0.0 };
    shader.uniforms.uHvsParams = sharedUniforms.uHvsParams || hvsFallback.uHvsParams;
    shader.uniforms.uHvsGlare = sharedUniforms.uHvsGlare || hvsFallback.uHvsGlare;

    shader.vertexShader = [
      'varying vec3 vOmniWorldPos;',
      'varying vec3 vOmniWorldNormal;',
      'varying vec2 vOmniUV;',
      'uniform float uTime;',
      'uniform float uWindSwayAmp;',
      'uniform vec3 uWindVec;',
      shader.vertexShader
    ]
      .join('\n')
      .replace(
        '#include <begin_vertex>',
        [
          '#include <begin_vertex>',
          '#ifdef USE_INSTANCING',
          '  mat4 omniWorldMat = modelMatrix * instanceMatrix;',
          '#else',
          '  mat4 omniWorldMat = modelMatrix;',
          '#endif',
          'vOmniUV = uv;',
          'vOmniWorldPos = (omniWorldMat * vec4(transformed, 1.0)).xyz;',
          'mat3 omniM3 = mat3(omniWorldMat);',
          'vec3 omniAdjN = objectNormal.x * cross(omniM3[1], omniM3[2])',
          '              + objectNormal.y * cross(omniM3[2], omniM3[0])',
          '              + objectNormal.z * cross(omniM3[0], omniM3[1]);',
          'vOmniWorldNormal = normalize(length(omniAdjN) > 1e-7 ? omniAdjN : (omniM3 * objectNormal));'
        ].join('\n')
      )
      .replace(
        '#include <project_vertex>',
        [
          '#include <project_vertex>',
          'if (uWindSwayAmp > 0.5) {',
          '  vec3 wpSway = vOmniWorldPos;',
          '  float hSway = clamp((wpSway.y - 0.25) / 10.5, 0.0, 1.45);',
          '  float windMag = length(uWindVec.xz);',
          '  vec2 windDir = windMag > 0.001 ? (uWindVec.xz / windMag) : vec2(0.85, 0.52);',
          '  float gustWave = 0.68 + 0.32 * sin(uTime * 1.95 - dot(wpSway.xz, windDir) * 0.075)',
          '                        + 0.16 * cos(uTime * 3.4 + wpSway.x * 0.11 - wpSway.z * 0.09);',
          '  float beamBend = hSway * hSway * (0.09 + 0.17 * windMag) * gustWave;',
          '  float leafFlutter = smoothstep(2.2, 5.8, wpSway.y) * (0.015 + 0.036 * windMag);',
          '  vec3 worldOffset = vec3(',
          '    windDir.x * beamBend + sin(uTime * 6.1 + wpSway.y * 2.1 + wpSway.z * 1.4) * leafFlutter,',
          '    -beamBend * 0.13 * hSway,',
          '    windDir.y * beamBend + cos(uTime * 5.6 + wpSway.x * 1.5 + wpSway.y * 1.9) * leafFlutter',
          '  );',
          '  vOmniWorldPos += worldOffset;',
          '  mvPosition.xyz += (viewMatrix * vec4(worldOffset, 0.0)).xyz;',
          '  gl_Position = projectionMatrix * mvPosition;',
          '}'
        ].join('\n')
      );

    shader.fragmentShader = [
      `#define OMNI_PROFILE ${profId}`,
      'varying vec3 vOmniWorldPos;',
      'varying vec3 vOmniWorldNormal;',
      'varying vec2 vOmniUV;',
      'uniform float uYSegStart;',
      'uniform float uSegStepH;',
      'uniform float uSegCount;',
      'uniform float uSurfaceProfile;',
      'uniform float uBumpScale;',
      'uniform float uIsNight;',
      'uniform float uNightFactor;',
      'uniform float uDuskFactor;',
      'uniform float uCloudCover;',
      'uniform vec3 uSunDir;',
      'uniform vec3 uSunColor;',
      'uniform float uShaderQuality;',
      'uniform float uZebraInspectMode;',
      'uniform float uTime;',
      'uniform float uWetness;',
      'uniform float uRainIntensity;',
      'uniform float uLightningFlash;',
      'uniform sampler2D uCollisionMap;',
      'uniform float uCollisionReady;',
      'uniform vec4 uHvsParams;',
      'uniform vec4 uHvsGlare;',
      HumanEyeHDROptics.getGLSLKernel(),
      'const mat2 OMNI_ROT = mat2(0.80, 0.60, -0.60, 0.80);',
      'float omniHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }',
      'vec2 omniHash2(vec2 p) {',
      '  return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453);',
      '}',
      'float omniNoise(vec2 p) {',
      '  vec2 i = floor(p); vec2 f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);',
      '  return mix(mix(omniHash(i), omniHash(i + vec2(1.0, 0.0)), u.x),',
      '             mix(omniHash(i + vec2(0.0, 1.0)), omniHash(i + vec2(1.0, 1.0)), u.x), u.y);',
      '}',
      'float omniFbm2(vec2 p) {',
      '  float v = 0.54 * omniNoise(p);',
      '  p = OMNI_ROT * p * 2.07 + vec2(11.3, 23.7);',
      '  v += 0.30 * omniNoise(p);',
      '  p = OMNI_ROT * p * 2.03 + vec2(7.9, 17.1);',
      '  v += 0.16 * omniNoise(p);',
      '  return v;',
      '}',
      'float omniFbm4(vec2 p) {',
      '  float v = 0.50 * omniNoise(p);',
      '  p = OMNI_ROT * p * 2.05 + vec2(13.1, 7.7);',
      '  v += 0.26 * omniNoise(p);',
      '  p = OMNI_ROT * p * 2.09 + vec2(5.3, 19.4);',
      '  v += 0.15 * omniNoise(p);',
      '  p = OMNI_ROT * p * 2.13 + vec2(23.9, 3.1);',
      '  v += 0.09 * omniNoise(p);',
      '  return v;',
      '}',
      // Cellular Voronoi River-Stone ('Batu Kali') Masonry Evaluator -> vec2(edgeMortarDist, cellIdHash)
      '#if OMNI_PROFILE == 9',
      'vec2 omniVoronoiStone(vec2 x) {',
      '  vec2 n = floor(x);',
      '  vec2 f = fract(x);',
      '  float md1 = 8.0;',
      '  float md2 = 8.0;',
      '  float idHash = 0.5;',
      '  for (int j = -1; j <= 1; j++) {',
      '    for (int i = -1; i <= 1; i++) {',
      '      vec2 g = vec2(float(i), float(j));',
      '      vec2 o = 0.15 + 0.70 * omniHash2(n + g);',
      '      vec2 r = g + o - f;',
      '      float d = dot(r, r);',
      '      if (d < md1) {',
      '        md2 = md1;',
      '        md1 = d;',
      '        idHash = o.x;',
      '      } else if (d < md2) {',
      '        md2 = d;',
      '      }',
      '    }',
      '  }',
      '  return vec2(sqrt(md2) - sqrt(md1), idHash);',
      '}',
      '#endif',
      'float omniSampleSkyExposure(vec3 wp) {',
      '  if (uCollisionReady < 0.5) return 1.0;',
      '  vec2 cuv = (wp.xz + 185.0) / 370.0;',
      '  if (cuv.x < 0.002 || cuv.x > 0.998 || cuv.y < 0.002 || cuv.y > 0.998) return 1.0;',
      '  vec4 cTex = texture2D(uCollisionMap, cuv);',
      '  float yTop = -6.0 + ((cTex.r * 255.0 * 256.0 + cTex.g * 255.0) / 65535.0) * 60.0;',
      '  float isCanopyAbove = step(0.98, cTex.a);',
      '  float shelter = smoothstep(0.55, 1.65, yTop - wp.y);',
      '  return mix(1.0, isCanopyAbove * 0.28, shelter);',
      '}',
      // =========================================================================
      // SEAMLESS 3D ISOTROPIC BAND-LIMITED NOISE & ARCHITECTURAL SURFACE KERNEL
      // (Zero triUV coordinate-blending seams, zero Nyquist TV-static, zero 2x2 blockiness!)
      // =========================================================================
      'float omniHash3(vec3 p) {',
      '  return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453123);',
      '}',
      'float omniNoise3(vec3 p) {',
      '  vec3 i = floor(p);',
      '  vec3 f = fract(p);',
      '  vec3 u = f * f * (3.0 - 2.0 * f);',
      '  float n000 = omniHash3(i);',
      '  float n100 = omniHash3(i + vec3(1.0, 0.0, 0.0));',
      '  float n010 = omniHash3(i + vec3(0.0, 1.0, 0.0));',
      '  float n110 = omniHash3(i + vec3(1.0, 1.0, 0.0));',
      '  float n001 = omniHash3(i + vec3(0.0, 0.0, 1.0));',
      '  float n101 = omniHash3(i + vec3(1.0, 0.0, 1.0));',
      '  float n011 = omniHash3(i + vec3(0.0, 1.0, 1.0));',
      '  float n111 = omniHash3(i + vec3(1.0, 1.0, 1.0));',
      '  return mix(',
      '    mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),',
      '    mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),',
      '    u.z',
      '  );',
      '}',
      // Nyquist anti-aliasing band-limiter: smoothly fades out any octave before < 4.5 pixels/wave
      'float omniBand(float pxSize, float freq) {',
      '  return 1.0 - smoothstep(0.055, 0.21, pxSize * freq);',
      '}',
      // Seamless 3D 2-octave band-limited fBm centered at 0.0 (never aliases, never stretches on curves!)
      'float omniFbm3BL(vec3 p, float baseFreq, float pxSize) {',
      '  float w1 = omniBand(pxSize, baseFreq);',
      '  if (w1 <= 0.001) return 0.0;',
      '  float v = 0.66 * w1 * (omniNoise3(p * baseFreq) - 0.5);',
      '  float w2 = omniBand(pxSize, baseFreq * 2.07);',
      '  if (w2 > 0.001) {',
      '    v += 0.34 * w2 * (omniNoise3(p * (baseFreq * 2.07) + vec3(13.7, 29.3, 7.1)) - 0.5);',
      '  }',
      '  return v;',
      '}',
      'float omniJoint1D(float coord, float period, float halfWidth) {',
      '  float f = fract(coord / period);',
      '  return smoothstep(0.0, halfWidth, f) * smoothstep(0.0, halfWidth, 1.0 - f);',
      '}',
      // Returns vec4(heightMeters, albedoScale, roughnessDelta, microCavityAO)
      'vec4 omniEvalSurfaceDetail(float prof, vec3 wp, vec3 wn, vec3 baseCol, float pxSize) {',
      '  vec3 aw = abs(wn);',
      '  vec2 wWallXZ = pow(aw.zx, vec2(6.0));',
      '  wWallXZ /= max(wWallXZ.x + wWallXZ.y, 1e-4);',
      '  float h = 0.0;',
      '  float alb = 1.0;',
      '  float rDelta = 0.0;',
      '  float ao = 1.0;',
      '',
      '#if OMNI_PROFILE == 1',
      '    // PROFILE 1: REFINED ARCHITECTURAL PRECAST CONCRETE (Value-blended seams + close-up mineral finish)',
      '    float bJoint = omniBand(pxSize, 1.8);',
      '    float seamAlongX = omniJoint1D(wp.x, 1.60, 0.014);',
      '    float seamAlongZ = omniJoint1D(wp.z, 1.60, 0.014);',
      '    float seamH = omniJoint1D(wp.y, 1.625, 0.014);',
      '    float rawSeam = (seamAlongX * wWallXZ.x + seamAlongZ * wWallXZ.y) * seamH;',
      '    float panelSeam = mix(1.0, mix(rawSeam, 1.0, step(0.75, aw.y)), bJoint);',
      '    float mineral = omniFbm3BL(wp, 5.5, pxSize);',
      '    float closePore = omniFbm3BL(wp, 18.0, pxSize);',
      '    h = -(1.0 - panelSeam) * 0.0022 + mineral * 0.0007 + closePore * 0.00028;',
      '    alb = (0.985 + 0.015 * panelSeam) + mineral * 0.022 + closePore * 0.014;',
      '    rDelta = (1.0 - panelSeam) * 0.04 - mineral * 0.02;',
      '    ao = mix(0.92, 1.0, panelSeam);',
      '',
      '#elif OMNI_PROFILE == 2',
      '    // PROFILE 2: JAVANESE TERRACOTTA CLAY ROOF TILES (Smooth Genteng Kodok S-waves, zero grit)',
      '    float bTile = omniBand(pxSize, 4.0);',
      '    vec2 horizDir = normalize(vec2(-wn.z, wn.x) + vec2(1e-4, 0.0));',
      '    float uTile = dot(wp.xz, horizDir) / 0.24;',
      '    float vSlope = (wp.y / max(0.32, sqrt(max(0.05, 1.0 - wn.y * wn.y)))) / 0.29;',
      '    float rowIdx = floor(vSlope);',
      '    float colIdx = floor(uTile + mod(rowIdx, 2.0) * 0.15);',
      '    float fU = fract(uTile + mod(rowIdx, 2.0) * 0.15);',
      '    float fV = fract(vSlope);',
      '    float waveRib = (sin(fU * 6.2831853) * 0.5 + 0.5) * bTile;',
      '    float overlapStep = mix(1.0, smoothstep(0.0, 0.14, fV), bTile);',
      '    float tileHash = (omniHash(vec2(colIdx, rowIdx)) - 0.5) * bTile;',
      '    float patina = omniFbm3BL(wp, 0.85, pxSize);',
      '    h = waveRib * 0.0032 + overlapStep * 0.0028;',
      '    alb = (1.0 + 0.065 * tileHash) * (0.98 + 0.04 * waveRib) + patina * 0.05;',
      '    rDelta = -waveRib * 0.04 + tileHash * 0.04;',
      '    ao = mix(0.86, 1.0, overlapStep);',
      '',
      '#elif OMNI_PROFILE == 3',
      '    // PROFILE 3: REKTORAT UI COURTYARD ◇◇◇ STONE PAVERS (Clean Andesite grid + honed white diamond)',
      '    float baseLuma = dot(baseCol, vec3(0.299, 0.587, 0.114));',
      '    float isWhiteDiamond = smoothstep(0.52, 0.72, baseLuma);',
      '    float bGrout = omniBand(pxSize, 2.2);',
      '    float gX = omniJoint1D(wp.x, 0.60, 0.024);',
      '    float gZ = omniJoint1D(wp.z, 0.60, 0.024);',
      '    float grout = mix(1.0, mix(gX * gZ, 1.0, isWhiteDiamond * 0.70), bGrout);',
      '    float stoneGrain = omniFbm3BL(wp, 8.5, pxSize) * mix(1.0, 0.25, isWhiteDiamond);',
      '    h = -(1.0 - grout) * 0.0018 + stoneGrain * 0.00055;',
      '    alb = 1.0 + stoneGrain * 0.028;',
      '    rDelta = mix(0.03, -0.15, isWhiteDiamond) + (1.0 - grout) * 0.06;',
      '    ao = mix(0.89, 1.0, grout);',
      '',
      '#elif OMNI_PROFILE == 6',
      '    // PROFILE 6: WEATHERED TROPICAL FLAT CONCRETE ROOF DECK & FOUNDATION',
      '    float bSlab = omniBand(pxSize, 1.0);',
      '    float jnt = mix(1.0, omniJoint1D(wp.x, 2.20, 0.014) * omniJoint1D(wp.z, 2.20, 0.014), bSlab * aw.y);',
      '    float screed = omniFbm3BL(wp, 3.5, pxSize);',
      '    h = -(1.0 - jnt) * 0.0020 + screed * 0.0008;',
      '    alb = 1.0 + screed * 0.038;',
      '    rDelta = screed * 0.04;',
      '    ao = mix(0.90, 1.0, jnt);',
      '',
      '#elif OMNI_PROFILE == 7',
      '    // PROFILE 7: RUNNING-BOND PRESSED TERRACOTTA BRICK MASONRY',
      '    float bBrick = omniBand(pxSize, 8.0);',
      '    float bRow = floor(wp.y / 0.085);',
      '    float mX = omniJoint1D(wp.x + mod(bRow, 2.0) * 0.12, 0.24, 0.06);',
      '    float mZ = omniJoint1D(wp.z + mod(bRow, 2.0) * 0.12, 0.24, 0.06);',
      '    float mY = omniJoint1D(wp.y, 0.085, 0.10);',
      '    float mtr = mix(1.0, (mX * wWallXZ.x + mZ * wWallXZ.y) * mY, bBrick);',
      '    float clayFine = omniFbm3BL(wp, 10.0, pxSize);',
      '    h = mtr * 0.0016 + clayFine * 0.0004;',
      '    alb = mix(0.92, 1.02, mtr) + clayFine * 0.025;',
      '    rDelta = (1.0 - mtr) * 0.06;',
      '    ao = mix(0.88, 1.0, mtr);',
      '',
      '#elif OMNI_PROFILE == 8',
      '    // PROFILE 8: SATIN ARCHITECTURAL BRONZE TRIM & CARVED MAKARA STONE',
      '    float satin = omniFbm3BL(wp, 12.0, pxSize);',
      '    h = satin * 0.00025;',
      '    alb = 1.0 + satin * 0.018;',
      '    rDelta = satin * 0.02;',
      '',
      '#elif OMNI_PROFILE == 9',
      '    // PROFILE 9: VELVET CAMPUS GREEN LAWNS + RIPARIAN PEBBLE/MOSS BANKS + VOLCANIC BASALT CRAGS + 360-DEG VORONOI TURAP MASONRY',
      '    float macroClover = omniFbm3BL(wp, 0.45, pxSize);',
      '    float midHumus = omniFbm3BL(wp, 2.35, pxSize);',
      '    float softTurf = omniFbm3BL(wp, 6.8, pxSize);',
      '    float greenDominance = baseCol.g - max(baseCol.r, baseCol.b);',
      '    float isEarthyOrCobble = 1.0 - smoothstep(-0.015, 0.065, greenDominance);',
      '    h = softTurf * 0.00085 + midHumus * 0.00055;',
      '    alb = 1.0 + macroClover * 0.058 + midHumus * 0.032 + softTurf * 0.026;',
      '    rDelta = -softTurf * 0.02;',
      '    ao = 0.97 + 0.03 * softTurf;',
      '    float isTurapMasonry = 1.0 - smoothstep(0.62, 0.86, wn.y);',
      '    float isCoastalStoneTint = 1.0 - smoothstep(0.025, 0.075, baseCol.g - baseCol.r);',
      '    if (wp.y < 0.58 && isTurapMasonry > 0.20 && isCoastalStoneTint > 0.25 && length(wp.xz - vec2(-18.0, 34.0)) > 62.0) {',
      '      float turapMask = smoothstep(0.58, 0.26, wp.y) * smoothstep(0.20, 0.50, isTurapMasonry) * isCoastalStoneTint;',
      '      float bStone = omniBand(pxSize, 2.3);',
      '      vec2 turapTan = normalize(vec2(-wn.z, wn.x) + vec2(1e-4, 0.0));',
      '      float uTurap = dot(wp.xz, turapTan);',
      '      vec2 vStone = omniVoronoiStone(vec2(uTurap * 2.15, wp.y * 2.55));',
      '      float mortar = mix(1.0, smoothstep(0.035, 0.12, vStone.x), bStone);',
      '      float stoneDome = smoothstep(0.03, 0.36, vStone.x) * bStone;',
      '      float rockSoft = omniFbm3BL(wp, 6.5, pxSize);',
      '      float waterMoss = smoothstep(0.15, -1.25, wp.y) * smoothstep(0.35, 0.78, omniFbm2(wp.xz * 1.4));',
      '      float hTurap = stoneDome * 0.0055 + rockSoft * 0.0006;',
      '      float albTurap = mix(0.86, 0.96 + 0.10 * vStone.y, mortar) * (1.0 - waterMoss * 0.16) + rockSoft * 0.025;',
      '      float rTurap = (1.0 - mortar) * 0.06 - waterMoss * 0.14;',
      '      float aoTurap = mix(0.83, 1.0, mortar);',
      '      h = mix(h, hTurap, turapMask);',
      '      alb = mix(alb, albTurap, turapMask);',
      '      rDelta = mix(rDelta, rTurap, turapMask);',
      '      ao = mix(ao, aoTurap, turapMask);',
      '    } else if (wp.z > 536.0 && wp.y > 1.8) {',
      '      float volcanoProx = smoothstep(134.0, 66.0, length(wp.xz - vec2(-32.0, 698.0)));',
      '      float rockMask = clamp(max(smoothstep(12.5, 22.0, wp.y) * (0.35 + 0.65 * volcanoProx), smoothstep(0.78, 0.46, wn.y) * isEarthyOrCobble), 0.0, 1.0);',
      '      if (rockMask > 0.01) {',
      '        float cragMacro = omniFbm3BL(vec3(wp.x * 0.78, wp.y * 1.35, wp.z * 0.78), 1.25, pxSize);',
      '        float cragRidge = omniFbm3BL(vec3(wp.x * 1.65 + cragMacro * 0.9, wp.y * 2.4, wp.z * 1.65), 1.9, pxSize);',
      '        float vesicular = omniFbm3BL(wp, 5.6, pxSize);',
      '        float crevice = smoothstep(-0.34, 0.18, cragRidge);',
      '        float hRock = cragMacro * 0.0075 + cragRidge * 0.0042 + vesicular * 0.0014;',
      '        float albRock = mix(0.83, 1.06, crevice) + cragMacro * 0.055 + vesicular * 0.028;',
      '        float rRock = 0.05 + (1.0 - crevice) * 0.05;',
      '        float aoRock = mix(0.80, 1.0, crevice) * (0.94 + 0.06 * cragMacro);',
      '        h = mix(h, hRock, rockMask);',
      '        alb = mix(alb, albRock, rockMask);',
      '        rDelta = mix(rDelta, rRock, rockMask);',
      '        ao = mix(ao, aoRock, rockMask);',
      '      }',
      '    } else if (isEarthyOrCobble > 0.05) {',
      '      float cobMacro = omniFbm3BL(wp, 3.1, pxSize);',
      '      float pebGrain = omniFbm3BL(wp, 11.5, pxSize);',
      '      float dampSheen = smoothstep(0.08, -0.02, greenDominance) * smoothstep(0.28, 0.14, dot(baseCol, vec3(0.333)));',
      '      float hCob = cobMacro * 0.0036 + pebGrain * 0.0014;',
      '      float albCob = 1.0 + cobMacro * 0.085 + pebGrain * 0.045;',
      '      float rCob = -0.16 * dampSheen + pebGrain * 0.03;',
      '      float aoCob = 0.92 + 0.08 * smoothstep(-0.35, 0.35, cobMacro);',
      '      h = mix(h, hCob, isEarthyOrCobble);',
      '      alb = mix(alb, albCob, isEarthyOrCobble);',
      '      rDelta = mix(rDelta, rCob, isEarthyOrCobble);',
      '      ao = mix(ao, aoCob, isEarthyOrCobble);',
      '    }',
      '',
      '#elif OMNI_PROFILE == 10',
      '    // PROFILE 10: SMOOTH 3D VERTICAL TREE BARK, TEAK BENCH GRAIN, & ROLLED ASPHALT',
      '    float isVerticalBark = 1.0 - smoothstep(0.35, 0.72, aw.y);',
      '    if (isVerticalBark > 0.5) {',
      '      float barkRidge = omniFbm3BL(vec3(wp.x * 3.8, wp.y * 0.85, wp.z * 3.8), 1.0, pxSize);',
      '      float barkClose = omniFbm3BL(vec3(wp.x * 9.5, wp.y * 2.2, wp.z * 9.5), 1.0, pxSize * 3.5);',
      '      h = barkRidge * 0.0028 + barkClose * 0.0006;',
      '      alb = 1.0 + barkRidge * 0.065 + barkClose * 0.025;',
      '      ao = 0.95 + 0.05 * barkRidge;',
      '    } else {',
      '      float isTeak = smoothstep(0.14, 0.28, baseCol.r - baseCol.b);',
      '      float teakGrain = omniFbm3BL(vec3(wp.x * 2.2, wp.y * 12.0, wp.z * 12.0), 1.0, pxSize * 3.0);',
      '      float asphaltSoft = omniFbm3BL(wp, 9.0, pxSize);',
      '      h = mix(asphaltSoft * 0.0005, teakGrain * 0.0004, isTeak);',
      '      alb = 1.0 + mix(asphaltSoft * 0.028, teakGrain * 0.038, isTeak);',
      '      rDelta = mix(-asphaltSoft * 0.03, -0.10, isTeak);',
      '    }',
      '',
      '#elif OMNI_PROFILE == 11',
      '    // PROFILE 11: LUSH VELVET BOTANICAL FOLIAGE CANOPY (Soft 3D leaf cushions, zero harsh grain!)',
      '    float softBough = omniFbm3BL(wp, 1.4, pxSize);',
      '    float softLeaf = omniFbm3BL(wp, 4.2, pxSize);',
      '    h = softBough * 0.0018 + softLeaf * 0.0006;',
      '    alb = 1.0 + softBough * 0.048 + softLeaf * 0.022;',
      '    rDelta = -0.04 - softLeaf * 0.02;',
      '    ao = 0.96 + 0.04 * softBough;',
      '',
      '#else',
      '    float g1 = omniFbm3BL(wp, 5.0, pxSize);',
      '    h = g1 * 0.0006;',
      '    alb = 1.0 + g1 * 0.022;',
      '#endif',
      '  return vec4(h, alb, rDelta, ao);',
      '}',
      '',
      // =========================================================================
      // TRUE 3D RAY-BOX PARALLAX INTERIOR ROOM MAPPING FOR ARCHITECTURAL WINDOWS!
      // Renders 3D office ceiling grids, glowing LED troffers, partition walls,
      // teak bookshelves/desks, & Venetian window blinds moving in 3D parallax!
      // =========================================================================
      '#if OMNI_PROFILE == 4 || OMNI_PROFILE == 5',
      'vec3 omniEvalInteriorParallaxRoom(vec3 wp, vec3 wn, vec3 viewDirOut, float isAtrium) {',
      '  vec3 roomOut = normalize(vec3(wn.x, 0.0, wn.z) + vec3(1e-5, 0.0, 0.0));',
      '  vec3 roomUp = vec3(0.0, 1.0, 0.0);',
      '  vec3 roomRight = normalize(cross(roomUp, roomOut));',
      '',
      '  float roomW = isAtrium > 0.5 ? 4.50 : 2.95;',
      '  float roomH = 3.25;',
      '  float depthRatio = isAtrium > 0.5 ? 1.85 : 1.32;',
      '  float uHoriz = dot(wp, roomRight);',
      '  float yRel = wp.y - 2.40;',
      '  float roomIdxX = floor(uHoriz / roomW);',
      '  float floorIdx = floor(yRel / roomH);',
      '  float fx = fract(uHoriz / roomW);',
      '  float fy = fract(yRel / roomH);',
      '  float rSeed = omniHash(vec2(roomIdxX * 1.37 + roomOut.x * 7.0, floorIdx * 2.11 + roomOut.z * 11.0));',
      '',
      '  // Semi-drawn Horizontal Venetian Window Blinds on ~34% of administrative office rooms!',
      '  float hasBlinds = step(0.66, rSeed) * (1.0 - isAtrium);',
      '  float blindDrop = 0.42 + 0.42 * fract(rSeed * 7.3);',
      '  if (hasBlinds > 0.5 && fy > (1.0 - blindDrop)) {',
      '    float slat = 0.5 + 0.5 * sin(fy * 190.0);',
      '    float litBlind = step(0.42, fract(rSeed * 3.7));',
      '    vec3 dayBlind = vec3(0.46, 0.45, 0.42) * (0.65 + 0.35 * slat);',
      '    vec3 nightBlind = litBlind * vec3(0.95, 0.72, 0.38) * (0.35 + 0.65 * slat);',
      '    return mix(dayBlind * 0.42, nightBlind, uIsNight);',
      '  }',
      '',
      '  // Transform incoming camera->window ray into 3D Room Space',
      '  vec3 rayW = -viewDirOut;',
      '  float dR = dot(rayW, roomRight);',
      '  float dU = rayW.y;',
      '  float dIn = max(0.06, dot(rayW, -roomOut));',
      '',
      '  float tBack = depthRatio / dIn;',
      '  float tSide = ((dR >= 0.0 ? (1.0 - fx) : -fx)) / (abs(dR) > 1e-4 ? dR : (dR >= 0.0 ? 1e-4 : -1e-4));',
      '  float tVert = ((dU >= 0.0 ? (1.0 - fy) : -fy)) / (abs(dU) > 1e-4 ? dU : (dU >= 0.0 ? 1e-4 : -1e-4));',
      '  tSide = max(0.0, tSide);',
      '  tVert = max(0.0, tVert);',
      '  float tHit = min(tBack, min(tSide, tVert));',
      '  vec3 hit = vec3(clamp(fx + dR * tHit, 0.0, 1.0), clamp(fy + dU * tHit, 0.0, 1.0), dIn * tHit);',
      '',
      '  vec3 surfCol = vec3(0.42, 0.40, 0.37);',
      '  float trofferGlow = 0.0;',
      '',
      '  if (tVert <= tBack && tVert <= tSide && dU > 0.0) {',
      '    // 3D Suspended Acoustic Ceiling Tile Grid + Twin Linear LED Troffer Fixtures!',
      '    vec2 cGrid = fract(vec2(hit.x * 4.0, hit.z * 3.0));',
      '    float tBar = smoothstep(0.0, 0.06, cGrid.x) * smoothstep(0.0, 0.06, 1.0 - cGrid.x)',
      '               * smoothstep(0.0, 0.06, cGrid.y) * smoothstep(0.0, 0.06, 1.0 - cGrid.y);',
      '    float ledX = smoothstep(0.20, 0.24, abs(hit.x - 0.5));',
      '    float ledPanel = (1.0 - smoothstep(0.11, 0.14, abs(abs(hit.x - 0.5) - 0.22)))',
      '                   * smoothstep(0.22, 0.26, hit.z) * (1.0 - smoothstep(0.95, 1.02, hit.z));',
      '    trofferGlow = clamp(ledPanel, 0.0, 1.0);',
      '    surfCol = vec3(0.58, 0.56, 0.52) * (0.82 + 0.18 * tBar);',
      '  } else if (tVert <= tBack && tVert <= tSide && dU <= 0.0) {',
      '    // 3D Interior Office Floor + Desk Silhouette',
      '    float desk = step(0.35, hit.z) * step(hit.z, 0.82) * step(0.18, hit.x) * step(hit.x, 0.82);',
      '    surfCol = mix(vec3(0.26, 0.25, 0.24), vec3(0.38, 0.25, 0.15), desk);',
      '  } else if (tBack <= tSide) {',
      '    // 3D Back Wall: Warm Cream Wall + Teak Academic Bookshelves / Credenza!',
      '    float isShelf = step(hit.y, 0.56) * step(0.12, hit.x) * step(hit.x, 0.88);',
      '    float shelfRow = fract(hit.y * 7.0);',
      '    float books = fract(hit.x * 18.0 + floor(hit.y * 7.0) * 3.1);',
      '    vec3 bookTint = mix(vec3(0.34, 0.20, 0.12), vec3(0.56, 0.44, 0.30), step(0.45, books));',
      '    bookTint *= smoothstep(0.0, 0.14, shelfRow);',
      '    vec3 wallPaint = mix(vec3(0.62, 0.59, 0.53), vec3(0.54, 0.56, 0.58), fract(rSeed * 5.1));',
      '    // Subtle framed certificate/painting on upper back wall',
      '    float frame = step(0.64, hit.y) * step(hit.y, 0.86) * step(0.34, hit.x) * step(hit.x, 0.66);',
      '    wallPaint = mix(wallPaint, vec3(0.38, 0.32, 0.24), frame * 0.55);',
      '    surfCol = mix(wallPaint, bookTint, isShelf);',
      '  } else {',
      '    // 3D Left / Right Partition Wall with Doorway Frame!',
      '    float door = step(hit.y, 0.74) * step(0.55, hit.z) * step(hit.z, 0.92);',
      '    surfCol = mix(vec3(0.56, 0.54, 0.49), vec3(0.34, 0.22, 0.14), door * 0.75);',
      '  }',
      '',
      '  // Corner & depth ambient occlusion inside 3D room',
      '  float roomAO = (1.0 - 0.34 * (hit.z / depthRatio))',
      '               * (0.78 + 0.22 * smoothstep(0.0, 0.18, min(hit.x, 1.0 - hit.x)))',
      '               * (0.80 + 0.20 * smoothstep(0.0, 0.18, hit.y));',
      '  bool isRoomLit = fract(rSeed * 13.7) < (isAtrium > 0.5 ? 0.88 : 0.56);',
      '  float lightLevelDay = isRoomLit ? 0.52 : 0.26;',
      '  float lightLevelNight = isRoomLit ? 1.15 : 0.03;',
      '  float activeLight = mix(lightLevelDay, lightLevelNight, uIsNight);',
      '  vec3 warmLamp = mix(vec3(0.92, 0.94, 0.98), vec3(1.0, 0.80, 0.48), uIsNight * 0.85 + 0.15);',
      '  vec3 litRoom = surfCol * roomAO * activeLight * warmLamp;',
      '  if (isRoomLit) {',
      '    litRoom += vec3(1.0, 0.92, 0.76) * trofferGlow * mix(0.45, 1.45, uIsNight);',
      '  }',
      '  return litRoom;',
      '}',
      '#endif',
      shader.fragmentShader
    ]
      .join('\n')
      // 1. PRE-LIGHTING BAND-LIMITED SURFACE MICRO-ALBEDO & SUBTLE CAVITY AO
      .replace(
        '#include <color_fragment>',
        [
          '#include <color_fragment>',
          'float omniPxSize = max(length(dFdx(vOmniWorldPos)), length(dFdy(vOmniWorldPos)));',
          'vec4 omniSurfDetail = vec4(0.0, 1.0, 0.0, 1.0);',
          '#if OMNI_PROFILE != 4 && OMNI_PROFILE != 5',
          'if (uShaderQuality > 0.5 && uZebraInspectMode < 0.5) {',
          '  omniSurfDetail = omniEvalSurfaceDetail(uSurfaceProfile, vOmniWorldPos, normalize(vOmniWorldNormal), diffuseColor.rgb, omniPxSize);',
          '  diffuseColor.rgb *= clamp(omniSurfDetail.y, 0.84, 1.16) * omniSurfDetail.w;',
          '}',
          '#endif'
        ].join('\n')
      )
      // 2. MICRO-SURFACE & WET-WEATHER PBR ROUGHNESS MODULATION
      .replace(
        '#include <roughnessmap_fragment>',
        [
          '#include <roughnessmap_fragment>',
          'if (uShaderQuality > 0.5 && uZebraInspectMode < 0.5) {',
          '  roughnessFactor = clamp(roughnessFactor + omniSurfDetail.z, 0.04, 0.98);',
          '}',
          'float omniSkyExp = 1.0;',
          'float omniPuddleMask = 0.0;',
          'if (uWetness > 0.005 && uZebraInspectMode < 0.5) {',
          '  omniSkyExp = omniSampleSkyExposure(vOmniWorldPos);',
          '  float effWetR = uWetness * (0.16 + 0.84 * omniSkyExp);',
          '  float upnessR = smoothstep(0.90, 0.992, normalize(vOmniWorldNormal).y);',
          '  float pFieldR = omniFbm2(vOmniWorldPos.xz * 0.095);',
          '  omniPuddleMask = upnessR * omniSkyExp * smoothstep(0.65 - effWetR * 0.20, 0.78 - effWetR * 0.14, pFieldR) * step(-1.2, vOmniWorldPos.y);',
          '  roughnessFactor = mix(roughnessFactor, max(0.08, roughnessFactor * 0.34), effWetR * 0.84);',
          '  roughnessFactor = mix(roughnessFactor, 0.032, omniPuddleMask * min(1.0, effWetR * 1.25));',
          '}'
        ].join('\n')
      )
      // 3. PER-PIXEL ANALYTICAL TANGENT-PLANE MICRO-NORMAL PERTURBATION (0 2x2 Quad Static, Soft Clamped <= 6.3 deg!)
      .replace(
        '#include <normal_fragment_maps>',
        [
          '#include <normal_fragment_maps>',
          '#if OMNI_PROFILE != 4 && OMNI_PROFILE != 5',
          'if (uShaderQuality > 0.5 && uZebraInspectMode < 0.5 && uBumpScale > 0.01) {',
          '  float camDist = length(vViewPosition);',
          '  float bumpFade = (1.0 - smoothstep(26.0, 82.0, camDist)) * (1.0 - omniPuddleMask * 0.88);',
          '  if (bumpFade > 0.005) {',
          '    vec3 wN = normalize(vOmniWorldNormal);',
          '    vec3 upRef = abs(wN.y) < 0.92 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);',
          '    vec3 tU = normalize(cross(upRef, wN));',
          '    vec3 tV = cross(wN, tU);',
          '    float eps = clamp(omniPxSize * 1.65, 0.004, 0.09);',
          '    float hU = omniEvalSurfaceDetail(uSurfaceProfile, vOmniWorldPos + tU * eps, wN, diffuseColor.rgb, omniPxSize).x;',
          '    float hV = omniEvalSurfaceDetail(uSurfaceProfile, vOmniWorldPos + tV * eps, wN, diffuseColor.rgb, omniPxSize).x;',
          '    float slopeU = clamp(((hU - omniSurfDetail.x) / eps) * uBumpScale * bumpFade, -0.11, 0.11);',
          '    float slopeV = clamp(((hV - omniSurfDetail.x) / eps) * uBumpScale * bumpFade, -0.11, 0.11);',
          '    vec3 worldGrad = slopeU * tU + slopeV * tV;',
          '    vec3 viewGrad = (viewMatrix * vec4(worldGrad, 0.0)).xyz;',
          '    normal = normalize(normal - viewGrad);',
          '  }',
          '}',
          '#endif'
        ].join('\n')
      )
      // 4. ULTRA LINEAR-HDR PHYSICAL SHADING (PRE-TONEMAPPING: Oren-Nayar, Botanical SSS,
      //    3D Interior Parallax + Solar Glass Fresnel, Wet Dielectric Clearcoat, Puddle Ripples, & Mie Aerial Perspective!)
      .replace(
        '#include <tonemapping_fragment>',
        [
          'if (uShaderQuality > 0.5 && uZebraInspectMode < 0.5) {',
          '  vec3 wNorm = normalize(vOmniWorldNormal);',
          '  vec3 viewDirW = normalize(cameraPosition - vOmniWorldPos);',
          '  vec3 sunL = normalize(uSunDir);',
          '  float NdotV = max(dot(wNorm, viewDirW), 0.0);',
          '  float NdotL = max(dot(wNorm, sunL), 0.0);',
          '  float sunVisUltra = clamp(1.0 - uCloudCover * 0.55 - uRainIntensity * 0.48, 0.10, 1.0) * (1.0 - uIsNight * 0.84);',
          '',
          '#if OMNI_PROFILE != 4 && OMNI_PROFILE != 5',
          '  // A. FUJII OREN-NAYAR MICROFACET ROUGH DIFFUSE RETRO-REFLECTION (Concrete, Stone, Clay, Bark)',
          '  if (roughnessFactor > 0.32) {',
          '    float sON = dot(sunL, viewDirW) - NdotL * NdotV;',
          '    float tON = sON > 0.0 ? max(NdotL, NdotV) + 1e-4 : 1.0;',
          '    float sig2 = roughnessFactor * roughnessFactor;',
          '    float orenRetro = 0.30 * (sig2 / (sig2 + 0.13)) * max(0.0, sON / tON) * NdotL;',
          '    gl_FragColor.rgb += diffuseColor.rgb * uSunColor * orenRetro * sunVisUltra * 0.38;',
          '  }',
          '#endif',
          '',
          '#if OMNI_PROFILE == 11',
          '  // B. BEER-LAMBERT BOTANICAL SUBSURFACE SCATTERING (LEAF TRANSLUCENCY) & CUTICLE FRESNEL',
          '  {',
          '    float VdotNegL = clamp(dot(viewDirW, -sunL), 0.0, 1.0);',
          '    float leafTrans = pow(VdotNegL, 2.8) * 0.56 + clamp((dot(wNorm, sunL) + 0.52) / 1.52, 0.0, 1.0) * 0.20;',
          '    vec3 chlorophyll = diffuseColor.rgb * vec3(1.16, 1.44, 0.48);',
          '    gl_FragColor.rgb += chlorophyll * uSunColor * leafTrans * sunVisUltra * (0.38 + 0.62 * omniSkyExp) * 0.44;',
          '    float cuticleSheen = pow(1.0 - NdotV, 4.0) * clamp(wNorm.y * 0.5 + 0.5, 0.0, 1.0) * 0.042 * (1.0 - uIsNight * 0.78);',
          '    gl_FragColor.rgb += mix(vec3(0.40, 0.56, 0.74), vec3(0.78, 0.48, 0.28), uDuskFactor) * cuticleSheen;',
          '  }',
          '#endif',
          '',
          '#if OMNI_PROFILE == 4 || OMNI_PROFILE == 5',
          '  // C. TRUE 3D RAY-BOX PARALLAX INTERIOR ROOMS + EXTERIOR FRESNEL SKY/CLOUD/SUN GLASS OPTICS',
          '  if (abs(wNorm.y) < 0.45) {',
          '    float baseGlassLuma = dot(diffuseColor.rgb, vec3(0.299, 0.587, 0.114));',
          '    float paneMask = 1.0 - smoothstep(0.22, 0.38, baseGlassLuma);',
          '    if (paneMask > 0.02) {',
          '      vec3 room3D = omniEvalInteriorParallaxRoom(vOmniWorldPos, wNorm, viewDirW, OMNI_PROFILE == 5 ? 1.0 : 0.0);',
          '      float fresnel = 0.08 + 0.90 * pow(1.0 - NdotV, 4.2);',
          '      vec3 reflDir = reflect(-viewDirW, wNorm);',
          '      vec3 dayHorizon = mix(vec3(0.54, 0.70, 0.86), vec3(0.92, 0.48, 0.24), uDuskFactor);',
          '      vec3 dayZenith = mix(vec3(0.16, 0.38, 0.72), vec3(0.18, 0.20, 0.42), uDuskFactor);',
          '      vec3 daySkyRefl = mix(vec3(0.16, 0.22, 0.20), mix(dayHorizon, dayZenith, clamp(reflDir.y, 0.0, 1.0)), smoothstep(-0.10, 0.24, reflDir.y));',
          '      float cloudRefl = smoothstep(0.44, 0.82, omniFbm2(reflDir.xz / max(0.18, abs(reflDir.y) + 0.15) * 1.4));',
          '      daySkyRefl = mix(daySkyRefl, mix(vec3(0.88, 0.91, 0.95), vec3(0.95, 0.62, 0.40), uDuskFactor), cloudRefl * step(0.02, reflDir.y) * (1.0 - uRainIntensity * 0.7));',
          '      vec3 extRefl = mix(daySkyRefl, vec3(0.025, 0.042, 0.082), uIsNight);',
          '      float glassSunSpec = pow(max(dot(reflDir, sunL), 0.0), 180.0) * sunVisUltra * 2.2;',
          '      extRefl += uSunColor * glassSunSpec + vec3(0.75, 0.86, 1.0) * uLightningFlash * 0.75;',
          '      vec3 glassComp = mix(room3D, extRefl, fresnel * mix(0.78, 0.32, uIsNight));',
          '      gl_FragColor.rgb = mix(gl_FragColor.rgb, glassComp, paneMask * 0.88);',
          '    }',
          '  }',
          '#endif',
          '',
          '  // D. ANALYTICAL ARCHITECTURAL EAVE & GROUND CONTACT AO',
          '  float contactAO = 1.0;',
          '  if (vOmniWorldPos.y > uYSegStart && vOmniWorldPos.y < uYSegStart + uSegStepH * uSegCount && abs(wNorm.y) < 0.7) {',
          '    float phase = fract((vOmniWorldPos.y - uYSegStart) / uSegStepH);',
          '    contactAO *= mix(0.48, 1.0, smoothstep(0.98, 0.54, phase));',
          '  }',
          '  if (vOmniWorldPos.y > 0.01 && vOmniWorldPos.y < 1.8) {',
          '    contactAO *= mix(0.70, 1.0, smoothstep(0.0, 1.8, vOmniWorldPos.y));',
          '  }',
          '',
          '#if OMNI_PROFILE == 9',
          '  // D2. 360-DEGREE COASTAL STONE TURAP WAVE-SPLASH & SWASH WET BOULDER BAND (PULAU U & PULAU I)',
          '  if (vOmniWorldPos.y < -0.78 && length(vOmniWorldPos.xz - vec2(-18.0, 34.0)) > 62.0) {',
          '    float swashPhase = dot(vOmniWorldPos.xz, vec2(0.052, 0.038));',
          '    float swashLine = -1.04 + 0.11 * sin(uTime * 1.10 + swashPhase) + 0.06 * cos(uTime * 0.79 - swashPhase * 0.82);',
          '    float splashNoise = omniNoise(vOmniWorldPos.xz * 2.4) * 0.12;',
          '    float splashWet = smoothstep(swashLine + splashNoise, swashLine - 0.34, vOmniWorldPos.y);',
          '    if (splashWet > 0.01) {',
          '      gl_FragColor.rgb *= mix(1.0, 0.54, splashWet);',
          '      vec3 spRefl = reflect(-viewDirW, wNorm);',
          '      float spFres = (0.0204 + 0.9796 * pow(1.0 - NdotV, 5.0)) * splashWet * 0.48;',
          '      vec3 spSky = mix(mix(vec3(0.34, 0.46, 0.56), vec3(0.68, 0.42, 0.26), uDuskFactor), vec3(0.04, 0.07, 0.13), uIsNight);',
          '      float spSun = pow(max(dot(spRefl, sunL), 0.0), 110.0) * sunVisUltra * splashWet * 0.65;',
          '      gl_FragColor.rgb = mix(gl_FragColor.rgb, spSky, spFres) + uSunColor * spSun;',
          '    }',
          '  }',
          '#endif',
          '',
          '  // E. DUAL-LAYER WET DIELECTRIC CLEARCOAT (IOR 1.333), PUDDLE RIPPLES, & GRAVITY RIVULETS',
          '  if (uWetness > 0.005) {',
          '    float effWet = uWetness * (0.16 + 0.84 * omniSkyExp);',
          '    float effRain = uRainIntensity * omniSkyExp;',
          '    float porous = smoothstep(0.26, 0.76, roughnessFactor);',
          '    gl_FragColor.rgb *= mix(1.0, 0.66, effWet * porous * 0.88);',
          '',
          '    // Thin-film dielectric water clearcoat reflection on wet roofs, concrete, & stone!',
          '#if OMNI_PROFILE != 4 && OMNI_PROFILE != 5',
          '    {',
          '      float waterFres = (0.0204 + 0.9796 * pow(1.0 - NdotV, 5.0)) * effWet * 0.42;',
          '      vec3 wRefl = reflect(-viewDirW, wNorm);',
          '      vec3 wetSkyCol = mix(mix(vec3(0.36, 0.44, 0.54), vec3(0.68, 0.42, 0.28), uDuskFactor), vec3(0.05, 0.08, 0.14), uIsNight);',
          '      float wetSunGlint = pow(max(dot(wRefl, sunL), 0.0), 96.0) * sunVisUltra * effWet * 0.55;',
          '      gl_FragColor.rgb = mix(gl_FragColor.rgb, wetSkyCol, waterFres) + uSunColor * wetSunGlint;',
          '    }',
          '#endif',
          '',
          '    float pStrength = omniPuddleMask * min(1.0, effWet * 1.25);',
          '    if (pStrength > 0.015) {',
          '      float ripX = 0.0;',
          '      float ripZ = 0.0;',
          '      float ringGlint = 0.0;',
          '      if (effRain > 0.03) {',
          '        for (int rl = 0; rl < 2; rl++) {',
          '          float gScale = rl == 0 ? 1.55 : 2.65;',
          '          vec2 gp = vOmniWorldPos.xz * gScale;',
          '          vec2 gi = floor(gp);',
          '          vec2 gf = fract(gp) - 0.5;',
          '          float hSeed = omniHash(gi + float(rl) * 37.0);',
          '          vec2 rCenter = (vec2(omniHash(gi + 5.1), omniHash(gi + 13.7)) - 0.5) * 0.48;',
          '          float life = fract(uTime * (1.28 + float(rl) * 0.48) + hSeed);',
          '          vec2 dv = gf - rCenter;',
          '          float dist = length(dv) + 1e-4;',
          '          float front = life * 0.44;',
          '          float env = exp(-abs(dist - front) * 28.0) * (1.0 - life) * smoothstep(0.0, 0.06, life);',
          '          float wv = sin((dist - front) * 96.0) * env * min(1.3, effRain);',
          '          ripX += (dv.x / dist) * wv;',
          '          ripZ += (dv.y / dist) * wv;',
          '          ringGlint += max(0.0, wv) * 0.55;',
          '        }',
          '      }',
          '      vec3 pNorm = normalize(vec3(ripX * 0.36, 1.0, ripZ * 0.36));',
          '      float fres = 0.0204 + 0.9796 * pow(1.0 - max(dot(viewDirW, pNorm), 0.0), 4.6);',
          '      vec3 puddleSkyDay = mix(vec3(0.42, 0.52, 0.62), vec3(0.22, 0.26, 0.32), clamp(uRainIntensity, 0.0, 1.0));',
          '      puddleSkyDay = mix(puddleSkyDay, vec3(0.76, 0.44, 0.24), uDuskFactor * (1.0 - uRainIntensity * 0.6));',
          '      vec3 puddleSky = mix(puddleSkyDay, vec3(0.04, 0.07, 0.13), uIsNight);',
          '      puddleSky += vec3(0.75, 0.86, 1.0) * uLightningFlash * 0.95;',
          '      gl_FragColor.rgb = mix(gl_FragColor.rgb * 0.80, puddleSky, clamp(pStrength * (0.30 + 0.68 * fres), 0.0, 0.90));',
          '      gl_FragColor.rgb += vec3(0.76, 0.84, 0.94) * ringGlint * pStrength * (0.40 - 0.20 * uIsNight + 0.50 * uLightningFlash);',
          '    }',
          '    if (effRain > 0.06 && wNorm.y < 0.88 && wNorm.y > -0.15) {',
          '      vec2 horizTan = normalize(vec2(-wNorm.z, wNorm.x) + vec2(1e-4, 0.0));',
          '      float lat = dot(vOmniWorldPos.xz, horizTan);',
          '      float flowY = vOmniWorldPos.y + uTime * (1.45 + 1.25 * effRain) * (1.0 - max(0.0, wNorm.y) * 0.42);',
          '      float rivTrack = omniNoise(vec2(lat * 10.5 + sin(vOmniWorldPos.y * 2.4) * 0.32, flowY * 1.35));',
          '      float rivMask = smoothstep(0.74, 0.93, rivTrack) * min(1.0, effRain);',
          '      float graze = pow(1.0 - NdotV, 2.6);',
          '      vec3 rivTint = mix(vec3(0.55, 0.64, 0.74), vec3(0.16, 0.22, 0.32), uIsNight) + vec3(0.75, 0.86, 1.0) * uLightningFlash;',
          '      gl_FragColor.rgb += rivTint * rivMask * (0.06 + 0.16 * graze);',
          '    }',
          '  }',
          '',
          '  if (uLightningFlash > 0.01) {',
          '    gl_FragColor.rgb += vec3(0.68, 0.80, 1.0) * uLightningFlash * omniSkyExp * (0.10 + 0.18 * max(0.0, wNorm.y));',
          '  }',
          '',
          '  float surfLuma = dot(gl_FragColor.rgb, vec3(0.299, 0.587, 0.114));',
          '  float nightMod = mix(1.0, mix(0.72, 1.0, smoothstep(0.06, 0.28, surfLuma)), uIsNight);',
          '  gl_FragColor.rgb *= contactAO * nightMod;',
          '',
          '#if OMNI_PROFILE == 9',
          '  // E1B. PULAU I GUNUNG GEDE BOGOR — KAWAH RATU WEATHERED ANDESITE & HYDROTHERMAL MINERAL CRUST (ZERO RED MAGMA!)',
          '  if (vOmniWorldPos.z > 670.0 && vOmniWorldPos.y > 42.5) {',
          '    vec2 dVol = vOmniWorldPos.xz - vec2(-32.0, 698.0);',
          '    float rVol = length(dVol);',
          '    if (rVol < 21.5) {',
          '      float floorMask = smoothstep(14.8, 9.6, rVol) * smoothstep(49.5, 46.2, vOmniWorldPos.y);',
          '      vec2 cellK = omniVoronoiStone(dVol * 0.52);',
          '      float mineralNoise = omniFbm2(dVol * 0.68);',
          '      if (floorMask > 0.005) {',
          '        // Quiet geothermal mineral-silica floor & damp fumarole andesite gravel (0.0 red fire emissive!)',
          '        float silicaVein = (1.0 - smoothstep(0.02, 0.16, cellK.x)) * (0.55 + 0.45 * mineralNoise);',
          '        vec3 dampAndesite = gl_FragColor.rgb * 0.84;',
          '        vec3 hydrothermalSilica = gl_FragColor.rgb * vec3(1.14, 1.15, 1.04);',
          '        gl_FragColor.rgb = mix(dampAndesite, hydrothermalSilica, clamp(silicaVein * 0.72, 0.0, 0.85) * floorMask);',
          '      }',
          '      float calderaWall = smoothstep(21.5, 14.5, rVol) * smoothstep(59.0, 45.0, vOmniWorldPos.y);',
          '      float strata = 0.5 + 0.5 * sin(vOmniWorldPos.y * 2.4 + mineralNoise * 3.1);',
          '      gl_FragColor.rgb *= mix(1.0, 0.90 + 0.16 * strata, calderaWall * 0.65);',
          '    }',
          '  }',
          '#endif',
          '',
          '#if OMNI_PROFILE == 10',
          '  // E2. TRUE EMISSIVE NIGHTTIME ICAO HELIPAD, PULAU I RUNWAY EDGE/THRESHOLD/PAPI & WINDSOCK BEACONS (0.0 during DAY!)',
          '  if (uIsNight > 0.05) {',
          '    if (diffuseColor.g > 0.78 && diffuseColor.g > diffuseColor.r * 2.4) {',
          '      gl_FragColor.rgb += vec3(0.08, 2.25, 0.82) * uIsNight;',
          '    } else if (diffuseColor.r > 0.82 && diffuseColor.r > diffuseColor.g * 2.8 && vOmniWorldPos.y > 5.2) {',
          '      float beaconPulse = 0.30 + 0.70 * smoothstep(0.35, 0.55, sin(uTime * 4.8) * 0.5 + 0.5);',
          '      gl_FragColor.rgb += vec3(2.85, 0.18, 0.10) * beaconPulse * uIsNight;',
          '    } else if (vOmniWorldPos.z > 655.0 && vOmniWorldPos.y > 0.72 && vOmniWorldPos.y < 1.35) {',
          '      if (diffuseColor.r > 0.86 && diffuseColor.g > 0.76 && diffuseColor.b > 0.42) {',
          '        gl_FragColor.rgb += vec3(2.45, 2.15, 1.45) * uIsNight;',
          '      } else if (diffuseColor.r > 0.82 && diffuseColor.r > diffuseColor.g * 2.6) {',
          '        gl_FragColor.rgb += vec3(2.40, 0.18, 0.10) * uIsNight;',
          '      }',
          '    }',
          '  }',
          '#endif',
          '',
          '  // F. DIRECTIONAL HENYEY-GREENSTEIN MIE AERIAL PERSPECTIVE IN-SCATTERING',
          '  float camDistAP = length(cameraPosition - vOmniWorldPos);',
          '  if (camDistAP > 30.0) {',
          '    float cosSunV = max(dot(-viewDirW, sunL), 0.0);',
          '    float miePhase = pow(cosSunV, 5.0) * (1.0 - uIsNight * 0.85) * (1.0 - uRainIntensity * 0.65);',
          '    float airScatter = 1.0 - exp(-max(0.0, camDistAP - 30.0) * 0.00085);',
          '    vec3 mieTint = mix(vec3(0.76, 0.84, 0.95), vec3(0.98, 0.56, 0.26), uDuskFactor);',
          '    gl_FragColor.rgb += mieTint * airScatter * miePhase * 0.14;',
          '  }',
          '}',
          'if (uZebraInspectMode < 0.5) {',
          '  vec3 hvsSunL = normalize(uSunDir);',
          '  gl_FragColor.rgb = nexusApplyBoundaryLayerAerialPerspective(',
          '    gl_FragColor.rgb,',
          '    cameraPosition,',
          '    vOmniWorldPos,',
          '    hvsSunL,',
          '    uDuskFactor,',
          '    uNightFactor,',
          '    uCloudCover,',
          '    uRainIntensity,',
          '    uLightningFlash',
          '  );',
          '  vec3 hvsRayW = normalize(vOmniWorldPos - cameraPosition);',
          '  gl_FragColor.rgb = nexusApplyHumanEyeHDR(gl_FragColor.rgb, hvsRayW, hvsSunL, uSunColor, uHvsParams, uHvsGlare);',
          '}'
        ].join('\n')
      )
      // 5. STUDIO ZEBRA / CLAY DIAGNOSTIC OVERRIDE (POST-TONEMAPPING)
      .replace(
        '#include <dithering_fragment>',
        [
          'if (uZebraInspectMode > 1.5) {',
          '  vec3 nClay = normalize(vOmniWorldNormal);',
          '  vec3 keyDir = normalize(vec3(0.52, 0.76, 0.38));',
          '  vec3 fillDir = normalize(vec3(-0.48, 0.32, -0.55));',
          '  float diffKey = max(dot(nClay, keyDir), 0.0);',
          '  float diffFill = max(dot(nClay, fillDir), 0.0) * 0.35;',
          '  float rim = pow(1.0 - max(dot(nClay, normalize(cameraPosition - vOmniWorldPos)), 0.0), 2.5) * 0.18;',
          '  gl_FragColor.rgb = vec3(0.76, 0.73, 0.69) * (0.24 + 0.72 * diffKey + diffFill) + vec3(rim);',
          '} else if (uZebraInspectMode > 0.5) {',
          '  vec3 viewDir = normalize(cameraPosition - vOmniWorldPos);',
          '  vec3 refl = reflect(-viewDir, normalize(vOmniWorldNormal));',
          '  float stripe = smoothstep(-0.1, 0.1, sin(refl.y * 28.0 + refl.x * 8.0));',
          '  gl_FragColor.rgb = mix(vec3(0.12, 0.14, 0.18), vec3(0.95, 0.97, 1.0), stripe);',
          '}',
          '#include <dithering_fragment>'
        ].join('\n')
      );
  };

  return material;
}

function makeProceduralCanvasTexture(width, height, drawFn, repeatX = 1, repeatY = 1) {
  if (typeof document !== 'undefined' && typeof document.createElement === 'function') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      drawFn(ctx, width, height);
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(repeatX, repeatY);
    tex.anisotropy = 16;
    tex.needsUpdate = true;
    return tex;
  }

  const data = new Uint8Array(16 * 16 * 4);
  const tex = new THREE.DataTexture(data, 16, 16);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(repeatX, repeatY);
  return tex;
}

/**
 * 1. 2048x1024 Crisp Off-White Architectural Precast Concrete (chamfered joints, tie-rod pins, mineral speckle)
 */
export function createPrecastConcreteTexture() {
  return makeProceduralCanvasTexture(
    2048,
    1024,
    (ctx, w, h) => {
      ctx.fillStyle = '#eae5db';
      ctx.fillRect(0, 0, w, h);

      const cols = 16;
      const rows = 6;
      const cw = w / cols;
      const ch = h / rows;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const jitter = ((c * 37 + r * 19) % 11) - 5;
          const base = 233 + jitter;
          const x0 = c * cw;
          const y0 = r * ch;

          const pGrad = ctx.createLinearGradient(x0, y0, x0, y0 + ch);
          pGrad.addColorStop(0.0, `rgb(${base + 4}, ${base + 1}, ${base - 7})`);
          pGrad.addColorStop(0.85, `rgb(${base + 1}, ${base - 2}, ${base - 10})`);
          pGrad.addColorStop(1.0, `rgb(${base - 5}, ${base - 8}, ${base - 16})`);
          ctx.fillStyle = pGrad;
          ctx.fillRect(x0 + 2, y0 + 2, cw - 4, ch - 4);

          // Chamfered top/left sun highlight & bottom/right shadow bevel
          ctx.fillStyle = 'rgba(255, 253, 248, 0.45)';
          ctx.fillRect(x0 + 2, y0 + 2, cw - 4, 2);
          ctx.fillRect(x0 + 2, y0 + 2, 2, ch - 4);

          // Formwork tie-rod plugged cone dimples at 4 corners of each precast slab!
          for (const tx of [x0 + 18, x0 + cw - 18]) {
            for (const ty of [y0 + 18, y0 + ch - 18]) {
              ctx.fillStyle = 'rgba(135, 128, 116, 0.42)';
              ctx.beginPath();
              ctx.arc(tx, ty, 3.2, 0, Math.PI * 2);
              ctx.fill();
              ctx.fillStyle = 'rgba(248, 245, 238, 0.55)';
              ctx.beginPath();
              ctx.arc(tx - 0.8, ty - 0.8, 1.8, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        }
      }

      // Fine mineral limestone/quartz aggregate speckles
      for (let i = 0; i < 4200; i++) {
        const px = ((i * 7919) % w);
        const py = ((i * 6151) % h);
        const dark = i % 3 !== 0;
        ctx.fillStyle = dark ? 'rgba(130, 124, 112, 0.16)' : 'rgba(255, 254, 250, 0.24)';
        ctx.fillRect(px, py, (i % 2) + 1.2, (i % 2) + 1.2);
      }

      // Recessed architectural expansion reveal grooves ('Tali Air')
      ctx.strokeStyle = 'rgba(128, 121, 109, 0.52)';
      ctx.lineWidth = 3;
      for (let c = 0; c <= cols; c++) {
        ctx.beginPath();
        ctx.moveTo(c * cw, 0);
        ctx.lineTo(c * cw, h);
        ctx.stroke();
      }
      for (let r = 0; r <= rows; r++) {
        ctx.beginPath();
        ctx.moveTo(0, r * ch);
        ctx.lineTo(w, r * ch);
        ctx.stroke();
      }
    },
    2,
    2
  );
}

/**
 * 2. 1024x1024 Weathered Tropical Flat Concrete Roof Deck Slab
 */
export function createWeatheredFlatRoofTexture() {
  return makeProceduralCanvasTexture(
    1024,
    1024,
    (ctx, w, h) => {
      ctx.fillStyle = '#8b8f92';
      ctx.fillRect(0, 0, w, h);

      const grid = 12;
      const cell = w / grid;
      for (let y = 0; y < grid; y++) {
        for (let x = 0; x < grid; x++) {
          const v = 132 + (((x * 29 + y * 43) % 23) - 11);
          ctx.fillStyle = `rgb(${v}, ${v + 3}, ${v + 5})`;
          ctx.fillRect(x * cell + 2.5, y * cell + 2.5, cell - 5, cell - 5);
          ctx.strokeStyle = 'rgba(72, 76, 78, 0.55)';
          ctx.lineWidth = 2;
          ctx.strokeRect(x * cell + 1.5, y * cell + 1.5, cell - 3, cell - 3);
        }
      }

      for (let i = 0; i < 36; i++) {
        const px = ((i * 97) % 100) * 0.01 * w;
        const py = ((i * 53) % 100) * 0.01 * h;
        const rad = 36 + (i % 6) * 24;
        const g = ctx.createRadialGradient(px, py, 3, px, py, rad);
        g.addColorStop(0, 'rgba(62, 66, 63, 0.36)');
        g.addColorStop(1, 'rgba(62, 66, 63, 0.0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(px, py, rad, 0, Math.PI * 2);
        ctx.fill();
      }

      for (let i = 0; i < 3200; i++) {
        const px = (i * 4999) % w;
        const py = (i * 3571) % h;
        ctx.fillStyle = i % 2 === 0 ? 'rgba(55,58,60,0.22)' : 'rgba(205,210,212,0.18)';
        ctx.fillRect(px, py, 2, 2);
      }
    },
    1,
    1
  );
}

/**
 * 3. 2048x1024 Pressed Terracotta Brick & Mortar Coursework
 */
export function createTerracottaBrickTexture() {
  return makeProceduralCanvasTexture(
    2048,
    1024,
    (ctx, w, h) => {
      ctx.fillStyle = '#8b7e6e'; // Recessed sand-lime mortar bed
      ctx.fillRect(0, 0, w, h);

      const rows = 24;
      const cols = 20;
      const bh = h / rows;
      const bw = w / cols;

      for (let r = 0; r < rows; r++) {
        const offset = (r % 2) * (bw * 0.5);
        for (let c = -1; c <= cols; c++) {
          const shade = ((r * 23 + c * 41) % 25) - 12;
          const R = Math.min(255, Math.max(70, 156 + shade));
          const G = Math.min(255, Math.max(40, 83 + Math.round(shade * 0.6)));
          const B = Math.min(255, Math.max(30, 66 + Math.round(shade * 0.5)));
          const x0 = c * bw + offset + 3;
          const y0 = r * bh + 3;
          ctx.fillStyle = `rgb(${R}, ${G}, ${B})`;
          ctx.fillRect(x0, y0, bw - 6, bh - 6);

          // Top kiln-brick edge highlight & bottom mortar shadow
          ctx.fillStyle = 'rgba(235, 165, 138, 0.25)';
          ctx.fillRect(x0, y0, bw - 6, 2);
          ctx.fillStyle = 'rgba(48, 18, 12, 0.38)';
          ctx.fillRect(x0, y0 + bh - 8, bw - 6, 2.5);
        }
      }
    },
    3,
    3
  );
}

/**
 * 4. 2048x1024 Authentic Interlocking Javanese Clay Roof Tiles ('Genteng Kodok / Plentong UI Depok')
 */
export function createJavaneseClayTileTexture() {
  return makeProceduralCanvasTexture(
    2048,
    1024,
    (ctx, w, h) => {
      ctx.fillStyle = '#752f23';
      ctx.fillRect(0, 0, w, h);

      const tileCols = 64;
      const tileRows = 20;
      const tw = w / tileCols;
      const th = h / tileRows;

      for (let r = 0; r < tileRows; r++) {
        const stag = (r % 2) * (tw * 0.12);
        for (let c = -1; c <= tileCols; c++) {
          const v = ((r * 29 + c * 53) % 25) - 12;
          const sunBleach = ((r * 13 + c * 7) % 15) === 0 ? 15 : 0;
          const x0 = c * tw + stag;
          const y0 = r * th;

          // Curved S-profile Genteng Kodok shading + drainage flute channel
          const grad = ctx.createLinearGradient(x0, y0, x0 + tw, y0);
          grad.addColorStop(0.0, `rgb(${112 + v}, ${45 + Math.round(v * 0.4)}, ${35})`);
          grad.addColorStop(0.24, `rgb(${154 + v + sunBleach}, ${67 + Math.round(v * 0.5) + sunBleach}, ${52 + sunBleach})`);
          grad.addColorStop(0.54, `rgb(${182 + v + sunBleach}, ${82 + Math.round(v * 0.5) + sunBleach}, ${65 + sunBleach})`);
          grad.addColorStop(0.84, `rgb(${136 + v}, ${57 + Math.round(v * 0.4)}, ${44})`);
          grad.addColorStop(1.0, `rgb(${94 + v}, ${36 + Math.round(v * 0.3)}, ${28})`);
          ctx.fillStyle = grad;
          ctx.fillRect(x0, y0, tw - 0.8, th - 2.5);

          // Upper interlocking nib highlight & deep lower course overlap drop-shadow!
          ctx.fillStyle = 'rgba(235, 148, 120, 0.22)';
          ctx.fillRect(x0 + tw * 0.18, y0 + th - 6.0, tw * 0.55, 2.2);
          ctx.fillStyle = 'rgba(34, 10, 7, 0.62)';
          ctx.fillRect(x0, y0 + th - 3.6, tw, 3.6);
        }
      }
    },
    5,
    4
  );
}

/**
 * 5. 2048x2048 Signature Rektorat UI Courtyard Paving: Split-Face Andesite Grid + Crisp Honed White Diamonds (◇◇◇)
 */
export function createHorizontalStonePaverTexture() {
  return makeProceduralCanvasTexture(
    2048,
    2048,
    (ctx, w, h) => {
      // Warm Palimanan buff-limestone structural grid bands (Mavic Nadir #3 & #5!)
      ctx.fillStyle = '#a89f82';
      ctx.fillRect(0, 0, w, h);

      const grid = 8;
      const cell = w / grid;
      const pad = 13;

      for (let y = 0; y < grid; y++) {
        for (let x = 0; x < grid; x++) {
          const cx = (x + 0.5) * cell;
          const cy = (y + 0.5) * cell;
          const bx = x * cell + pad;
          const by = y * cell + pad;
          const bw = cell - pad * 2;

          // Recessed perimeter mortar groove around each courtyard square
          ctx.strokeStyle = 'rgba(54, 52, 45, 0.65)';
          ctx.lineWidth = 3;
          ctx.strokeRect(bx - 1.5, by - 1.5, bw + 3, bw + 3);

          // 4x4 Individual Split-Face Batu Andesit Sub-Pavers inside each square!
          const subN = 4;
          const subW = bw / subN;
          for (let sy = 0; sy < subN; sy++) {
            for (let sx = 0; sx < subN; sx++) {
              const delta = (((x * 4 + sx) * 31 + (y * 4 + sy) * 17) % 15) - 7;
              const v = 93 + delta;
              ctx.fillStyle = `rgb(${v}, ${v + 2}, ${v - 3})`;
              ctx.fillRect(bx + sx * subW + 1.5, by + sy * subW + 1.5, subW - 3, subW - 3);
            }
          }

          // Signature Honed White-Marble Diamond Inlay Band (◇◇◇)
          const isDiamondBand = y === 2 || y === 3 || (x === 3 && y % 2 === 0);
          if (isDiamondBand) {
            const outerD = cell * 0.43;
            const midD = cell * 0.39;
            const innerD = cell * 0.21;

            // Recessed dark mortar border around diamond inlay
            ctx.strokeStyle = 'rgba(46, 48, 44, 0.72)';
            ctx.lineWidth = 4;
            ctx.fillStyle = '#efeCE2';
            ctx.beginPath();
            ctx.moveTo(cx, cy - outerD);
            ctx.lineTo(cx + outerD, cy);
            ctx.lineTo(cx, cy + outerD);
            ctx.lineTo(cx - outerD, cy);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();

            // Subtle inner honed stone chamfer line
            ctx.strokeStyle = 'rgba(185, 180, 168, 0.65)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cx, cy - midD);
            ctx.lineTo(cx + midD, cy);
            ctx.lineTo(cx, cy + midD);
            ctx.lineTo(cx - midD, cy);
            ctx.closePath();
            ctx.stroke();

            // Dark honed andesite diamond core
            ctx.fillStyle = '#4e514c';
            ctx.beginPath();
            ctx.moveTo(cx, cy - innerD);
            ctx.lineTo(cx + innerD, cy);
            ctx.lineTo(cx, cy + innerD);
            ctx.lineTo(cx - innerD, cy);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
          } else if ((x + y) % 4 === 0) {
            const sm = cell * 0.115;
            ctx.fillStyle = '#e5e2d8';
            ctx.fillRect(cx - sm, cy - 5, sm * 2, 10);
            ctx.fillRect(cx - 5, cy - sm, 10, sm * 2);
          }
        }
      }

      // Chiselled stone mineral speckles across entire 2K plaza atlas
      for (let i = 0; i < 6500; i++) {
        const px = (i * 7877) % w;
        const py = (i * 5437) % h;
        ctx.fillStyle = i % 2 === 0 ? 'rgba(35,36,34,0.18)' : 'rgba(245,243,236,0.16)';
        ctx.fillRect(px, py, 2, 2);
      }
    },
    1,
    1
  );
}

/**
 * 6. 1024x256 Weathered Black-and-White Zebra-Striped Concrete Curb Texture
 */
export function createZebraCurbTexture() {
  return makeProceduralCanvasTexture(
    1024,
    256,
    (ctx, w, h) => {
      const stripes = 8;
      const sw = w / stripes;
      for (let i = 0; i < stripes; i++) {
        ctx.fillStyle = i % 2 === 0 ? '#f1efe9' : '#1e2125';
        ctx.fillRect(i * sw, 0, sw, h);
        // Concrete joint line between curb blocks
        ctx.fillStyle = 'rgba(90, 88, 84, 0.45)';
        ctx.fillRect(i * sw - 1.5, 0, 3, h);
      }
      // Chamfer top highlight & road-gutter tire grime AO along base
      ctx.fillStyle = 'rgba(255, 255, 255, 0.22)';
      ctx.fillRect(0, 12, w, 6);
      const grime = ctx.createLinearGradient(0, h - 42, 0, h);
      grime.addColorStop(0.0, 'rgba(20, 20, 18, 0.0)');
      grime.addColorStop(1.0, 'rgba(20, 20, 18, 0.42)');
      ctx.fillStyle = grime;
      ctx.fillRect(0, h - 42, w, 42);
    },
    1,
    1
  );
}

/**
 * 7. 2048x512 6-Bay White-Mullioned Dark Architectural Ribbon Glass + Strict #000000 Night Mask
 */
export function createRibbonGlassTextures() {
  const albedo = makeProceduralCanvasTexture(
    2048,
    512,
    (ctx, w, h) => {
      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0.0, '#0a1216');
      grad.addColorStop(0.5, '#132129');
      grad.addColorStop(1.0, '#0b1419');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      const panes = 24;
      const pw = w / panes;
      for (let i = 0; i < panes; i++) {
        // Dark EPDM rubber glazing gasket around each window pane
        ctx.strokeStyle = '#070b0d';
        ctx.lineWidth = 4;
        ctx.strokeRect(i * pw + 8, 22, pw - 16, h - 46);

        // Crisp off-white architectural mullion with side shadow bevel
        ctx.fillStyle = '#e7e2d7';
        ctx.fillRect(i * pw - 6, 0, 12, h);
        ctx.fillStyle = 'rgba(95, 90, 82, 0.45)';
        ctx.fillRect(i * pw + 5, 0, 2, h);
      }
      ctx.fillStyle = '#e7e2d7';
      ctx.fillRect(0, 0, w, 20);
      ctx.fillRect(0, h - 22, w, 22);
    },
    1,
    1
  );

  const emissiveMap = makeProceduralCanvasTexture(
    2048,
    512,
    (ctx, w, h) => {
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, w, h);

      const panes = 24;
      const pw = w / panes;
      for (let i = 0; i < panes; i++) {
        const hash = (i * 73 + 19) % 100;
        if (hash < 54) {
          const warm = ctx.createLinearGradient(0, 24, 0, h - 24);
          warm.addColorStop(0.0, '#ffe3a8');
          warm.addColorStop(0.68, '#dfa44e');
          warm.addColorStop(1.0, '#593812');
          ctx.fillStyle = warm;
          ctx.fillRect(i * pw + 11, 26, pw - 22, h - 52);
        }
      }
    },
    1,
    1
  );

  return { map: albedo, emissiveMap };
}

/**
 * 8. 1024x2048 Central Recessed Vertical Dark-Glass X-Diamond Atrium Curtain Wall
 */
export function createAtriumDiamondGlassTextures() {
  const albedo = makeProceduralCanvasTexture(
    1024,
    2048,
    (ctx, w, h) => {
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0.0, '#091014');
      grad.addColorStop(0.5, '#14222a');
      grad.addColorStop(1.0, '#091014');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      const tiers = 6;
      const th = h / tiers;
      ctx.fillStyle = 'rgba(34, 46, 54, 0.58)';
      for (let t = 0; t < tiers; t++) {
        ctx.fillRect(0, t * th - 16, w, 32);
      }

      ctx.strokeStyle = '#38464e';
      ctx.lineWidth = 18;
      for (let t = 0; t < tiers; t++) {
        const y0 = t * th;
        const y1 = (t + 1) * th;
        ctx.beginPath();
        ctx.moveTo(0, y0);
        ctx.lineTo(w * 0.5, (y0 + y1) * 0.5);
        ctx.lineTo(w, y0);
        ctx.moveTo(0, y1);
        ctx.lineTo(w * 0.5, (y0 + y1) * 0.5);
        ctx.lineTo(w, y1);
        ctx.stroke();
      }

      ctx.fillStyle = '#222c32';
      ctx.fillRect(0, 0, 24, h);
      ctx.fillRect(w - 24, 0, 24, h);
    },
    1,
    1
  );

  const emissiveMap = makeProceduralCanvasTexture(
    1024,
    2048,
    (ctx, w, h) => {
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, w, h);

      const tiers = 6;
      const th = h / tiers;
      for (let t = 0; t < tiers; t++) {
        const yMid = (t + 0.5) * th;
        const g = ctx.createRadialGradient(w * 0.5, yMid, 14, w * 0.5, yMid, w * 0.44);
        g.addColorStop(0.0, '#ffd48a');
        g.addColorStop(0.65, '#8c5b22');
        g.addColorStop(1.0, '#000000');
        ctx.fillStyle = g;
        ctx.fillRect(36, t * th + 32, w - 72, th - 64);
      }

      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 26;
      for (let t = 0; t < tiers; t++) {
        const y0 = t * th;
        const y1 = (t + 1) * th;
        ctx.beginPath();
        ctx.moveTo(0, y0);
        ctx.lineTo(w * 0.5, (y0 + y1) * 0.5);
        ctx.lineTo(w, y0);
        ctx.moveTo(0, y1);
        ctx.lineTo(w * 0.5, (y0 + y1) * 0.5);
        ctx.lineTo(w, y1);
        ctx.stroke();
      }
    },
    1,
    1
  );

  return { map: albedo, emissiveMap };
}

/**
 * 9. 1024x1024 Sculpted Bas-Relief Makara UI Emblem on Crisp Off-White Concrete Crown Panel
 */
export function createMakaraEmblemTextures() {
  const drawMakaraMotif = (ctx, w, h, isEmissive) => {
    const cx = w * 0.5;
    const cy = h * 0.52;
    const rOuter = Math.min(w, h) * 0.36;

    if (isEmissive) {
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, w, h);
    } else {
      ctx.fillStyle = '#eae5db';
      ctx.fillRect(0, 0, w, h);

      // Coffered architectural recessed frame around Crown panel
      ctx.strokeStyle = 'rgba(138, 132, 120, 0.55)';
      ctx.lineWidth = w * 0.016;
      ctx.strokeRect(w * 0.045, h * 0.045, w * 0.91, h * 0.91);
      ctx.strokeStyle = 'rgba(165, 158, 145, 0.40)';
      ctx.lineWidth = w * 0.006;
      ctx.strokeRect(w * 0.075, h * 0.075, w * 0.85, h * 0.85);

      const shGrad = ctx.createRadialGradient(cx, cy + 8, rOuter * 0.3, cx, cy + 8, rOuter * 1.08);
      shGrad.addColorStop(0.0, 'rgba(84, 79, 70, 0.36)');
      shGrad.addColorStop(0.85, 'rgba(84, 79, 70, 0.20)');
      shGrad.addColorStop(1.0, 'rgba(84, 79, 70, 0.0)');
      ctx.fillStyle = shGrad;
      ctx.beginPath();
      ctx.arc(cx, cy + 5, rOuter * 1.08, 0, Math.PI * 2);
      ctx.fill();
    }

    const reliefMain = isEmissive ? '#ffd24c' : '#857f73';
    const reliefHighlight = isEmissive ? '#fff0a6' : '#b2ab9c';
    const reliefShadow = isEmissive ? '#c99322' : '#585349';

    ctx.strokeStyle = reliefShadow;
    ctx.lineWidth = w * 0.036;
    ctx.beginPath();
    ctx.arc(cx, cy + 3, rOuter * 0.92, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = reliefMain;
    ctx.lineWidth = w * 0.026;
    ctx.beginPath();
    ctx.arc(cx, cy, rOuter * 0.92, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = reliefMain;
    ctx.beginPath();
    ctx.moveTo(cx - rOuter * 0.12, cy + rOuter * 0.62);
    ctx.quadraticCurveTo(cx - rOuter * 0.05, cy, cx - rOuter * 0.06, cy - rOuter * 0.46);
    ctx.lineTo(cx + rOuter * 0.06, cy - rOuter * 0.46);
    ctx.quadraticCurveTo(cx + rOuter * 0.05, cy, cx + rOuter * 0.12, cy + rOuter * 0.62);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = reliefHighlight;
    ctx.beginPath();
    ctx.ellipse(cx, cy - rOuter * 0.62, rOuter * 0.11, rOuter * 0.19, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = reliefMain;
    ctx.lineWidth = w * 0.028;
    for (let tier = 0; tier < 4; tier++) {
      const yB = cy - rOuter * (0.34 - tier * 0.21);
      const span = rOuter * (0.34 + tier * 0.13);
      for (const sign of [-1, 1]) {
        ctx.beginPath();
        ctx.moveTo(cx, yB + rOuter * 0.06);
        ctx.quadraticCurveTo(cx + sign * span * 0.55, yB - rOuter * 0.1, cx + sign * span, yB - rOuter * 0.02);
        ctx.stroke();

        ctx.fillStyle = reliefHighlight;
        ctx.beginPath();
        ctx.arc(cx + sign * span, yB - rOuter * 0.02, rOuter * 0.072, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.strokeStyle = reliefHighlight;
    ctx.lineWidth = w * 0.03;
    ctx.beginPath();
    ctx.arc(cx, cy + rOuter * 0.44, rOuter * 0.38, 0.16 * Math.PI, 0.84 * Math.PI);
    ctx.stroke();
  };

  const map = makeProceduralCanvasTexture(1024, 1024, (ctx, w, h) => drawMakaraMotif(ctx, w, h, false), 1, 1);
  const emissiveMap = makeProceduralCanvasTexture(1024, 1024, (ctx, w, h) => drawMakaraMotif(ctx, w, h, true), 1, 1);
  return { map, emissiveMap };
}
