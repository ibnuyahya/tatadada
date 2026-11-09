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
    uFogFactor: { value: 0.0 },
    uLightningFlash: { value: 0.0 },
    uStreetlampPower: { value: 0.0 },
    uWindVec: { value: new THREE.Vector3(0.65, 0.0, 0.35) },
    uCollisionMap: { value: RainCollisionBake.createFallbackTexture() },
    uCollisionBounds: {
      value: new THREE.Vector4(
        RainCollisionBake.DEFAULT_MIN_X,
        RainCollisionBake.DEFAULT_MIN_Z,
        RainCollisionBake.DEFAULT_MAX_X - RainCollisionBake.DEFAULT_MIN_X,
        RainCollisionBake.DEFAULT_MAX_Z - RainCollisionBake.DEFAULT_MIN_Z
      )
    },
    uCollisionHeightRange: {
      value: new THREE.Vector2(
        RainCollisionBake.HEIGHT_MIN_Y,
        RainCollisionBake.HEIGHT_MAX_Y - RainCollisionBake.HEIGHT_MIN_Y
      )
    },
    uCollisionReady: { value: 0.0 },
    uVideotronPortalAlpha: { value: 0.0 },
    uVideotronShortMode: { value: 0.0 },
    uFountainNightGlow: { value: new THREE.Vector4(0.0, 0.0, 0.0, 0.0) },
    uFleetKinematics: { value: new THREE.Vector4(0.0, 0.0, 0.0, 0.0) },
    uAiFarStaggerDist: { value: 9999.0 },
    uCockpitGlassCutRadius: { value: 0.0 },
    uCloseDerivativeDist: { value: 22.0 },
    uCloudShadowPhase: { value: new THREE.Vector2(0.0, 0.0) },
    uSolarDirectTrans: { value: 1.0 },
    uBoneMatrixTex: { value: null },
    uGpuSkinningEnabled: { value: 1.0 },
    uHvsParams: hvsUniforms.uHvsParams,
    uHvsGlare: hvsUniforms.uHvsGlare,
    uGenshinMode: hvsUniforms.uGenshinMode
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
  const hvsFallback = HumanEyeHDROptics.createUniforms();
  const rawProfId = Math.round(Number(surfaceProfile) || 0);
  const omniGroup =
    rawProfId === 4 || rawProfId === 5
      ? 4
      : rawProfId === 9
        ? 9
        : rawProfId === 10
          ? 10
          : rawProfId === 11
            ? 11
            : rawProfId === 1 || rawProfId === 2 || rawProfId === 3 || rawProfId === 7
              ? 1
              : 6;
  material.customProgramCacheKey = () => `nexus-omni-v55-g${omniGroup}`;

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
    shader.uniforms.uCloseDerivativeDist = sharedUniforms.uCloseDerivativeDist || { value: 22.0 };
    shader.uniforms.uZebraInspectMode = sharedUniforms.uZebraInspectMode || { value: 0.0 };
    shader.uniforms.uTime = sharedUniforms.uTime || { value: 0.0 };
    shader.uniforms.uWetness = sharedUniforms.uWetness || { value: 0.0 };
    shader.uniforms.uRainIntensity = sharedUniforms.uRainIntensity || { value: 0.0 };
    shader.uniforms.uFogFactor = sharedUniforms.uFogFactor || { value: 0.0 };
    shader.uniforms.uLightningFlash = sharedUniforms.uLightningFlash || { value: 0.0 };
    shader.uniforms.uStreetlampPower = sharedUniforms.uStreetlampPower || { value: 0.0 };
    shader.uniforms.uWindVec =
      sharedUniforms.uWindVec || { value: new THREE.Vector3(0.65, 0.0, 0.35) };
    shader.uniforms.uCollisionMap =
      sharedUniforms.uCollisionMap || { value: RainCollisionBake.createFallbackTexture() };
    shader.uniforms.uCollisionBounds =
      sharedUniforms.uCollisionBounds || {
        value: new THREE.Vector4(
          RainCollisionBake.DEFAULT_MIN_X,
          RainCollisionBake.DEFAULT_MIN_Z,
          RainCollisionBake.DEFAULT_MAX_X - RainCollisionBake.DEFAULT_MIN_X,
          RainCollisionBake.DEFAULT_MAX_Z - RainCollisionBake.DEFAULT_MIN_Z
        )
      };
    shader.uniforms.uCollisionHeightRange =
      sharedUniforms.uCollisionHeightRange || {
        value: new THREE.Vector2(
          RainCollisionBake.HEIGHT_MIN_Y,
          RainCollisionBake.HEIGHT_MAX_Y - RainCollisionBake.HEIGHT_MIN_Y
        )
      };
    shader.uniforms.uCollisionReady = sharedUniforms.uCollisionReady || { value: 0.0 };
    shader.uniforms.uVideotronPortalAlpha = sharedUniforms.uVideotronPortalAlpha || { value: 0.0 };
    shader.uniforms.uVideotronShortMode = sharedUniforms.uVideotronShortMode || { value: 0.0 };
    shader.uniforms.uFountainNightGlow =
      sharedUniforms.uFountainNightGlow || { value: new THREE.Vector4(0.0, 0.0, 0.0, 0.0) };
    shader.uniforms.uCloudShadowPhase =
      sharedUniforms.uCloudShadowPhase || { value: new THREE.Vector2(0.0, 0.0) };
    shader.uniforms.uSolarDirectTrans = sharedUniforms.uSolarDirectTrans || { value: 1.0 };
    shader.uniforms.uHvsParams = sharedUniforms.uHvsParams || hvsFallback.uHvsParams;
    shader.uniforms.uHvsGlare = sharedUniforms.uHvsGlare || hvsFallback.uHvsGlare;
    shader.uniforms.uGenshinMode = sharedUniforms.uGenshinMode || hvsFallback.uGenshinMode;

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
      `#define OMNI_GROUP ${omniGroup}`,
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
      'uniform float uCloseDerivativeDist;',
      'uniform float uZebraInspectMode;',
      'uniform float uTime;',
      'uniform float uWetness;',
      'uniform float uRainIntensity;',
      'uniform float uFogFactor;',
      'uniform float uLightningFlash;',
      'uniform float uStreetlampPower;',
      'uniform sampler2D uCollisionMap;',
      'uniform vec4 uCollisionBounds;',
      'uniform vec2 uCollisionHeightRange;',
      'uniform float uCollisionReady;',
      'uniform float uVideotronPortalAlpha;',
      'uniform float uVideotronShortMode;',
      'uniform vec4 uFountainNightGlow;',
      'uniform vec2 uCloudShadowPhase;',
      'uniform float uSolarDirectTrans;',
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
      'float omniEvalCloudShadow(vec3 wp, vec3 sunDir) {',
      '  if (uNightFactor > 0.94 || uIsNight > 0.92 || uRainIntensity > 0.85 || uCloudCover < 0.03) return 1.0;',
      '  float cloudAlt = max(240.0, 920.0 - wp.y);',
      '  vec2 rayProj = wp.xz + sunDir.xz * (cloudAlt / max(0.24, sunDir.y));',
      '  vec2 cUV = rayProj * 0.00205 + uCloudShadowPhase * 0.0148 + vec2(5.7, -3.4);',
      '  float fMacro = omniFbm4(cUV);',
      '  float fPuff = omniFbm2(OMNI_ROT * cUV * 2.35 + vec2(11.4, 6.2));',
      '  float cField = fMacro * 0.72 + fPuff * 0.28;',
      '  float thresh = mix(0.60, 0.28, clamp(uCloudCover, 0.0, 1.0));',
      '  float occ = smoothstep(thresh, thresh + 0.26, cField);',
      '  float dayStrength = (1.0 - uIsNight * 0.92) * (1.0 - clamp(uRainIntensity * 0.85, 0.0, 0.85));',
      '  float maxDarken = mix(0.28, 0.42, clamp(uCloudCover, 0.0, 1.0)) * dayStrength;',
      '  return 1.0 - occ * maxDarken;',
      '}',
      // Cellular Voronoi River-Stone ('Batu Kali') Masonry Evaluator -> vec2(edgeMortarDist, cellIdHash)
      '#if OMNI_GROUP == 9',
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
      '  if (wp.y < -3.65 && abs(wp.x + 356.0) < 5.4 && wp.z > 325.8 && wp.z < 624.2) return 0.0;',
      '  if (uCollisionReady < 0.5) return 1.0;',
      '  vec2 cuv = (wp.xz - uCollisionBounds.xy) / uCollisionBounds.zw;',
      '  if (cuv.x < 0.002 || cuv.x > 0.998 || cuv.y < 0.002 || cuv.y > 0.998) return 1.0;',
      '  vec4 cTex = texture2D(uCollisionMap, cuv);',
      '  float yTop = uCollisionHeightRange.x + ((cTex.r * 255.0 * 256.0 + cTex.g * 255.0) / 65535.0) * uCollisionHeightRange.y;',
      '  float isCanopyAbove = step(0.98, cTex.a);',
      '  float shelter = smoothstep(0.55, 1.65, yTop - wp.y);',
      '  return mix(1.0, isCanopyAbove * 0.28, shelter);',
      '}',
      '#if OMNI_GROUP != 4 && OMNI_GROUP != 11',
      'float omniSampleOverhangTap(vec2 uvTap, float wpY) {',
      '  vec4 ct = texture2D(uCollisionMap, uvTap);',
      '  float yT = uCollisionHeightRange.x + ((ct.r * 255.0 * 256.0 + ct.g * 255.0) / 65535.0) * uCollisionHeightRange.y;',
      '  float dh = yT - wpY;',
      '  float isLeaf = step(0.98, ct.a);',
      '  float sh = smoothstep(0.55, 2.40, dh) * (1.0 - smoothstep(14.0, 30.0, dh));',
      '  return sh * mix(0.36, 0.24, isLeaf);',
      '}',
      'float omniEvalWorldOverhangAO(vec3 wp, vec3 wn) {',
      '  float aoOut = 1.0;',
      '  float upWeight = clamp(wn.y * 0.65 + 0.35, 0.0, 1.0);',
      '  // 1. Soft 5-tap Canopy & Building Overhang Ambient Contact Shadow (zero mutable array for fast D3D11 fxc!)',
      '  if (uCollisionReady > 0.5 && upWeight > 0.08) {',
      '    vec2 cuv0 = (wp.xz - uCollisionBounds.xy) / uCollisionBounds.zw;',
      '    if (cuv0.x > 0.004 && cuv0.x < 0.996 && cuv0.y > 0.004 && cuv0.y < 0.996) {',
      '      vec2 dUV = vec2(1.65) / uCollisionBounds.zw;',
      '      float occSum = omniSampleOverhangTap(cuv0, wp.y)',
      '                   + omniSampleOverhangTap(cuv0 + vec2(dUV.x, 0.0), wp.y)',
      '                   + omniSampleOverhangTap(cuv0 - vec2(dUV.x, 0.0), wp.y)',
      '                   + omniSampleOverhangTap(cuv0 + vec2(0.0, dUV.y), wp.y)',
      '                   + omniSampleOverhangTap(cuv0 - vec2(0.0, dUV.y), wp.y);',
      '      float dapple = 0.88 + 0.12 * omniNoise(wp.xz * 0.85);',
      '      aoOut *= 1.0 - (occSum * 0.20) * upWeight * dapple;',
      '    }',
      '  }',
      '  // 2. Analytical Soft Ground/Approach Contact Shadow beneath Jembatan Teksas UI Deck & Girders',
      '  if (wp.z > 149.0 && wp.z < 342.0 && abs(wp.x + 28.0) < 5.2) {',
      '    float uBr = clamp((wp.z - 245.65) / 74.0, -1.3, 1.3);',
      '    float deckYEst = 4.15 + 4.95 * max(0.0, 1.0 - uBr * uBr);',
      '    if (wp.y < deckYEst - 1.45) {',
      '      float latMask = 1.0 - smoothstep(2.2, 4.8, abs(wp.x + 28.0));',
      '      float hFade = 1.0 - smoothstep(2.0, 10.5, deckYEst - wp.y);',
      '      aoOut *= 1.0 - 0.34 * latMask * (0.45 + 0.55 * hFade) * upWeight;',
      '    }',
      '  }',
      '  return clamp(aoOut, 0.52, 1.0);',
      '}',
      // =========================================================================
      // O(1) ANALYTIC PHOTOCELL STREETLAMP SENSOR & 40-LAMP GROUND LIGHT POOL KERNEL
      // =========================================================================
      'float omniEvalLampPhotocellPower(vec2 lampXZ, float globalSensor) {',
      '  if (globalSensor <= 0.001) return 0.0;',
      '  float tol = (omniHash(floor(lampXZ * 0.4 + 0.5)) - 0.5) * 0.14;',
      '  return smoothstep(clamp(0.08 + tol, 0.005, 0.22), clamp(0.42 + tol, 0.25, 0.65), globalSensor);',
      '}',
      'float omniIslandHeadRadius(float th) {',
      '  float eastCove = -46.0 * pow(0.5 + 0.5 * cos(th + 0.25), 4.4);',
      '  float neHead = 23.5 * pow(0.5 + 0.5 * cos(th + 1.04), 8.5);',
      '  float sLobe = 15.5 * pow(0.5 + 0.5 * cos(th - 1.12), 2.3);',
      '  float swPen = 20.5 * pow(0.5 + 0.5 * cos(th - 2.44), 6.5);',
      '  float meand = 2.15 * sin(3.0 * th + 0.40) + 1.20 * cos(5.0 * th - 0.70) + 0.55 * sin(8.0 * th + 1.10);',
      '  return 121.0 + eastCove + neHead + sLobe + swPen + meand;',
      '}',
      'vec3 omniEvalNearestStreetlamp(vec2 xz) {',
      '  vec2 bestP = vec2(9999.0);',
      '  float bestD2 = 1e9;',
      '  if (xz.y < 165.0 && xz.y > -72.0 && xz.x > -104.0 && xz.x < 105.0) {',
      '    // A. 12 Rotunda Ring Lamps (O(1) angular snap)',
      '    vec2 dRot = xz - vec2(-2.5, 106.5);',
      '    float ang = atan(dRot.y, dRot.x) - 0.26179939;',
      '    float idxA = floor(ang / 0.52359878 + 0.5);',
      '    float snapA = idxA * 0.52359878 + 0.26179939;',
      '    vec2 pA = vec2(-2.5, 106.5) + vec2(cos(snapA), sin(snapA)) * 34.9;',
      '    float d2A = dot(xz - pA, xz - pA);',
      '    bestP = pA; bestD2 = d2A;',
      '    // B. 4 South Rektorat Approach Lamps',
      '    if (xz.y > 35.0 && xz.y < 84.0 && abs(xz.x + 2.5) < 36.0) {',
      '      bool upperRow = xz.y > 60.0;',
      '      bool eastSide = xz.x > -2.5;',
      '      vec2 pB = upperRow ? vec2(eastSide ? 6.5 : -11.5, 68.5) : vec2(eastSide ? 16.2 : -21.2, 51.5);',
      '      float d2B = dot(xz - pB, xz - pB);',
      '      if (d2B < bestD2) { bestD2 = d2B; bestP = pB; }',
      '    }',
      '    // C. 6 West Helipad Boulevard Lamps',
      '    if (xz.x < -34.0 && xz.y > 85.0 && xz.y < 128.0) {',
      '      float cx = xz.x < -77.6 ? -83.2 : (xz.x < -61.0 ? -72.0 : -50.0);',
      '      float czOff = cx < -80.0 ? 6.9 : 5.7;',
      '      float cz = 106.5 + (xz.y >= 106.5 ? czOff : -czOff);',
      '      vec2 pC = vec2(cx, cz);',
      '      float d2C = dot(xz - pC, xz - pC);',
      '      if (d2C < bestD2) { bestD2 = d2C; bestP = pC; }',
      '    }',
      '    // D. 6 West Courtyard / Golf-Cart Shuttle Boulevard Lamps (x = -55.4, z = -22..88)',
      '    if (xz.x < -38.0 && xz.x > -74.0 && xz.y > -38.0 && xz.y < 104.0) {',
      '      float kD = clamp(floor((xz.y + 22.0) / 22.0 + 0.5), 0.0, 5.0);',
      '      vec2 pD = vec2(-55.4, -22.0 + kD * 22.0);',
      '      float d2D = dot(xz - pD, xz - pD);',
      '      if (d2D < bestD2) { bestD2 = d2D; bestP = pD; }',
      '    }',
      '    // E. 8 East Lakeside Promenade Lamps (z = -52..48)',
      '    if (xz.x > 36.0 && xz.y > -68.0 && xz.y < 64.0) {',
      '      float stepE = 100.0 / 7.0;',
      '      float kE = clamp(floor((xz.y + 52.0) / stepE + 0.5), 0.0, 7.0);',
      '      float lz = -52.0 + kE * stepE;',
      '      float dz = lz - 34.0;',
      '      float rH = omniIslandHeadRadius(atan(dz, 76.0));',
      '      float lx = -18.0 + sqrt(max(4.0, rH * rH - dz * dz));',
      '      vec2 pE = vec2(lx - 0.55, lz);',
      '      float d2E = dot(xz - pE, xz - pE);',
      '      if (d2E < bestD2) { bestD2 = d2E; bestP = pE; }',
      '    }',
      '    // G1. 2 Jembatan Teksas UI West Forecourt Lamps (x = -22.2 / -33.8, z = 148.5)',
      '    if (xz.y > 136.0 && xz.y < 164.0 && abs(xz.x + 28.0) < 22.0) {',
      '      vec2 pG1 = vec2(xz.x >= -28.0 ? -22.2 : -33.8, 148.5);',
      '      float d2G1 = dot(xz - pG1, xz - pG1);',
      '      if (d2G1 < bestD2) { bestD2 = d2G1; bestP = pG1; }',
      '    }',
      '  } else if (xz.y > 328.0 && xz.y < 362.0 && abs(xz.x + 28.0) < 24.0) {',
      '    // G2. 2 Jembatan Teksas UI East FIB Overlook Plaza Lamps (x = -22.2 / -33.8, z = 343.5)',
      '    bestP = vec2(xz.x >= -28.0 ? -22.2 : -33.8, 343.5);',
      '    bestD2 = dot(xz - bestP, xz - bestP);',
      '  } else if (xz.y > 642.0 && xz.y < 676.0 && xz.x > -294.0 && xz.x < -214.0) {',
      '    // F. 4 Pulau I West Aircraft Apron Lamps (z = 658.4, x = -276.5..-231.5)',
      '    float kF = clamp(floor((xz.x + 276.5) / 15.0 + 0.5), 0.0, 3.0);',
      '    bestP = vec2(-276.5 + kF * 15.0, 658.4);',
      '    bestD2 = dot(xz - bestP, xz - bestP);',
      '  }',
      '  // H1. 6 Pulau U West-Arm Arterial Lamps (x = -96..-326 step -46, z = 71.3..89.1)',
      '  if (xz.x < -76.0 && xz.x > -346.0 && xz.y > 54.0 && xz.y < 110.0) {',
      '    float kH1 = clamp(floor((-96.0 - xz.x) / 46.0 + 0.5), 0.0, 5.0);',
      '    vec2 pH1 = vec2(-96.0 - kH1 * 46.0, 71.3 + 2.36 * kH1 + 0.24 * kH1 * kH1);',
      '    float d2H1 = dot(xz - pH1, xz - pH1);',
      '    if (d2H1 < bestD2) { bestD2 = d2H1; bestP = pH1; }',
      '  }',
      '  // H2. 4 Pulau U Southern Arterial Lamps (z = 128..248 step 40)',
      '  if (xz.x < -334.0 && xz.x > -384.0 && xz.y > 106.0 && xz.y < 272.0) {',
      '    float kH2 = clamp(floor((xz.y - 128.0) / 40.0 + 0.5), 0.0, 3.0);',
      '    vec2 pH2 = vec2(-355.2 - 9.6 * kH2 + 3.0 * kH2 * kH2, 128.0 + kH2 * 40.0);',
      '    float d2H2 = dot(xz - pH2, xz - pH2);',
      '    if (d2H2 < bestD2) { bestD2 = d2H2; bestP = pH2; }',
      '  }',
      '  // H3. 4 Underpass Bawah Laut West & East Portal Gateway Lamps (z = 324.5 & z = 625.5)',
      '  if (abs(xz.x + 356.0) < 24.0 && ((xz.y > 306.0 && xz.y < 344.0) || (xz.y > 607.0 && xz.y < 645.0))) {',
      '    vec2 pH3 = vec2(xz.x >= -356.0 ? -348.1 : -363.9, xz.y < 480.0 ? 324.5 : 625.5);',
      '    float d2H3 = dot(xz - pH3, xz - pH3);',
      '    if (d2H3 < bestD2) { bestD2 = d2H3; bestP = pH3; }',
      '  }',
      '  // H4. 6 Northern Ferry Terminals (Dermaga U & Dermaga I) + Right-Arm Junction & North Headland Lamps',
      '  if (xz.y > 444.0 && xz.y < 482.0 && abs(xz.x + 48.0) < 24.0) {',
      '    vec2 pDU = vec2(xz.x >= -48.0 ? -41.2 : -54.8, 462.5);',
      '    float d2DU = dot(xz - pDU, xz - pDU);',
      '    if (d2DU < bestD2) { bestD2 = d2DU; bestP = pDU; }',
      '  } else if (xz.y > 592.0 && xz.y < 630.0 && abs(xz.x + 98.0) < 24.0) {',
      '    vec2 pDI = vec2(xz.x >= -98.0 ? -91.2 : -104.8, 611.0);',
      '    float d2DI = dot(xz - pDI, xz - pDI);',
      '    if (d2DI < bestD2) { bestD2 = d2DI; bestP = pDI; }',
      '  } else if (xz.y > 418.0 && xz.y < 460.0 && abs(xz.x + 164.5) < 22.0) {',
      '    vec2 pSJ = vec2(-164.5, 438.2);',
      '    float d2SJ = dot(xz - pSJ, xz - pSJ);',
      '    if (d2SJ < bestD2) { bestD2 = d2SJ; bestP = pSJ; }',
      '  } else if (xz.y > 376.0 && xz.y < 416.0 && abs(xz.x - 37.2) < 22.0) {',
      '    vec2 pNH = vec2(37.2, 396.0);',
      '    float d2NH = dot(xz - pNH, xz - pNH);',
      '    if (d2NH < bestD2) { bestD2 = d2NH; bestP = pNH; }',
      '  }',
      '  return vec3(bestP, bestD2);',
      '}',
      '#endif',
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
      '#if OMNI_GROUP != 4',
      // Returns vec4(heightMeters, albedoScale, roughnessDelta, microCavityAO)
      'vec4 omniEvalSurfaceDetail(float prof, vec3 wp, vec3 wn, vec3 baseCol, float pxSize) {',
      '  int pId = int(prof + 0.5);',
      '  vec3 aw = abs(wn);',
      '  vec2 wWallXZ = pow(aw.zx, vec2(6.0));',
      '  wWallXZ /= max(wWallXZ.x + wWallXZ.y, 1e-4);',
      '  float h = 0.0;',
      '  float alb = 1.0;',
      '  float rDelta = 0.0;',
      '  float ao = 1.0;',
      '',
      '#if OMNI_GROUP == 1',
      '  if (pId == 1) {',
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
      '  } else if (pId == 2) {',
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
      '  } else if (pId == 3) {',
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
      '  } else if (pId == 7) {',
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
      '  }',
      '#elif OMNI_GROUP == 6',
      '  if (pId == 6) {',
      '    // PROFILE 6: WEATHERED TROPICAL FLAT CONCRETE ROOF DECK & FOUNDATION',
      '    float bSlab = omniBand(pxSize, 1.0);',
      '    float jnt = mix(1.0, omniJoint1D(wp.x, 2.20, 0.014) * omniJoint1D(wp.z, 2.20, 0.014), bSlab * aw.y);',
      '    float screed = omniFbm3BL(wp, 3.5, pxSize);',
      '    h = -(1.0 - jnt) * 0.0020 + screed * 0.0008;',
      '    alb = 1.0 + screed * 0.038;',
      '    rDelta = screed * 0.04;',
      '    ao = mix(0.90, 1.0, jnt);',
      '',
      '  } else if (pId == 8) {',
      '    // PROFILE 8: SATIN ARCHITECTURAL BRONZE TRIM & CARVED MAKARA STONE',
      '    float satin = omniFbm3BL(wp, 12.0, pxSize);',
      '    h = satin * 0.00025;',
      '    alb = 1.0 + satin * 0.018;',
      '    rDelta = satin * 0.02;',
      '',
      '  } else {',
      '    float g1 = omniFbm3BL(wp, 5.0, pxSize);',
      '    h = g1 * 0.0006;',
      '    alb = 1.0 + g1 * 0.022;',
      '  }',
      '#elif OMNI_GROUP == 9',
      '  {',
      '    // PROFILE 9: VELVET CAMPUS GREEN LAWNS + RIPARIAN PEBBLE/MOSS BANKS + VOLCANIC BASALT CRAGS + 360-DEG VORONOI TURAP MASONRY',
      '    float macroClover = omniFbm3BL(wp, 0.38, pxSize);',
      '    float midHumus = omniFbm3BL(wp, 1.85, pxSize);',
      '    float softTurf = omniFbm3BL(wp, 6.8, pxSize);',
      '    float greenDominance = baseCol.g - max(baseCol.r, baseCol.b);',
      '    float isEarthyOrCobble = 1.0 - smoothstep(-0.015, 0.065, greenDominance);',
      '    h = midHumus * 0.0042 + softTurf * 0.0022;',
      '    alb = 1.0 + macroClover * 0.115 + midHumus * 0.068 + softTurf * 0.046;',
      '    rDelta = -softTurf * 0.035;',
      '    ao = 0.935 + 0.065 * smoothstep(-0.35, 0.35, midHumus + softTurf * 0.6);',
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
      '        vec2 vOff = wp.xz - vec2(-32.0, 698.0);',
      '        float vAng = atan(vOff.y, vOff.x);',
      '        float gully = omniFbm2(vec2(vAng * 7.5, wp.y * 0.095)) - 0.5;',
      '        float cragMacro = omniFbm3BL(vec3(wp.x * 0.52, wp.y * 0.88, wp.z * 0.52), 1.0, pxSize);',
      '        float cragRidge = omniFbm3BL(vec3(wp.x * 1.55 + cragMacro * 0.85, wp.y * 1.95, wp.z * 1.55), 1.8, pxSize);',
      '        float strataBand = omniFbm3BL(vec3(wp.x * 0.28, wp.y * 1.45 + cragMacro * 2.2, wp.z * 0.28), 1.0, pxSize) * (1.0 - smoothstep(0.52, 0.82, wn.y));',
      '        float vesicular = omniFbm3BL(wp, 5.6, pxSize);',
      '        float crevice = smoothstep(-0.32, 0.22, cragRidge + gully * 0.45 + strataBand * 0.35);',
      '        float hRock = cragMacro * 0.022 + cragRidge * 0.013 + gully * 0.010 + strataBand * 0.008 + vesicular * 0.003;',
      '        float albRock = mix(0.76, 1.12, crevice) + cragMacro * 0.072 + strataBand * 0.045 + vesicular * 0.032;',
      '        float rRock = 0.05 + (1.0 - crevice) * 0.07;',
      '        float aoRock = mix(0.72, 1.0, crevice) * (0.91 + 0.09 * cragMacro);',
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
      '  }',
      '#elif OMNI_GROUP == 10',
      '  {',
      '    // PROFILE 10: 3D BARK & RINGED PALMS, TEAK GRAIN, TIRE-WORN ROLLED ASPHALT, VIDEOTRON, & JEMBATAN TEKSAS UI STEEL/SKYLIGHTS',
      '    bool isStaticSpec = false;',
      '#ifndef USE_INSTANCING',
      '    if (vOmniUV.y < -1000.0) {',
      '      return vec4(0.0, 1.0, -0.45, 1.0);',
      '    }',
      '    bool inTeksasZone = (wp.z > 147.0 && wp.z < 349.0 && abs(wp.x + 28.0) < 12.0 && wp.y > 0.25);',
      '    if (inTeksasZone && baseCol.b > baseCol.r * 1.65 && baseCol.b > 0.28) {',
      '      // Multi-wall architectural polycarbonate skylight micro-flutes + wave surface sheen',
      '      float fluteBand = omniBand(pxSize, 24.0);',
      '      float fluteRib = sin(wp.x * 150.0) * 0.5 * fluteBand;',
      '      float polyWave = omniFbm3BL(vec3(wp.x * 4.0, wp.y * 4.0, wp.z * 8.0), 1.0, pxSize * 2.0);',
      '      h = polyWave * 0.00020 + fluteRib * 0.00018;',
      '      alb = 1.0 + polyWave * 0.016 + fluteRib * 0.022;',
      '      rDelta = -0.56;',
      '      ao = 0.985 + 0.015 * fluteRib;',
      '      isStaticSpec = true;',
      '    } else if (inTeksasZone && (max(baseCol.r, baseCol.g) > baseCol.b * 1.28 || (baseCol.r < 0.30 && baseCol.b < 0.32 && wp.y > 0.30))) {',
      '      // Weathered Krakatau BTKC Structural Steel: welded plate seams, brushed mill scale & subtle rain-runoff patina',
      '      float bWeld = omniBand(pxSize, 1.2);',
      '      float weldZ = mix(1.0, omniJoint1D(wp.z, 2.64, 0.016), bWeld);',
      '      float steelBrush = omniFbm3BL(vec3(wp.x * 6.5, wp.y * 6.5, wp.z * 13.0), 1.0, pxSize * 2.0);',
      '      float rainPatina = omniFbm3BL(vec3(wp.x * 4.2, wp.y * 1.15, wp.z * 4.2), 1.0, pxSize);',
      '      h = -(1.0 - weldZ) * 0.0014 + steelBrush * 0.00028;',
      '      alb = (0.985 + 0.015 * weldZ) + steelBrush * 0.024 + rainPatina * 0.032;',
      '      rDelta = -0.20 + (1.0 - weldZ) * 0.08 - rainPatina * 0.03;',
      '      ao = mix(0.90, 0.99, weldZ);',
      '      isStaticSpec = true;',
      '    } else if (vOmniUV.y > -422.0 && vOmniUV.y < -418.0) {',
      '      // Hot-Mix Rolled Asphalt Roadway: crushed basalt/quartz aggregate micro-relief + twin polished wheel-track lanes',
      '      float roadF = clamp(vOmniUV.y + 420.0, -1.0, 1.0);',
      '      float absRF = abs(roadF);',
      '      float wheelWear = exp(-pow((absRF - 0.38) * 6.6, 2.0)) + exp(-pow((absRF - 0.68) * 6.6, 2.0));',
      '      float gutterDust = smoothstep(0.84, 0.99, absRF);',
      '      float aggCoarse = omniFbm3BL(wp, 9.5, pxSize);',
      '      float aggFine = omniFbm3BL(wp, 24.0, pxSize);',
      '      float macroPatch = omniFbm3BL(wp, 0.55, pxSize);',
      '      h = (aggCoarse * 0.00068 + aggFine * 0.00032) * (1.0 - 0.35 * wheelWear);',
      '      alb = (1.0 - 0.055 * wheelWear + 0.065 * gutterDust) + macroPatch * 0.045 + aggCoarse * 0.042 + aggFine * 0.028;',
      '      rDelta = -0.14 * wheelWear + 0.06 * gutterDust - aggFine * 0.03;',
      '      ao = 0.95 + 0.05 * smoothstep(-0.35, 0.35, aggCoarse);',
      '      isStaticSpec = true;',
      '    }',
      '#endif',
      '    if (!isStaticSpec) {',
      '      float isVerticalBark = 1.0 - smoothstep(0.35, 0.72, aw.y);',
      '      if (isVerticalBark > 0.5) {',
      '        float isPalmTrunk = smoothstep(0.46, 0.58, baseCol.r) * (1.0 - smoothstep(0.06, 0.14, abs(baseCol.r - baseCol.b)));',
      '        if (isPalmTrunk > 0.5) {',
      '          float bRing = omniBand(pxSize, 4.2);',
      '          float palmRing = mix(1.0, omniJoint1D(wp.y, 0.24, 0.10), bRing);',
      '          float palmFib = omniFbm3BL(vec3(wp.x * 6.0, wp.y * 1.8, wp.z * 6.0), 1.0, pxSize * 2.0);',
      '          h = -(1.0 - palmRing) * 0.0018 + palmFib * 0.0005;',
      '          alb = mix(0.88, 1.03, palmRing) + palmFib * 0.035;',
      '          rDelta = (1.0 - palmRing) * 0.05;',
      '          ao = mix(0.88, 1.0, palmRing);',
      '        } else {',
      '          float barkRidge = omniFbm3BL(vec3(wp.x * 4.2, wp.y * 0.82, wp.z * 4.2), 1.0, pxSize);',
      '          float barkClose = omniFbm3BL(vec3(wp.x * 11.0, wp.y * 2.4, wp.z * 11.0), 1.0, pxSize * 3.0);',
      '          float baseMoss = smoothstep(2.4, 0.25, wp.y) * smoothstep(0.15, 0.55, barkRidge + 0.35);',
      '          h = barkRidge * 0.0034 + barkClose * 0.00075;',
      '          alb = (1.0 - baseMoss * 0.08) + barkRidge * 0.082 + barkClose * 0.032;',
      '          rDelta = 0.03 - baseMoss * 0.05;',
      '          ao = 0.92 + 0.08 * smoothstep(-0.35, 0.35, barkRidge);',
      '        }',
      '      } else {',
      '        float isTeak = smoothstep(0.14, 0.28, baseCol.r - baseCol.b);',
      '        float teakGrain = omniFbm3BL(vec3(wp.x * 2.2, wp.y * 12.0, wp.z * 12.0), 1.0, pxSize * 3.0);',
      '        float asphaltSoft = omniFbm3BL(wp, 10.5, pxSize);',
      '        float asphaltMacro = omniFbm3BL(wp, 0.85, pxSize);',
      '        h = mix(asphaltSoft * 0.00058, teakGrain * 0.00045, isTeak);',
      '        alb = 1.0 + mix(asphaltSoft * 0.036 + asphaltMacro * 0.035, teakGrain * 0.042, isTeak);',
      '        rDelta = mix(-asphaltSoft * 0.03, -0.12, isTeak);',
      '      }',
      '    }',
      '  }',
      '#elif OMNI_GROUP == 11',
      '  {',
      '    // PROFILE 11: MULTI-SPECIES BOTANICAL FOLIAGE CANOPY (Crisp 3D leaf-blade cluster relief, deep inter-bough crevices & waxy cuticle)',
      '    float softBough = omniFbm3BL(wp, 1.08, pxSize);',
      '    float midCluster = omniFbm3BL(wp, 3.45, pxSize);',
      '    float fineLeaf = omniFbm3BL(wp, 9.6, pxSize);',
      '    float microBlade = omniFbm3BL(vec3(wp.x * 18.5, wp.y * 14.0, wp.z * 18.5), 1.0, pxSize * 1.35);',
      '    float leafPocket = smoothstep(-0.36, 0.26, softBough * 0.62 + midCluster * 0.54 + fineLeaf * 0.22);',
      '    h = softBough * 0.026 + midCluster * 0.0115 + fineLeaf * 0.0042 + microBlade * 0.0016;',
      '    alb = mix(0.70, 1.20, leafPocket) + midCluster * 0.105 + fineLeaf * 0.068 + microBlade * 0.038;',
      '    rDelta = -0.095 - fineLeaf * 0.052;',
      '    ao = mix(0.60, 1.0, leafPocket) * (0.90 + 0.10 * smoothstep(-0.3, 0.3, fineLeaf));',
      '  }',
      '#endif',
      '  return vec4(h, alb, rDelta, ao);',
      '}',
      '#endif',
      '',
      // =========================================================================
      // TRUE 3D RAY-BOX PARALLAX INTERIOR ROOM MAPPING FOR ARCHITECTURAL WINDOWS!
      // Renders 3D office ceiling grids, glowing LED troffers, partition walls,
      // teak bookshelves/desks, & Venetian window blinds moving in 3D parallax!
      // =========================================================================
      '#if OMNI_GROUP == 4',
      'vec3 omniEvalInteriorParallaxRoom(vec3 wp, vec3 wn, vec3 viewDirOut, float isAtrium, float pxSize) {',,
      '  // Quantize facade normal to exact 45-degree architectural octants so vertex-normal interpolation',
      '  // across master-surface corners NEVER jitters roomOut, uHoriz, or integer room hash seeds!',
      '  float rawAz = atan(wn.z, wn.x);',
      '  float octIdx = floor(rawAz / 0.78539816 + 0.5);',
      '  float octKey = mod(octIdx + 8.0, 8.0);',
      '  float snapAz = octIdx * 0.78539816;',
      '  vec3 roomOut = vec3(cos(snapAz), 0.0, sin(snapAz));',
      '  vec3 roomUp = vec3(0.0, 1.0, 0.0);',
      '  vec3 roomRight = vec3(-roomOut.z, 0.0, roomOut.x);',
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
      '  float rSeed = omniHash(vec2(roomIdxX + octKey * 37.0, floorIdx * 3.0 + octKey * 11.0));',
      '',
      '  // Semi-drawn Horizontal Venetian Window Blinds on ~34% of administrative office rooms (Nyquist band-limited)!',
      '  float hasBlinds = step(0.66, rSeed) * (1.0 - isAtrium);',
      '  float blindDrop = 0.42 + 0.42 * fract(rSeed * 7.3);',
      '  if (hasBlinds > 0.5 && fy > (1.0 - blindDrop)) {',
      '    float blindBand = omniBand(pxSize, 9.3);',
      '    float slat = 0.5 + 0.5 * sin(fy * 190.0) * blindBand;',
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
      '    // 3D Interior Office Floor + Teak Desk + Computer Monitor Silhouette',
      '    float desk = step(0.35, hit.z) * step(hit.z, 0.82) * step(0.18, hit.x) * step(hit.x, 0.82);',
      '    float plant = step(0.24, hit.z) * step(hit.z, 0.44) * step(0.83, hit.x) * step(hit.x, 0.95) * step(0.45, fract(rSeed * 9.1));',
      '    surfCol = mix(vec3(0.26, 0.25, 0.24), vec3(0.38, 0.25, 0.15), desk);',
      '    surfCol = mix(surfCol, vec3(0.16, 0.34, 0.14), plant);',
      '  } else if (tBack <= tSide) {',
      '    // 3D Back Wall: Warm Cream Wall + Teak Academic Bookshelves / Credenza + Glowing Workstation Monitor!',
      '    float isShelf = step(hit.y, 0.56) * step(0.12, hit.x) * step(hit.x, 0.88);',
      '    float shelfRow = fract(hit.y * 7.0);',
      '    float books = fract(hit.x * 18.0 + floor(hit.y * 7.0) * 3.1);',
      '    vec3 bookTint = mix(vec3(0.34, 0.20, 0.12), vec3(0.56, 0.44, 0.30), step(0.45, books));',
      '    bookTint *= smoothstep(0.0, 0.14, shelfRow);',
      '    vec3 wallPaint = mix(vec3(0.62, 0.59, 0.53), vec3(0.54, 0.56, 0.58), fract(rSeed * 5.1));',
      '    float frame = step(0.64, hit.y) * step(hit.y, 0.86) * step(0.34, hit.x) * step(hit.x, 0.66);',
      '    wallPaint = mix(wallPaint, vec3(0.38, 0.32, 0.24), frame * 0.55);',
      '    surfCol = mix(wallPaint, bookTint, isShelf);',
      '    float hasMon = step(0.35, fract(rSeed * 17.3)) * (1.0 - isAtrium);',
      '    float monMask = hasMon * step(0.30, hit.y) * step(hit.y, 0.45) * step(0.40, hit.x) * step(hit.x, 0.62);',
      '    trofferGlow += monMask * 0.42;',
      '    surfCol = mix(surfCol, vec3(0.52, 0.72, 0.94), monMask * 0.75);',
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
          'int omniPId = int(uSurfaceProfile + 0.5);',
          'float omniPxSize = max(length(dFdx(vOmniWorldPos)), length(dFdy(vOmniWorldPos)));',
          'vec4 omniSurfDetail = vec4(0.0, 1.0, 0.0, 1.0);',
          '#if OMNI_GROUP != 4',
          'if (uShaderQuality > 0.5 && uZebraInspectMode < 0.5) {',
          '  omniSurfDetail = omniEvalSurfaceDetail(uSurfaceProfile, vOmniWorldPos, normalize(vOmniWorldNormal), diffuseColor.rgb, omniPxSize);',
          '  diffuseColor.rgb *= clamp(omniSurfDetail.y, 0.64, 1.26) * omniSurfDetail.w;',
          '}',
          '#endif'
        ].join('\n')
      )
      // 2. MICRO-SURFACE, TOKSVIG GEOMETRIC SPECULAR AA & WET-WEATHER PBR ROUGHNESS MODULATION
      .replace(
        '#include <roughnessmap_fragment>',
        [
          '#include <roughnessmap_fragment>',
          'vec3 omniDNdx = dFdx(vOmniWorldNormal);',
          'vec3 omniDNdy = dFdy(vOmniWorldNormal);',
          'if (uShaderQuality > 0.5 && uZebraInspectMode < 0.5) {',
          '  float geomRoughAA = min(0.14, dot(omniDNdx, omniDNdx) + dot(omniDNdy, omniDNdy));',
          '  roughnessFactor = clamp(sqrt(roughnessFactor * roughnessFactor + geomRoughAA * 0.30) + omniSurfDetail.z, 0.04, 0.98);',
          '}',
          'float omniSkyExp = 1.0;',
          'float omniPuddleMask = 0.0;',
          '#if OMNI_GROUP != 4',
          'if (uWetness > 0.005 && uZebraInspectMode < 0.5) {',
          '  omniSkyExp = omniSampleSkyExposure(vOmniWorldPos);',
          '  float effWetR = uWetness * (0.16 + 0.84 * omniSkyExp);',
          '  float upnessR = smoothstep(0.90, 0.992, normalize(vOmniWorldNormal).y);',
          '  float pFieldR = omniFbm2(vOmniWorldPos.xz * 0.095);',
          '  omniPuddleMask = upnessR * omniSkyExp * smoothstep(0.65 - effWetR * 0.20, 0.78 - effWetR * 0.14, pFieldR) * step(-1.2, vOmniWorldPos.y);',
          '  roughnessFactor = mix(roughnessFactor, max(0.08, roughnessFactor * 0.34), effWetR * 0.84);',
          '  roughnessFactor = mix(roughnessFactor, 0.032, omniPuddleMask * min(1.0, effWetR * 1.25));',
          '}',
          '#endif'
        ].join('\n')
      )
      // 3. DISTANCE-ADAPTIVE TANGENT / SCREEN-SPACE DERIVATIVE MICRO-NORMAL PERTURBATION
      //    (Screen-space derivatives dFdx/dFdy are evaluated in uniform control flow for strict D3D11/ANGLE compliance)
      .replace(
        '#include <normal_fragment_maps>',
        [
          '#include <normal_fragment_maps>',
          '#if OMNI_GROUP != 4',
          'float omniDHdx = dFdx(omniSurfDetail.x);',
          'float omniDHdy = dFdy(omniSurfDetail.x);',
          'vec3 omniDPdx = dFdx(vViewPosition);',
          'vec3 omniDPdy = dFdy(vViewPosition);',
          'if (uShaderQuality > 0.5 && uZebraInspectMode < 0.5 && uBumpScale > 0.01) {',
          '  float camDist = length(vViewPosition);',
          '  float bumpFade = (1.0 - smoothstep(65.0, 145.0, camDist)) * (1.0 - omniPuddleMask * 0.88);',
          '  if (bumpFade > 0.005) {',
          '    float closeBoost = mix(1.06, 0.78, smoothstep(uCloseDerivativeDist * 0.45, uCloseDerivativeDist, camDist));',
          '    vec3 r1 = cross(omniDPdy, normal);',
          '    vec3 r2 = cross(normal, omniDPdx);',
          '    float det = dot(omniDPdx, r1);',
          '    vec3 surfGrad = sign(det) * (omniDHdx * r1 + omniDHdy * r2) / max(abs(det), 1e-6);',
          '    normal = normalize(normal - clamp(surfGrad * (uBumpScale * closeBoost * bumpFade), vec3(-0.32), vec3(0.32)));',
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
          '  float worldCloudTrans = omniEvalCloudShadow(vOmniWorldPos, sunL);',
          '  float sunVisUltra = clamp(1.0 - uCloudCover * 0.55 - uRainIntensity * 0.48, 0.04, 1.0) * (1.0 - uIsNight * 0.84) * worldCloudTrans * clamp(uSolarDirectTrans, 0.0, 1.0);',
          '#if OMNI_GROUP != 4',
          '  if (vOmniWorldPos.y > -3.5) {',
          '    float sunWeight = mix(0.32, 0.88, NdotL);',
          '    gl_FragColor.rgb *= mix(1.0, worldCloudTrans, sunWeight);',
          '  }',
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
          '#if OMNI_GROUP == 11',
          '  // B. BEER-LAMBERT BOTANICAL SUBSURFACE SCATTERING (LEAF TRANSLUCENCY) & CUTICLE FRESNEL',
          '  {',
          '    float VdotNegL = clamp(dot(viewDirW, -sunL), 0.0, 1.0);',
          '    float crownRim = pow(1.0 - NdotV, 2.2) * clamp(dot(wNorm, sunL) * 0.5 + 0.5, 0.0, 1.0);',
          '    float leafTrans = pow(VdotNegL, 2.6) * 0.64 + clamp((dot(wNorm, sunL) + 0.52) / 1.52, 0.0, 1.0) * 0.24 + crownRim * 0.22;',
          '    vec3 chlorophyll = diffuseColor.rgb * vec3(1.22, 1.54, 0.44);',
          '    gl_FragColor.rgb += chlorophyll * uSunColor * leafTrans * sunVisUltra * (0.38 + 0.62 * omniSkyExp) * 0.48;',
          '    float cuticleSheen = pow(1.0 - NdotV, 4.0) * clamp(wNorm.y * 0.5 + 0.5, 0.0, 1.0) * 0.052 * (1.0 - uIsNight * 0.78);',
          '    gl_FragColor.rgb += mix(vec3(0.40, 0.56, 0.74), vec3(0.78, 0.48, 0.28), uDuskFactor) * cuticleSheen;',
          '  }',
          '#endif',
          '',
          '#if OMNI_GROUP == 4',
          '  // C. TRUE 3D RAY-BOX PARALLAX INTERIOR ROOMS + EXTERIOR FRESNEL SKY/CLOUD/SUN GLASS OPTICS',
          '  if (abs(wNorm.y) < 0.45) {',
          '    float baseGlassLuma = dot(diffuseColor.rgb, vec3(0.299, 0.587, 0.114));',
          '    float paneMask = 1.0 - smoothstep(0.22, 0.38, baseGlassLuma);',
          '    if (paneMask > 0.02) {',
          '      vec3 room3D = omniEvalInteriorParallaxRoom(vOmniWorldPos, wNorm, viewDirW, omniPId == 5 ? 1.0 : 0.0, omniPxSize);',
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
          '  // D. ANALYTICAL ARCHITECTURAL EAVE, CANOPY OVERHANG & JEMBATAN TEKSAS GROUND CONTACT AO',
          '  float contactAO = 1.0;',
          '  if (vOmniWorldPos.y > uYSegStart && vOmniWorldPos.y < uYSegStart + uSegStepH * uSegCount && abs(wNorm.y) < 0.7) {',
          '    float phase = fract((vOmniWorldPos.y - uYSegStart) / uSegStepH);',
          '    contactAO *= mix(0.46, 1.0, smoothstep(0.98, 0.52, phase));',
          '  }',
          '#if OMNI_GROUP != 9',
          '  if (vOmniWorldPos.y > 0.01 && vOmniWorldPos.y < 1.8 && abs(wNorm.y) < 0.78) {',
          '    contactAO *= mix(0.70, 1.0, smoothstep(0.0, 1.8, vOmniWorldPos.y));',
          '  }',
          '#endif',
          '#if OMNI_GROUP != 4 && OMNI_GROUP != 11',
          '  contactAO *= omniEvalWorldOverhangAO(vOmniWorldPos, wNorm);',
          '#endif',
          '',
          '#if OMNI_GROUP == 9',
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
          '  bool omniSubseaBore = (vOmniWorldPos.y < -3.65 && abs(vOmniWorldPos.x + 356.0) < 5.35 && vOmniWorldPos.z > 325.8 && vOmniWorldPos.z < 624.2);',
          '#if OMNI_GROUP != 4',
          '  bool isGroundOrPavedProf = (omniPId == 1 || omniPId == 3 || omniPId == 6 || omniPId == 9 || omniPId == 10);',
          '  if (uWetness > 0.005 && !omniSubseaBore) {',
          '    float effWet = uWetness * (0.16 + 0.84 * omniSkyExp);',
          '    float effRain = uRainIntensity * omniSkyExp;',
          '    float porous = smoothstep(0.26, 0.76, roughnessFactor);',
          '    gl_FragColor.rgb *= mix(1.0, 0.66, effWet * porous * 0.88);',
          '',
          '    // Thin-film dielectric water clearcoat reflection on wet roofs, concrete, & stone!',
          '    float waterFres = (0.0204 + 0.9796 * pow(1.0 - NdotV, 5.0)) * effWet * 0.42;',
          '    vec3 wRefl = reflect(-viewDirW, wNorm);',
          '    vec3 wetSkyCol = mix(mix(vec3(0.36, 0.44, 0.54), vec3(0.68, 0.42, 0.28), uDuskFactor), vec3(0.05, 0.08, 0.14), uIsNight);',
          '    float wetSunGlint = pow(max(dot(wRefl, sunL), 0.0), 96.0) * sunVisUltra * effWet * 0.55;',
          '    gl_FragColor.rgb = mix(gl_FragColor.rgb, wetSkyCol, waterFres) + uSunColor * wetSunGlint;',
          '',
          '#if OMNI_GROUP != 11 && !(OMNI_GROUP == 10 && defined(USE_INSTANCING))',
          '    if (isGroundOrPavedProf) {',
          '      float pStrength = omniPuddleMask * min(1.0, effWet * 1.25);',
          '      if (pStrength > 0.015) {',
          '        float ripX = 0.0;',
          '        float ripZ = 0.0;',
          '        float ringGlint = 0.0;',
          '        if (effRain > 0.03) {',
          '          for (int rl = 0; rl < 2; rl++) {',
          '            float gScale = rl == 0 ? 1.55 : 2.65;',
          '            vec2 gp = vOmniWorldPos.xz * gScale;',
          '            vec2 gi = floor(gp);',
          '            vec2 gf = fract(gp) - 0.5;',
          '            float hSeed = omniHash(gi + float(rl) * 37.0);',
          '            vec2 rCenter = (vec2(omniHash(gi + 5.1), omniHash(gi + 13.7)) - 0.5) * 0.48;',
          '            float life = fract(uTime * (1.28 + float(rl) * 0.48) + hSeed);',
          '            vec2 dv = gf - rCenter;',
          '            float dist = length(dv) + 1e-4;',
          '            float front = life * 0.44;',
          '            float env = exp(-abs(dist - front) * 28.0) * (1.0 - life) * smoothstep(0.0, 0.06, life);',
          '            float wv = sin((dist - front) * 96.0) * env * min(1.3, effRain);',
          '            ripX += (dv.x / dist) * wv;',
          '            ripZ += (dv.y / dist) * wv;',
          '            ringGlint += max(0.0, wv) * 0.55;',
          '          }',
          '        }',
          '        vec3 pNorm = normalize(vec3(ripX * 0.36, 1.0, ripZ * 0.36));',
          '        float fres = 0.0204 + 0.9796 * pow(1.0 - max(dot(viewDirW, pNorm), 0.0), 4.6);',
          '        vec3 puddleSkyDay = mix(vec3(0.42, 0.52, 0.62), vec3(0.22, 0.26, 0.32), clamp(uRainIntensity, 0.0, 1.0));',
          '        puddleSkyDay = mix(puddleSkyDay, vec3(0.76, 0.44, 0.24), uDuskFactor * (1.0 - uRainIntensity * 0.6));',
          '        vec3 puddleSky = mix(puddleSkyDay, vec3(0.04, 0.07, 0.13), uIsNight);',
          '        puddleSky += vec3(0.75, 0.86, 1.0) * uLightningFlash * 0.95;',
          '        gl_FragColor.rgb = mix(gl_FragColor.rgb * 0.80, puddleSky, clamp(pStrength * (0.30 + 0.68 * fres), 0.0, 0.90));',
          '        gl_FragColor.rgb += vec3(0.76, 0.84, 0.94) * ringGlint * pStrength * (0.40 - 0.20 * uIsNight + 0.50 * uLightningFlash);',
          '      }',
          '    }',
          '#endif',
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
          '#endif',
          '',
          '  if (uLightningFlash > 0.01) {',
          '    gl_FragColor.rgb += vec3(0.68, 0.80, 1.0) * uLightningFlash * omniSkyExp * (0.10 + 0.18 * max(0.0, wNorm.y));',
          '  }',
          '',
          '  float surfLuma = dot(gl_FragColor.rgb, vec3(0.299, 0.587, 0.114));',
          '  float nightMod = mix(1.0, mix(0.72, 1.0, smoothstep(0.06, 0.28, surfLuma)), uIsNight);',
          '  gl_FragColor.rgb *= contactAO * nightMod;',
          '',
          '#if OMNI_GROUP == 9',
          '  // E1B. PULAU I GUNUNG GEDE BOGOR — KAWAH RATU WEATHERED ANDESITE & HYDROTHERMAL MINERAL CRUST (ZERO RED MAGMA!)',
          '  if (vOmniWorldPos.z > 670.0 && vOmniWorldPos.y > 42.5) {',
          '    vec2 dVol = vOmniWorldPos.xz - vec2(-32.0, 698.0);',
          '    float rVol = length(dVol);',
          '    if (rVol < 21.5) {',
          '      float floorMask = smoothstep(14.8, 9.6, rVol) * smoothstep(49.5, 46.2, vOmniWorldPos.y);',
          '      vec2 cellK = omniVoronoiStone(dVol * 0.52);',
          '      float mineralNoise = omniFbm2(dVol * 0.68);',
          '      if (floorMask > 0.005) {',
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
          '#if OMNI_GROUP == 6',
          '  // E2A. PHOTOCELL-CONTROLLED CAMPUS STREETLAMP FROSTED BOROSILICATE LANTERN & TAJUG HOOD BOUNCE',
          '  if (omniPId == 8 && vOmniUV.y > 1.20) {',
          '    float lampPwr = omniEvalLampPhotocellPower(vOmniWorldPos.xz, uStreetlampPower);',
          '    if (vOmniUV.y > 1.50) {',
          '      float glassFres = pow(1.0 - NdotV, 3.2) * 0.14 * (1.0 - uNightFactor * 0.8);',
          '      gl_FragColor.rgb += vec3(0.62, 0.72, 0.84) * glassFres;',
          '      if (lampPwr > 0.001) {',
          '        float coreHot = smoothstep(1.68, 2.00, vOmniUV.y);',
          '        vec3 warm2900K = mix(vec3(3.4, 2.15, 0.86), vec3(6.4, 4.45, 2.05), coreHot);',
          '        gl_FragColor.rgb += warm2900K * lampPwr;',
          '      }',
          '    } else if (lampPwr > 0.001) {',
          '      gl_FragColor.rgb += vec3(1.18, 0.72, 0.28) * lampPwr;',
          '    }',
          '  }',
          '#endif',
          '',
          '#if OMNI_GROUP != 4 && OMNI_GROUP != 11',
          '  if (isGroundOrPavedProf) {',
          '    // E2B. O(1) ANALYTIC 2900K STREETLAMP GROUND ILLUMINANCE POOLS & WET ROAD SPECULAR GLINTS',
          '    if (uStreetlampPower > 0.002 && vOmniWorldPos.y > -0.35 && vOmniWorldPos.y < 22.0) {',
          '      vec3 nl = omniEvalNearestStreetlamp(vOmniWorldPos.xz);',
          '      if (nl.z < 196.0) {',
          '        float lampPwr = omniEvalLampPhotocellPower(nl.xy, uStreetlampPower);',
          '        if (lampPwr > 0.002) {',
          '          float lampRelH = vOmniWorldPos.y > 2.2 ? 3.48 : (3.48 - max(0.0, vOmniWorldPos.y * 0.35) - vOmniWorldPos.y);',
          '          vec3 toLamp = vec3(nl.x - vOmniWorldPos.x, lampRelH, nl.y - vOmniWorldPos.z);',
          '          float d3Sq = dot(toLamp, toLamp);',
          '          vec3 lDir = toLamp * inversesqrt(max(0.25, d3Sq));',
          '          float postBaseShadow = 1.0 - 0.58 * exp(-nl.z * 2.2);',
          '          float radialCutoff = smoothstep(196.0, 14.0, nl.z) * postBaseShadow;',
          '          float atten = (11.5 / (5.5 + d3Sq)) * radialCutoff * lampPwr;',
          '          float nDotLamp = max(0.0, dot(wNorm, lDir));',
          '          vec3 lampTint = vec3(1.05, 0.70, 0.32);',
          '          gl_FragColor.rgb += diffuseColor.rgb * lampTint * (atten * nDotLamp * 1.35);',
          '          vec3 hLamp = normalize(lDir + viewDirW);',
          '          float specExp = mix(24.0, 110.0, 1.0 - clamp(roughnessFactor, 0.04, 0.95));',
          '          float lampSpec = pow(max(0.0, dot(wNorm, hLamp)), specExp) * (0.14 + 0.52 * uWetness);',
          '          gl_FragColor.rgb += lampTint * (atten * lampSpec * 0.95);',
          '        }',
          '      }',
          '    }',
          '  }',
          '#endif',
          '',
          '  // E2D. OUTDOOR CAMPUS MONOPOLE LED VIDEOTRON (1.47-MILLION-DIODE SMD P8 ARRAY @ 28.4, 78.2) VOLUMETRIC RADIATION, BEZEL GLOW, TREE CANOPY & ROTUNDA SPILL',
          '  {',
          '    float vtEmitPwr = clamp(0.24 + uNightFactor * 0.96 + uDuskFactor * 0.68 + uRainIntensity * 0.32 + uFogFactor * 0.28, 0.0, 1.35);',
          '    if (vOmniUV.y > -500.0 && vOmniWorldPos.x > -34.0 && vOmniWorldPos.x < 46.0 && vOmniWorldPos.z > 68.0 && vOmniWorldPos.z < 146.0 && vOmniWorldPos.y > -0.6 && vOmniWorldPos.y < 21.5) {',
          '      vec2 vtRel = vOmniWorldPos.xz - vec2(28.4, 78.2);',
          '      float vtFwd = dot(vtRel, vec2(-0.73745127, 0.67539916));',
          '      float vtLat = dot(vtRel, vec2(0.67539916, 0.73745127));',
          '      float nearLat = clamp(vtLat, -6.35, 6.35);',
          '      float nearY = clamp(vOmniWorldPos.y, 5.9, 13.05);',
          '      float bcastShift = 0.5 + 0.5 * sin(uTime * 0.85 + nearLat * 0.16);',
          '      vec3 vtCastCol = mix(vec3(0.42, 0.78, 1.25), vec3(1.12, 0.86, 0.46), bcastShift);',
          '      // 1. Direct Inner Bezel, Top Rain-Visor Hood, & Frame Perimeter SMD Spill',
          '      if (vtFwd > -0.55 && vtFwd < 1.35 && abs(vtLat) < 7.15 && vOmniWorldPos.y > 4.6 && vOmniWorldPos.y < 14.3) {',
          '        float dEdgeX = max(0.0, abs(vtLat) - 6.32);',
          '        float dEdgeY = max(0.0, max(5.88 - vOmniWorldPos.y, vOmniWorldPos.y - 13.08));',
          '        float dBezel = length(vec2(dEdgeX, dEdgeY)) + max(0.0, abs(vtFwd - 0.30));',
          '        float bezelGlow = exp(-dBezel * 2.35) * vtEmitPwr * 1.45;',
          '        gl_FragColor.rgb += (diffuseColor.rgb * 0.55 + vec3(0.28)) * vtCastCol * bezelGlow;',
          '      }',
          '      // 2. 64-Meter Forward Hemisphere Area-Light Throw onto Trees, Gazebo, Lawn, & Wet Road',
          '      if (vtFwd > 0.15 && vtFwd < 64.0) {',
          '        vec3 nearScrW = vec3(',
          '          28.4 + nearLat * 0.67539916 - 0.302 * 0.73745127,',
          '          nearY,',
          '          78.2 + nearLat * 0.73745127 + 0.302 * 0.67539916',
          '        );',
          '        vec3 toScr = nearScrW - vOmniWorldPos;',
          '        float dScrSq = dot(toScr, toScr);',
          '        vec3 lScr = toScr * inversesqrt(max(0.65, dScrSq));',
          '        float cosCone = max(0.0, dot(-lScr, vec3(-0.73745127, 0.0, 0.67539916)));',
          '        float nDotScr = max(0.0, dot(wNorm, lScr));',
          '        float leafTransScr = max(0.0, dot(-wNorm, lScr)) * 0.52 + 0.22;',
          '        float surfResp = max(nDotScr, (omniPId == 4 || omniPId == 5 || omniPId == 11) ? leafTransScr : nDotScr * 0.85 + 0.15);',
          '        float distFade = smoothstep(64.0, 3.0, vtFwd);',
          '        float areaAtten = (54.0 / (9.5 + dScrSq)) * pow(cosCone, 1.15) * distFade * vtEmitPwr;',
          '        gl_FragColor.rgb += (diffuseColor.rgb + vec3(0.06)) * vtCastCol * (areaAtten * surfResp * 1.38);',
          '        vec3 hScr = normalize(lScr + viewDirW);',
          '        float scrSpec = pow(max(0.0, dot(wNorm, hScr)), 32.0) * (0.14 + 0.58 * uWetness);',
          '        gl_FragColor.rgb += vtCastCol * (areaAtten * scrSpec * 0.92);',
          '      }',
          '    }',
          '  }',
          '',
          '#if OMNI_GROUP == 6 || OMNI_GROUP == 9 || OMNI_GROUP == 10',
          '  // E2E. ROTUNDA DANCING FOUNTAIN (-2.5, 106.5) SYNCHRONIZED NIGHT RGBW WATER-SHOW BASIN & BOULEVARD SPILL',
          '  if ((omniPId == 8 || omniPId == 9 || omniPId == 10) && uFountainNightGlow.w > 0.01 && vOmniUV.y > -500.0 && abs(vOmniWorldPos.x + 2.5) < 26.0 && abs(vOmniWorldPos.z - 106.5) < 26.0 && vOmniWorldPos.y > -0.4 && vOmniWorldPos.y < 7.5) {',
          '    vec3 ftnSrc = vec3(-2.5, 1.45, 106.5);',
          '    vec3 toFtn = ftnSrc - vOmniWorldPos;',
          '    float dFtnSq = dot(toFtn, toFtn);',
          '    if (dFtnSq < 576.0) {',
          '      vec3 lFtn = toFtn * inversesqrt(max(0.35, dFtnSq));',
          '      float rHoriz = length(vOmniWorldPos.xz - vec2(-2.5, 106.5));',
          '      float ftnFade = smoothstep(24.0, 2.2, rHoriz);',
          '      float ftnAtten = (14.5 / (3.2 + dFtnSq)) * ftnFade * uFountainNightGlow.w;',
          '      float nDotFtn = max(0.18, dot(wNorm, lFtn));',
          '      float caustRip = 0.82 + 0.18 * sin(rHoriz * 3.6 - uTime * 3.4 + atan(vOmniWorldPos.z - 106.5, vOmniWorldPos.x + 2.5) * 4.0);',
          '      vec3 ftnCol = uFountainNightGlow.rgb * caustRip;',
          '      gl_FragColor.rgb += diffuseColor.rgb * ftnCol * (ftnAtten * nDotFtn * 1.45);',
          '      vec3 hFtn = normalize(lFtn + viewDirW);',
          '      float ftnSpec = pow(max(0.0, dot(wNorm, hFtn)), 44.0) * (0.22 + 0.55 * uWetness);',
          '      gl_FragColor.rgb += ftnCol * (ftnAtten * ftnSpec * 0.90);',
          '    }',
          '  }',
          '#endif',
          '',
          '#if (OMNI_GROUP == 1 || OMNI_GROUP == 10) && !defined(USE_INSTANCING)',
          '  // E2C. UNDERPASS BAWAH LAUT SELATAN U-I: 24/7 3800K TUNNEL CEILING LUMINAIRES & SUBSEA OCEAN-CAUSTIC ACRYLIC WINDOWS',
          '  if ((omniPId == 1 || omniPId == 10) && omniSubseaBore) {',
          '    float lumRhythm = 0.76 + 0.24 * cos((vOmniWorldPos.z - 326.0) * 0.5235988);',
          '    if (vOmniWorldPos.y > -4.65 && diffuseColor.r > 0.92 && diffuseColor.g > 0.88 && diffuseColor.b > 0.72) {',
          '      gl_FragColor.rgb += vec3(2.45, 2.22, 1.78) * (0.88 + 0.12 * lumRhythm);',
          '    } else if (diffuseColor.b > diffuseColor.r * 2.2 && diffuseColor.b > 0.42) {',
          '      vec2 cUV1 = vec2(vOmniWorldPos.z * 0.52 + uTime * 0.38, vOmniWorldPos.y * 0.85 - uTime * 0.22);',
          '      vec2 cUV2 = vec2(vOmniWorldPos.z * 0.83 - uTime * 0.31, vOmniWorldPos.y * 1.15 + uTime * 0.27);',
          '      float cNet = pow(clamp(1.0 - abs(omniNoise(cUV1) - omniNoise(cUV2)) * 2.6, 0.0, 1.0), 2.4);',
          '      float depthGrad = clamp((vOmniWorldPos.y + 7.6) * 0.36, 0.0, 1.0);',
          '      float dayMarine = mix(1.0, 0.34, uIsNight);',
          '      vec3 deepSea = mix(vec3(0.03, 0.16, 0.28), vec3(0.07, 0.36, 0.54), depthGrad) * dayMarine;',
          '      vec3 causticGlow = vec3(0.20, 0.74, 0.92) * cNet * (0.45 + 0.55 * depthGrad) * dayMarine;',
          '      float acrylicFres = pow(1.0 - NdotV, 3.6) * 0.28;',
          '      gl_FragColor.rgb = deepSea + causticGlow + vec3(0.45, 0.65, 0.78) * acrylicFres;',
          '    } else {',
          '      float upFacing = 0.45 + 0.55 * clamp(wNorm.y * 0.5 + 0.5, 0.0, 1.0);',
          '      vec3 ledBounce = vec3(0.96, 0.88, 0.74) * (0.64 * lumRhythm * upFacing);',
          '      float winBay = max(0.0, cos((vOmniWorldPos.z - 344.6) * 0.16868)) * step(338.0, vOmniWorldPos.z) * step(vOmniWorldPos.z, 612.0);',
          '      vec3 seaSpill = vec3(0.05, 0.22, 0.32) * pow(winBay, 3.0) * mix(0.65, 0.22, uIsNight);',
          '      gl_FragColor.rgb += diffuseColor.rgb * (ledBounce + seaSpill);',
          '    }',
          '  }',
          '#endif',
          '',
          '#if OMNI_GROUP == 10 && !defined(USE_INSTANCING)',
          '  // E2F. ARCHITECTURAL WINDOW GLAZING (DPPD, PERPUSTAKAAN LAMA, STP-UI, ILRC, ATC KECIL, & CAMPUS KIOSKS)',
          '  if (vOmniUV.y < -600.0 && vOmniUV.y > -850.0) {',
          '    float fresArch = 0.10 + 0.88 * pow(1.0 - NdotV, 4.0);',
          '    vec3 reflArch = reflect(-viewDirW, wNorm);',
          '    vec3 dayHorizA = mix(vec3(0.54, 0.70, 0.86), vec3(0.92, 0.48, 0.24), uDuskFactor);',
          '    vec3 dayZenA = mix(vec3(0.16, 0.38, 0.72), vec3(0.18, 0.20, 0.42), uDuskFactor);',
          '    vec3 skyReflA = mix(vec3(0.14, 0.20, 0.18), mix(dayHorizA, dayZenA, clamp(reflArch.y, 0.0, 1.0)), smoothstep(-0.10, 0.25, reflArch.y));',
          '    float cldA = smoothstep(0.42, 0.82, omniFbm2(reflArch.xz / max(0.18, abs(reflArch.y) + 0.15) * 1.35));',
          '    skyReflA = mix(skyReflA, mix(vec3(0.88, 0.91, 0.96), vec3(0.95, 0.62, 0.40), uDuskFactor), cldA * step(0.02, reflArch.y) * (1.0 - uRainIntensity * 0.7));',
          '    vec3 extReflA = mix(skyReflA, vec3(0.025, 0.042, 0.082), uIsNight);',
          '    float sunGlintA = pow(max(dot(reflArch, sunL), 0.0), 160.0) * sunVisUltra * 2.0;',
          '    extReflA += uSunColor * sunGlintA;',
          '    vec2 roomCell = floor(vec2(dot(vOmniWorldPos.xz, vec2(0.42, 0.37)), vOmniWorldPos.y * 0.36));',
          '    float occHash = omniHash(roomCell);',
          '    float nightPwrA = clamp(max(uIsNight, uStreetlampPower * 0.85), 0.0, 1.0);',
          '    float roomOn = step(0.20, occHash) * nightPwrA;',
          '    float ceilGrad = 0.62 + 0.38 * fract(vOmniWorldPos.y * 0.36);',
          '    vec3 warmOffice = mix(vec3(1.55, 1.16, 0.64), vec3(1.25, 1.32, 1.42), step(0.82, occHash)) * ceilGrad;',
          '    vec3 dayInterior = diffuseColor.rgb * (0.28 + 0.24 * sunVisUltra);',
          '    vec3 intColA = mix(dayInterior, warmOffice, roomOn);',
          '    gl_FragColor.rgb = mix(intColA, extReflA, fresArch * mix(0.76, 0.26, roomOn));',
          '  } else if (vOmniUV.y <= -860.0 && vOmniUV.y > -980.0) {',
          '    if (vOmniUV.y < -905.0) {',
          '      if (uFountainNightGlow.w > 0.01) {',
          '        gl_FragColor.rgb += uFountainNightGlow.rgb * (3.2 * uFountainNightGlow.w);',
          '      }',
          '    } else if (uStreetlampPower > 0.005) {',
          '      gl_FragColor.rgb += vec3(2.45, 1.95, 1.18) * uStreetlampPower;',
          '    }',
          '  }',
          '  // E2. JEMBATAN TEKSAS UI SMOKED-CERULEAN WAVE SKYLIGHT FRESNEL/SKY OPTICS, DARK NIGHT BASELINE, & MULTI-COLOR ARCHITECTURAL SPOTLIGHTS',
          '  if (vOmniWorldPos.z > 148.0 && vOmniWorldPos.z < 344.0 && abs(vOmniWorldPos.x + 28.0) < 9.5) {',
          '    float teksasDarkFactor = clamp(uIsNight * 0.88 + uDuskFactor * 0.42 + uRainIntensity * 0.32, 0.0, 0.92);',
          '    gl_FragColor.rgb *= mix(1.0, 0.22, teksasDarkFactor);',
          '  }',
          '  if (vOmniWorldPos.z > 170.0 && vOmniWorldPos.z < 321.0 && abs(vOmniWorldPos.x + 28.0) < 4.5 && diffuseColor.b > diffuseColor.r * 1.65 && diffuseColor.b > 0.28) {',
          '    vec3 reflRoof = reflect(-viewDirW, wNorm);',
          '    float fresRoof = 0.06 + 0.88 * pow(1.0 - NdotV, 4.0);',
          '    vec3 roofHoriz = mix(vec3(0.52, 0.68, 0.84), vec3(0.88, 0.48, 0.26), uDuskFactor);',
          '    vec3 roofZen = mix(vec3(0.15, 0.36, 0.66), vec3(0.16, 0.20, 0.38), uDuskFactor);',
          '    vec3 roofSky = mix(roofHoriz, roofZen, clamp(abs(reflRoof.y), 0.0, 1.0));',
          '    float cldRoof = smoothstep(0.44, 0.82, omniFbm2(reflRoof.xz / max(0.20, abs(reflRoof.y) + 0.16) * 1.25));',
          '    roofSky = mix(roofSky, mix(vec3(0.86, 0.90, 0.95), vec3(0.94, 0.60, 0.38), uDuskFactor), cldRoof * (1.0 - uRainIntensity * 0.65));',
          '    roofSky = mix(roofSky * (1.0 - uRainIntensity * 0.55), vec3(0.015, 0.028, 0.055), uIsNight);',
          '    float roofSunSpec = pow(max(dot(reflRoof, sunL), 0.0), 140.0) * sunVisUltra * 1.65;',
          '    float backTrans = (0.14 + 0.26 * max(0.0, sunL.y)) * sunVisUltra * (wNorm.y < 0.0 ? 1.25 : 0.65);',
          '    float purlinBar = 0.86 + 0.14 * smoothstep(0.08, 0.32, abs(fract((vOmniWorldPos.z - 175.05) / 4.707) - 0.5));',
          '    gl_FragColor.rgb = mix(gl_FragColor.rgb * purlinBar + vec3(0.05, 0.19, 0.34) * backTrans, roofSky, fresRoof * 0.62) + uSunColor * roofSunSpec;',
          '  }',
          '  if (uStreetlampPower > 0.002 && vOmniWorldPos.z > 148.0 && vOmniWorldPos.z < 344.0 && abs(vOmniWorldPos.x + 28.0) < 9.5) {',
          '    if (diffuseColor.r > 0.92 && diffuseColor.g > 0.85 && diffuseColor.b > 0.68 && vOmniWorldPos.y > 1.0) {',
          '      float stIdxL = mod(floor(max(0.0, vOmniWorldPos.z - 172.0) / 14.5), 4.0);',
          '      vec3 lensCol = stIdxL < 0.5 ? vec3(2.8, 2.0, 0.9) : (stIdxL < 1.5 ? vec3(0.6, 2.1, 3.0) : (stIdxL < 2.5 ? vec3(0.5, 2.6, 1.7) : vec3(2.6, 0.8, 2.4)));',
          '      gl_FragColor.rgb += lensCol * uStreetlampPower;',
          '    }',
          '    if (diffuseColor.r > 0.80 && diffuseColor.g > 0.55 && diffuseColor.b < 0.22 && (abs(vOmniWorldPos.z - 171.65) < 4.2 || abs(vOmniWorldPos.z - 319.65) < 4.2)) {',
          '      float zPyl = vOmniWorldPos.z < 245.0 ? 171.65 : 319.65;',
          '      float sSide = vOmniWorldPos.x + 28.0 >= 0.0 ? 1.0 : -1.0;',
          '      vec3 spotPosP = vec3(-28.0 + sSide * 2.75, 4.15, zPyl + (vOmniWorldPos.z < 245.0 ? -1.1 : 1.1));',
          '      vec3 toPyl = vOmniWorldPos - spotPosP;',
          '      float dPyl = length(toPyl);',
          '      vec3 lPyl = toPyl / max(dPyl, 1e-3);',
          '      vec3 aimPyl = normalize(vec3(-sSide * 0.22, 0.95, vOmniWorldPos.z < 245.0 ? 0.14 : -0.14));',
          '      float conePyl = pow(max(0.0, dot(lPyl, aimPyl)), 6.5);',
          '      float nDotPyl = max(0.15, dot(wNorm, -lPyl));',
          '      float attenPyl = (26.0 / (3.5 + dPyl * dPyl)) * conePyl * nDotPyl * smoothstep(23.5, 4.2, vOmniWorldPos.y);',
          '      vec3 pylSpotCol = sSide > 0.0 ? vec3(1.65, 1.15, 0.42) : vec3(0.45, 1.25, 1.85);',
          '      gl_FragColor.rgb += (diffuseColor.rgb * 0.75 + vec3(0.18)) * pylSpotCol * attenPyl * uStreetlampPower * 2.2;',
          '    }',
          '    if (vOmniWorldPos.z > 172.0 && vOmniWorldPos.z < 319.0 && abs(vOmniWorldPos.x + 28.0) < 3.8 && vOmniWorldPos.y > 3.6 && vOmniWorldPos.y < 13.8) {',
          '      float bayCoord = (vOmniWorldPos.z - 173.5) / 14.5;',
          '      float bayIdx = mod(floor(bayCoord + 0.5), 4.0);',
          '      float dzSpot = (fract(bayCoord + 0.5) - 0.5) * 14.5;',
          '      float spotPoolZ = exp(-dzSpot * dzSpot * 0.26);',
          '      if (spotPoolZ > 0.015) {',
          '        vec3 spotCol = bayIdx < 0.5',
          '          ? vec3(1.65, 1.12, 0.38)',
          '          : (bayIdx < 1.5',
          '            ? vec3(0.28, 1.18, 1.85)',
          '            : (bayIdx < 2.5 ? vec3(0.24, 1.48, 0.92) : vec3(1.52, 0.44, 1.38)));',
          '        float isCanopyTop = step(0.35, wNorm.y) * step(diffuseColor.r * 1.5, diffuseColor.b);',
          '        float verticalFocus = exp(-pow((vOmniWorldPos.y - 7.8) * 0.42, 2.0));',
          '        float dirMask = mix(1.0, 0.14, isCanopyTop);',
          '        float spotIntensity = spotPoolZ * verticalFocus * dirMask * uStreetlampPower * 1.45;',
          '        gl_FragColor.rgb += (diffuseColor.rgb * 0.65 + vec3(0.22)) * spotCol * spotIntensity;',
          '      }',
          '    }',
          '  }',
          '  // Photocell architectural uplights on Underpass Portals, Dermaga Penyeberangan Utara Tajug Pavilions, & Videotron UI Makara Crown',
          '  if (uStreetlampPower > 0.002 && ((abs(vOmniWorldPos.x + 356.0) < 10.5 && vOmniWorldPos.y > -3.5) || (abs(vOmniWorldPos.x + 48.0) < 16.0 && vOmniWorldPos.z > 445.0 && vOmniWorldPos.z < 494.0) || (abs(vOmniWorldPos.x + 98.0) < 16.0 && vOmniWorldPos.z > 579.0 && vOmniWorldPos.z < 626.0) || (abs(vOmniWorldPos.x - 28.4) < 8.0 && abs(vOmniWorldPos.z - 78.2) < 8.0 && vOmniUV.y > -500.0))) {',
          '    if (diffuseColor.r > 0.92 && diffuseColor.g > 0.85 && diffuseColor.b > 0.68 && vOmniWorldPos.y > 0.9) {',
          '      gl_FragColor.rgb += vec3(2.30, 1.82, 1.08) * uStreetlampPower;',
          '    } else if (diffuseColor.r > 0.80 && diffuseColor.g > 0.55 && diffuseColor.b < 0.22) {',
          '      gl_FragColor.rgb += diffuseColor.rgb * vec3(1.12, 0.88, 0.52) * 0.78 * uStreetlampPower;',
          '    }',
          '  }',
          '  if (uIsNight > 0.05) {',
          '    if (diffuseColor.g > 0.78 && diffuseColor.g > diffuseColor.r * 2.4) {',
          '      gl_FragColor.rgb += vec3(0.08, 2.25, 0.82) * uIsNight;',
          '    } else if (diffuseColor.r > 0.90 && diffuseColor.r > diffuseColor.g * 5.0 && (vOmniWorldPos.y > 5.2 || (vOmniWorldPos.y > 2.4 && vOmniWorldPos.z > 480.0 && vOmniWorldPos.z < 595.0))) {',
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
          '  // E3. GENSHIN IMPACT STYLIZED RENDERING PIPELINE (TAHAP 1: WARM-COOL CEL RAMP, TAHAP 2: PAINTERLY FOLIAGE & MEADOWS, TAHAP 3: ANIME FRESNEL RIM)',
          '  if (uGenshinMode > 0.001 && !omniSubseaBore) {',
          '    float gMode = clamp(uGenshinMode, 0.0, 1.0);',
          '    float gHalfL = dot(wNorm, sunL) * 0.5 + 0.5;',
          '    float gAlbLum = max(dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722)), 0.035);',
          '    float gCurLum = dot(gl_FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));',
          '    float gLitRatio = clamp(gCurLum / gAlbLum, 0.0, 2.6);',
          '    float gCelBand = smoothstep(0.42, 0.66, gLitRatio * 0.72 + gHalfL * 0.28);',
          '    float gDayFactor = 1.0 - uIsNight * 0.82;',
          '#if OMNI_GROUP == 11',
          '    // Tahap 2A: Ghibli/Genshin Fluffy Spherical Canopy Normals, Deep Jade Underside/Shadow + Rich Emerald-Chartreuse Sunlit Crown',
          '    vec3 gPuffN = normalize(mix(wNorm, normalize(vec3(sin(vOmniWorldPos.x * 0.36 + vOmniWorldPos.z * 0.14), 1.25, cos(vOmniWorldPos.z * 0.36 - vOmniWorldPos.x * 0.14))), 0.55));',
          '    float gLeafHalf = dot(gPuffN, sunL) * 0.5 + 0.5;',
          '    float gLeafCel = smoothstep(0.38, 0.64, gLitRatio * 0.62 + gLeafHalf * 0.38);',
          '    float gCanopyTop = smoothstep(2.2, 13.5, vOmniWorldPos.y) * smoothstep(-0.15, 0.65, wNorm.y);',
          '    vec3 gJadeShadow = mix(vec3(0.038, 0.165, 0.155), vec3(0.058, 0.235, 0.195), gCanopyTop);',
          '    vec3 gSpringCrown = mix(vec3(0.145, 0.415, 0.125), vec3(0.265, 0.555, 0.145), gCanopyTop);',
          '    vec3 gStylizedFoliage = mix(gJadeShadow, gSpringCrown, gLeafCel) * clamp(gLitRatio * 0.58 + 0.42, 0.32, 1.22);',
          '    float gBackGlow = pow(clamp(dot(viewDirW, -sunL), 0.0, 1.0), 2.4) * sunVisUltra * gLeafCel;',
          '    gStylizedFoliage += vec3(0.30, 0.56, 0.12) * gBackGlow * 0.24;',
          '    gl_FragColor.rgb = mix(gl_FragColor.rgb, mix(gl_FragColor.rgb * 1.04, gStylizedFoliage, 0.74), gMode);',
          '#elif OMNI_GROUP == 9',
          '    // Tahap 2B: Mondstadt Emerald-Chartreuse Meadow Turf Preserving Crisp Cast Shadows + Sweeping Wind-Wave Sunlight Bands',
          '    if (wNorm.y > 0.68 && vOmniWorldPos.y > -0.35 && vOmniWorldPos.y < 38.0 && diffuseColor.g > diffuseColor.r * 1.04) {',
          '      float gWindWave = 0.5 + 0.5 * sin(dot(vOmniWorldPos.xz, vec2(0.075, 0.048)) - uTime * 1.75 + omniNoise(vOmniWorldPos.xz * 0.08) * 2.2);',
          '      float gWaveCrest = smoothstep(0.54, 0.88, gWindWave);',
          '      vec3 gMeadowShadow = vec3(0.058, 0.235, 0.168);',
          '      vec3 gMeadowLit = mix(vec3(0.155, 0.445, 0.138), vec3(0.235, 0.535, 0.152), gWaveCrest * 0.60 + 0.40 * gCelBand);',
          '      vec3 gMeadowCol = mix(gMeadowShadow, gMeadowLit, gCelBand) * clamp(gLitRatio * 0.68 + 0.34, 0.34, 1.18);',
          '      gl_FragColor.rgb = mix(gl_FragColor.rgb, mix(gl_FragColor.rgb * 1.04, gMeadowCol, 0.68), gMode);',
          '    } else {',
          '      vec3 gShadowed = gl_FragColor.rgb * vec3(0.84, 0.94, 1.16) + diffuseColor.rgb * vec3(0.04, 0.08, 0.14) * gDayFactor;',
          '      vec3 gSunlit = gl_FragColor.rgb * vec3(1.06, 1.02, 0.94) + diffuseColor.rgb * uSunColor * vec3(0.08, 0.06, 0.04) * sunVisUltra;',
          '      gl_FragColor.rgb = mix(gl_FragColor.rgb, mix(gShadowed, gSunlit, gCelBand), gMode * 0.85);',
          '    }',
          '#else',
          '    // Tahap 1: Architectural Warm Golden-Cream Key & Cool Sky-Cerulean Shadow Bounce + Rich Terracotta Roof Depth',
          '    vec3 gArchShadow = gl_FragColor.rgb * vec3(0.82, 0.93, 1.18) + diffuseColor.rgb * vec3(0.045, 0.085, 0.155) * gDayFactor;',
          '    vec3 gArchSunlit = gl_FragColor.rgb * vec3(1.06, 1.02, 0.94) + diffuseColor.rgb * uSunColor * vec3(0.09, 0.07, 0.04) * sunVisUltra;',
          '    vec3 gArchFinal = mix(gArchShadow, gArchSunlit, gCelBand);',
          '    if (omniPId == 2 || omniPId == 7 || (diffuseColor.r > diffuseColor.g * 1.35 && diffuseColor.r > 0.25)) {',
          '      vec3 gRoofShadow = gl_FragColor.rgb * vec3(0.78, 0.72, 0.92);',
          '      vec3 gRoofLit = gl_FragColor.rgb * vec3(1.08, 0.92, 0.82);',
          '      gArchFinal = mix(gRoofShadow, gRoofLit, gCelBand);',
          '    }',
          '    gl_FragColor.rgb = mix(gl_FragColor.rgb, gArchFinal, gMode * 0.85);',
          '#endif',
          '    // Tahap 3: Subtle Directional Anime Fresnel Rim-Light on Sunlit Edges',
          '    float gFresnelRim = smoothstep(0.68, 0.94, 1.0 - NdotV) * gCelBand * clamp(wNorm.y + 0.35, 0.0, 1.0);',
          '    gl_FragColor.rgb += uSunColor * vec3(1.05, 0.98, 0.82) * gFresnelRim * 0.12 * sunVisUltra * gMode;',
          '  } else if (uGenshinMode < -0.001 && !omniSubseaBore) {',
          '    // E3B. FORZATECH NEXT-GEN PHOTOREALISTIC HDR: MICRO-BITUMEN ROAD SHEEN, AUTOMOTIVE CLEARCOAT, & WAXY BOTANICAL CUTICLE',
          '    float fMode = clamp(-uGenshinMode, 0.0, 1.0);',
          '    vec3 fHalfV = normalize(sunL + viewDirW);',
          '    float fNdotH = max(0.0, dot(wNorm, fHalfV));',
          '    float fGrazing = pow(clamp(1.0 - NdotV, 0.0, 1.0), 4.2);',
          '    vec3 fSkyCoat = mix(mix(vec3(0.42, 0.66, 0.96), vec3(0.96, 0.46, 0.18), uDuskFactor), vec3(0.05, 0.09, 0.17), uIsNight);',
          '#if OMNI_GROUP == 11',
          '    float fCanopyAO = mix(0.76, 1.08, smoothstep(1.2, 11.5, vOmniWorldPos.y));',
          '    float fCuticleSpec = pow(fNdotH, 32.0) * 0.22 * sunVisUltra + fGrazing * 0.08;',
          '    gl_FragColor.rgb = mix(gl_FragColor.rgb, gl_FragColor.rgb * fCanopyAO + uSunColor * fCuticleSpec + fSkyCoat * fGrazing * 0.06, fMode);',
          '#elif OMNI_GROUP == 1 || OMNI_GROUP == 3 || OMNI_GROUP == 6',
          '    float fBitumenSparkle = pow(fNdotH, 52.0) * (0.26 + 0.54 * uWetness) * sunVisUltra;',
          '    float fRoadSheen = fGrazing * (0.16 + 0.42 * uWetness) * max(0.0, wNorm.y);',
          '    gl_FragColor.rgb += (uSunColor * fBitumenSparkle + fSkyCoat * fRoadSheen) * fMode;',
          '#else',
          '    float fClearcoat = pow(fNdotH, 84.0) * (0.24 + 0.38 * uWetness) * sunVisUltra;',
          '    float fArchFres = fGrazing * 0.12;',
          '    gl_FragColor.rgb = mix(gl_FragColor.rgb, gl_FragColor.rgb * 1.04 + uSunColor * fClearcoat + fSkyCoat * fArchFres, fMode);',
          '#endif',
          '  }',
          '',
          '  // F. DIRECTIONAL HENYEY-GREENSTEIN MIE AERIAL PERSPECTIVE IN-SCATTERING',
          '  float camDistAP = length(cameraPosition - vOmniWorldPos);',
          '  if (camDistAP > 30.0 && !omniSubseaBore) {',
          '    float cosSunV = max(dot(-viewDirW, sunL), 0.0);',
          '    float miePhase = pow(cosSunV, 5.0) * (1.0 - uIsNight * 0.85) * (1.0 - uRainIntensity * 0.65);',
          '    float airScatter = 1.0 - exp(-max(0.0, camDistAP - 30.0) * 0.00085);',
          '    vec3 mieTint = mix(vec3(0.76, 0.84, 0.95), vec3(0.98, 0.56, 0.26), uDuskFactor);',
          '    gl_FragColor.rgb += mieTint * airScatter * miePhase * 0.14;',
          '  }',
          '}',
          'if (uZebraInspectMode < 0.5) {',
          '  vec3 hvsSunL = normalize(uSunDir);',
          '  bool hvsInTunnel = (vOmniWorldPos.y < -3.65 && abs(vOmniWorldPos.x + 356.0) < 5.35 && vOmniWorldPos.z > 325.8 && vOmniWorldPos.z < 624.2);',
          '  if (!hvsInTunnel) {',
          '    gl_FragColor.rgb = nexusApplyBoundaryLayerAerialPerspective(',
          '      gl_FragColor.rgb,',
          '      cameraPosition,',
          '      vOmniWorldPos,',
          '      hvsSunL,',
          '      uDuskFactor,',
          '      uNightFactor,',
          '      uCloudCover,',
          '      uRainIntensity,',
          '      uLightningFlash,',
          '      uFogFactor',
          '    );',
          '  }',
          '  vec3 hvsRayW = normalize(vOmniWorldPos - cameraPosition);',
          '  gl_FragColor.rgb = nexusApplyHumanEyeHDR(gl_FragColor.rgb, hvsRayW, hvsSunL, uSunColor, uHvsParams, uHvsGlare);',
          '}'
        ].join('\n')
      )
      // 4B. CONSISTENT IEC sRGB OUTPUT ACROSS DIRECT CANVAS, 4x MSAA POSTFX TARGET, & PLANAR WATER REFLECTION TARGET
      .replace(
        '#include <colorspace_fragment>',
        'gl_FragColor.rgb = nexusLinearToSRGB(gl_FragColor.rgb);'
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
          '#include <dithering_fragment>',
          '#if OMNI_GROUP == 10 && !defined(USE_INSTANCING)',
          'if (vOmniUV.y < -1000.0 && uZebraInspectMode < 0.5) {',
          '  vec2 vtRelS = vOmniWorldPos.xz - vec2(28.4, 78.2);',
          '  float lxS = dot(vtRelS, vec2(0.67539916, 0.73745127));',
          '  float lyS = vOmniWorldPos.y - 5.88;',
          '  vec2 scrUV = clamp(vec2(lxS / 12.80 + 0.5, lyS / 7.20), 0.0, 1.0);',
          '  float rainScrC = clamp(uRainIntensity, 0.0, 1.35);',
          '  float fogScrC = clamp(uFogFactor, 0.0, 1.0);',
          '  float cloudScrC = clamp(uCloudCover, 0.0, 1.0);',
          '  float duskScrC = clamp(uDuskFactor, 0.0, 1.0);',
          '  float nightScrC = clamp(uNightFactor, 0.0, 1.0);',
          '  float gScrC = clamp(uGenshinMode, 0.0, 1.0);',
          '  // 1. Nyquist Band-Limited 16x9 Modular Outdoor Die-Cast LED Cabinet Seams (0.80m x 0.80m)',
          '  float bCab = omniBand(omniPxSize, 1.25);',
          '  float cabGrid = omniJoint1D(lxS + 6.40, 0.80, 0.005) * omniJoint1D(lyS, 0.80, 0.005);',
          '  float cabOpen = mix(1.0, mix(0.84, 1.0, cabGrid), bCab);',
          '  // 2. Nyquist Band-Limited SMD P8 (8mm Pitch = 125 LEDs/m) RGB Diode Matrix & Anti-Glare Micro-Louvers',
          '  float bPix = omniBand(omniPxSize, 125.0);',
          '  vec2 pCell = fract(vec2((lxS + 6.40) * 125.0, lyS * 125.0)) - 0.5;',
          '  float smdCapsule = 1.0 - smoothstep(0.28, 0.46, max(abs(pCell.x), abs(pCell.y)));',
          '  float microLouver = smoothstep(-0.48, -0.34, pCell.y);',
          '  float diodeOpen = mix(0.978, mix(0.22, 1.0, smdCapsule * microLouver), bPix);',
          '  // 3. Viewing-Angle Louver Gain, Sky/Sun Specular Sheen, Rainwater Rivulets, & 3D Boundary-Layer Aerial Haze',
          '  vec3 scrNormW = vec3(-0.73745127, 0.0, 0.67539916);',
          '  vec3 viewRayToCam = cameraPosition - vOmniWorldPos;',
          '  float distScr = length(viewRayToCam);',
          '  vec3 viewDirScr = viewRayToCam / max(distScr, 1e-4);',
          '  vec3 camToScrDir = -viewDirScr;',
          '  float nDotVScr = clamp(dot(scrNormW, viewDirScr), 0.0, 1.0);',
          '  float louverGain = mix(0.74, 1.0, smoothstep(0.08, 0.58, nDotVScr));',
          '  vec3 sunDirScr = normalize(uSunDir);',
          '  vec3 reflScr = reflect(camToScrDir, scrNormW);',
          '  vec3 skyHorizS = nexusEvalHorizonSkySRGB(',
          '    vec3(camToScrDir.x, 0.0, camToScrDir.z),',
          '    sunDirScr,',
          '    duskScrC,',
          '    nightScrC,',
          '    cloudScrC,',
          '    rainScrC,',
          '    clamp(rainScrC * 0.65, 0.0, 1.0),',
          '    fogScrC,',
          '    0.25 * nightScrC',
          '  );',
          '  float epoxyFres = (0.048 + 0.24 * pow(1.0 - nDotVScr, 3.6)) * (1.0 + 0.55 * uWetness) * (1.0 - nightScrC * 0.72);',
          '  float sunLouvSpec = pow(max(dot(reflScr, sunDirScr), 0.0), mix(32.0, 18.0, gScrC)) * (1.0 - nightScrC) * (1.0 - min(1.0, rainScrC) * 0.72) * (0.14 + 0.12 * gScrC);',
          '  float celRimScr = smoothstep(0.42, 0.88, 1.0 - nDotVScr) * gScrC * (0.08 + 0.06 * (1.0 - nightScrC));',
          '  vec3 glintRGB = skyHorizS * epoxyFres + mix(uSunColor, vec3(1.0, 0.92, 0.76), gScrC * 0.5) * sunLouvSpec + vec3(0.65, 0.85, 1.0) * celRimScr;',
          '  // 3B. Physical Rainwater Rivulets & Falling Rain Sheet on LED Cabinet Surface',
          '  float scrRivMask = 0.0;',
          '  vec3 rivGlintRGB = vec3(0.0);',
          '  if (rainScrC > 0.04 || uWetness > 0.15) {',
          '    float flowYScr = lyS * 1.45 + uTime * (2.1 + 1.55 * rainScrC);',
          '    float rivNoise = omniNoise(vec2(lxS * 9.2 + sin(lyS * 2.6) * 0.28, flowYScr));',
          '    scrRivMask = smoothstep(0.68, 0.91, rivNoise) * clamp(rainScrC * 0.85 + uWetness * 0.25, 0.0, 1.0);',
          '    rivGlintRGB = (skyHorizS * 0.85 + vec3(0.75, 0.86, 1.0) * uLightningFlash) * scrRivMask * (0.14 + 0.22 * (1.0 - nDotVScr));',
          '  }',
          '  // 3C. Unified 3D Boundary-Layer Atmospheric Extinction (H_BL = 46m) + Near-Field Fog/Rain Mist',
          '  float ucScr = max(0.0, cameraPosition.y) / 46.0;',
          '  float upScr = max(0.0, vOmniWorldPos.y) / 46.0;',
          '  float duScr = upScr - ucScr;',
          '  float meanBLScr = abs(duScr) > 0.006 ? (exp(-ucScr) - exp(-upScr)) / duScr : exp(-0.5 * (ucScr + upScr));',
          '  float effDistClear = max(0.0, distScr - 12.0);',
          '  float effDistWeather = max(0.0, distScr - 3.5);',
          '  float tauClear = effDistClear * (0.00032 + 0.00028 * cloudScrC) * clamp(meanBLScr, 0.15, 1.0) * mix(1.0, 1.42, gScrC);',
          '  float tauWeather = effDistWeather * (0.0046 * rainScrC + 0.0088 * fogScrC) * clamp(meanBLScr + 0.25, 0.35, 1.0);',
          '  float airVeil = clamp(1.0 - exp(-(tauClear + tauWeather)), 0.0, mix(0.85, 0.90, fogScrC));',
          '  float mieScr = pow(max(dot(camToScrDir, sunDirScr), 0.0), mix(4.5, 2.8, gScrC)) * (1.0 - nightScrC * 0.85) * (1.0 - min(1.0, rainScrC) * 0.65);',
          '  vec3 mieTintScr = mix(mix(vec3(0.78, 0.86, 0.96), vec3(1.0, 0.92, 0.74), gScrC * 0.7), vec3(1.0, 0.56, 0.28), duskScrC);',
          '  vec3 veilColSRGB = clamp(skyHorizS + mieTintScr * mieScr * (0.22 + 0.18 * gScrC + 0.14 * fogScrC) + vec3(0.65, 0.82, 1.0) * uLightningFlash * 0.55, 0.0, 1.0);',
          '  float edgeBorder = smoothstep(0.36, 0.50, max(abs(scrUV.x - 0.5), abs(scrUV.y - 0.5)));',
          '  float grazeRad = pow(1.0 - nDotVScr, 1.8) * (0.28 + 0.52 * nightScrC + 0.35 * rainScrC + 0.42 * fogScrC);',
          '  float radPulse = 0.5 + 0.5 * sin(uTime * 0.85 + scrUV.x * 2.2);',
          '  vec3 smdRadTint = mix(vec3(0.38, 0.76, 1.18), vec3(1.05, 0.84, 0.42), radPulse);',
          '  vec3 smdRadGlow = smdRadTint * (grazeRad + edgeBorder * (0.18 + 0.28 * nightScrC + 0.22 * rainScrC));',
          '  if (uVideotronPortalAlpha > 0.5) {',
          '    // Premultiplied-Alpha Optical Punch-Through + 1.47M SMD Diode Atmospheric Radiation Glow!',
          '    float weatherDim = 1.0 - scrRivMask * 0.20 - clamp(rainScrC * 0.06 + fogScrC * 0.08, 0.0, 0.16);',
          '    float videoTrans = clamp(diodeOpen * cabOpen * louverGain * weatherDim * (1.0 - airVeil) * (1.0 - grazeRad * 0.35), 0.0, 0.968);',
          '    float maskAlpha = 1.0 - videoTrans;',
          '    vec3 genshinTintVeil = mix(vec3(0.0), mix(vec3(0.03, 0.04, 0.06), vec3(0.06, 0.03, 0.05), duskScrC), gScrC);',
          '    vec3 premulOverlay = (glintRGB + rivGlintRGB + smdRadGlow + genshinTintVeil) * (1.0 - airVeil) + (veilColSRGB + smdRadTint * (0.26 * nightScrC + 0.18 * rainScrC)) * airVeil;',
          '    gl_FragColor = vec4(clamp(premulOverlay, 0.0, 1.0), clamp(maskAlpha, 0.032, 1.0));',
          '  } else {',
          '    // Rich Procedural Universe Island Broadcast SMD Display (Active when YouTube iframe is offline / pre-handshake / headless)',
          '    float waveA = 0.5 + 0.5 * sin(scrUV.x * 5.2 - uTime * 1.15 + scrUV.y * 2.4);',
          '    float waveB = 0.5 + 0.5 * cos(scrUV.x * 8.4 + uTime * 0.82 - scrUV.y * 3.1);',
          '    vec3 uiNavy = mix(vec3(0.04, 0.09, 0.22), vec3(0.09, 0.24, 0.48), waveA * 0.72 + waveB * 0.28);',
          '    vec3 uiGold = vec3(0.96, 0.78, 0.18);',
          '    vec3 uiWhite = vec3(0.94, 0.96, 1.0);',
          '    // Center 8-Ray Star-Compass Ring + Stylized Menara Utama Silhouette + Live Equalizer Bars',
          '    vec2 mUV = (scrUV - vec2(0.50, 0.55)) * vec2(1.778, 1.0);',
          '    float mR = length(mUV);',
          '    float mAng = atan(mUV.y, mUV.x);',
          '    float sunRays = smoothstep(0.32, 0.16, mR) * step(0.16, mR) * (0.55 + 0.45 * cos(mAng * 8.0 - uTime * 0.65));',
          '    float crestRing = (1.0 - smoothstep(0.008, 0.020, abs(mR - 0.235))) + (1.0 - smoothstep(0.006, 0.015, abs(mR - 0.165))) * 0.75;',
          '    float tajugTier1 = step(abs(mUV.x), max(0.0, (0.64 - scrUV.y) * 0.85)) * step(0.45, scrUV.y) * step(scrUV.y, 0.64);',
          '    float tajugTier2 = step(abs(mUV.x), max(0.0, (0.73 - scrUV.y) * 0.58)) * step(0.61, scrUV.y) * step(scrUV.y, 0.73);',
          '    float tajugSpire = step(abs(mUV.x), 0.012) * step(0.70, scrUV.y) * step(scrUV.y, 0.79);',
          '    float tajugMask = clamp(tajugTier1 + tajugTier2 + tajugSpire, 0.0, 1.0);',
          '    // Top Header Bar ("UNIVERSE BROADCAST · PULAU U & I") & Bottom Scrolling Ticker Strip',
          '    float topBar = smoothstep(0.845, 0.858, scrUV.y);',
          '    float botBar = 1.0 - smoothstep(0.135, 0.148, scrUV.y);',
          '    float goldRule = (1.0 - smoothstep(0.004, 0.010, abs(scrUV.y - 0.845))) + (1.0 - smoothstep(0.004, 0.010, abs(scrUV.y - 0.145)));',
          '    float liveBadge = topBar * step(0.04, scrUV.x) * step(scrUV.x, 0.14) * step(0.875, scrUV.y) * step(scrUV.y, 0.955);',
          '    float headerBlocks = topBar * step(0.17, scrUV.x) * step(scrUV.x, 0.62) * step(0.895, scrUV.y) * step(scrUV.y, 0.935)',
          '                       * step(0.32, fract(scrUV.x * 34.0));',
          '    float tickerScroll = botBar * step(0.045, scrUV.y) * step(scrUV.y, 0.095)',
          '                       * step(0.38, omniNoise(vec2(floor((scrUV.x + uTime * 0.055) * 58.0), 3.7)));',
          '    float eqCol = floor((scrUV.x - 0.70) * 44.0);',
          '    float eqH = 0.22 + 0.18 * (0.5 + 0.5 * sin(eqCol * 0.9 + uTime * 4.2)) * (0.5 + 0.5 * cos(eqCol * 0.4 - uTime * 2.7));',
          '    float eqMask = step(0.70, scrUV.x) * step(scrUV.x, 0.94) * step(0.20, scrUV.y) * step(scrUV.y, 0.20 + eqH)',
          '                 * step(0.26, fract((scrUV.x - 0.70) * 44.0));',
          '    vec3 bcastRGB = uiNavy;',
          '    bcastRGB += uiGold * (sunRays * 0.34 + crestRing * 0.88 + tajugMask * 0.92 + eqMask * 0.75);',
          '    bcastRGB = mix(bcastRGB, vec3(0.025, 0.055, 0.13), clamp(topBar + botBar, 0.0, 1.0) * 0.88);',
          '    bcastRGB = mix(bcastRGB, vec3(0.86, 0.14, 0.12), liveBadge * (0.78 + 0.22 * sin(uTime * 3.8)));',
          '    bcastRGB = mix(bcastRGB, uiWhite, headerBlocks * 0.88);',
          '    bcastRGB = mix(bcastRGB, uiGold, clamp(goldRule * 0.92 + tickerScroll * 0.84, 0.0, 1.0));',
          '    float shortBand = step(abs(scrUV.x - 0.5), 0.1582);',
          '    if (uVideotronShortMode > 0.5 && shortBand < 0.5) {',
          '      bcastRGB = mix(uiNavy * 0.58, bcastRGB * 0.48, 0.35);',
          '    }',
          '    if (gScrC > 0.01) {',
          '      float bLuma = dot(bcastRGB, vec3(0.2126, 0.7152, 0.0722));',
          '      vec3 bVibrant = mix(vec3(bLuma), bcastRGB, 1.34) * vec3(1.06, 1.04, 1.10);',
          '      bcastRGB = mix(bcastRGB, clamp(bVibrant, 0.0, 1.0), gScrC);',
          '    }',
          '    float smdBoost = mix(1.18, 0.96, nightScrC) * (1.0 - scrRivMask * 0.18);',
          '    vec3 finalFallback = bcastRGB * smdBoost * (diodeOpen * cabOpen * louverGain) + glintRGB + rivGlintRGB;',
          '    gl_FragColor = vec4(clamp(mix(finalFallback, veilColSRGB, airVeil), 0.0, 1.0), 1.0);',
          '  }',
          '}',
          '#endif'
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

          // Subtle tropical rain-runoff weathering drip streaks below upper reveal joint
          for (let ds = 0; ds < 3; ds++) {
            const dx = x0 + 14 + (((c * 53 + r * 29 + ds * 41) % Math.max(12, Math.floor(cw - 28))));
            const dLen = ch * (0.22 + ((c + r + ds) % 4) * 0.11);
            const dripG = ctx.createLinearGradient(0, y0 + 2, 0, y0 + 2 + dLen);
            dripG.addColorStop(0.0, 'rgba(118, 112, 98, 0.16)');
            dripG.addColorStop(1.0, 'rgba(118, 112, 98, 0.0)');
            ctx.fillStyle = dripG;
            ctx.fillRect(dx, y0 + 2, 3.5 + (ds % 2) * 2.0, dLen);
          }

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
 * 2. 2048x2048 Weathered Tropical Flat Concrete Roof Deck Slab
 */
export function createWeatheredFlatRoofTexture() {
  return makeProceduralCanvasTexture(
    2048,
    2048,
    (ctx, w, h) => {
      ctx.fillStyle = '#8b8f92';
      ctx.fillRect(0, 0, w, h);

      const grid = 16;
      const cell = w / grid;
      for (let y = 0; y < grid; y++) {
        for (let x = 0; x < grid; x++) {
          const v = 132 + (((x * 29 + y * 43) % 23) - 11);
          ctx.fillStyle = `rgb(${v}, ${v + 3}, ${v + 5})`;
          ctx.fillRect(x * cell + 3, y * cell + 3, cell - 6, cell - 6);
          ctx.strokeStyle = 'rgba(68, 72, 74, 0.58)';
          ctx.lineWidth = 2.5;
          ctx.strokeRect(x * cell + 1.5, y * cell + 1.5, cell - 3, cell - 3);
        }
      }

      for (let i = 0; i < 56; i++) {
        const px = ((i * 97) % 100) * 0.01 * w;
        const py = ((i * 53) % 100) * 0.01 * h;
        const rad = 52 + (i % 7) * 34;
        const g = ctx.createRadialGradient(px, py, 4, px, py, rad);
        g.addColorStop(0, 'rgba(56, 62, 57, 0.34)');
        g.addColorStop(1, 'rgba(56, 62, 57, 0.0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(px, py, rad, 0, Math.PI * 2);
        ctx.fill();
      }

      for (let i = 0; i < 7200; i++) {
        const px = (i * 4999) % w;
        const py = (i * 3571) % h;
        ctx.fillStyle = i % 2 === 0 ? 'rgba(52,55,58,0.22)' : 'rgba(208,212,214,0.18)';
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

          // Upper interlocking nib highlight, subtle tropical lichen/moss valley patina, & course overlap shadow!
          if (((r * 17 + c * 31) % 9) < 2) {
            ctx.fillStyle = 'rgba(58, 66, 42, 0.24)';
            ctx.fillRect(x0 + 1, y0 + th - 8.5, tw * 0.34, 5.0);
          }
          ctx.fillStyle = 'rgba(235, 148, 120, 0.22)';
          ctx.fillRect(x0 + tw * 0.18, y0 + th - 6.0, tw * 0.55, 2.2);
          ctx.fillStyle = 'rgba(34, 10, 7, 0.66)';
          ctx.fillRect(x0, y0 + th - 3.8, tw, 3.8);
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
 * 9. 2048x2048 Sculpted Bas-Relief Universe Island Star-Compass Emblem on Crisp Off-White Crown Panel
 */
export function createMakaraEmblemTextures() {
  const drawUniverseStarMotif = (ctx, w, h, isEmissive) => {
    const cx = w * 0.5;
    const cy = h * 0.50;
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

      const shGrad = ctx.createRadialGradient(cx, cy + 8, rOuter * 0.25, cx, cy + 8, rOuter * 1.08);
      shGrad.addColorStop(0.0, 'rgba(84, 79, 70, 0.36)');
      shGrad.addColorStop(0.85, 'rgba(84, 79, 70, 0.20)');
      shGrad.addColorStop(1.0, 'rgba(84, 79, 70, 0.0)');
      ctx.fillStyle = shGrad;
      ctx.beginPath();
      ctx.arc(cx, cy + 5, rOuter * 1.08, 0, Math.PI * 2);
      ctx.fill();
    }

    const reliefMain = isEmissive ? '#ffd24c' : '#857f73';
    const reliefHighlight = isEmissive ? '#fff2b2' : '#b5ae9f';
    const reliefShadow = isEmissive ? '#c99322' : '#585349';

    // Outer & Inner Celestial Orbital Rings
    ctx.strokeStyle = reliefShadow;
    ctx.lineWidth = w * 0.034;
    ctx.beginPath();
    ctx.arc(cx, cy + 3, rOuter * 0.92, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = reliefMain;
    ctx.lineWidth = w * 0.025;
    ctx.beginPath();
    ctx.arc(cx, cy, rOuter * 0.92, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = reliefMain;
    ctx.lineWidth = w * 0.015;
    ctx.beginPath();
    ctx.arc(cx, cy - rOuter * 0.03, rOuter * 0.56, 0, Math.PI * 2);
    ctx.stroke();

    // 8-Pointed Universe Star-Compass Rays
    for (let r = 0; r < 8; r++) {
      const ang = (r * Math.PI) / 4.0 - Math.PI * 0.5;
      const isCard = r % 2 === 0;
      const tipR = rOuter * (isCard ? 0.84 : 0.62);
      const shoulderR = rOuter * (isCard ? 0.19 : 0.14);
      const aL = ang - 0.26;
      const aR = ang + 0.26;

      ctx.fillStyle = isCard ? reliefHighlight : reliefMain;
      ctx.beginPath();
      ctx.moveTo(cx, cy - rOuter * 0.03);
      ctx.lineTo(cx + Math.cos(aL) * shoulderR, cy - rOuter * 0.03 + Math.sin(aL) * shoulderR);
      ctx.lineTo(cx + Math.cos(ang) * tipR, cy - rOuter * 0.03 + Math.sin(ang) * tipR);
      ctx.lineTo(cx + Math.cos(aR) * shoulderR, cy - rOuter * 0.03 + Math.sin(aR) * shoulderR);
      ctx.closePath();
      ctx.fill();
    }

    // Central Compass Sun-Core & 4 Orbital Satellite Nodes
    ctx.fillStyle = reliefShadow;
    ctx.beginPath();
    ctx.arc(cx, cy - rOuter * 0.03, rOuter * 0.20, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = reliefHighlight;
    ctx.beginPath();
    ctx.arc(cx, cy - rOuter * 0.03, rOuter * 0.13, 0, Math.PI * 2);
    ctx.fill();

    for (let k = 0; k < 4; k++) {
      const ang = (k * Math.PI) / 2.0 + Math.PI / 4.0;
      ctx.fillStyle = reliefHighlight;
      ctx.beginPath();
      ctx.arc(
        cx + Math.cos(ang) * rOuter * 0.56,
        cy - rOuter * 0.03 + Math.sin(ang) * rOuter * 0.56,
        rOuter * 0.055,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }

    // Twin-Archipelago Strait Wave Arc at Base
    ctx.strokeStyle = reliefHighlight;
    ctx.lineWidth = w * 0.024;
    ctx.beginPath();
    ctx.arc(cx, cy + rOuter * 0.36, rOuter * 0.42, 0.18 * Math.PI, 0.82 * Math.PI);
    ctx.stroke();
  };

  const map = makeProceduralCanvasTexture(2048, 2048, (ctx, w, h) => drawUniverseStarMotif(ctx, w, h, false), 1, 1);
  const emissiveMap = makeProceduralCanvasTexture(2048, 2048, (ctx, w, h) => drawUniverseStarMotif(ctx, w, h, true), 1, 1);
  return { map, emissiveMap };
}
