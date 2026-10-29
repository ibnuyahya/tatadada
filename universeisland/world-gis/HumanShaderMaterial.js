import * as THREE from 'three';
import { RainCollisionBake } from '../core/RainCollisionBake.js';
import { HumanEyeHDROptics } from '../core/HumanEyeHDROptics.js';
import { createSharedSurfaceUniforms } from '../core/ProceduralTextureFactory.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.4
 * ./world-gis/HumanShaderMaterial.js
 *
 * Dedicated Multi-Class Photorealistic Human PBR, Epidermal Subsurface Scattering (SSS),
 * Corneal Eye Catchlight, Kajiya-Kay Anisotropic Hair Sheen, Charlie Cloth Microfiber Sheen,
 * Garment-Locked Procedural Javanese Batik Kawung/Parang & UI Jakun Cotton-Twill Weave,
 * and Self-Illuminated Smartphone Screen Shader — all in STRICTLY 1 SINGLE DRAW CALL!
 */

export const HUMAN_MAT = {
  SKIN: 0.10,
  EYE_CORNEA: 0.20,
  METAL_ACCENT: 0.25,
  HAIR: 0.30,
  JAKUN_TWILL: 0.40,
  CREST_GOLD: 0.45,
  HIJAB_SILK: 0.50,
  BATIK_MOTIF: 0.60,
  FABRIC_LOWER: 0.70,
  SHOE_BAG: 0.80,
  PHONE_GLOW: 0.90,
  UMBRELLA_CANOPY: 0.96,
  CAT_FUR: 1.15,
  CAT_NOSE_EAR: 1.25,
  CAT_EYE_TAPETUM: 1.35
};

/**
 * Creates the single unified MeshStandardMaterial for all 24 Sivitas Akademika humans
 * with per-vertex material-class optics (`aHumanMeta` + garment-locked `uv`).
 *
 * @param {Object} [sharedUniforms]
 * @returns {THREE.MeshStandardMaterial}
 */
export function createPhotorealHumanMaterial(sharedUniforms = null) {
  const uniforms = sharedUniforms || createSharedSurfaceUniforms();

  const mat = new THREE.MeshStandardMaterial({
    name: 'SivitasAkademikaUIPhotorealHumanMat',
    color: 0xffffff,
    vertexColors: true,
    roughness: 0.68,
    metalness: 0.04,
    envMapIntensity: 0.16
  });

  const hvsFallback = HumanEyeHDROptics.createUniforms();
  mat.userData.omniShaderInjected = true;
  mat.customProgramCacheKey = () => 'nexus-photoreal-human-feline-hvs-v36';

  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uIsNight = uniforms.uIsNight || { value: 0.0 };
    shader.uniforms.uNightFactor = uniforms.uNightFactor || { value: 0.0 };
    shader.uniforms.uDuskFactor = uniforms.uDuskFactor || { value: 0.0 };
    shader.uniforms.uCloudCover = uniforms.uCloudCover || { value: 0.22 };
    shader.uniforms.uSunDir =
      uniforms.uSunDir || { value: new THREE.Vector3(0.52, 0.72, 0.45).normalize() };
    shader.uniforms.uSunColor = uniforms.uSunColor || { value: new THREE.Color(0xfff5e4) };
    shader.uniforms.uShaderQuality = uniforms.uShaderQuality || { value: 1.0 };
    shader.uniforms.uZebraInspectMode = uniforms.uZebraInspectMode || { value: 0.0 };
    shader.uniforms.uTime = uniforms.uTime || { value: 0.0 };
    shader.uniforms.uWetness = uniforms.uWetness || { value: 0.0 };
    shader.uniforms.uRainIntensity = uniforms.uRainIntensity || { value: 0.0 };
    shader.uniforms.uLightningFlash = uniforms.uLightningFlash || { value: 0.0 };
    shader.uniforms.uCollisionMap =
      uniforms.uCollisionMap || { value: RainCollisionBake.createFallbackTexture() };
    shader.uniforms.uCollisionReady = uniforms.uCollisionReady || { value: 0.0 };
    shader.uniforms.uHvsParams = uniforms.uHvsParams || hvsFallback.uHvsParams;
    shader.uniforms.uHvsGlare = uniforms.uHvsGlare || hvsFallback.uHvsGlare;

    shader.vertexShader = [
      'attribute vec3 aHumanMeta;',
      'varying vec3 vHumanMeta;',
      'varying vec2 vHumanUV;',
      'varying vec3 vHumanWorldPos;',
      'varying vec3 vHumanWorldNorm;',
      shader.vertexShader
    ]
      .join('\n')
      .replace(
        '#include <begin_vertex>',
        [
          '#include <begin_vertex>',
          'vHumanMeta = aHumanMeta;',
          'vHumanUV = uv;',
          'vHumanWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;',
          'vHumanWorldNorm = normalize(mat3(modelMatrix) * objectNormal);'
        ].join('\n')
      );

    shader.fragmentShader = [
      'varying vec3 vHumanMeta;',
      'varying vec2 vHumanUV;',
      'varying vec3 vHumanWorldPos;',
      'varying vec3 vHumanWorldNorm;',
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
      'float humanHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }',
      'float humanNoise(vec2 p) {',
      '  vec2 i = floor(p); vec2 f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);',
      '  return mix(mix(humanHash(i), humanHash(i + vec2(1.0, 0.0)), u.x),',
      '             mix(humanHash(i + vec2(0.0, 1.0)), humanHash(i + vec2(1.0, 1.0)), u.x), u.y);',
      '}',
      'float humanSkyExposure(vec3 wp) {',
      '  if (uCollisionReady < 0.5) return 1.0;',
      '  vec2 cuv = (wp.xz + 185.0) / 370.0;',
      '  if (cuv.x < 0.002 || cuv.x > 0.998 || cuv.y < 0.002 || cuv.y > 0.998) return 1.0;',
      '  vec4 cTex = texture2D(uCollisionMap, cuv);',
      '  float yTop = -6.0 + ((cTex.r * 255.0 * 256.0 + cTex.g * 255.0) / 65535.0) * 60.0;',
      '  float isCanopyAbove = step(0.98, cTex.a);',
      '  float shelter = smoothstep(0.65, 1.85, yTop - wp.y);',
      '  return mix(1.0, isCanopyAbove * 0.25, shelter);',
      '}',
      shader.fragmentShader
    ]
      .join('\n')
      // 1. Garment-Locked Micro-Weave, Javanese Batik Kawung Motif, & UI Makara Pocket Crest
      .replace(
        '#include <color_fragment>',
        [
          '#include <color_fragment>',
          'float mCls = vHumanMeta.x;',
          'float cavAO = clamp(vHumanMeta.y, 0.25, 1.15);',
          'if (uZebraInspectMode < 0.5 && uShaderQuality > 0.5) {',
          '  if (abs(mCls - 0.60) < 0.03) {',
          '    // Procedural High-Resolution Javanese Batik Kawung & Isen-Isen Medallion Motif!',
          '    vec2 bCell = fract(vHumanUV * vec2(10.0, 13.0)) - 0.5;',
          '    float d1 = length(bCell - vec2(0.5, 0.5));',
          '    float d2 = length(bCell - vec2(-0.5, 0.5));',
          '    float d3 = length(bCell - vec2(0.5, -0.5));',
          '    float d4 = length(bCell - vec2(-0.5, -0.5));',
          '    float petal = smoothstep(0.055, 0.020, abs(d1 - 0.50))',
          '                + smoothstep(0.055, 0.020, abs(d2 - 0.50))',
          '                + smoothstep(0.055, 0.020, abs(d3 - 0.50))',
          '                + smoothstep(0.055, 0.020, abs(d4 - 0.50));',
          '    float cecek = smoothstep(0.085, 0.045, length(bCell))',
          '                + smoothstep(0.065, 0.030, length(abs(bCell) - vec2(0.23, 0.23)));',
          '    float waxLine = clamp(petal * 0.72 + cecek * 0.85, 0.0, 1.0);',
          '    vec3 malamGold = vec3(0.84, 0.68, 0.38);',
          '    vec3 indigoAccent = diffuseColor.rgb * 0.62;',
          '    float innerLobe = step(0.47, min(min(d1, d2), min(d3, d4)));',
          '    diffuseColor.rgb = mix(diffuseColor.rgb, indigoAccent, innerLobe * 0.45);',
          '    diffuseColor.rgb = mix(diffuseColor.rgb, malamGold, waxLine * 0.78);',
          '  } else if (abs(mCls - 0.40) < 0.025) {',
          '    // Iconic UI Alma Mater Yellow Jacket ("Jakun") 45-deg Cotton-Twill Micro-Weave',
          '    float twill = sin((vHumanUV.x * 1.35 + vHumanUV.y) * 290.0) * 0.028;',
          '    float weft = sin((vHumanUV.x - vHumanUV.y * 1.35) * 180.0) * 0.014;',
          '    diffuseColor.rgb *= (1.0 + twill + weft);',
          '  } else if (abs(mCls - 0.45) < 0.025) {',
          '    // Embroidered Golden Makara UI Crest & Lanyard Weave Detail',
          '    vec2 cUV = vHumanUV * 2.0 - 1.0;',
          '    float rCrest = length(cUV);',
          '    float rays = 0.5 + 0.5 * cos(atan(cUV.y, cUV.x) * 9.0);',
          '    float emb = smoothstep(0.88, 0.72, rCrest) * (0.78 + 0.30 * rays);',
          '    diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.95, 0.77, 0.22), emb * 0.45);',
          '  } else if (abs(mCls - 0.70) < 0.03) {',
          '    // Tailored Chino / Denim / Pleated Skirt Micro-Warp',
          '    float weave = (humanNoise(vHumanUV * vec2(190.0, 210.0)) - 0.5) * 0.055;',
          '    diffuseColor.rgb *= (1.0 + weave);',
          '  } else if (abs(mCls - 0.10) < 0.03) {',
          '    // Natural Living Skin Micro-Tone Variation',
          '    float pore = (humanNoise(vHumanUV * 150.0) - 0.5) * 0.022;',
          '    diffuseColor.rgb *= (1.0 + pore);',
          '  } else if (abs(mCls - 1.15) < 0.04) {',
          '    // Photorealistic Feline Fur Micro-Strands & 12 Genetic Coat Patterns (Kucing Kepulauan UI!)',
          '    float patCode = floor(vHumanMeta.z + 0.001);',
          '    float cSeed = fract(vHumanMeta.z) * 73.17;',
          '    float uBody = vHumanUV.x;',
          '    float vCirc = vHumanUV.y;',
          '    float absV = abs(vCirc);',
          '    float isLimb = step(1.02, uBody);',
          '    float limbT = clamp((uBody - 1.05) / 0.40, 0.0, 1.0);',
          '    float isTail = (1.0 - isLimb) * step(0.72, uBody);',
          '    float furFiber = (humanNoise(vec2(uBody * 96.0 + cSeed, vCirc * 280.0)) - 0.5) * 0.11',
          '                   + (humanNoise(vec2(uBody * 210.0, vCirc * 520.0 - cSeed)) - 0.5) * 0.06;',
          '    vec3 baseCoat = diffuseColor.rgb;',
          '    float warp = humanNoise(vec2(uBody * 9.5 + cSeed, vCirc * 4.2)) * 1.65;',
          '',
          '    if (patCode > 0.5 && patCode < 4.5) {',
          '      // 1: Oyen Mackerel Tabby, 2: Oyen Bicolor White, 3: Mujair Brown Tabby, 4: Silver Blotched Tabby',
          '      vec3 darkStripeCol = (patCode < 2.5) ? baseCoat * vec3(0.56, 0.36, 0.18) : baseCoat * 0.24;',
          '      vec3 creamBelly = mix(baseCoat, vec3(0.91, 0.87, 0.79), (patCode < 2.5) ? 0.62 : 0.48);',
          '      float spineLine = (1.0 - isLimb) * exp(-vCirc * vCirc * 22.0) * smoothstep(0.08, 0.22, uBody);',
          '      float flankWave = 0.0;',
          '      if (abs(patCode - 4.0) < 0.4) {',
          '        // Classic Swirling Marble / Bullseye Blotched Tabby',
          '        vec2 swirlP = vec2(uBody * 14.0 + warp * 0.9, absV * 6.5);',
          '        flankWave = smoothstep(0.15, 0.65, sin(length(fract(swirlP) - 0.5) * 18.8 + warp));',
          '      } else {',
          '        // Mackerel Tiger Stripes + Forehead M-Mark',
          '        flankWave = smoothstep(0.18, 0.72, sin(uBody * 74.0 + absV * 7.2 + warp * 1.4));',
          '      }',
          '      float tailRings = isTail * smoothstep(0.10, 0.68, sin((uBody - 0.72) * 118.0));',
          '      float tailTip = isTail * smoothstep(0.92, 0.99, uBody);',
          '      float legBands = isLimb * smoothstep(0.22, 0.75, sin(limbT * 28.0 + warp * 0.5)) * (1.0 - limbT * 0.45);',
          '      float browM = (1.0 - isLimb) * smoothstep(0.15, 0.06, abs(uBody - 0.10)) * smoothstep(0.46, 0.10, absV)',
          '                  * smoothstep(0.0, 0.7, sin(vCirc * 26.0));',
          '      float stripeMask = clamp(spineLine * 0.88 + flankWave * (1.0 - isLimb) * (1.0 - isTail) * 0.78 + tailRings * 0.85 + tailTip + legBands * 0.72 + browM * 0.85, 0.0, 1.0);',
          '      float bellyFade = (1.0 - isLimb) * (1.0 - isTail) * smoothstep(0.58, 0.90, absV);',
          '      baseCoat = mix(baseCoat, darkStripeCol, stripeMask * (1.0 - bellyFade * 0.75));',
          '      baseCoat = mix(baseCoat, creamBelly, bellyFade);',
          '      if (abs(patCode - 2.0) < 0.4) {',
          '        // Oyen-Putih Bicolor Piebald: Crisp White Tuxedo Bib, Muzzle Blaze, Belly & Socks!',
          '        float whiteBib = (1.0 - isLimb) * (1.0 - isTail) * smoothstep(0.50, 0.64, absV + (humanNoise(vec2(uBody * 7.0 + cSeed, vCirc * 3.0)) - 0.5) * 0.18);',
          '        float whiteBlaze = (1.0 - isLimb) * smoothstep(0.085, 0.035, uBody) * smoothstep(0.22, 0.48, absV);',
          '        float whiteSocks = isLimb * smoothstep(0.60, 0.74, limbT);',
          '        baseCoat = mix(baseCoat, vec3(0.93, 0.92, 0.89), clamp(whiteBib + whiteBlaze + whiteSocks, 0.0, 1.0));',
          '      }',
          '    } else if (abs(patCode - 5.0) < 0.4) {',
          '      // 5: Calico / Belang Tiga (Snowy White Base + Domain-Warped Ginger Tabby & Jet-Black Patches)',
          '      vec3 whiteFur = vec3(0.93, 0.92, 0.89);',
          '      vec3 gingerFur = vec3(0.80, 0.41, 0.13) * (0.86 + 0.24 * sin(uBody * 72.0 + warp));',
          '      vec3 blackFur = vec3(0.055, 0.052, 0.058);',
          '      float pA = humanNoise(vec2(uBody * 4.6 + cSeed, vCirc * 2.3 + warp * 0.35));',
          '      float pB = humanNoise(vec2(uBody * 5.2 - cSeed * 0.7, vCirc * 2.7 - warp * 0.30));',
          '      float dorsalBias = (1.0 - isLimb * 0.65) * (1.0 - smoothstep(0.55, 0.86, absV) * 0.75);',
          '      float orangeMask = smoothstep(0.46, 0.56, pA * dorsalBias + isTail * 0.22);',
          '      float blackMask = smoothstep(0.48, 0.58, pB * dorsalBias + isTail * 0.28) * (1.0 - orangeMask * 0.85);',
          '      baseCoat = mix(whiteFur, gingerFur, orangeMask);',
          '      baseCoat = mix(baseCoat, blackFur, blackMask);',
          '    } else if (abs(patCode - 6.0) < 0.4) {',
          '      // 6: Formal Tuxedo Cat (Glossy Panther Black + Snowy White Bib, Chin & 4 White Socks)',
          '      vec3 tuxBlack = vec3(0.052, 0.052, 0.060);',
          '      vec3 tuxWhite = vec3(0.93, 0.92, 0.90);',
          '      float bib = (1.0 - isLimb) * smoothstep(0.12, 0.18, uBody) * smoothstep(0.50, 0.38, uBody) * smoothstep(0.58, 0.70, absV);',
          '      float chin = (1.0 - isLimb) * smoothstep(0.075, 0.025, uBody) * smoothstep(0.42, 0.58, absV);',
          '      float socks = isLimb * smoothstep(0.68, 0.78, limbT);',
          '      baseCoat = mix(tuxBlack, tuxWhite, clamp(bib + chin + socks, 0.0, 1.0));',
          '    } else if (abs(patCode - 7.0) < 0.4) {',
          '      // 7: Tortoiseshell / Tortie (Mottled Obsidian & Fiery Amber-Gold Mosaic)',
          '      float m1 = humanNoise(vec2(uBody * 16.0 + cSeed, vCirc * 11.0 + warp));',
          '      float m2 = humanNoise(vec2(uBody * 34.0 - cSeed, vCirc * 24.0));',
          '      float blaze = smoothstep(0.44, 0.66, m1 * 0.7 + m2 * 0.45);',
          '      baseCoat = mix(vec3(0.060, 0.052, 0.050), vec3(0.78, 0.39, 0.12), blaze);',
          '    } else if (abs(patCode - 8.0) < 0.4) {',
          '      // 8: Siamese / Himalayan Seal-Point (Creamy Fawn Body + Thermal Dark Espresso Points)',
          '      vec3 fawnBody = vec3(0.85, 0.78, 0.67);',
          '      vec3 sealPoint = vec3(0.12, 0.082, 0.068);',
          '      float faceMask = (1.0 - isLimb) * smoothstep(0.115, 0.035, uBody) * (0.45 + 0.55 * smoothstep(0.15, 0.65, absV));',
          '      float earMask = (1.0 - isLimb) * smoothstep(0.14, 0.19, uBody) * smoothstep(0.24, 0.17, uBody) * (1.0 - smoothstep(0.25, 0.55, absV));',
          '      float tailMask = isTail * smoothstep(0.72, 0.82, uBody);',
          '      float pawMask = isLimb * smoothstep(0.28, 0.72, limbT);',
          '      float pointAmt = clamp(max(max(faceMask, earMask), max(tailMask, pawMask)), 0.0, 1.0);',
          '      baseCoat = mix(fawnBody, sealPoint, pointAmt);',
          '    } else if (abs(patCode - 12.0) < 0.4) {',
          '      // 12: Cow-Cat Harlequin (High-White Coat + Bold Black Spots & Black Tail)',
          '      float spotN = humanNoise(vec2(uBody * 5.0 + cSeed, vCirc * 2.5 + warp * 0.25));',
          '      float spots = smoothstep(0.56, 0.64, spotN * (1.0 - isLimb * 0.5)) + isTail * 0.95;',
          '      baseCoat = mix(vec3(0.93, 0.92, 0.90), vec3(0.055, 0.055, 0.062), clamp(spots, 0.0, 1.0));',
          '    }',
          '    diffuseColor.rgb = clamp(baseCoat * (1.0 + furFiber), vec3(0.02), vec3(0.98));',
          '  } else if (abs(mCls - 1.25) < 0.04) {',
          '    // Damp Rhinarium Nose Leather & Delicate Inner Ear Concha Micro-Grain',
          '    float pebble = (humanNoise(vHumanUV * 220.0) - 0.5) * 0.06;',
          '    diffuseColor.rgb *= (1.0 + pebble);',
          '  } else if (abs(mCls - 1.35) < 0.04) {',
          '    // Slanted Feline Iris Stroma Fibers & Diurnal Vertical-Slit Pupil!',
          '    vec2 eUV = vHumanUV * 2.0 - 1.0;',
          '    float rEye = length(eUV);',
          '    float angEye = atan(eUV.y, eUV.x);',
          '    float stroma = 0.84 + 0.22 * sin(angEye * 32.0 + humanNoise(eUV * 12.0) * 3.0);',
          '    float limbal = smoothstep(0.98, 0.74, rEye);',
          '    float pupilW = mix(0.13, 0.46, clamp(uIsNight * 0.85 + uDuskFactor * 0.35, 0.0, 1.0));',
          '    float slitDist = length(vec2(eUV.x / max(0.08, pupilW), eUV.y / 0.74));',
          '    float pupilMask = 1.0 - smoothstep(0.84, 1.02, slitDist);',
          '    vec3 irisCol = diffuseColor.rgb * stroma * (0.42 + 0.68 * limbal);',
          '    diffuseColor.rgb = mix(irisCol, vec3(0.015, 0.015, 0.022), pupilMask);',
          '  }',
          '  diffuseColor.rgb *= cavAO;',
          '}'
        ].join('\n')
      )
      // 2. Per-Material-Class Roughness + Rain Wetness Modulation
      .replace(
        '#include <roughnessmap_fragment>',
        [
          '#include <roughnessmap_fragment>',
          'float mClsR = vHumanMeta.x;',
          'if (abs(mClsR - 0.10) < 0.03) roughnessFactor = 0.43;',        // Human skin dual-lobe
          'else if (abs(mClsR - 0.20) < 0.03) roughnessFactor = 0.032;',  // Wet eye cornea!
          'else if (abs(mClsR - 0.25) < 0.03) roughnessFactor = 0.20;',   // Eyeglasses / Watch metal
          'else if (abs(mClsR - 0.30) < 0.03) roughnessFactor = 0.46;',   // Hair locks
          'else if (abs(mClsR - 0.40) < 0.025) roughnessFactor = 0.74;',  // UI Jakun cotton twill
          'else if (abs(mClsR - 0.45) < 0.025) roughnessFactor = 0.34;',  // Gold Makara embroidery
          'else if (abs(mClsR - 0.50) < 0.03) roughnessFactor = 0.60;',   // Voal / Silk Hijab
          'else if (abs(mClsR - 0.60) < 0.03) roughnessFactor = 0.56;',   // Pressed Batik shirt
          'else if (abs(mClsR - 0.70) < 0.03) roughnessFactor = 0.82;',   // Trousers / Pleated skirt
          'else if (abs(mClsR - 0.80) < 0.03) roughnessFactor = 0.48;',   // Sneakers & Nylon bag
          'else if (abs(mClsR - 0.90) < 0.03) roughnessFactor = 0.07;',   // Glass smartphone screen
          'else if (abs(mClsR - 0.96) < 0.03) roughnessFactor = 0.32;',   // Taut nylon umbrella canopy
          'else if (abs(mClsR - 1.15) < 0.04) roughnessFactor = 0.78;',   // Velvety feline fur coat
          'else if (abs(mClsR - 1.25) < 0.04) roughnessFactor = 0.28;',   // Damp cat nose & inner ear
          'else if (abs(mClsR - 1.35) < 0.04) roughnessFactor = 0.024;',  // Wet feline eye cornea!
          'float hSkyExp = 1.0;',
          'if (uWetness > 0.005 && uZebraInspectMode < 0.5) {',
          '  hSkyExp = humanSkyExposure(vHumanWorldPos);',
          '  float effWetH = uWetness * (0.12 + 0.88 * hSkyExp);',
          '  roughnessFactor = mix(roughnessFactor, max(0.05, roughnessFactor * 0.38), effWetH * 0.80);',
          '}'
        ].join('\n')
      )
      // 3. Per-Material-Class Metalness
      .replace(
        '#include <metalnessmap_fragment>',
        [
          '#include <metalnessmap_fragment>',
          'float mClsM = vHumanMeta.x;',
          'if (abs(mClsM - 0.25) < 0.03) metalnessFactor = 0.82;',
          'else if (abs(mClsM - 0.45) < 0.025) metalnessFactor = 0.42;',
          'else metalnessFactor = 0.03;'
        ].join('\n')
      )
      // 4. ULTRA LINEAR-HDR HUMAN OPTICS (PRE-TONEMAPPING: Epidermal SSS, Corneal Eye Catchlight,
      //    Kajiya-Kay Anisotropic Hair, Charlie Cloth Sheen, & Live Smartphone Viewfinder!)
      .replace(
        '#include <tonemapping_fragment>',
        [
          'if (uShaderQuality > 0.5 && uZebraInspectMode < 0.5) {',
          '  vec3 N = normalize(vHumanWorldNorm);',
          '  vec3 V = normalize(cameraPosition - vHumanWorldPos);',
          '  vec3 L = normalize(uSunDir);',
          '  float NdotV = max(dot(N, V), 0.0);',
          '  float NdotL = dot(N, L);',
          '  float dayFactor = 1.0 - uIsNight * 0.72;',
          '  float mClsP = vHumanMeta.x;',
          '',
          '  if (abs(mClsP - 0.10) < 0.03) {',
          '    // Pre-Integrated Epidermal Subsurface Scattering coupled to live Sun/Dusk/Moon radiance',
          '    float wrapL = clamp((NdotL + 0.50) / 1.50, 0.0, 1.0);',
          '    float sssBand = wrapL * (1.0 - smoothstep(0.22, 0.84, max(NdotL, 0.0)));',
          '    float backTrans = pow(clamp(-NdotL * 0.65 + (1.0 - NdotV) * 0.55, 0.0, 1.0), 2.3);',
          '    vec3 dermisGlow = vec3(0.28, 0.072, 0.036) * uSunColor * (sssBand * 0.92 + backTrans * 0.48) * dayFactor;',
          '    gl_FragColor.rgb += dermisGlow * clamp(vHumanMeta.y, 0.4, 1.0);',
          '  } else if (abs(mClsP - 0.20) < 0.03) {',
          '    // Living Wet Corneal Eye HDR Specular Catchlight Sparkle!',
          '    vec3 R = reflect(-V, N);',
          '    float glintA = pow(max(dot(R, L), 0.0), 120.0) * 1.45;',
          '    float glintB = pow(max(dot(R, normalize(vec3(-0.35, 0.72, 0.58))), 0.0), 76.0) * 0.48;',
          '    gl_FragColor.rgb += uSunColor * glintA + vec3(0.92, 0.96, 1.0) * glintB;',
          '  } else if (abs(mClsP - 0.30) < 0.03) {',
          '    // Dual-Shift Kajiya-Kay / Marschner Anisotropic Hair Crown Strand Sheen',
          '    vec3 T = normalize(cross(N, vec3(0.0, 1.0, 0.01)) + vec3(1e-4));',
          '    vec3 H = normalize(L + V);',
          '    float TdotH = dot(T, H);',
          '    float sinTH = sqrt(max(0.0, 1.0 - TdotH * TdotH));',
          '    float hairPrimary = pow(sinTH, 36.0) * max(NdotL + 0.3, 0.0) * 0.12;',
          '    float hairSecondary = pow(sinTH, 14.0) * max(NdotL + 0.3, 0.0) * 0.055;',
          '    gl_FragColor.rgb += (vec3(0.65, 0.58, 0.50) * hairPrimary + vec3(0.45, 0.32, 0.22) * hairSecondary) * uSunColor * dayFactor;',
          '  } else if (abs(mClsP - 0.40) < 0.025 || abs(mClsP - 0.50) < 0.03) {',
          '    // Charlie Microfiber / Voal Silk Grazing-Angle Cloth Sheen',
          '    float clothRim = pow(1.0 - NdotV, 3.4) * (0.11 + 0.06 * max(NdotL, 0.0)) * dayFactor;',
          '    gl_FragColor.rgb += (gl_FragColor.rgb + vec3(0.04) * uSunColor) * clothRim;',
          '  } else if (abs(mClsP - 0.90) < 0.03) {',
          '    // Self-Illuminated Smartphone Live Camera Viewfinder (Mini Rektorat Tajug Preview!)',
          '    vec2 sUV = vHumanUV;',
          '    vec3 skyPrev = mix(vec3(0.62, 0.80, 0.96), vec3(0.24, 0.52, 0.88), clamp(sUV.y, 0.0, 1.0));',
          '    skyPrev = mix(skyPrev, vec3(0.92, 0.52, 0.28), uDuskFactor * 0.65);',
          '    float tajugSil = step(abs(sUV.x - 0.5), max(0.0, (0.76 - sUV.y) * 0.55)) * step(0.22, sUV.y);',
          '    vec3 screenImg = mix(skyPrev, vec3(0.72, 0.34, 0.25), tajugSil);',
          '    screenImg = mix(vec3(0.14, 0.22, 0.16), screenImg, step(0.18, sUV.y));',
          '    gl_FragColor.rgb = mix(gl_FragColor.rgb, screenImg * 1.15, 0.82);',
          '  } else if (abs(mClsP - 1.15) < 0.04) {',
          '    // Charlie / Kajiya-Kay Soft Feline Fur Grazing Halo & Guard-Hair Sheen',
          '    float furRim = pow(1.0 - NdotV, 3.0) * (0.16 + 0.14 * max(NdotL + 0.25, 0.0)) * dayFactor;',
          '    vec3 Tfur = normalize(cross(N, vec3(0.0, 1.0, 0.02)) + vec3(1e-4));',
          '    vec3 Hfur = normalize(L + V);',
          '    float sinFH = sqrt(max(0.0, 1.0 - dot(Tfur, Hfur) * dot(Tfur, Hfur)));',
          '    float guardSheen = pow(sinFH, 24.0) * max(NdotL + 0.2, 0.0) * 0.065 * dayFactor;',
          '    gl_FragColor.rgb += (gl_FragColor.rgb * 0.75 + vec3(0.08) * uSunColor) * furRim + uSunColor * guardSheen;',
          '  } else if (abs(mClsP - 1.25) < 0.04) {',
          '    // Thin Translucent Rose Feline Ear Pinna Backlit Subsurface Scattering',
          '    float earTrans = pow(clamp(-NdotL * 0.70 + (1.0 - NdotV) * 0.45, 0.0, 1.0), 2.1);',
          '    gl_FragColor.rgb += vec3(0.36, 0.09, 0.07) * uSunColor * earTrans * 0.65 * dayFactor;',
          '  } else if (abs(mClsP - 1.35) < 0.04) {',
          '    // Wet Feline Corneal Sun Catchlight + Strictly Nighttime Tapetum Lucidum Retro-Reflection!',
          '    vec3 Rf = reflect(-V, N);',
          '    float glint1 = pow(max(dot(Rf, L), 0.0), 135.0) * 1.55;',
          '    float glint2 = pow(max(dot(Rf, normalize(vec3(-0.32, 0.75, 0.56))), 0.0), 82.0) * 0.52;',
          '    float tapetum = uIsNight * pow(NdotV, 4.2) * 0.36;',
          '    gl_FragColor.rgb += uSunColor * glint1 + vec3(0.92, 0.96, 1.0) * glint2 + vec3(0.58, 0.84, 0.42) * tapetum;',
          '  }',
          '',
          '  // Wet-weather fabric darkening, rain-bead umbrella specular, & lightning reflection',
          '  if (uWetness > 0.005 && abs(mClsP - 0.90) > 0.03) {',
          '    float effW = uWetness * (0.12 + 0.88 * hSkyExp);',
          '    float absorb = (abs(mClsP - 0.96) < 0.03 || abs(mClsP - 0.20) < 0.03) ? 0.15 : 0.72;',
          '    gl_FragColor.rgb *= mix(1.0, 0.70, effW * absorb);',
          '    if (abs(mClsP - 0.96) < 0.03) {',
          '      float bead = pow(1.0 - NdotV, 3.2) * effW * 0.26;',
          '      gl_FragColor.rgb += vec3(0.62, 0.72, 0.86) * bead;',
          '    }',
          '  }',
          '  if (uLightningFlash > 0.01) {',
          '    float lRim = pow(1.0 - NdotV, 2.2) * max(0.15, N.y * 0.5 + 0.5);',
          '    gl_FragColor.rgb += vec3(0.55, 0.68, 0.96) * uLightningFlash * lRim * 0.42;',
          '  }',
          '}',
          'if (uZebraInspectMode < 0.5) {',
          '  vec3 hvsSunH = normalize(uSunDir);',
          '  gl_FragColor.rgb = nexusApplyBoundaryLayerAerialPerspective(',
          '    gl_FragColor.rgb,',
          '    cameraPosition,',
          '    vHumanWorldPos,',
          '    hvsSunH,',
          '    uDuskFactor,',
          '    uNightFactor,',
          '    uCloudCover,',
          '    uRainIntensity,',
          '    uLightningFlash',
          '  );',
          '  vec3 hvsRayH = normalize(vHumanWorldPos - cameraPosition);',
          '  gl_FragColor.rgb = nexusApplyHumanEyeHDR(gl_FragColor.rgb, hvsRayH, hvsSunH, uSunColor, uHvsParams, uHvsGlare);',
          '}'
        ].join('\n')
      )
      // 5. STUDIO ZEBRA / CLAY DIAGNOSTIC OVERRIDE
      .replace(
        '#include <dithering_fragment>',
        [
          'if (uZebraInspectMode > 1.5) {',
          '  vec3 nClay = normalize(vHumanWorldNorm);',
          '  vec3 keyDir = normalize(vec3(0.52, 0.76, 0.38));',
          '  vec3 fillDir = normalize(vec3(-0.48, 0.32, -0.55));',
          '  float diffKey = max(dot(nClay, keyDir), 0.0);',
          '  float diffFill = max(dot(nClay, fillDir), 0.0) * 0.35;',
          '  float rim = pow(1.0 - max(dot(nClay, normalize(cameraPosition - vHumanWorldPos)), 0.0), 2.5) * 0.18;',
          '  gl_FragColor.rgb = vec3(0.76, 0.73, 0.69) * (0.24 + 0.72 * diffKey + diffFill) + vec3(rim);',
          '} else if (uZebraInspectMode > 0.5) {',
          '  vec3 viewDirZ = normalize(cameraPosition - vHumanWorldPos);',
          '  vec3 reflZ = reflect(-viewDirZ, normalize(vHumanWorldNorm));',
          '  float stripe = smoothstep(-0.1, 0.1, sin(reflZ.y * 28.0 + reflZ.x * 8.0));',
          '  gl_FragColor.rgb = mix(vec3(0.12, 0.14, 0.18), vec3(0.95, 0.97, 1.0), stripe);',
          '}',
          '#include <dithering_fragment>'
        ].join('\n')
      );
  };

  return mat;
}
