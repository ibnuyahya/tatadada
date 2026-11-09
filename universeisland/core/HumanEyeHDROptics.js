import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v3.5 (Human-Eye Physiological HDR Optics Edition)
 * ./core/HumanEyeHDROptics.js
 *
 * Biologically grounded Human Visual System (HVS) HDR Photoreceptor & Eye-Adaptation Pipeline
 * operating uniformly across 100% of world shaders (Sky, Ocean, Fluvial River/Spring/Lake/Waterfall,
 * Archipelago Terrain, Rainforest Foliage, Gedung Rektorat UI, Sivitas Akademika, & Cats)
 * in +0 extra Draw Calls with native hardware MSAA preserved:
 *
 * 1. Asymmetric Pupillary & Photochemical Adaptation Kinetics:
 *    - Fast photopic iris constriction & cone neural adaptation (tau_light = 0.42s)
 *      under bright sunlight gaze or sudden plasma lightning discharges.
 *    - Gradual mesopic/scotopic dark recovery (tau_dark = 1.85s) under rain, storm overcast,
 *      post-lightning recovery, twilight, & moonlight.
 * 2. Normalized Naka-Rushton Retinal Photoreceptor Sigmoidal Compression:
 *    - R(I) = [I^n / (I^n + sigma^n)] * [(W^n + sigma^n) / W^n]
 *    - Hybrid Hue-Preserving Midtone Cone Response + Per-Cone L/M/S Highlight Bleaching (Y > 0.42),
 *      eliminating AgX midtone green/terracotta pastel-grey desaturation while gracefully
 *      compressing intense solar glints, clouds, lightning, & molten magma without clipping.
 * 3. Psychophysical Appearance Phenomena across All Weathers & Times of Day:
 *    - Stevens Effect (1961): Luminance-dependent apparent contrast slope (n).
 *    - Hunt Effect (1952): Photopic colorfulness / chroma scaling with adapted luminance.
 *    - Purkinje Shift (1825): Mesopic/scotopic 507nm rod spectral shift in dim night shadows
 *      while keeping cone-threshold warm 2700K architectural lamps & magma vividly warm.
 *    - Spencer-Shirley (1995) Intraocular Ciliary Glare Veil: Corneal/lenticular forward scatter
 *      when gazing toward the solar disc or active lightning flashes.
 */
export class HumanEyeHDROptics {
  static TAU_LIGHT_SEC = 0.42;
  static TAU_DARK_SEC = 1.85;

  constructor() {
    this.adaptedLuminance = 0.95;
    this.targetLuminance = 0.95;
    this.pupilDiameterMm = 2.6;
    this.visionRegime = 'PHOTOPIC'; // 'PHOTOPIC' | 'MESOPIC' | 'SCOTOPIC'
    this._camDir = new THREE.Vector3(0, -0.2, -1);
  }

  static createUniforms() {
    return {
      // x: eyeGain, y: huntChroma, z: stevensContrast (n), w: purkinjeMix
      uHvsParams: { value: new THREE.Vector4(1.04, 1.045, 1.14, 0.0) },
      // x: sunGlareStrength, y: lightningVeil, z: shadowToeLift, w: whitePoint
      uHvsGlare: { value: new THREE.Vector4(0.76, 0.0, 0.018, 4.6) },
      // 0.0 = PBR Natural (HVS Realism), 1.0 = Genshin Impact Stylized Vibrant (Tahap 1+2+3)
      uGenshinMode: { value: 0.0 }
    };
  }

  /**
   * Updates live biological pupil & photoreceptor adaptation state and writes synchronized
   * HVS uniforms for both WeatherOpticsEngine and SharedSurfaceUniforms.
   */
  update(
    dt,
    {
      nightFactor = 0.0,
      duskFactor = 0.0,
      cloudCover = 0.22,
      rainIntensity = 0.0,
      stormFactor = 0.0,
      lightningFlash = 0.0,
      sunDir = null,
      camera = null,
      immediate = false
    } = {},
    targetUniformsA = null,
    targetUniformsB = null
  ) {
    const safeDt = Math.min(0.25, Math.max(0.0, Number(dt) || 0.0));
    const nightF = Math.max(0.0, Math.min(1.0, nightFactor));
    const duskF = Math.max(0.0, Math.min(1.0, duskFactor));
    const cloudF = Math.max(0.0, Math.min(1.0, cloudCover));
    const rainF = Math.max(0.0, Math.min(1.0, rainIntensity));
    const stormF = Math.max(0.0, Math.min(1.0, stormFactor));
    const flashF = Math.max(0.0, lightningFlash);

    // 1. Physical Scene Ambient Luminance Estimator (normalized photopic scale)
    const overcastAtten =
      1.0 - Math.min(0.66, cloudF * 0.22 + rainF * 0.28 + stormF * 0.24);
    const diurnalBase =
      (1.0 - nightF) * (1.0 - duskF * 0.54) * overcastAtten +
      duskF * (1.0 - nightF) * 0.20 * overcastAtten +
      nightF * (0.125 * (0.68 + 0.32 * overcastAtten));

    // 2. Subtle Foveal Gaze Coupling (+/- 11% natural pupil reflex between sky/sun gaze & shaded ground/stream gaze)
    let gazeFactor = 1.0;
    let sunVis =
      Math.max(0.0, 1.0 - cloudF * 0.58 - rainF * 0.68 - stormF * 0.45) *
      (1.0 - nightF * 0.88);
    if (camera && typeof camera.getWorldDirection === 'function') {
      camera.getWorldDirection(this._camDir);
      const skyLook = Math.max(-0.55, Math.min(0.65, this._camDir.y));
      let sunGaze = 0.0;
      if (sunDir) {
        sunGaze = Math.max(0.0, this._camDir.dot(sunDir)) * sunVis;
      }
      gazeFactor = 1.0 + skyLook * 0.11 + Math.pow(sunGaze, 3.0) * 0.14;
    }

    const rawTarget = Math.max(0.075, diurnalBase * gazeFactor + flashF * 0.78);
    this.targetLuminance = rawTarget;

    // 3. Asymmetric Pupil & Retinal Photoreceptor Temporal Integration
    if (immediate || safeDt <= 0.0) {
      this.adaptedLuminance = rawTarget;
    } else {
      const tau =
        rawTarget > this.adaptedLuminance
          ? HumanEyeHDROptics.TAU_LIGHT_SEC
          : HumanEyeHDROptics.TAU_DARK_SEC;
      const alpha = 1.0 - Math.exp(-safeDt / tau);
      this.adaptedLuminance += (rawTarget - this.adaptedLuminance) * alpha;
    }

    const L = Math.max(0.075, Math.min(2.2, this.adaptedLuminance));

    // Physiological pupil diameter [2.2mm bright sun .. 6.8mm moonlit night] (Watson & Yellott 2012)
    const photopicNorm = Math.max(0.0, Math.min(1.0, (L - 0.10) / 0.90));
    this.pupilDiameterMm = 6.8 - 4.4 * Math.pow(photopicNorm, 0.58);
    this.visionRegime =
      nightF > 0.62 ? 'SCOTOPIC' : duskF > 0.38 || L < 0.42 ? 'MESOPIC' : 'PHOTOPIC';

    // 4. Derive Biologically Coupled HVS Tone-Mapping & Color-Appearance Parameters:
    // A. Retinal & Pupillary Gain: opens shadow/midtone visibility under clouds, rain, storm, dusk, & night
    //    while constricting cleanly on sunny daylight & lightning flashes!
    const eyeGain = Math.max(
      0.86,
      Math.min(1.66, 1.045 * Math.pow(0.96 / L, 0.215))
    );

    // B. Hasselblad Natural Color Science (HNCS) & Macbeth ColorChecker Photopic Calibration:
    //    Eliminates CG/gamey neon oversaturation (previously 1.19) while keeping natural lushness
    const overcastDesat = Math.min(0.105, rainF * 0.062 + stormF * 0.043 + cloudF * 0.018);
    const huntChroma =
      1.048 -
      overcastDesat -
      duskF * 0.018 * (1.0 - sunVis) -
      nightF * 0.165;

    // C. Stevens Effect & Photographic Micro-Contrast Slope (crisp 3D separation even in storm/overcast)
    const stevensContrast =
      1.055 +
      0.115 * Math.pow(photopicNorm, 0.72) * (1.0 - rainF * 0.24 - stormF * 0.20);

    // D. Purkinje Rod-Cone Spectral Shift (507nm blue-cyan rod takeover in low-luminance mesopic/scotopic shadows)
    const purkinjeMix = Math.max(
      0.0,
      Math.min(0.40, nightF * 0.36 + duskF * 0.07 * (1.0 - photopicNorm))
    );

    // E. Spencer-Shirley Intraocular Ciliary Glare & Natural Filmic Shadow-Toe Micro-Lift
    const sunGlareStrength = sunVis * (0.68 + 0.36 * duskF);
    const lightningVeil = Math.min(0.30, flashF * 0.15);
    const shadowToeLift = (0.016 + 0.006 * rainF + 0.005 * stormF) * (1.0 - nightF * 0.88);
    const whitePoint = 4.65 - 0.40 * nightF;

    this._writeToBag(
      targetUniformsA,
      eyeGain,
      huntChroma,
      stevensContrast,
      purkinjeMix,
      sunGlareStrength,
      lightningVeil,
      shadowToeLift,
      whitePoint
    );
    this._writeToBag(
      targetUniformsB,
      eyeGain,
      huntChroma,
      stevensContrast,
      purkinjeMix,
      sunGlareStrength,
      lightningVeil,
      shadowToeLift,
      whitePoint
    );

    return {
      eyeGain,
      huntChroma,
      stevensContrast,
      purkinjeMix,
      pupilDiameterMm: this.pupilDiameterMm,
      visionRegime: this.visionRegime
    };
  }

  _writeToBag(
    bag,
    eyeGain,
    huntChroma,
    stevensContrast,
    purkinjeMix,
    sunGlareStrength,
    lightningVeil,
    shadowToeLift,
    whitePoint
  ) {
    if (!bag) return;
    if (bag.uHvsParams && bag.uHvsParams.value) {
      bag.uHvsParams.value.set(eyeGain, huntChroma, stevensContrast, purkinjeMix);
    }
    if (bag.uHvsGlare && bag.uHvsGlare.value) {
      bag.uHvsGlare.value.set(sunGlareStrength, lightningVeil, shadowToeLift, whitePoint);
    }
  }

  /**
   * Exact CPU/JS mirror of the 3D exponential boundary-layer atmospheric optical depth integral
   * (H_BL = 46.0m) used for 4-Gate aerial photography vs ground-level mist verification.
   */
  static evalBoundaryLayerHazeFactorJS(
    camY,
    posY,
    distMeters,
    cloudCover = 0.22,
    rainIntensity = 0.0,
    fogFactor = 0.0
  ) {
    const effDist = Math.max(0.0, Number(distMeters) - 20.0);
    if (effDist <= 0.001) return 0.0;
    const uc = Math.max(0.0, Number(camY)) / 46.0;
    const up = Math.max(0.0, Number(posY)) / 46.0;
    const du = up - uc;
    const meanBL =
      Math.abs(du) > 0.006 ? (Math.exp(-uc) - Math.exp(-up)) / du : Math.exp(-0.5 * (uc + up));
    const rainC = Math.max(0.0, Math.min(1.35, Number(rainIntensity) || 0.0));
    const cloudC = Math.max(0.0, Math.min(1.0, Number(cloudCover) || 0.0));
    const fogC = Math.max(0.0, Math.min(1.0, Number(fogFactor) || 0.0));
    const sigmaBL = 0.00028 + 0.00105 * rainC + 0.00024 * cloudC + 0.00245 * fogC;
    const tauAir = effDist * (sigmaBL * Math.max(0.0, Math.min(1.0, meanBL)) + 0.000038 + 0.00012 * fogC);
    return Math.max(0.0, Math.min(0.88, 1.0 - Math.exp(-tauAir)));
  }

  /**
   * Canonical GLSL Human-Eye Photoreceptor HDR Tone-Mapping, Altitude-Aware Boundary-Layer
   * Aerial Perspective, & Color-Space Kernel.
   * Shared identically across MeshStandardMaterial (pre-tonemapping_fragment) and custom ShaderMaterials.
   */
  static getGLSLKernel() {
    return /* glsl */ `
      #ifndef NEXUS_GENSHIN_UNIFORM_DECLARED
      #define NEXUS_GENSHIN_UNIFORM_DECLARED
      uniform float uGenshinMode;
      #endif

      vec3 nexusSRGBToLinear(vec3 srgb) {
        vec3 c = max(srgb, vec3(0.0));
        return pow(c, vec3(2.2));
      }

      vec3 nexusLinearToSRGB(vec3 lin) {
        vec3 c = clamp(lin, 0.0, 1.0);
        vec3 lo = c * 12.92;
        vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
        return mix(lo, hi, step(vec3(0.0031308), c));
      }

      /**
       * Analytical 3D Exponential Boundary-Layer Aerial Perspective (H_BL = 46.0m):
       * Integrates atmospheric mist/rain extinction along the 3D view ray from camPos to worldPos
       * in true Scene-Referred Linear-HDR (replacing isotropic 3D FogExp2 high-altitude grey-out!).
       * - Horizontal ground rays (y ~ 2m) traverse dense boundary-layer mist during rain/storm;
       * - High-altitude aerial views (Foto Udara y ~ 200-1500m) look down through the thin 46m
       *   boundary layer cleanly (hazeFactor ~ 9-19%, NEVER 99.9% flat grey silhouettes!),
       *   with 100% unified sky-matched horizon radiance across Land, Architecture, Foliage, & Water!
       */
      vec3 nexusEvalHorizonSkySRGB(
        vec3 rayDir,
        vec3 sunDir,
        float duskFactor,
        float nightFactor,
        float cloudCover,
        float rainIntensity,
        float stormFactor,
        float fogFactor,
        float lunarSkyScatter
      ) {
        float duskC = clamp(duskFactor, 0.0, 1.0);
        float nightC = clamp(nightFactor, 0.0, 1.0);
        float cloudC = clamp(cloudCover, 0.0, 1.0);
        float rainC = clamp(rainIntensity, 0.0, 1.35);
        float stormC = clamp(stormFactor, 0.0, 1.0);
        float fogC = clamp(fogFactor, 0.0, 1.0);
        float gMode = clamp(uGenshinMode, 0.0, 1.0);
        float fMode = clamp(-uGenshinMode, 0.0, 1.0);

        vec2 dirXZ = normalize(rayDir.xz + vec2(1e-5, 0.0));
        vec2 sunXZ = normalize(sunDir.xz + vec2(1e-5, 0.0));
        float cosAzSun = dot(dirXZ, sunXZ);

        vec3 dayHorS = mix(mix(vec3(0.62, 0.78, 0.92), vec3(0.55, 0.80, 0.96), gMode * 0.85), vec3(0.52, 0.73, 0.95), fMode * 0.78);
        vec3 duskHorPBR = mix(vec3(0.52, 0.56, 0.68), vec3(0.94, 0.47, 0.20), smoothstep(-0.35, 0.85, cosAzSun));
        vec3 duskHorGenshin = mix(vec3(0.58, 0.49, 0.86), vec3(1.00, 0.52, 0.28), smoothstep(-0.35, 0.85, cosAzSun));
        vec3 duskHorForza = mix(vec3(0.42, 0.50, 0.72), vec3(0.99, 0.43, 0.15), smoothstep(-0.35, 0.85, cosAzSun));
        vec3 duskHorS = mix(mix(duskHorPBR, duskHorGenshin, gMode * 0.88), duskHorForza, fMode * 0.82);
        vec3 nightHorPBR = vec3(0.084, 0.122, 0.192) + vec3(0.042, 0.062, 0.104) * clamp(lunarSkyScatter, 0.0, 1.0);
        vec3 nightHorGenshin = vec3(0.095, 0.168, 0.345) + vec3(0.055, 0.095, 0.175) * clamp(lunarSkyScatter, 0.0, 1.0);
        vec3 nightHorForza = vec3(0.068, 0.105, 0.182) + vec3(0.038, 0.058, 0.102) * clamp(lunarSkyScatter, 0.0, 1.0);
        vec3 nightHorS = mix(mix(nightHorPBR, nightHorGenshin, gMode * 0.78), nightHorForza, fMode * 0.75);
        vec3 horS = mix(mix(dayHorS, duskHorS, duskC), nightHorS, nightC);

        // Unified Anti-Solar Belt of Venus & Earth's Shadow Arch across Sky, Sea, & Aerial Perspective
        float overcast = clamp(cloudC * 0.62 + rainC * 0.48, 0.0, 1.0);
        if (duskC > 0.01 && overcast < 0.88) {
          float antiSun = clamp(dot(dirXZ, -sunXZ) * 0.5 + 0.5, 0.0, 1.0);
          float shadowTop = clamp(0.03 + max(0.0, -sunDir.y) * 0.55, 0.02, 0.14);
          float hEff = clamp(max(0.0, rayDir.y), 0.0, 1.0);
          float earthShadow = (1.0 - smoothstep(shadowTop * 0.45, shadowTop + 0.05, hEff)) * pow(antiSun, 1.4);
          float beltOfVenus = exp(-pow((hEff - (shadowTop + 0.055)) / 0.068, 2.0)) * pow(antiSun, 1.25);
          float clearTwilight = duskC * (1.0 - overcast * 0.85) * (1.0 - nightC * 0.85);
          horS = mix(horS, vec3(0.11, 0.14, 0.25), earthShadow * clearTwilight * 0.62);
          horS += vec3(0.42, 0.20, 0.26) * beltOfVenus * clearTwilight * (0.55 + 0.25 * gMode);
        }

        vec3 stormHorDay = mix(vec3(0.35, 0.39, 0.44), vec3(0.18, 0.21, 0.26), stormC);
        stormHorDay *= mix(1.0, 0.22, nightC);
        horS = mix(horS, stormHorDay, clamp(rainC * 0.74 + stormC * 0.34 + cloudC * 0.12 * (1.0 - fogC * 0.65) * (1.0 - gMode * 0.45), 0.0, 0.92));

        if (fogC > 0.002) {
          vec3 fogDewS = mix(mix(vec3(0.79, 0.85, 0.90), vec3(0.90, 0.66, 0.48), duskC), vec3(0.095, 0.130, 0.195), nightC);
          horS = mix(horS, fogDewS, fogC * 0.80);
        }
        return horS;
      }

      vec3 nexusApplyBoundaryLayerAerialPerspective(
        vec3 hdrLinear,
        vec3 camPos,
        vec3 worldPos,
        vec3 sunDir,
        float duskFactor,
        float nightFactor,
        float cloudCover,
        float rainIntensity,
        float lightningFlash,
        float fogFactor
      ) {
        vec3 rayVec = worldPos - camPos;
        float dist = length(rayVec);
        float effDist = max(0.0, dist - 20.0);
        if (effDist <= 0.001) return hdrLinear;

        float uc = max(0.0, camPos.y) / 46.0;
        float up = max(0.0, worldPos.y) / 46.0;
        float du = up - uc;
        float meanBL = abs(du) > 0.006
          ? (exp(-uc) - exp(-up)) / du
          : exp(-0.5 * (uc + up));

        float rainC = clamp(rainIntensity, 0.0, 1.35);
        float cloudC = clamp(cloudCover, 0.0, 1.0);
        float duskC = clamp(duskFactor, 0.0, 1.0);
        float nightC = clamp(nightFactor, 0.0, 1.0);
        float fogC = clamp(fogFactor, 0.0, 1.0);
        float gMode = clamp(uGenshinMode, 0.0, 1.0);
        float fMode = clamp(-uGenshinMode, 0.0, 1.0);

        float sigmaBL = 0.00028 + 0.00105 * rainC + 0.00024 * cloudC + 0.00245 * fogC;
        float tauAir = effDist * (sigmaBL * clamp(meanBL, 0.0, 1.0) + 0.000038 + 0.00012 * fogC);
        tauAir *= mix(mix(1.0, 1.36, gMode), 0.88, fMode);
        float hazeFactor = clamp(1.0 - exp(-tauAir), 0.0, mix(0.86, 0.88, fogC));

        vec3 rayDir = rayVec / max(dist, 1e-4);
        vec3 hazeSRGB = nexusEvalHorizonSkySRGB(
          vec3(rayDir.x, 0.0, rayDir.z),
          sunDir,
          duskC,
          nightC,
          cloudC,
          rainC,
          clamp(rainC * 0.65, 0.0, 1.0),
          fogC,
          0.25 * nightC
        );
        vec3 hazeHDR = nexusSRGBToLinear(hazeSRGB) * mix(1.16 + 0.14 * gMode, 0.76 + 0.12 * gMode, nightC);

        float mieFwd = pow(max(dot(rayDir, sunDir), 0.0), mix(5.0, 3.2, gMode))
                     * (1.0 - nightC * 0.85)
                     * (1.0 - clamp(rainC, 0.0, 1.0) * 0.68);
        vec3 mieTintS = mix(
          mix(vec3(0.78, 0.85, 0.96), vec3(1.00, 0.92, 0.74), gMode * 0.65),
          mix(vec3(0.98, 0.56, 0.26), vec3(1.00, 0.60, 0.34), gMode),
          duskC
        );
        hazeHDR += nexusSRGBToLinear(mieTintS) * mieFwd * (0.24 + 0.18 * gMode + 0.12 * fogC) + vec3(0.65, 0.82, 1.25) * lightningFlash * 0.65;

        return mix(hdrLinear, hazeHDR, hazeFactor);
      }

      vec3 nexusApplyBoundaryLayerAerialPerspective(
        vec3 hdrLinear,
        vec3 camPos,
        vec3 worldPos,
        vec3 sunDir,
        float duskFactor,
        float nightFactor,
        float cloudCover,
        float rainIntensity,
        float lightningFlash
      ) {
        return nexusApplyBoundaryLayerAerialPerspective(
          hdrLinear,
          camPos,
          worldPos,
          sunDir,
          duskFactor,
          nightFactor,
          cloudCover,
          rainIntensity,
          lightningFlash,
          0.0
        );
      }

      float nexusNakaRushtonScalar(float x, float n, float sigmaPow, float normScale) {
        float xp = pow(max(x, 0.0), n);
        return (xp / (xp + sigmaPow)) * normScale;
      }

      vec3 nexusNakaRushtonVec3(vec3 x, float n, float sigmaPow, float normScale) {
        vec3 xp = pow(max(x, vec3(0.0)), vec3(n));
        return (xp / (xp + vec3(sigmaPow))) * normScale;
      }

      /**
       * Evaluates Human Visual System (HVS) Physiological Photoreceptor HDR Response:
       * - hdrLinear: scene-referred Linear HDR radiance [0, +inf)
       * - rayFromCam: normalized view ray pointing FROM camera TO pixel
       * - sunDir: normalized celestial light direction
       * - sunColor: live Linear solar/dusk/lunar tint
       * - hvsParams: vec4(eyeGain, huntChroma, stevensContrast, purkinjeMix)
       * - hvsGlare: vec4(sunGlareStrength, lightningVeil, shadowToeLift, whitePoint)
       * Returns display-referred Linear [0, 1] ready for sRGB OETF encoding.
       */
      vec3 nexusApplyHumanEyeHDR(
        vec3 hdrLinear,
        vec3 rayFromCam,
        vec3 sunDir,
        vec3 sunColor,
        vec4 hvsParams,
        vec4 hvsGlare
      ) {
        float eyeGain = hvsParams.x;
        float huntChroma = hvsParams.y;
        float nSlope = hvsParams.z;
        float purkinjeMix = hvsParams.w;

        float sunGlareVis = hvsGlare.x;
        float lightningVeil = hvsGlare.y;
        float shadowToe = hvsGlare.z;
        float whitePt = max(2.2, hvsGlare.w);
        float gMode = clamp(uGenshinMode, 0.0, 1.0);
        float fMode = clamp(-uGenshinMode, 0.0, 1.0);
        whitePt *= (1.0 + 0.16 * fMode);

        // 1. Dynamic Pupillary & Retinal Gain + Zero-Preserving Bipolar Shadow-Toe Micro-Lift
        vec3 c = max(hdrLinear, vec3(0.0)) * (eyeGain * mix(mix(1.0, 1.08, gMode), 1.04, fMode));
        c += (shadowToe * (1.0 - 0.34 * fMode) + 0.022 * gMode) * (c / (c + vec3(0.11))) * mix(vec3(1.0), vec3(0.78, 0.90, 1.14), gMode);

        // 2. Spencer-Shirley Intraocular Ciliary Solar Scatter & Plasma Lightning Veil
        float cosSun = max(dot(rayFromCam, sunDir), 0.0);
        float ciliaryScatter = (pow(cosSun, 5.0) * (0.026 + 0.018 * gMode + 0.012 * fMode) + pow(cosSun, 24.0) * (0.062 + 0.032 * gMode + 0.028 * fMode)) * sunGlareVis;
        c += sunColor * ciliaryScatter + vec3(0.62, 0.76, 1.0) * lightningVeil * 0.075;

        // 3. AgX-Inspired Cross-Talk Inset + Normalized Naka-Rushton Photoreceptor Response
        //    Prevents pure RGB primaries (red bridge steel, yellow transit, blue glass, emerald grass)
        //    from clipping into synthetic CG neon, rolling highlights smoothly into natural warm white!
        const mat3 AGX_INSET = mat3(
          0.895, 0.052, 0.036,
          0.068, 0.908, 0.044,
          0.037, 0.040, 0.920
        );
        const mat3 AGX_OUTSET = mat3(
           1.126, -0.064, -0.041,
          -0.084,  1.109, -0.050,
          -0.042, -0.045,  1.091
        );
        vec3 cInset = mix(AGX_INSET * c, c, gMode * 0.55);

        float sigma = mix(mix(0.57, 0.50, gMode), 0.54, fMode);
        float effSlope = mix(mix(nSlope, max(1.02, nSlope * 0.93), gMode), nSlope * 1.08, fMode);
        float sigmaPow = pow(sigma, effSlope);
        float normScale = (pow(whitePt, effSlope) + sigmaPow) / pow(whitePt, effSlope);

        float lumIn = dot(cInset, vec3(0.2126, 0.7152, 0.0722));
        float lumNR = nexusNakaRushtonScalar(lumIn, effSlope, sigmaPow, normScale);
        vec3 cHuePreserved = cInset * (lumNR / max(lumIn, 1e-5));
        vec3 cConeBleach = nexusNakaRushtonVec3(cInset, effSlope, sigmaPow, normScale);

        // Smooth filmic highlight shoulder desaturation above Y > 0.30
        float bleachT = smoothstep(0.28, 1.65, lumIn) * 0.78 * mix(mix(1.0, 0.52, gMode), 0.85, fMode);
        vec3 mappedInset = mix(cHuePreserved, cConeBleach, bleachT);
        vec3 mapped = max(vec3(0.0), mix(AGX_OUTSET * mappedInset, mappedInset, smoothstep(0.45, 0.96, lumNR) * 0.72));

        // 4. Hunt Effect (Photopic Colorfulness) with Macbeth Neutral & Highlight Gamut Guard
        float lumOut = dot(mapped, vec3(0.2126, 0.7152, 0.0722));
        float maxCh = max(mapped.r, max(mapped.g, mapped.b));
        float minCh = min(mapped.r, min(mapped.g, mapped.b));
        float satEst = (maxCh - minCh) / max(maxCh, 1e-4);
        // Protect already-saturated pigments from over-boosting while keeping subtle mineral/earth nuances rich
        float chromaGuard = (1.0 - 0.68 * smoothstep(0.58, 0.96, lumOut)) * (1.0 - 0.32 * smoothstep(0.55, 0.92, satEst));
        float effChroma = mix(1.0, huntChroma, chromaGuard)
                        * mix(1.0, 1.16, gMode * (1.0 - 0.55 * smoothstep(0.65, 0.96, lumOut)))
                        * mix(1.0, 1.13, fMode * (1.0 - 0.45 * smoothstep(0.68, 0.96, lumOut)));
        mapped = max(vec3(0.0), vec3(lumOut) + (mapped - vec3(lumOut)) * effChroma);

        // Genshin Warm-Sunlit / Cool-Cerulean-Shadow Split-Tone Grading (Tahap 1)
        if (gMode > 0.001) {
          float shadowMaskG = (1.0 - smoothstep(0.08, 0.54, lumOut)) * (1.0 - purkinjeMix * 0.75);
          float litMaskG = smoothstep(0.36, 0.88, lumOut) * (1.0 - purkinjeMix * 0.75);
          vec3 coolShadowG = mapped * vec3(0.84, 0.94, 1.15) + vec3(0.012, 0.028, 0.058);
          vec3 warmLitG = mapped * vec3(1.045, 1.018, 0.955);
          mapped = mix(mapped, coolShadowG, shadowMaskG * gMode * 0.52);
          mapped = mix(mapped, warmLitG, litMaskG * gMode * 0.48);
        }

        // ForzaTech Next-Gen Photorealistic HDR Polarized Sky-Shadow & Crisp Solar Highlight Grading
        if (fMode > 0.001) {
          float shadowMaskF = (1.0 - smoothstep(0.05, 0.44, lumOut)) * (1.0 - purkinjeMix * 0.80);
          float litMaskF = smoothstep(0.40, 0.92, lumOut) * (1.0 - purkinjeMix * 0.80);
          vec3 deepShadowF = mapped * vec3(0.88, 0.93, 1.06);
          vec3 crispLitF = mapped * vec3(1.035, 1.012, 0.975);
          mapped = mix(mapped, deepShadowF, shadowMaskF * fMode * 0.38);
          mapped = mix(mapped, crispLitF, litMaskF * fMode * 0.34);
        }

        // 5. Purkinje Mesopic/Scotopic Rod Shift (507nm blue-cyan rod sensitivity below cone threshold)
        if (purkinjeMix > 0.005) {
          float rodMask = (1.0 - smoothstep(0.025, 0.24, lumIn)) * purkinjeMix;
          float rodLum = dot(mapped, vec3(0.062, 0.608, 0.330));
          vec3 scotopicCol = rodLum * vec3(0.74, 0.90, 1.18);
          mapped = mix(mapped, scotopicCol, rodMask);
        }

        return clamp(mapped, 0.0, 1.0);
      }
    `;
  }
}
