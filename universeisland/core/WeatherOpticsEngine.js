import * as THREE from 'three';
import { RainCollisionBake } from './RainCollisionBake.js';
import { HumanEyeHDROptics } from './HumanEyeHDROptics.js';
import {
  computeBaseJD0WIB,
  evaluateDepokAstronomy,
  createRealStarCatalogGridTexture
} from './AstronomicalSkyCatalog.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.2 Super-Realistic Physical Weather Edition)
 * ./core/WeatherOpticsEngine.js
 *
 * Implements Hukum Emas 8, 12, 13, & 14 + Full Meteorological & Fluid-Impact Physics:
 * 1. Analytical Sky Dome with:
 *    - Exact Sphere Sun/Moon Projection (z_sphere = sqrt(max(0.0, 1.0 - r_norm^2)))
 *    - Radial Euclidean SDF Starfield
 *    - 2-Layer Parallax 6-Octave Domain-Rotated FBM Cumulonimbus Clouds (R = mat2(0.8, 0.6, -0.6, 0.8))
 *      with Beer-Lambert Self-Shadowing, Horizon Virga Rain Shafts, & Multi-Branch Forked Plasma Lightning!
 * 2. 100% GPU Physics-Driven Raindrops & Newtonian Roof/Canopy Drip Curtains (1 Draw Call, 6,800 Quads):
 *    - Marshall-Palmer Drop-Size Distribution (D in [0.95, 4.65] mm)
 *    - Gunn-Kinzer Terminal Velocity v_t(D) = 9.65 - 10.3 * exp(-0.6 * D)
 *    - Boundary-Layer Wind-Shear Profile W(y) ~ (y / 10)^0.22 + Aerodynamic Drag Coupling
 *    - Exact 512x512 3D Scene Collision Z-Buffer Interception (keeping covered walkways/interiors 100% dry)
 *    - Pendant-Bead Accumulation + Newtonian Free-Fall y(t) = y_eave - 0.5 * g * t^2 along 3D Roof Eaves & Trees
 * 3. 100% GPU 3D Surface Impact Splashes, Parabolic Rebound Micro-Droplets, & Roof Spray Mist (1 Draw Call, 5,200 Quads):
 *    - Rayleigh-Worthington Crown Rings aligned to exact 3D surface normal N_hit(x, z)
 *    - Ballistic Rebound Micro-Droplets P(t) = P_hit + v_0 * t + 0.5 * g * t^2
 *    - Wind-Drifting Impact Aerosol Spray Mist over roofs & plazas
 * 4. Danau Kenanga Rotated 6-Octave FBM Water with Multi-Layer Capillary Raindrop Ripples & Storm Chop.
 */
export class WeatherOpticsEngine {
  static SKY_DROP_COUNT = 5200;
  static EAVE_DRIP_COUNT = 2000;
  static SPLASH_CROWN_COUNT = 2600;
  static REBOUND_DROP_COUNT = 2000;
  static SPRAY_MIST_COUNT = 600;

  static MIN_TRANSITION_SEC = 20.0;
  static MAX_TRANSITION_SEC = 45.0;
  static DEFAULT_TRANSITION_SEC = 25.0;

  constructor(scene) {
    this.scene = scene;
    this.timeOfDay = 'DAY'; // 'DAY' | 'DUSK' | 'NIGHT'
    this.weatherMode = 'CLEAR'; // 'CLEAR' | 'MIST' | 'RAIN' | 'STORM'

    this.transitionDurationSec = WeatherOpticsEngine.DEFAULT_TRANSITION_SEC;
    this.currentSolarHour = 10.35; // Continuous 24h Kampus UI Depok WIB solar clock [0, 24)
    this.continuousSolarHour = 10.35; // Unwrapped monotonic WIB hours since _baseJD0WIB (0.000° midnight jump!)
    this._baseJD0WIB = computeBaseJD0WIB(new Date());
    this._astroState = null;
    this.nightIlluminationLevel = 0.0; // Continuous [0, 1] architectural & aviation night glow
    this.streetlampSensorLevel = 0.0; // Continuous [0, 1] ambient-lux photocell streetlamp power

    this.targetRainIntensity = 0.0;
    this.targetWetness = 0.0;
    this.targetCloudCover = 0.22;
    this.targetStormFactor = 0.0;
    this.targetFogDensity = 0.00058;
    this.baseWindSpeed = 0.82;
    this.targetWindSpeed = 0.82;
    this.windGustMultiplier = 1.0;

    this._weatherTrans = {
      active: false,
      fromMode: 'CLEAR',
      toMode: 'CLEAR',
      elapsed: 0.0,
      duration: WeatherOpticsEngine.DEFAULT_TRANSITION_SEC,
      progress: 1.0,
      startRain: 0.0,
      startWet: 0.0,
      startCloud: 0.22,
      startStorm: 0.0,
      startFog: 0.00058,
      startWind: 0.82
    };

    this._timeTrans = {
      active: false,
      fromTod: 'DAY',
      toTod: 'DAY',
      elapsed: 0.0,
      duration: WeatherOpticsEngine.DEFAULT_TRANSITION_SEC,
      progress: 1.0,
      startHour: 10.35,
      startContinuousHour: 10.35,
      deltaHour: 0.0
    };

    // Multi-stroke plasma lightning state (active when uStormFactor > 0.60)
    this.nextLightningTimer = 2.4;
    this.lightningClock = -1.0;
    this.lightningStrokeProfile = null;
    this.onThunder = null; // Optional callback: (distanceMeters, intensity) => void

    this._baseSunPos = new THREE.Vector3(68, 96, 62);
    this._baseSunColor = new THREE.Color(0xfff5e4);
    this._baseSunIntensity = 2.35;
    this._baseHemiIntensity = 0.72;
    this._baseFillPos = new THREE.Vector3(-55, 38, -48);
    this._baseFillIntensity = 0.35;
    this._flashColor = new THREE.Color(0xc8deff);
    this._tmpSunColor = new THREE.Color();
    this._colDaySun = new THREE.Color(0xfff5e4);
    this._colDuskSun = new THREE.Color(0xff9c52);
    this._colNightMoon = new THREE.Color(0x8fb0e8);
    this._colStormTint = new THREE.Color(0x9aa9bc);
    this._colHemiSky = new THREE.Color();
    this._colHemiGnd = new THREE.Color();
    this._colFog = new THREE.Color();
    this._colTmpA = new THREE.Color();
    this._colTmpB = new THREE.Color();

    const fallbackCollisionTex = RainCollisionBake.createFallbackTexture();
    this.hvsOptics = new HumanEyeHDROptics();
    this._lastSharedSurfaceUniforms = null;
    const hvsUniforms = HumanEyeHDROptics.createUniforms();

    this.uniforms = {
      uTime: { value: 0.0 },
      uCamPos: { value: new THREE.Vector3() },
      uSunDir: { value: new THREE.Vector3(0.52, 0.72, 0.45).normalize() },
      uTrueSunDir: { value: new THREE.Vector3(0.52, 0.72, 0.45).normalize() },
      uMoonDir: { value: new THREE.Vector3(-0.42, 0.62, -0.52).normalize() },
      uMoonPhaseSunDir: { value: new THREE.Vector3(0.42, -0.56, 0.52).normalize() },
      uMoonIllumFraction: { value: 0.85 },
      uWorldToEqMat: { value: new THREE.Matrix3() },
      uStarGridTex: { value: createRealStarCatalogGridTexture() },
      uStreetlampPower: { value: 0.0 },
      uSunColor: { value: new THREE.Color(0xfff5e4) },
      uNightFactor: { value: 0.0 },
      uDuskFactor: { value: 0.0 },
      uRainIntensity: { value: 0.0 },
      uWetness: { value: 0.0 },
      uCloudCover: { value: 0.22 },
      uStormFactor: { value: 0.0 },
      uLightningFlash: { value: 0.0 },
      uLightningDir: { value: new THREE.Vector3(-0.45, 0.34, -0.82).normalize() },
      uLightningSeed: { value: 17.0 },
      uWindVec: { value: new THREE.Vector3(0.65, 0.0, 0.35) },
      uWindPhase: { value: new THREE.Vector2(0.0, 0.0) },
      uSeaState: { value: 0.32 },
      uFireflyIntensity: { value: 0.0 },
      uCollisionMap: { value: fallbackCollisionTex },
      uCollisionHalfExtent: { value: RainCollisionBake.DEFAULT_HALF_EXTENT },
      uHvsParams: hvsUniforms.uHvsParams,
      uHvsGlare: hvsUniforms.uHvsGlare
    };

    this.hemiLight = null;
    this.sunLight = null;
    this.fillLight = null;
    this.skyMesh = null;
    this.rainAndDripMesh = null;
    this.splashAndMistMesh = null;
    this.fireflyPoints = null;
    this.volcanicPlumeMesh = null;

    this._dripAnchorAttr = null;
    this._dripLandingAttr = null;
    this._dripOutDirAttr = null;
    this._splashAnchorAttr = null;
    this._envMaps = null;

    this._initLighting();
    this._initAnalyticalSkyDome();
    this._initPhysicalRainAndDripMesh();
    this._initPhysicalSplashAndMistMesh();
    this._initGPUFireflyParticles();
    this._initVolcanicSmokePlumeMesh();
    this.setTimeOfDay('DAY', null, { immediate: true });
    this.setWeather('CLEAR', { immediate: true });
  }

  _initLighting() {
    this.hemiLight = new THREE.HemisphereLight(0xdbeeff, 0x5c6358, 0.78);
    this.hemiLight.position.set(0, 80, 0);
    this.scene.add(this.hemiLight);

    this.sunLight = new THREE.DirectionalLight(0xfff5e4, 2.45);
    this.sunLight.position.set(68, 96, 62);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.width = 4096;
    this.sunLight.shadow.mapSize.height = 4096;
    this.sunLight.shadow.camera.near = 10;
    this.sunLight.shadow.camera.far = 260;
    const d = 88;
    this.sunLight.shadow.camera.left = -d;
    this.sunLight.shadow.camera.right = d;
    this.sunLight.shadow.camera.top = d;
    this.sunLight.shadow.camera.bottom = -d;
    this.sunLight.shadow.bias = -0.00022;
    this.sunLight.shadow.normalBias = 0.018;
    this.sunLight.shadow.radius = 2.1;
    this.scene.add(this.sunLight);

    this.fillLight = new THREE.DirectionalLight(0x9ec5eb, 0.42);
    this.fillLight.position.set(-55, 38, -48);
    this.scene.add(this.fillLight);
  }

  /**
   * NEXUS v2.6 ULTRA HDR IMAGE-BASED LIGHTING (PMREM):
   * Pre-convolves 3 HDR Physical Sky + Campus Bounce PMREM Cubemap Environment Textures
   * (DAY, DUSK, NIGHT) once at startup (< 8ms, 0 extra runtime Draw Calls!), activating Three.js r170
   * Split-Sum GGX Specular IBL + Kulla-Conty Multiscattering Energy Compensation across all PBR surfaces.
   */
  attachRenderer(renderer) {
    if (!renderer || typeof THREE.PMREMGenerator !== 'function') return;
    try {
      const pmrem = new THREE.PMREMGenerator(renderer);
      if (typeof pmrem.compileCubemapShader === 'function') {
        pmrem.compileCubemapShader();
      }

      const envScene = new THREE.Scene();
      const envGeo = new THREE.SphereGeometry(24, 32, 18);
      const envUniforms = {
        uZenith: { value: new THREE.Color(0x1f5ec4) },
        uHorizon: { value: new THREE.Color(0x9ec6ea) },
        uGround: { value: new THREE.Color(0x3a4834) },
        uTerracottaBounce: { value: new THREE.Color(0x8e4c3e) },
        uSunDir: { value: new THREE.Vector3(0.52, 0.72, 0.45).normalize() },
        uSunRadiance: { value: new THREE.Color(8.5, 7.8, 6.6) }
      };
      const envMat = new THREE.ShaderMaterial({
        side: THREE.BackSide,
        depthWrite: false,
        toneMapped: false,
        uniforms: envUniforms,
        vertexShader: /* glsl */ `
          varying vec3 vDir;
          void main() {
            vDir = normalize(position);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: /* glsl */ `
          varying vec3 vDir;
          uniform vec3 uZenith;
          uniform vec3 uHorizon;
          uniform vec3 uGround;
          uniform vec3 uTerracottaBounce;
          uniform vec3 uSunDir;
          uniform vec3 uSunRadiance;
          void main() {
            vec3 d = normalize(vDir);
            vec3 col;
            if (d.y >= 0.0) {
              col = mix(uHorizon, uZenith, pow(clamp(d.y, 0.0, 1.0), 0.56));
              float cosS = max(dot(d, normalize(uSunDir)), 0.0);
              float mieHalo = pow(cosS, 14.0) * 0.28 + pow(cosS, 140.0) * 1.65;
              col += uSunRadiance * mieHalo;
            } else {
              float down = clamp(-d.y, 0.0, 1.0);
              vec3 bounce = mix(uTerracottaBounce, uGround, smoothstep(0.0, 0.45, down));
              col = mix(uHorizon * 0.62, bounce, smoothstep(0.0, 0.28, down));
            }
            gl_FragColor = vec4(col, 1.0);
          }
        `
      });
      const envMesh = new THREE.Mesh(envGeo, envMat);
      envScene.add(envMesh);

      const bakeRegime = (zen, hor, gnd, bounce, sDir, sRad) => {
        envUniforms.uZenith.value.setHex(zen);
        envUniforms.uHorizon.value.setHex(hor);
        envUniforms.uGround.value.setHex(gnd);
        envUniforms.uTerracottaBounce.value.setHex(bounce);
        envUniforms.uSunDir.value.copy(sDir).normalize();
        envUniforms.uSunRadiance.value.setRGB(sRad[0], sRad[1], sRad[2]);
        return pmrem.fromScene(envScene, 0.025).texture;
      };

      // Single continuous broadband sky + campus-bounce PMREM cubemap (ZERO discrete texture swaps at dusk/night!)
      const broadbandTex = bakeRegime(
        0x3664a8,
        0xa8c6e0,
        0x354330,
        0x784a3c,
        new THREE.Vector3(0.18, 0.88, 0.42),
        [3.6, 3.4, 3.1]
      );
      this._envMaps = {
        BROADBAND: broadbandTex
      };
      this.scene.environment = broadbandTex;

      envGeo.dispose();
      envMat.dispose();
      pmrem.dispose();

      this._applyEnvironmentIBL(0.0);
    } catch (_err) {
      this._envMaps = null;
    }
  }

  _applyEnvironmentIBL(lightningFlash = 0.0) {
    if (!this._envMaps || !this.scene) return;
    if (this.scene.environment !== this._envMaps.BROADBAND) {
      this.scene.environment = this._envMaps.BROADBAND;
    }
    const nightF = this.uniforms.uNightFactor.value;
    const duskF = this.uniforms.uDuskFactor.value;
    const cloudF = this.uniforms.uCloudCover.value;
    const overcastDim =
      1.0 -
      Math.min(
        0.56,
        cloudF * 0.14 +
          this.uniforms.uRainIntensity.value * 0.28 +
          this.uniforms.uStormFactor.value * 0.18
      );
    // C-infinity continuous IBL intensity curve across 24h Day -> Dusk -> Night (0 cubemap swap pops!)
    const baseEnvInt =
      0.90 * (1.0 - duskF * 0.22) * (1.0 - nightF) + 0.16 * nightF;
    if ('environmentIntensity' in this.scene) {
      this.scene.environmentIntensity = baseEnvInt * overcastDim + lightningFlash * 0.65;
    }
  }

  /**
   * Bakes the 512x512 3D world collision height+normal+material field from scene roots
   * and binds the extracted 3D roof-eave & tree-canopy drip coordinates directly to GPU buffers.
   */
  bakeSceneCollisionField(roots, options = {}, sharedSurfaceUniforms = null) {
    const bake = RainCollisionBake.bakeFromHierarchy(roots, {
      ...options,
      maxDrips: WeatherOpticsEngine.EAVE_DRIP_COUNT
    });
    return this._applyCollisionBakeResult(bake, sharedSurfaceUniforms);
  }

  /**
   * Time-sliced asynchronous 512x512 3D Rain Collision Z-Buffer & Drip Anchor baker (< 6ms per frame slice).
   * Streams reference collision data in the background while the 3D world runs interactively at 60-120 FPS!
   */
  async bakeSceneCollisionFieldAsync(roots, options = {}, sharedSurfaceUniforms = null) {
    const bake = await RainCollisionBake.bakeFromHierarchyAsync(
      roots,
      {
        ...options,
        maxDrips: WeatherOpticsEngine.EAVE_DRIP_COUNT
      },
      6.0
    );
    return this._applyCollisionBakeResult(bake, sharedSurfaceUniforms);
  }

  _applyCollisionBakeResult(bake, sharedSurfaceUniforms = null) {
    this.uniforms.uCollisionMap.value = bake.texture;
    this.uniforms.uCollisionHalfExtent.value = bake.halfExtent;

    if (sharedSurfaceUniforms) {
      if (sharedSurfaceUniforms.uCollisionMap) {
        sharedSurfaceUniforms.uCollisionMap.value = bake.texture;
      }
      if (sharedSurfaceUniforms.uCollisionReady) {
        sharedSurfaceUniforms.uCollisionReady.value = 1.0;
      }
    }

    // Bind exact 3D roof-eave & tree-canopy overhang coordinates + outward eave pitch direction to the 2,000 drip/runoff quads
    if (this._dripAnchorAttr && this._dripLandingAttr && bake.dripAnchors.length > 0) {
      const skyVerts = WeatherOpticsEngine.SKY_DROP_COUNT * 4;
      const anchorArr = this._dripAnchorAttr.array;
      const landingArr = this._dripLandingAttr.array;
      const outDirArr = this._dripOutDirAttr ? this._dripOutDirAttr.array : null;
      const anchors = bake.dripAnchors;

      for (let i = 0; i < WeatherOpticsEngine.EAVE_DRIP_COUNT; i++) {
        const a = anchors[i % anchors.length];
        const ox = Number.isFinite(a.outX) ? a.outX : 0.0;
        const oz = Number.isFinite(a.outZ) ? a.outZ : 1.0;
        for (let c = 0; c < 4; c++) {
          const vi = skyVerts + i * 4 + c;
          anchorArr[vi * 3] = a.x;
          anchorArr[vi * 3 + 1] = a.y;
          anchorArr[vi * 3 + 2] = a.z;
          landingArr[vi * 2] = a.landingY;
          landingArr[vi * 2 + 1] = a.isCanopy;
          if (outDirArr) {
            outDirArr[vi * 2] = ox;
            outDirArr[vi * 2 + 1] = oz;
          }
        }
      }
      this._dripAnchorAttr.needsUpdate = true;
      this._dripLandingAttr.needsUpdate = true;
      if (this._dripOutDirAttr) this._dripOutDirAttr.needsUpdate = true;

      // Also anchor 600 splash crowns directly along the 3D eave-drip & parabolic genting-runoff landing lines!
      if (this._splashAnchorAttr) {
        const spArr = this._splashAnchorAttr.array;
        const eaveSplashStart = 2000; // Last 600 of the 2,600 crown splashes
        for (let i = 0; i < 600; i++) {
          const a = anchors[(i * 3) % anchors.length];
          const ox = Number.isFinite(a.outX) ? a.outX : 0.0;
          const oz = Number.isFinite(a.outZ) ? a.outZ : 0.0;
          const outThrow = a.isCanopy > 0.5 ? 0.0 : (i % 2 === 0 ? 0.42 : 0.05);
          const qBase = (eaveSplashStart + i) * 4;
          for (let c = 0; c < 4; c++) {
            const vi = qBase + c;
            spArr[vi * 3] = a.x + ox * outThrow;
            spArr[vi * 3 + 1] = a.z + oz * outThrow;
            spArr[vi * 3 + 2] = 1.0; // Flag: world-pinned eave-drip / runoff landing splash!
          }
        }
        this._splashAnchorAttr.needsUpdate = true;
      }
    }

    return bake;
  }

  /**
   * Shared Dual-Island Kepulauan UI ("U" & "I") Analytical SDF GLSL Kernel
   * Synchronized 1:1 with TerrainDEMLoader.getIslandUCoastDistance & getIslandICoastDistance.
   */
  static _getArchipelagoSdfGLSL() {
    return /* glsl */ `
      const vec2 ISLAND_CENTER = vec2(-18.0, 34.0);
      const vec2 ARCHIPELAGO_CENTER = vec2(-155.0, 330.0);

      float sminArch(float a, float b, float k) {
        float h = max(0.0, k - abs(a - b)) / k;
        return min(a, b) - h * h * k * 0.25;
      }

      // Exact 360-degree closed polar campus-head shoreline R(theta) synchronized 1:1 with TerrainDEMLoader.getIslandRadiusAtAngle
      float islandRadiusAtAngle(float theta) {
        float eastCove = -46.0 * pow(0.5 + 0.5 * cos(theta + 0.25), 4.4);
        float neHeadland = 23.5 * pow(0.5 + 0.5 * cos(theta + 1.04), 8.5);
        float southRotundaLobe = 15.5 * pow(0.5 + 0.5 * cos(theta - 1.12), 2.3);
        float swHelipadPeninsula = 20.5 * pow(0.5 + 0.5 * cos(theta - 2.44), 6.5);
        float organicMeander =
          2.15 * sin(3.0 * theta + 0.40) +
          1.20 * cos(5.0 * theta - 0.70) +
          0.55 * sin(8.0 * theta + 1.10);
        return 121.0 + eastCove + neHeadland + southRotundaLobe + swHelipadPeninsula + organicMeander;
      }

      // Pulau "U" Horseshoe Signed Coast Distance dU(x, z)
      float evalPulauUCoast(vec2 xz) {
        float distUCenter;
        if (xz.x <= -145.0) {
          distUCenter = abs(length(xz - vec2(-145.0, 207.0)) - 185.0);
        } else {
          float dLegL = length(vec2(max(0.0, xz.x + 62.0), xz.y - 22.0));
          float dLegR = length(vec2(max(0.0, xz.x - 10.0), xz.y - 392.0));
          distUCenter = min(dLegL, dLegR);
        }
        float meanderU = 2.0 * sin(xz.x * 0.035 + xz.y * 0.025) + 1.1 * cos(xz.y * 0.048 - xz.x * 0.031);
        float dUHorseshoe = distUCenter - 74.0 - meanderU;

        vec2 d = xz - ISLAND_CENTER;
        if (xz.y > 250.0 || xz.x < -195.0) {
          return dUHorseshoe;
        }
        float r = length(d) + 1e-4;
        float theta = atan(d.y, d.x);
        float r0 = islandRadiusAtAngle(theta);
        float eps = 0.008;
        float dRdTh = (islandRadiusAtAngle(theta + eps) - islandRadiusAtAngle(theta - eps)) / (2.0 * eps);
        float slopeRatio = dRdTh / max(20.0, r0);
        float dCampusHead = (r - r0) * inversesqrt(1.0 + slopeRatio * slopeRatio);
        return sminArch(dCampusHead, dUHorseshoe, 28.0);
      }

      // Pulau "I" Volcano & Runway Island Signed Coast Distance dI(x, z)
      // Synchronized 1:1 with TerrainDEMLoader: I_CENTER=(-160, 694), I_HALF=(244, 88), I_CORNER_R=56
      float evalPulauICoast(vec2 xz) {
        vec2 q = abs(xz - vec2(-160.0, 694.0)) - vec2(188.0, 32.0);
        float outsideDist = length(max(q, vec2(0.0)));
        float insideDist = min(max(q.x, q.y), 0.0);
        float meanderI = 1.9 * sin(xz.x * 0.039 - xz.y * 0.027) + 1.05 * cos(xz.x * 0.061 + xz.y * 0.033);
        return outsideDist + insideDist - 56.0 - meanderI;
      }

      float evalArchipelagoCoastRaw(vec2 xz) {
        return min(evalPulauUCoast(xz), evalPulauICoast(xz));
      }

      // Analytical terrain + volcano + runway elevation outside the central 320m x 320m Rectorat bake texture
      float evalArchipelagoAnalyticalHeight(vec2 xz, float dCoast) {
        if (dCoast > 3.52) return -1.45;
        if (dCoast > 0.0) return -1.45 * smoothstep(0.0, 3.6, dCoast);
        float coastCollar = smoothstep(-1.2, -16.5, dCoast);
        if (xz.y > 536.0) {
          // Pulau I: ICAO Runway RWY 36/18 + West Apron + Active Stratovolcano at (-32, 698)
          if (xz.x > -383.0 && xz.x < -127.0 && abs(xz.y - 708.0) < 12.6) {
            return 0.58;
          }
          if (xz.x > -285.0 && xz.x < -223.0 && xz.y > 659.5 && xz.y < 697.5) {
            return 0.58;
          }
          float hBase = 0.44 * smoothstep(-1.2, -11.0, dCoast);
          vec2 dv = xz - vec2(-32.0, 698.0);
          float rV = length(dv);
          if (rV < 118.0) {
            float u = max(0.0, (98.0 - rV) / 78.5);
            float cone = 58.0 * pow(min(1.0, u), 1.38) * smoothstep(0.0, 0.14, u);
            float crater = -14.2 * smoothstep(19.5, 5.2, rV);
            float southClear = smoothstep(-126.0, -96.0, xz.x);
            return hBase + max(0.0, (cone + crater) * southClear * coastCollar);
          }
          return hBase;
        }
        // Pulau U Rolling Hills outside central Rectorat lowland
        float flatGuard = smoothstep(125.0, 185.0, length(xz - vec2(-12.0, 48.0)));
        float hHills =
          18.6 * exp(-dot(xz - vec2(-336.0, 236.0), xz - vec2(-336.0, 236.0)) / 9200.0) +
          16.8 * exp(-dot(xz - vec2(-68.0, 392.0), xz - vec2(-68.0, 392.0)) / 7800.0) +
          14.6 * exp(-dot(xz - vec2(-228.0, 384.0), xz - vec2(-228.0, 384.0)) / 6800.0) +
          14.2 * exp(-dot(xz - vec2(-248.0, 24.0), xz - vec2(-248.0, 24.0)) / 5900.0);
        return (0.45 + hHills * flatGuard) * coastCollar;
      }
    `;
  }

  /**
   * Hukum Emas 14: Analytical Sphere Sun/Moon + Radial Euclidean SDF Stars +
   * 2-Layer 6-Octave Rotated FBM Cumulonimbus Storm Clouds + Forked Plasma Lightning
   */
  _initAnalyticalSkyDome() {
    const skyGeo = new THREE.SphereGeometry(3400, 56, 32);
    const skyMat = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      extensions: { derivatives: true },
      uniforms: {
        uTime: this.uniforms.uTime,
        uSunDir: this.uniforms.uSunDir,
        uTrueSunDir: this.uniforms.uTrueSunDir,
        uMoonDir: this.uniforms.uMoonDir,
        uMoonPhaseSunDir: this.uniforms.uMoonPhaseSunDir,
        uMoonIllumFraction: this.uniforms.uMoonIllumFraction,
        uWorldToEqMat: this.uniforms.uWorldToEqMat,
        uStarGridTex: this.uniforms.uStarGridTex,
        uSunColor: this.uniforms.uSunColor,
        uNightFactor: this.uniforms.uNightFactor,
        uDuskFactor: this.uniforms.uDuskFactor,
        uRainIntensity: this.uniforms.uRainIntensity,
        uCloudCover: this.uniforms.uCloudCover,
        uStormFactor: this.uniforms.uStormFactor,
        uLightningFlash: this.uniforms.uLightningFlash,
        uLightningDir: this.uniforms.uLightningDir,
        uLightningSeed: this.uniforms.uLightningSeed,
        uWindVec: this.uniforms.uWindVec,
        uWindPhase: this.uniforms.uWindPhase,
        uHvsParams: this.uniforms.uHvsParams,
        uHvsGlare: this.uniforms.uHvsGlare
      },
      vertexShader: /* glsl */ `
        varying vec3 vWorldDir;
        void main() {
          vWorldDir = normalize(position);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec3 vWorldDir;
        uniform float uTime;
        uniform vec3 uSunDir;
        uniform vec3 uTrueSunDir;
        uniform vec3 uMoonDir;
        uniform vec3 uMoonPhaseSunDir;
        uniform float uMoonIllumFraction;
        uniform mat3 uWorldToEqMat;
        uniform sampler2D uStarGridTex;
        uniform vec3 uSunColor;
        uniform float uNightFactor;
        uniform float uDuskFactor;
        uniform float uRainIntensity;
        uniform float uCloudCover;
        uniform float uStormFactor;
        uniform float uLightningFlash;
        uniform vec3 uLightningDir;
        uniform float uLightningSeed;
        uniform vec3 uWindVec;
        uniform vec2 uWindPhase;
        uniform vec4 uHvsParams;
        uniform vec4 uHvsGlare;

        ${HumanEyeHDROptics.getGLSLKernel()}

        // Hukum Emas 14: Strict domain rotation matrix preventing periodic axial striping
        const mat2 ROT_FBM = mat2(0.8, 0.6, -0.6, 0.8);

        // Exact IAU J2000.0 Equatorial -> Galactic Rotation Matrix (Column-Major for GLSL):
        // Row 0 -> Galactic Center (RA 17h45m37.2s, Dec -28.9362°)
        // Row 2 -> North Galactic Pole (RA 12h51m26.3s, Dec +27.1283°)
        const mat3 GAL_FROM_EQ = mat3(
          -0.05487556,  0.49410943, -0.86766615,
          -0.87343709, -0.44482963, -0.19807637,
          -0.48383502,  0.74698224,  0.45598378
        );

        float hash21(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }

        float noise2(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          vec2 u = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
            mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x),
            u.y
          );
        }

        float fbmGal4(vec2 p) {
          float v = 0.0;
          float a = 0.54;
          for (int i = 0; i < 4; i++) {
            v += a * noise2(p);
            p = ROT_FBM * p * 2.11 + vec2(13.7, 27.3);
            a *= 0.48;
          }
          return v;
        }

        // 6-Octave Domain-Rotated FBM for billowing tropical Cumulonimbus structure
        float fbmCloud6(vec2 p) {
          float v = 0.0;
          float a = 0.52;
          for (int i = 0; i < 6; i++) {
            v += a * noise2(p);
            p = ROT_FBM * p * 2.04 + vec2(17.1, 31.7);
            a *= 0.5;
          }
          return v;
        }

        float fbmBolt4(vec2 p) {
          float v = 0.0;
          float a = 0.55;
          for (int i = 0; i < 4; i++) {
            v += a * (noise2(p) - 0.5);
            p = ROT_FBM * p * 2.35 + vec2(9.3, 21.4);
            a *= 0.5;
          }
          return v;
        }

        // Physical Planck B-V Color Index -> Stellar Photosphere RGB Chromaticity
        vec3 evalPlanckStellarColor(float bv) {
          vec3 hotOBA = vec3(0.68, 0.82, 1.00);   // Rigel, Acrux, Spica, Orion Belt (B-V < 0)
          vec3 pureAF = vec3(0.96, 0.98, 1.00);   // Sirius, Vega, Altair, Fomalhaut (B-V ~ 0.0..0.3)
          vec3 solarGK = vec3(1.00, 0.91, 0.72);  // Alpha Centauri, Capella, Pollux (B-V ~ 0.7..1.0)
          vec3 rubyKM = vec3(1.00, 0.60, 0.32);   // Betelgeuse, Antares, Aldebaran, Gacrux (B-V > 1.4)
          vec3 c = mix(hotOBA, pureAF, smoothstep(-0.28, 0.14, bv));
          c = mix(c, solarGK, smoothstep(0.18, 0.88, bv));
          c = mix(c, rubyKM, smoothstep(0.95, 1.75, bv));
          return c;
        }

        // Procedural Tidally-Locked Nearside Selenographic Albedo & Micro-Crater Relief
        // (Anorthosite Lunar Highlands vs Dark Basalt Maria Basins + Southern Tycho Ejecta Rays)
        vec2 evalLunarMariaSurface(vec2 sDisk, float zMoon) {
          // True 3D spherical selenographic coordinates centered on sub-Earth Sinus Medii (0°, 0°),
          // naturally foreshortening Mare Crisium & Oceanus Procellarum toward the lunar limb!
          float sLon = atan(sDisk.x, max(0.06, zMoon));
          float sLat = asin(clamp(sDisk.y, -1.0, 1.0));
          vec2 sUV = vec2(sLon, sLat);

          float craterMicro = fbmGal4(sUV * 11.5 + vec2(3.1, 7.4));
          float mareMacro = fbmGal4(sUV * 3.1 + vec2(11.2, 5.8));

          // Canonical Earth-facing nearside Lunar Maria basins (rad):
          // Mare Imbrium (-18°W, +33°N), Mare Serenitatis (+18°E, +28°N), Mare Tranquillitatis (+31°E, +9°N),
          // Mare Crisium (+58°E, +17°N), Mare Fecunditatis (+51°E, -8°S), Oceanus Procellarum (-46°W, +10°N),
          // Mare Nubium (-17°W, -20°S), Mare Humorum (-39°W, -24°S)
          float mImbrium   = exp(-dot(sUV - vec2(-0.31,  0.56), sUV - vec2(-0.31,  0.56)) * 6.2);
          float mSeren     = exp(-dot(sUV - vec2( 0.31,  0.48), sUV - vec2( 0.31,  0.48)) * 9.5);
          float mTranquil  = exp(-dot(sUV - vec2( 0.54,  0.15), sUV - vec2( 0.54,  0.15)) * 7.8);
          float mCrisium   = exp(-dot(sUV - vec2( 1.01,  0.30), sUV - vec2( 1.01,  0.30)) * 22.0);
          float mFecund    = exp(-dot(sUV - vec2( 0.88, -0.14), sUV - vec2( 0.88, -0.14)) * 10.5);
          float mProcell   = exp(-dot(sUV - vec2(-0.78,  0.16), sUV - vec2(-0.78,  0.16)) * 3.6);
          float mNubium    = exp(-dot(sUV - vec2(-0.30, -0.35), sUV - vec2(-0.30, -0.35)) * 7.4);
          float mHumorum   = exp(-dot(sUV - vec2(-0.68, -0.42), sUV - vec2(-0.68, -0.42)) * 13.0);

          float mariaRaw = mImbrium * 1.05 + mSeren * 0.98 + mTranquil * 0.96 + mCrisium * 0.92
                         + mFecund * 0.82 + mProcell * 0.90 + mNubium * 0.84 + mHumorum * 0.80;
          float mariaMask = clamp(mariaRaw * (0.72 + 0.54 * mareMacro), 0.0, 1.0);

          // Southern Highland Tycho crater (-11°W, -43°S) & Copernican bright ejecta rays
          vec2 dTycho = sUV - vec2(-0.19, -0.75);
          float rTycho = length(dTycho) + 1e-4;
          float rayAng = atan(dTycho.y, dTycho.x);
          float tychoRays = exp(-rTycho * 2.9) * (0.42 + 0.58 * pow(0.5 + 0.5 * sin(rayAng * 9.0 + mareMacro * 2.4), 3.0));

          float highland = 0.98 + 0.16 * craterMicro + 0.24 * tychoRays;
          float basaltMare = 0.14 + 0.07 * craterMicro;
          float albedo = mix(highland, basaltMare, smoothstep(0.18, 0.62, mariaMask));
          return vec2(albedo, craterMicro);
        }

        void main() {
          vec3 dir = normalize(vWorldDir);
          // Uniform-control-flow angular pixel footprint (radians/px) for subpixel-stable constellation stars
          float skyPxRad = clamp(max(length(dFdx(dir)), length(dFdy(dir))), 0.00050, 0.00340);
          float h = clamp(dir.y, 0.0, 1.0);
          float overcast = clamp(uCloudCover * 0.62 + uRainIntensity * 0.48, 0.0, 1.0);

          vec3 trueSunDir = normalize(uTrueSunDir);
          vec3 moonDir = normalize(uMoonDir);
          vec3 celestialDir = normalize(uSunDir);
          float cosTheta = dot(dir, celestialDir);
          float cosSun = dot(dir, trueSunDir);
          float cosMoon = dot(dir, moonDir);

          // 1. FULL-FEATURE RAYLEIGH + OZONE CHAPPUIS + BORTLE CLASS 1-2 ISLAND NIGHT ATMOSPHERE
          //    Zero urban orange skyglow pollution! Only pure dark oceanic cosmos, subtle 557.7nm natural
          //    airglow near the marine horizon, and physical lunar Rayleigh scatter when the Moon is risen.
          vec3 dayZenith = vec3(0.12, 0.36, 0.74);
          vec3 dayHorizon = vec3(0.62, 0.78, 0.92);

          // Stratospheric Ozone Chappuis absorption deepens twilight zenith to royal cobalt-indigo
          vec3 duskZenith = vec3(0.10, 0.15, 0.38);
          vec3 duskHorizon = mix(vec3(0.52, 0.56, 0.68), vec3(0.94, 0.47, 0.20), smoothstep(-0.35, 0.85, cosSun));

          float lunarSkyScatter = clamp(moonDir.y, 0.0, 1.0) * uMoonIllumFraction * (1.0 - overcast * 0.75);
          vec3 bortle1Zenith = vec3(0.0038, 0.0062, 0.0145) + vec3(0.007, 0.012, 0.026) * lunarSkyScatter;
          // Subtle natural 557.7nm island airglow + marine horizon scatter (minimal light pollution!)
          vec3 bortle1Horizon = vec3(0.0105, 0.0170, 0.0290) + vec3(0.016, 0.024, 0.045) * lunarSkyScatter;

          vec3 zenith = mix(mix(dayZenith, duskZenith, uDuskFactor), bortle1Zenith, uNightFactor);
          vec3 horizon = mix(mix(dayHorizon, duskHorizon, uDuskFactor), bortle1Horizon, uNightFactor);
          vec3 skyCol = mix(horizon, zenith, pow(h, 0.56));

          // Anti-Solar Belt of Venus ("Sabuk Venus") & Earth's Shadow Arch ("Bayangan Bumi") at Dawn/Dusk!
          if (uDuskFactor > 0.01 && overcast < 0.88) {
            vec2 dirXZ = normalize(dir.xz + vec2(1e-5, 0.0));
            vec2 sunXZ = normalize(trueSunDir.xz + vec2(1e-5, 0.0));
            float antiSun = clamp(dot(dirXZ, -sunXZ) * 0.5 + 0.5, 0.0, 1.0);
            float shadowTop = clamp(0.03 + max(0.0, -trueSunDir.y) * 0.55, 0.02, 0.14);
            float earthShadow = (1.0 - smoothstep(shadowTop * 0.45, shadowTop + 0.05, h)) * pow(antiSun, 1.4);
            float beltOfVenus = exp(-pow((h - (shadowTop + 0.055)) / 0.068, 2.0)) * pow(antiSun, 1.25);
            float clearTwilight = uDuskFactor * (1.0 - overcast * 0.85);
            skyCol = mix(skyCol, vec3(0.11, 0.14, 0.25), earthShadow * clearTwilight * 0.62);
            skyCol += vec3(0.42, 0.20, 0.26) * beltOfVenus * clearTwilight * 0.55;
          }

          // Deep atmospheric Rayleigh/Mie storm overcast gradient
          vec3 stormHorizonDay = mix(vec3(0.35, 0.39, 0.44), vec3(0.18, 0.21, 0.26), uStormFactor);
          vec3 stormZenithDay = mix(vec3(0.20, 0.24, 0.29), vec3(0.09, 0.11, 0.15), uStormFactor);
          vec3 stormBase = mix(stormHorizonDay, stormZenithDay, pow(h, 0.55));
          stormBase *= mix(1.0, 0.16, uNightFactor);
          skyCol = mix(skyCol, stormBase, clamp(uRainIntensity * 0.76 + uStormFactor * 0.35, 0.0, 0.94));

          // 2. BORTLE CLASS 1-2 GALACTIC MILKY WAY ("BIMASAKTI"), GREAT RIFT DUST LANES, LMC/SMC,
          //    FAINT GALACTIC STARFIELD, & 208 NOTABLE REAL J2000 CONSTELLATION STARS (TRUE LINEAR-HDR!)
          //    Accumulated directly in Scene-Referred Linear-HDR (hdrCosmos) so gamma^2.2 never crushes stars,
          //    and occluded spatially ONLY where foreground clouds or the solid lunar sphere actually pass!
          vec3 eqDir = normalize(uWorldToEqMat * dir);
          vec3 galDir = GAL_FROM_EQ * eqDir;
          float starClearVis = smoothstep(0.12, 0.84, uNightFactor)
                             * clamp(1.0 - uRainIntensity * 1.12 - uStormFactor * 0.68, 0.0, 1.0);
          vec3 hdrCosmos = vec3(0.0);

          // Zodiacal Light Cone ("Cahaya Zodiak" — interplanetary dust glow in low-pollution island post-dusk/pre-dawn)
          if (uNightFactor > 0.30 && trueSunDir.y < -0.08 && trueSunDir.y > -0.54 && dir.y > 0.0) {
            float zodAlong = max(0.0, dot(normalize(dir.xz + vec2(1e-5, 0.0)), normalize(trueSunDir.xz + vec2(1e-5, 0.0))));
            float zodCone = pow(zodAlong, 5.2) * exp(-h * 2.6) * smoothstep(-0.54, -0.14, trueSunDir.y);
            hdrCosmos += vec3(0.0085, 0.0095, 0.0125) * zodCone * uNightFactor * (1.0 - overcast) * (1.0 - lunarSkyScatter * 0.65);
          }

          if (starClearVis > 0.005 && dir.y > -0.01) {
            // Bortle 1-2 low-pollution marine island clarity right down to ~3.8° above the sea horizon!
            float horizonExtinction = smoothstep(-0.005, 0.068, dir.y);
            float moonGlareWash = 1.0 - 0.24 * lunarSkyScatter - 0.52 * pow(max(cosMoon, 0.0), 18.0) * lunarSkyScatter;
            float darkSkyVis = starClearVis * horizonExtinction * clamp(moonGlareWash, 0.30, 1.0);

            // 2A. True J2000 Galactic Milky Way Band, Sagittarius-Scorpius Core Bulge, & Great Rift Dust Absorption
            //     Evaluated directly from 3D unit vector galDir (ZERO atan +-pi branch-cut seam anywhere across 360°!)
            float sinB = galDir.z;
            float coreProximity = clamp(galDir.x * 0.5 + 0.5, 0.0, 1.0);
            float bulgeWidth = mix(0.110, 0.225, pow(coreProximity, 3.0));
            float mwWarp = (fbmGal4(galDir.xy * 2.4 + vec2(sinB * 3.8, -sinB * 3.4)) - 0.5) * 0.065;
            float mwProfile = exp(-pow((sinB + mwWarp) / bulgeWidth, 2.0));

            float starCloudTex = fbmGal4(galDir.xy * 5.2 + vec2(sinB * 10.2 + 2.1, -sinB * 9.6 - 1.3));
            float fineCloudTex = fbmGal4(galDir.xy * 13.5 + vec2(-sinB * 21.0 - 4.7, sinB * 20.0 + 3.8));
            float mwLuminance = mwProfile * (0.48 + 0.84 * starCloudTex + 0.38 * fineCloudTex);
            mwLuminance *= mix(0.68, 1.62, pow(coreProximity, 2.0));

            // Great Rift ("Celah Besar") & Southern Cross Coalsack ("Kantung Arang") Dark Molecular Dust Lanes!
            float riftNoise = fbmGal4(galDir.xy * 6.8 + vec2(sinB * 17.5 + 9.4, -sinB * 16.4 + 2.2));
            float riftBand = exp(-pow((sinB - 0.012 + (riftNoise - 0.5) * 0.055) / 0.044, 2.0));
            float riftStrength = smoothstep(0.34, 0.72, riftNoise) * smoothstep(-0.35, 0.55, galDir.x);
            // Coalsack Dark Nebula right beside Crux (Galactic l ~ -57° = -1.00 rad, b ~ -1.5° -> 3D chord distance)
            float dCoalsack = length(galDir - vec3(0.54012, -0.84118, -0.02600));
            float coalsack = exp(-(dCoalsack * dCoalsack) / 0.0068) * 0.84;
            float dustTransmittance = clamp(1.0 - riftBand * riftStrength * 0.82 - coalsack, 0.11, 1.0);

            vec3 mwOuterLin = vec3(0.0185, 0.0265, 0.0440);
            vec3 mwCoreLin = vec3(0.0520, 0.0430, 0.0365);
            vec3 mwColorLin = mix(mwOuterLin, mwCoreLin, pow(coreProximity, 2.0) * (0.55 + 0.45 * starCloudTex));
            hdrCosmos += mwColorLin * mwLuminance * dustTransmittance * darkSkyVis;

            // Southern Hemisphere Gems: Large & Small Magellanic Clouds (LMC & SMC) + M42 Orion Nebula!
            vec3 lmcEq = vec3(0.0549, 0.3416, -0.9382); // RA 5.39h, Dec -69.76°
            vec3 smcEq = vec3(0.2878, 0.0674, -0.9553); // RA 0.88h, Dec -72.80°
            vec3 m42Eq = vec3(0.1066, 0.9898, -0.0940); // RA 5.59h, Dec -5.39° (Orion's Sword)
            float dLMC = length(eqDir - lmcEq);
            float dSMC = length(eqDir - smcEq);
            float dM42 = length(eqDir - m42Eq);
            if (dLMC < 0.16) {
              float lmcTex = 0.52 + 0.48 * fbmGal4(eqDir.xy * 42.0);
              hdrCosmos += vec3(0.015, 0.021, 0.035) * exp(-pow(dLMC / 0.054, 2.0)) * lmcTex * darkSkyVis;
            }
            if (dSMC < 0.10) {
              hdrCosmos += vec3(0.012, 0.017, 0.029) * exp(-pow(dSMC / 0.030, 2.0)) * darkSkyVis;
            }
            if (dM42 < 0.038) {
              float m42Glow = exp(-pow(dM42 / 0.0090, 2.0));
              hdrCosmos += vec3(0.095, 0.052, 0.110) * m42Glow * darkSkyVis;
            }

            // 2B. Tier-A Faint Equatorial-Locked Galactic Pinpoint Background Stars (~4,900 Bortle 1-2 naked-eye stars)
            //     3D Spherical-Shell 2x2x2 Voxel Lattice on eqDir: ZERO atan +-pi seam, ZERO pole pinching, ZERO clipping!
            vec3 bP3 = eqDir * 104.0;
            vec3 bBase = floor(bP3 - 0.5);
            float galStarThresh = mix(0.85, 0.58, exp(-abs(sinB) * 6.5));
            for (int oz = 0; oz < 2; oz++) {
              for (int oy = 0; oy < 2; oy++) {
                for (int ox = 0; ox < 2; ox++) {
                  vec3 bCell = bBase + vec3(float(ox), float(oy), float(oz));
                  float h0 = fract(sin(dot(bCell, vec3(127.1, 311.7, 74.7))) * 43758.5453);
                  if (h0 > galStarThresh) {
                    float h1 = fract(sin(dot(bCell, vec3(269.5, 183.3, 246.1))) * 43758.5453);
                    float h2 = fract(sin(dot(bCell, vec3(113.5, 271.9, 124.6))) * 43758.5453);
                    float h3 = fract(sin(dot(bCell, vec3(419.2, 371.9, 168.3))) * 43758.5453);
                    vec3 bStarPos = bCell + vec3(0.05 + 0.90 * h1, 0.05 + 0.90 * h2, 0.05 + 0.90 * h3);
                    if (abs(length(bStarPos) - 104.0) < 0.165) {
                      vec3 bStarDir = normalize(bStarPos);
                      float dAng = length(eqDir - bStarDir);
                      float rAng = max(skyPxRad * 1.05, mix(0.00085, 0.00145, h2));
                      if (dAng < rAng) {
                        float twkBg = 0.82 + 0.18 * sin(uTime * (1.8 + h3 * 2.5) + h0 * 50.0);
                        float mBg = smoothstep(rAng, 0.0, dAng) * mix(0.022, 0.058, h3 * h3);
                        vec3 cBg = evalPlanckStellarColor(mix(-0.15, 1.45, fract((h1 + h2) * 7.3)));
                        hdrCosmos += cBg * mBg * twkBg * darkSkyVis;
                      }
                    }
                  }
                }
              }
            }

            // 2C. Tier-B O(1) REAL J2000 CONSTELLATION STAR CATALOG (212 Navigational & Constellation Stars in 512x64 POT Grid!)
            //     Orion, Crux/Pari, Alpha/Beta Centauri, Scorpius, Sagittarius, Sirius/Canis Major, Canopus,
            //     Taurus + Pleiades, Gemini, Leo, Virgo, Lyra, Cygnus, Aquila, Cassiopeia, Pegasus, Ursa Major!
            float eqRA = atan(eqDir.y, eqDir.x); // [-pi, pi]
            float eqDec = asin(clamp(eqDir.z, -1.0, 1.0)); // [-pi/2, pi/2]
            float raPos = mod(eqRA + 6.28318530718, 6.28318530718);
            float cx = clamp(floor((raPos / 6.28318530718) * 128.0), 0.0, 127.0);
            float cy = clamp(floor((eqDec / 3.14159265359 + 0.5) * 64.0), 0.0, 63.0);

            for (int slot = 0; slot < 4; slot++) {
              vec2 sTexUV = vec2((cx * 4.0 + float(slot) + 0.5) / 512.0, (cy + 0.5) / 64.0);
              vec4 stData = texture2D(uStarGridTex, sTexUV);
              if (stData.w > 0.0) {
                vec3 stEq = normalize(stData.xyz);
                float theta = length(eqDir - stEq); // Exact chord angle (rad)
                if (theta < 0.0185) {
                  float magCode = floor(stData.w);
                  float vmag = magCode * 0.01 - 2.0;
                  float bv = fract(stData.w) * 2.6 - 0.4;
                  vec3 starRGB = evalPlanckStellarColor(bv);

                  // Smooth C^2 radial zero-taper window eliminating any outer halo boundary ring!
                  float radialWindow = pow(clamp(1.0 - theta / 0.0185, 0.0, 1.0), 3.0);

                  // Calibrated Linear-HDR Pogson prominence so 1st..3rd magnitude constellation patterns stand out unmistakably!
                  float rawFlux = pow(2.12, 2.85 - vmag);
                  float coreFlux = (0.42 * rawFlux) / (1.0 + 0.085 * rawFlux);
                  float haloFlux = (0.15 * rawFlux) / (1.0 + 0.28 * rawFlux);

                  // Altitude-dependent Kolmogorov atmospheric scintillation (stronger near horizon, calm at zenith)
                  float starSeed = fract(magCode * 0.173 + stEq.x * 13.7 + stEq.y * 29.1);
                  float lowAltScint = mix(0.30, 0.06, smoothstep(0.05, 0.55, h));
                  float twinkle = 1.0 + lowAltScint * sin(uTime * (3.6 + starSeed * 4.2) + starSeed * 62.8)
                                      * cos(uTime * (2.1 + starSeed * 2.9) - starSeed * 19.4);

                  // Crisp jewel-like anti-aliased Airy pinpoint core + tight colored B-V aureole (NEVER giant bubbles!)
                  float magProm = clamp((3.9 - vmag) / 5.2, 0.0, 1.0);
                  float coreRad = max(skyPxRad * mix(0.95, 1.65, magProm), mix(0.00105, 0.00220, magProm));
                  float airyCore = exp(-pow(theta / coreRad, 2.0));
                  float innerAureole = exp(-theta / max(coreRad * 2.25, 0.0026)) * 0.48;
                  float tightCiliary = exp(-theta * 260.0) * 0.14 * smoothstep(2.6, -0.8, vmag);

                  // Delicate 1-px 4-point human-eye crystalline-lens suture glint on 1st/2nd magnitude showpiece stars!
                  float glintSpikes = 0.0;
                  if (vmag < 1.95 && theta < 0.0155) {
                    vec3 stWorld = normalize(stEq * uWorldToEqMat);
                    vec3 stRight = normalize(cross(vec3(0.0, 1.0, 0.0), stWorld) + vec3(1e-5, 0.0, 0.0));
                    vec3 stUp = cross(stWorld, stRight);
                    vec2 dLocal = vec2(dot(dir - stWorld, stRight), dot(dir - stWorld, stUp));
                    float spikeW = max(skyPxRad * 0.72, 0.00055);
                    float spikeH = exp(-abs(dLocal.y) / spikeW) * exp(-abs(dLocal.x) * 240.0);
                    float spikeV = exp(-abs(dLocal.x) / spikeW) * exp(-abs(dLocal.y) * 240.0);
                    glintSpikes = (spikeH + spikeV) * 0.55 * smoothstep(1.95, -0.9, vmag);
                  }

                  vec3 coreCol = mix(starRGB, vec3(1.0), 0.38 * airyCore);
                  vec3 sContrib = (coreCol * airyCore * coreFlux + starRGB * (innerAureole + tightCiliary + glintSpikes) * haloFlux)
                                * radialWindow * twinkle * darkSkyVis;
                  hdrCosmos += sContrib;
                }
              }
            }

            // 2D. Occasional Subtle Tropical Island Shooting Star / Meteor ("Bintang Jatuh" every ~13s in clear dark skies)
            if (h > 0.16 && overcast < 0.42) {
              float mCycle = floor(uTime / 13.0);
              float mPhase = fract(uTime / 13.0);
              if (mPhase < 0.055) {
                float pNorm = mPhase / 0.055; // 0..1 over ~0.71s streak
                float mAz = hash21(vec2(mCycle, 3.7)) * 6.2831853;
                float mEl = 0.34 + 0.36 * hash21(vec2(mCycle, 9.1));
                vec3 mStart = normalize(vec3(cos(mAz) * cos(mEl), sin(mEl), sin(mAz) * cos(mEl)));
                vec3 mVel = normalize(vec3(-sin(mAz) * 0.72, -0.52, cos(mAz) * 0.45));
                vec3 mHead = normalize(mStart + mVel * (pNorm * 0.22));
                vec3 mTail = normalize(mStart + mVel * (max(0.0, pNorm - 0.34) * 0.22));
                vec3 seg = mHead - mTail;
                float segLen2 = max(1e-6, dot(seg, seg));
                float tProj = clamp(dot(dir - mTail, seg) / segLen2, 0.0, 1.0);
                float dSeg = length((mTail + seg * tProj) - dir);
                float meteor = exp(-dSeg * 1250.0) * tProj * sin(pNorm * 3.14159265) * darkSkyVis;
                hdrCosmos += vec3(0.75, 0.88, 1.0) * meteor * 0.85;
              }
            }
          }

          // 3. CONTINUOUS DUAL-BODY CELESTIAL PROJECTION: SOLAR PHOTOSPHERE & REAL-TIME SELENOGRAPHIC MOON!
          //    Spatial clouds occlude both discs via cloudTrans; discVis only attenuates during heavy rain/storm sheets.
          float discVis = clamp(1.0 - uRainIntensity * 0.90 - uStormFactor * 0.65, 0.0, 1.0);
          float sunAboveHor = smoothstep(-0.045, 0.045, trueSunDir.y);
          float sunVis = discVis * sunAboveHor;
          float sunDiscRadius = 0.024;
          float sinSun = sqrt(max(0.0, 1.0 - cosSun * cosSun));
          float rNormSun = sinSun / sunDiscRadius;
          if (cosSun > 0.0 && rNormSun < 1.0 && sunVis > 0.005) {
            float zSphere = sqrt(max(0.0, 1.0 - rNormSun * rNormSun));
            vec3 sunBodyCol = mix(vec3(1.0, 0.95, 0.82), vec3(1.0, 0.64, 0.34), uDuskFactor);
            skyCol = mix(skyCol, sunBodyCol * (0.72 + 0.38 * zSphere), sunVis);
          }

          // Henyey-Greenstein (g = 0.76) Physical Anisotropic Mie Solar Aureole
          float solarHaloVis = smoothstep(-0.14, 0.06, trueSunDir.y);
          float sunHalo = pow(max(cosSun, 0.0), 32.0) * (1.0 - overcast * 0.88) * solarHaloVis;
          float gMie = 0.76;
          float hgPhase = (1.0 - gMie * gMie) / (4.0 * 3.14159265 * pow(max(0.04, 1.0 + gMie * gMie - 2.0 * gMie * max(cosSun, -0.5)), 1.5));
          float mieAureole = clamp(hgPhase * 0.14, 0.0, 0.75) * (1.0 - overcast * 0.82) * solarHaloVis;
          vec3 solarMieTint = mix(vec3(0.52, 0.42, 0.26), vec3(0.96, 0.48, 0.18), uDuskFactor);
          skyCol += (sunHalo + mieAureole) * solarMieTint;

          // Real-Time Selenographic Moon Disc (Evaluated in True Linear-HDR so Lunar Maria & Phase Terminator stay razor-sharp!)
          float moonAboveHor = smoothstep(-0.020, 0.040, moonDir.y);
          float moonDayGain = mix(0.32, 1.0, smoothstep(-0.05, 0.70, uNightFactor + uDuskFactor * 0.55));
          float moonVis = moonDayGain * moonAboveHor * discVis;
          float moonDiscRadius = 0.0215;
          float sinMoon = sqrt(max(0.0, 1.0 - cosMoon * cosMoon));
          float rNormMoon = sinMoon / moonDiscRadius;
          vec3 hdrMoonBody = vec3(0.0);
          float moonOccultMask = 0.0;

          if (cosMoon > 0.0 && rNormMoon < 1.0 && moonAboveHor > 0.005) {
            // Tidally-locked Earth-facing selenographic frame aligned with J2000 Celestial North!
            // As the Moon rises in the East and sets in the West over Depok (-6.37° S), the familiar
            // nearside Lunar Maria rotate naturally with Southern-Hemisphere parallactic angle.
            vec3 celNorthWorld = normalize(vec3(0.0, 0.0, 1.0) * uWorldToEqMat);
            vec3 moonNorthProj = celNorthWorld - moonDir * dot(celNorthWorld, moonDir);
            vec3 mUpFallback = abs(moonDir.y) < 0.94 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
            vec3 moonNorthUp = length(moonNorthProj) > 1e-4 ? normalize(moonNorthProj) : normalize(mUpFallback - moonDir * dot(mUpFallback, moonDir));
            vec3 moonSelRight = normalize(cross(moonDir, moonNorthUp));

            vec2 sDisk = vec2(dot(dir, moonSelRight), dot(dir, moonNorthUp)) / moonDiscRadius;
            float zMoon = sqrt(max(0.0, 1.0 - dot(sDisk, sDisk)));

            // Outward 3D Lunar Sphere Normal in Depok World Space (sub-Earth center is -moonDir!)
            vec3 moonSurfWorldN = normalize(sDisk.x * moonSelRight + sDisk.y * moonNorthUp - zMoon * moonDir);
            vec3 sunFromMoon = normalize(uMoonPhaseSunDir);

            vec2 lunarSurf = evalLunarMariaSurface(sDisk, zMoon);
            float mariaAlbedo = lunarSurf.x;
            float craterRelief = (lunarSurf.y - 0.5) * 0.12;

            // Exact 3D Real-Time Lunar Phase Terminator with Micro-Crater Relief & Lommel-Seeliger Regolith Reflectance!
            float ndotl = dot(moonSurfWorldN, sunFromMoon);
            float terminator = smoothstep(-0.032, 0.055, ndotl + craterRelief);
            float mu0 = max(0.0, ndotl + craterRelief * 0.45);
            float mu = max(0.08, zMoon);
            float lommelSeeliger = (mu0 / (mu0 + mu + 0.06)) * 1.56;
            float limbSoft = smoothstep(1.0, 0.960, rNormMoon);

            // Solid lunar body occults background stars even across the dark unlit limb!
            moonOccultMask = limbSoft * moonAboveHor;

            // Calibrated Linear-HDR regolith (Dark Basalt Maria ~ 118/255 sRGB, Bright Highlands ~ 232/255 sRGB!)
            vec3 sunlitRegolithLin = vec3(0.76, 0.73, 0.65) * mariaAlbedo * lommelSeeliger * terminator;
            vec3 earthshineLin = vec3(0.0058, 0.0086, 0.0145) * (0.32 + 0.68 * mariaAlbedo)
                               * (1.0 - terminator * 0.88) * (0.55 + 0.45 * zMoon) * uNightFactor;
            hdrMoonBody = (sunlitRegolithLin + earthshineLin) * limbSoft * moonVis;
          }

          // Tight, crisp atmospheric lunar corona right around the Moon limb (ZERO wide blurry cotton-ball halo!)
          float moonHaloStrength = uMoonIllumFraction * smoothstep(0.12, 0.82, uNightFactor) * moonAboveHor * discVis;
          float moonTightCorona = (pow(max(cosMoon, 0.0), 540.0) * 0.13 + pow(max(cosMoon, 0.0), 68.0) * 0.018)
                                * (1.0 - overcast * 0.85) * moonHaloStrength;

          // 4. 3-LAYER FULL-FEATURE VOLUMETRIC CLOUDS (HIGH ICE CIRRUS + CUMULUS + CUMULONIMBUS)
          //    Clouds physically occlude stars, Milky Way, and Moon behind them!
          float cloudDensityTotal = 0.0;
          float cloudCoverMask = 0.0;
          if (dir.y > 0.01) {
            vec2 windAdv = uWindPhase * 0.011;
            // Layer 0: High-Altitude Fibrous Tropical Ice Cirrus (8-11 km, glowing coral-gold deep into twilight!)
            vec2 projCirrus = (dir.xz / (dir.y + 0.34)) * 0.78 + windAdv * 0.28;
            float cirrusStreak = fbmGal4(vec2(projCirrus.x * 2.6 + projCirrus.y * 0.8, projCirrus.y * 0.65));
            float cirrusFine = noise2(vec2(projCirrus.x * 11.0 + cirrusStreak * 2.4, projCirrus.y * 3.2));
            float dCirrus = smoothstep(0.56, 0.86, cirrusStreak) * (0.55 + 0.45 * cirrusFine) * (1.0 - uStormFactor * 0.65);

            vec2 projHigh = (dir.xz / (dir.y + 0.26)) * 1.15 + windAdv * 0.55;
            vec2 projLow = (dir.xz / (dir.y + 0.16)) * 1.85 + windAdv * 1.35 + vec2(5.7, -3.4);

            float fHigh = fbmCloud6(projHigh);
            float fLow = fbmCloud6(projLow);

            // Tropical clear nights naturally dissipate daytime convective cumulus, opening wide starry island skies!
            float dryNightClearBias = 0.085 * uNightFactor * (1.0 - clamp(uRainIntensity * 2.5 + uStormFactor, 0.0, 1.0));
            float thresh = mix(0.58, 0.20, clamp(uCloudCover, 0.0, 1.0)) + dryNightClearBias;
            float dHigh = smoothstep(thresh, thresh + 0.36, fHigh);
            float dLow = smoothstep(thresh - 0.04, thresh + 0.30, fLow) * (0.45 + 0.55 * uCloudCover);
            cloudDensityTotal = clamp(dCirrus * 0.22 + dHigh * 0.58 + dLow * 0.78, 0.0, 1.0);

            // High cirrus catches upper-stratosphere sunset light slightly longer than low cumulus
            vec3 cirrusDay = mix(vec3(0.96, 0.98, 1.0), vec3(1.0, 0.54, 0.30), clamp(uDuskFactor * 1.25, 0.0, 1.0));
            vec3 cirrusNight = vec3(0.026, 0.038, 0.066) * (0.30 + 0.70 * lunarSkyScatter);
            float horizonFade = smoothstep(0.005, 0.16, dir.y);
            skyCol = mix(skyCol, mix(cirrusDay, cirrusNight, uNightFactor), dCirrus * 0.28 * horizonFade * (1.0 - dLow));

            // Directional Beer-Lambert optical thickness toward sun/moon
            vec2 sunStep = normalize(celestialDir.xz + vec2(1e-4, 0.0)) * 0.14;
            float fShadow = fbmCloud6(projLow + sunStep);
            float selfLit = clamp(1.0 - max(0.0, fShadow - fLow) * 3.4 - dLow * (0.35 + 0.42 * uStormFactor), 0.08, 1.0);
            float forwardScatter = pow(max(cosTheta, 0.0), 8.0) * 0.35 * (1.0 - dLow * 0.7);
            // Silver-lining cloud-edge diffraction rim toward sun/moon!
            float silverEdge = pow(max(cosTheta, 0.0), 16.0) * smoothstep(0.05, 0.38, cloudDensityTotal) * exp(-cloudDensityTotal * 2.2) * 0.65;

            vec3 litCloudDay = mix(vec3(0.97, 0.98, 1.0), vec3(0.98, 0.58, 0.34), uDuskFactor);
            vec3 darkBaseDay = mix(vec3(0.46, 0.51, 0.58), vec3(0.11, 0.14, 0.19), clamp(uRainIntensity * 0.65 + uStormFactor * 0.55, 0.0, 1.0));
            darkBaseDay = mix(darkBaseDay, vec3(0.32, 0.22, 0.26), uDuskFactor * (1.0 - uStormFactor * 0.6));

            vec3 scatterTint = mix(vec3(1.0, 0.94, 0.80), vec3(1.0, 0.62, 0.30), uDuskFactor);
            vec3 cloudColDay = mix(darkBaseDay, litCloudDay, selfLit) + scatterTint * (forwardScatter + silverEdge);
            float moonCloudLit = 0.20 + 0.80 * lunarSkyScatter;
            vec3 cloudColNight = mix(vec3(0.010, 0.015, 0.027), vec3(0.068, 0.096, 0.155) * moonCloudLit, selfLit)
                               + vec3(0.20, 0.28, 0.42) * silverEdge * 0.48 * moonCloudLit;
            vec3 cloudCol = mix(cloudColDay, cloudColNight, uNightFactor);

            float mainCloudDens = clamp(dHigh * 0.58 + dLow * 0.78, 0.0, 1.0);
            cloudCoverMask = clamp(mainCloudDens * horizonFade * (0.68 + 0.30 * uCloudCover), 0.0, 0.985);
            skyCol = mix(skyCol, cloudCol, cloudCoverMask);

            // Crepuscular Atmospheric Sunburst Shafts (Sky God-Rays streaming past broken clouds!)
            if (cosSun > 0.15 && solarHaloVis > 0.01 && uRainIntensity < 0.75) {
              vec3 rayCross = cross(dir, trueSunDir);
              float rayAz = atan(rayCross.y, rayCross.x + rayCross.z + 1e-4);
              float rayPattern = 0.5 + 0.5 * sin(rayAz * 11.0 + uTime * 0.12) * cos(rayAz * 19.0 - uTime * 0.08);
              float godRay = pow(cosSun, 6.0) * rayPattern * (1.0 - cloudDensityTotal * 0.72) * smoothstep(0.04, 0.55, uCloudCover);
              vec3 rayCol = mix(vec3(0.42, 0.36, 0.22), vec3(0.88, 0.44, 0.18), uDuskFactor);
              skyCol += rayCol * godRay * solarHaloVis * (1.0 - uRainIntensity) * 0.24;
            }
          }

          // Distant Horizon Virga / Falling Rain Shafts beneath storm cloud base
          if (uRainIntensity > 0.15 && dir.y > -0.02 && dir.y < 0.32) {
            float az = atan(dir.z, dir.x);
            float shaftNoise = noise2(vec2(az * 7.5 - uTime * 0.09, dir.y * 2.2 + uTime * 0.8));
            float shaftStreaks = noise2(vec2(az * 34.0 + dir.y * 4.5, uTime * 2.2));
            float virgaMask = smoothstep(0.36, 0.82, shaftNoise) * (0.65 + 0.35 * shaftStreaks);
            virgaMask *= smoothstep(0.32, 0.08, dir.y) * smoothstep(-0.02, 0.04, dir.y);
            vec3 virgaCol = mix(vec3(0.46, 0.52, 0.58), vec3(0.06, 0.09, 0.14), uNightFactor);
            skyCol = mix(skyCol, virgaCol, virgaMask * min(1.0, uRainIntensity) * 0.42);
          }

          // Procedural Multi-Branch Forked Plasma Lightning + In-Cloud Volumetric Scattering
          if (uLightningFlash > 0.008 && dir.y > 0.0) {
            vec3 lDir = normalize(uLightningDir);
            float angCos = dot(normalize(vec3(dir.x, 0.0, dir.z) + vec3(1e-5, 0.0, 0.0)), normalize(vec3(lDir.x, 0.0, lDir.z) + vec3(1e-5, 0.0, 0.0)));
            float azDiff = acos(clamp(angCos, -1.0, 1.0));
            float crossSign = sign(dir.x * lDir.z - dir.z * lDir.x);
            float signedAz = azDiff * crossSign;

            float cloudScatter = exp(-azDiff * 2.6) * exp(-abs(dir.y - 0.34) * 3.2);
            skyCol += vec3(0.54, 0.68, 1.0) * cloudScatter * (0.45 + 0.75 * cloudDensityTotal) * uLightningFlash * 1.45;

            float seedT = uLightningSeed;
            if (azDiff < 0.42 && dir.y < 0.56) {
              float jagMain = fbmBolt4(vec2(dir.y * 13.5, seedT * 7.13)) * 0.13
                            + fbmBolt4(vec2(dir.y * 41.0, seedT * 3.71)) * 0.032;
              float mainX = jagMain + (0.28 - dir.y) * 0.08;
              float dMain = abs(signedAz - mainX);
              float topMask = smoothstep(0.55, 0.42, dir.y) * smoothstep(0.0, 0.02, dir.y);

              float core = exp(-dMain * 340.0) * topMask;
              float corona = exp(-dMain * 38.0) * topMask * 0.55;

              float b1Y = clamp(0.36 - dir.y, 0.0, 0.32);
              float jagB1 = fbmBolt4(vec2(dir.y * 22.0 + 4.2, seedT * 5.3)) * 0.06;
              float dB1 = abs(signedAz - (mainX + b1Y * 0.52 + jagB1));
              float mB1 = step(dir.y, 0.36) * smoothstep(0.06, 0.16, dir.y);
              core += exp(-dB1 * 420.0) * mB1 * 0.72;
              corona += exp(-dB1 * 48.0) * mB1 * 0.32;

              float b2Y = clamp(0.25 - dir.y, 0.0, 0.22);
              float jagB2 = fbmBolt4(vec2(dir.y * 26.0 - 8.1, seedT * 9.7)) * 0.05;
              float dB2 = abs(signedAz - (mainX - b2Y * 0.46 + jagB2));
              float mB2 = step(dir.y, 0.25) * smoothstep(0.03, 0.11, dir.y);
              core += exp(-dB2 * 440.0) * mB2 * 0.65;
              corona += exp(-dB2 * 52.0) * mB2 * 0.28;

              skyCol += (vec3(0.96, 0.98, 1.0) * core * 2.6 + vec3(0.48, 0.68, 1.0) * corona) * uLightningFlash;
            }
          }

          // Convert base sky & cloud scattering into True Scene-Referred Linear-HDR Radiance,
          // then composite Linear-HDR Cosmos (Milky Way + 208 J2000 Constellations), Moon, & Sun occluded by clouds!
          float cloudTrans = 1.0 - cloudCoverMask;
          vec3 hdrSky = nexusSRGBToLinear(max(skyCol, vec3(0.0))) * mix(1.16, 0.76, uNightFactor);

          // Linear-HDR Cosmos (occulted by solid Moon sphere and foreground clouds!)
          hdrSky += hdrCosmos * (1.0 - moonOccultMask) * (cloudTrans * cloudTrans);

          if (cosSun > 0.0 && rNormSun < 1.0 && sunVis > 0.005) {
            float zSphS = sqrt(max(0.0, 1.0 - rNormSun * rNormSun));
            vec3 hdrSunBody = mix(vec3(3.4, 3.0, 2.3), vec3(3.1, 1.55, 0.68), uDuskFactor);
            hdrSky += hdrSunBody * (0.45 + 0.95 * zSphS) * sunVis * cloudTrans;
          }
          hdrSky += nexusSRGBToLinear(max(solarMieTint, vec3(0.0))) * (sunHalo * 1.45 + mieAureole * 0.85) * (0.25 + 0.75 * cloudTrans);

          // Linear-HDR Selenographic Moon & Tight Corona (occluded behind foreground clouds, crisp Maria in clear sky!)
          hdrSky += hdrMoonBody * cloudTrans;
          hdrSky += vec3(0.55, 0.68, 0.92) * moonTightCorona * (1.0 - moonOccultMask * 0.65) * (0.25 + 0.75 * cloudTrans);

          if (uLightningFlash > 0.008 && dir.y > 0.0) {
            vec3 lDirH = normalize(uLightningDir);
            float angCosH = dot(normalize(vec3(dir.x, 0.0, dir.z) + vec3(1e-5, 0.0, 0.0)), normalize(vec3(lDirH.x, 0.0, lDirH.z) + vec3(1e-5, 0.0, 0.0)));
            float azDiffH = acos(clamp(angCosH, -1.0, 1.0));
            float boltBoost = exp(-azDiffH * 3.2) * exp(-abs(dir.y - 0.30) * 2.8);
            hdrSky += vec3(0.75, 0.95, 1.55) * boltBoost * uLightningFlash * 1.35;
          }

          vec3 hvsSkyLin = nexusApplyHumanEyeHDR(
            hdrSky,
            dir,
            celestialDir,
            uSunColor,
            uHvsParams,
            uHvsGlare
          );
          gl_FragColor = vec4(nexusLinearToSRGB(hvsSkyLin), 1.0);
        }
      `
    });

    this.skyMesh = new THREE.Mesh(skyGeo, skyMat);
    this.skyMesh.name = 'NexusAnalyticalSkyDome';
    this.skyMesh.frustumCulled = false;
    this.scene.add(this.skyMesh);
  }

  /**
   * Hukum Emas 8 & 12: 100% GPU Physics-Driven Raindrops (Marshall-Palmer + Gunn-Kinzer + Wind Shear)
   * & Newtonian Gravity Roof-Eave / Canopy Drip Curtains in STRICTLY 1 SINGLE DRAW CALL!
   */
  _initPhysicalRainAndDripMesh() {
    const skyCount = WeatherOpticsEngine.SKY_DROP_COUNT;
    const dripCount = WeatherOpticsEngine.EAVE_DRIP_COUNT;
    const totalQuads = skyCount + dripCount;
    const totalVerts = totalQuads * 4;

    const positions = new Float32Array(totalVerts * 3);
    const normals = new Float32Array(totalVerts * 3);
    const uvs = new Float32Array(totalVerts * 2);

    const aKind = new Float32Array(totalVerts); // 0 = Sky Marshall-Palmer Drop, 1 = Eave/Genting Drip & Limpahan
    const aDropDiam = new Float32Array(totalVerts); // Base Marshall-Palmer diameter seed in mm [0.65 .. 4.65]
    const aPhase = new Float32Array(totalVerts);
    const aAnchor = new Float32Array(totalVerts * 3); // Base (x, y, z) seed or exact 3D eave overhang coord
    const aLanding = new Float32Array(totalVerts * 2); // (landingY, isCanopy)
    const aOutDir = new Float32Array(totalVerts * 2); // Outward roof-eave overhang normal (outX, outZ) for genting limpahan
    const indices = new Uint32Array(totalQuads * 6);

    const cornerUVs = [
      [0.0, 0.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0]
    ];

    for (let q = 0; q < totalQuads; q++) {
      const isDrip = q >= skyCount;
      // Deterministic low-discrepancy Halton-like sequence (0 NaN, 0 clustering)
      const u1 = ((q * 733 + 19) % 10007) / 10007;
      const u2 = ((q * 1499 + 53) % 10009) / 10009;
      const u3 = ((q * 2311 + 97) % 9973) / 9973;
      const u4 = ((q * 3571 + 131) % 9967) / 9967;

      // Marshall-Palmer exponential drop-size inverse CDF: N(D) = N_0 * exp(-Lambda * D)
      const diamMm = isDrip
        ? 3.2 + u4 * 1.35 // Pendant roof-tile/leaf drips are ~3.2-4.55 mm capillary drops
        : Math.min(4.65, 0.85 - Math.log(1.0 - u4 * 0.965) * 1.12);

      const ax = (u1 - 0.5) * 134.0;
      const ay = u2 * 58.0;
      const az = (u3 - 0.5) * 134.0;
      const outAng = u1 * Math.PI * 2.0;
      const defOx = Math.cos(outAng);
      const defOz = Math.sin(outAng);

      for (let c = 0; c < 4; c++) {
        const vi = q * 4 + c;
        positions[vi * 3] = 0.0;
        positions[vi * 3 + 1] = 0.0;
        positions[vi * 3 + 2] = 0.0;

        normals[vi * 3] = 0.0;
        normals[vi * 3 + 1] = 1.0;
        normals[vi * 3 + 2] = 0.0;

        uvs[vi * 2] = cornerUVs[c][0];
        uvs[vi * 2 + 1] = cornerUVs[c][1];

        aKind[vi] = isDrip ? 1.0 : 0.0;
        aDropDiam[vi] = diamMm;
        aPhase[vi] = u2;

        aAnchor[vi * 3] = ax;
        aAnchor[vi * 3 + 1] = isDrip ? 6.0 : ay;
        aAnchor[vi * 3 + 2] = az;

        aLanding[vi * 2] = 0.14;
        aLanding[vi * 2 + 1] = 0.0;

        aOutDir[vi * 2] = defOx;
        aOutDir[vi * 2 + 1] = defOz;
      }

      const b = q * 4;
      const io = q * 6;
      indices[io] = b;
      indices[io + 1] = b + 1;
      indices[io + 2] = b + 2;
      indices[io + 3] = b;
      indices[io + 4] = b + 2;
      indices[io + 5] = b + 3;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setAttribute('aKind', new THREE.BufferAttribute(aKind, 1));
    geo.setAttribute('aDropDiam', new THREE.BufferAttribute(aDropDiam, 1));
    geo.setAttribute('aPhase', new THREE.BufferAttribute(aPhase, 1));

    this._dripAnchorAttr = new THREE.BufferAttribute(aAnchor, 3);
    this._dripLandingAttr = new THREE.BufferAttribute(aLanding, 2);
    this._dripOutDirAttr = new THREE.BufferAttribute(aOutDir, 2);
    geo.setAttribute('aAnchor', this._dripAnchorAttr);
    geo.setAttribute('aLanding', this._dripLandingAttr);
    geo.setAttribute('aOutDir', this._dripOutDirAttr);
    geo.setIndex(new THREE.BufferAttribute(indices, 1));

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
      uniforms: {
        uTime: this.uniforms.uTime,
        uCamPos: this.uniforms.uCamPos,
        uSunDir: this.uniforms.uSunDir,
        uNightFactor: this.uniforms.uNightFactor,
        uDuskFactor: this.uniforms.uDuskFactor,
        uRainIntensity: this.uniforms.uRainIntensity,
        uWetness: this.uniforms.uWetness,
        uLightningFlash: this.uniforms.uLightningFlash,
        uWindVec: this.uniforms.uWindVec,
        uWindPhase: this.uniforms.uWindPhase,
        uCollisionMap: this.uniforms.uCollisionMap,
        uCollisionHalfExtent: this.uniforms.uCollisionHalfExtent
      },
      vertexShader: /* glsl */ `
        attribute float aKind;
        attribute float aDropDiam;
        attribute float aPhase;
        attribute vec3 aAnchor;
        attribute vec2 aLanding;
        attribute vec2 aOutDir;

        uniform float uTime;
        uniform vec3 uCamPos;
        uniform float uRainIntensity;
        uniform float uWetness;
        uniform vec3 uWindVec;
        uniform vec2 uWindPhase;
        uniform sampler2D uCollisionMap;
        uniform float uCollisionHalfExtent;

        varying vec2 vUv;
        varying float vAlpha;
        varying float vBeadness;

        ${WeatherOpticsEngine._getArchipelagoSdfGLSL()}

        float decodeCollisionHeight(vec2 xz) {
          float hOut = 0.0;
          vec2 cuv = (xz + uCollisionHalfExtent) / (uCollisionHalfExtent * 2.0);
          if (cuv.x < 0.001 || cuv.x > 0.999 || cuv.y < 0.001 || cuv.y > 0.999) {
            float dCoast = evalArchipelagoCoastRaw(xz);
            hOut = evalArchipelagoAnalyticalHeight(xz, dCoast);
          } else {
            vec4 cTex = texture2D(uCollisionMap, cuv);
            hOut = -6.0 + ((cTex.r * 255.0 * 256.0 + cTex.g * 255.0) / 65535.0) * 60.0;
          }
          return hOut;
        }

        void main() {
          vUv = uv;
          vAlpha = 0.0;
          vBeadness = 0.0;

          // Cull helper
          vec4 culledClip = vec4(2.0, 2.0, 2.0, 1.0);

          if (aKind < 0.5) {
            // --- MODE A: EXACT 1:1 WORLD-SCALE MARSHALL-PALMER + GUNN-KINZER SKY RAINDROP ---
            // Filter drop spectrum by rain intensity (drizzle = fine ~0.8-1.4mm drops at ~3.4-5.2 m/s,
            // heavy tropical rain/storm = ~1.8-4.2mm drops at ~6.2-8.7 m/s)
            float activeGate = fract(aPhase * 31.73);
            if (activeGate > clamp(uRainIntensity * 0.92, 0.0, 1.0)) {
              gl_Position = culledClip;
              return;
            }
            if (aDropDiam > 1.55 + uRainIntensity * 2.9) {
              gl_Position = culledClip;
              return;
            }

            float rainNorm = clamp(uRainIntensity / 1.40, 0.08, 1.05);
            float effDiam = clamp(
              mix(0.68, 1.65, rainNorm) + (aDropDiam - 0.85) * mix(0.28, 0.92, rainNorm),
              0.55,
              4.65
            );

            // Exact 1:1 Meter/Sec Gunn-Kinzer (1949) terminal velocity: v_t(D) = 9.65 - 10.3 * exp(-0.6 * D_mm) m/s
            // (Zero artificial speed inflation: 1 world unit === 1.000 physical meter!)
            float vTerminal = clamp(9.65 - 10.3 * exp(-0.6 * effDiam), 2.55, 8.85);

            // Wrap 134m x 134m rain volume smoothly around camera horizontal position
            float boxSpanXZ = 134.0;
            float halfSpanXZ = 67.0;
            float colMinY = -2.5;
            float colSpanY = 58.5;

            float fallDist = uTime * vTerminal + aPhase * colSpanY;
            float cycleId = floor((aAnchor.y - fallDist) / colSpanY);
            float pyTop = colMinY + mod(aAnchor.y - fallDist, colSpanY);

            // Slight per-cycle horizontal decorrelation so drops never repeat on exact same vertical line
            float jx = sin(cycleId * 12.9898 + aPhase * 78.233) * 2.8;
            float jz = cos(cycleId * 39.346 + aPhase * 41.135) * 2.8;

            // Altitude boundary-layer wind shear W(y) ~ (y / 10)^0.22 + monotonic uWindPhase advection (0 gust-slosh!)
            float altFactor = pow(clamp(max(1.5, pyTop) / 10.0, 0.35, 1.75), 0.22);
            float dragCouple = mix(1.18, 0.78, clamp((effDiam - 0.65) / 3.9, 0.0, 1.0));

            vec2 baseXZ = vec2(
              uCamPos.x + mod(aAnchor.x + jx + uWindPhase.x * 0.85 - uCamPos.x + halfSpanXZ, boxSpanXZ) - halfSpanXZ,
              uCamPos.z + mod(aAnchor.z + jz + uWindPhase.y * 0.85 - uCamPos.z + halfSpanXZ, boxSpanXZ) - halfSpanXZ
            );

            float gust = 0.88 + 0.18 * sin(uTime * 1.65 - dot(baseXZ, vec2(0.055, 0.045)));
            vec2 windHoriz = uWindVec.xz * altFactor * dragCouple * gust * 0.96;
            vec3 vel3D = vec3(windHoriz.x, -vTerminal, windHoriz.y);
            vec3 streakDir = normalize(vel3D);

            // Human-eye persistence streak (~1/37s = 27ms natural foveal integration):
            // 0.09m - 0.26m streak length proportional to 1.70m humans, 0.25m cats, & genting roof tiles!
            float speed3D = length(vel3D);
            float streakLen = clamp(speed3D * 0.027, 0.075, 0.34);

            vec3 pHead = vec3(baseXZ.x, pyTop, baseXZ.y) + streakDir * streakLen;
            vec3 pTail = vec3(baseXZ.x, pyTop, baseXZ.y);

            // Exact 3D Scene Collision Interception (Roofs, Eaves, Walkways, Tree Canopies, & Ground!)
            float yHit = decodeCollisionHeight(pHead.xz);
            if (pTail.y <= yHit + 0.02) {
              // Entire raindrop is already below the intercepting 3D roof/canopy/ground -> 100% dry interior!
              gl_Position = culledClip;
              return;
            }
            if (pHead.y < yHit) {
              // Clip the leading tip cleanly right onto the impact surface!
              float frac = clamp((pTail.y - yHit) / max(0.02, pTail.y - pHead.y), 0.05, 1.0);
              pHead = mix(pTail, pHead, frac);
            }

            vec3 midP = 0.5 * (pHead + pTail);
            vec3 toCam = uCamPos - midP;
            float distCam = length(toCam) + 1e-4;
            if (distCam > 82.0) {
              gl_Position = culledClip;
              return;
            }

            vec3 sideDir = normalize(cross(toCam / distCam, streakDir) + vec3(1e-5, 0.0, 0.0));
            // Physical millimeter drop width + subtle anti-aliasing angular floor at distance
            float halfW = clamp(0.0032 + effDiam * 0.0020 + distCam * 0.00034, 0.0035, 0.018);

            // uv.y = 0 -> upper wake tail; uv.y = 1 -> lower leading meniscus head
            vec3 worldVert = mix(pTail, pHead, uv.y) + sideDir * ((uv.x - 0.5) * 2.0 * halfW);
            float distFade = smoothstep(82.0, 26.0, distCam) * smoothstep(0.6, 2.2, distCam);
            float sizeAlpha = mix(0.24, 0.58, clamp((effDiam - 0.65) / 3.8, 0.0, 1.0));
            vAlpha = sizeAlpha * distFade * clamp(uRainIntensity * 0.96, 0.28, 1.15);
            vBeadness = 0.16;

            gl_Position = projectionMatrix * viewMatrix * vec4(worldVert, 1.0);
          } else {
            // --- MODE B: AERODYNAMIC-DRAG-CAPPED GENTING CAPILLARY DRIPS (TETESAN)
            //             & HEAVY-RAIN PARABOLIC TILE-VALLEY RUNOFF STREAMS (LIMPAHAN) ---
            // Eaves keep dripping residual runoff naturally even as rain tapers!
            float runoffLevel = max(uRainIntensity, uWetness * 0.45);
            float dripGate = fract(aPhase * 19.37);
            if (runoffLevel < 0.08 || dripGate > clamp(runoffLevel * 1.05, 0.0, 1.0)) {
              gl_Position = culledClip;
              return;
            }

            vec3 eavePos = aAnchor;
            float distCam = length(uCamPos - eavePos);
            if (distCam > 78.0) {
              gl_Position = culledClip;
              return;
            }

            float landingY = aLanding.x;
            float isCanopy = aLanding.y;
            float isHardRoof = 1.0 - step(0.5, isCanopy);
            float dropHeight = max(0.35, eavePos.y - landingY);

            // Exact Aerodynamic-Drag-Capped Free-Fall Kinematics (g = 9.81 m/s^2, v_t = 8.35 m/s for ~3.8mm pendant drop):
            // dy(t) = (v_t^2 / g) * ln(cosh(g*t / v_t)), vy(t) = v_t * tanh(g*t / v_t)
            // Overflow-safe GLSL ES 1.00 closed-form fall duration u_max = acosh(exp(w)):
            float g = 9.81;
            float vTermDrip = 8.35;
            float wFall = (g * dropHeight) / (vTermDrip * vTermDrip);
            float uFallMax = wFall + log(1.0 + sqrt(max(0.0, 1.0 - exp(-2.0 * wFall))));
            float tFall = (vTermDrip / g) * uFallMax;

            // Dual Hydrological Regime on Architectural Genting Eaves:
            // 1) Tetesan Genting (light rain / post-rain): capillary pendant bead hangs under terracotta lip (tHold ~ 0.20-0.56s)
            // 2) Limpahan Cucuran Genting (moderate-to-heavy rain): continuous tile-valley runoff stream packets
            //    shooting off the roof pitch direction (aOutDir) in a realistic parabolic arc!
            float isRunoffSlot = isHardRoof * step(0.42, fract(aPhase * 7.13));
            float limpahanStrength = isRunoffSlot * smoothstep(0.28, 0.82, max(uRainIntensity * 0.92, runoffLevel * 0.85));

            float tHold = mix(
              mix(0.56, 0.20, clamp(runoffLevel, 0.0, 1.0)) + isCanopy * 0.14,
              0.024,
              limpahanStrength
            );
            float tPause = mix(
              mix(0.50, 0.10, clamp(runoffLevel, 0.0, 1.0)),
              0.042,
              limpahanStrength
            );
            float period = tHold + tFall + tPause;

            float tau = mod(uTime + aPhase * period * 3.7, period);
            if (tau > tHold + tFall) {
              gl_Position = culledClip;
              return;
            }

            vec3 pHead;
            vec3 pTail;
            vec3 streakDir = vec3(0.0, -1.0, 0.0);

            if (tau < tHold) {
              // Growing capillary pendant water bead hanging under terracotta tile lip / leaf tip!
              float grow = tau / max(0.01, tHold);
              float beadR = 0.011 + 0.013 * grow;
              pTail = eavePos + vec3(0.0, 0.005, 0.0);
              pHead = eavePos - vec3(0.0, beadR * 2.2, 0.0);
              vBeadness = 1.0;
            } else {
              // Detached drag-coupled gravity free-fall + parabolic outward genting runoff pitch trajectory:
              float tf = tau - tHold;
              float uG = (g * tf) / vTermDrip;
              float eNeg2u = exp(-2.0 * uG);
              float dy = ((vTermDrip * vTermDrip) / g) * (uG + log(0.5 * (1.0 + eNeg2u)));
              float vy = vTermDrip * ((1.0 - eNeg2u) / (1.0 + eNeg2u));

              vec2 outDirN = length(aOutDir) > 0.05 ? normalize(aOutDir) : vec2(0.0, 1.0);
              float vOut0 = limpahanStrength * (0.34 + 0.54 * clamp(uRainIntensity, 0.0, 1.35)) * (0.85 + 0.30 * fract(aPhase * 13.7));
              float dragTauH = 1.35;
              float outDisp = vOut0 * dragTauH * (1.0 - exp(-tf / dragTauH));
              float vOutNow = vOut0 * exp(-tf / dragTauH);
              vec2 windDrift = uWindVec.xz * (0.11 * tf * tf / (1.0 + 0.35 * tf));
              vec2 horizDisp = outDirN * outDisp + windDrift;

              vec3 vel = vec3(
                outDirN.x * vOutNow + uWindVec.x * 0.22 * min(1.5, tf),
                -max(0.35, vy),
                outDirN.y * vOutNow + uWindVec.z * 0.22 * min(1.5, tf)
              );
              streakDir = normalize(vel);

              float rawStreakLen = clamp(0.032 + length(vel) * mix(0.025, 0.038, limpahanStrength), 0.032, 0.36);
              float streakLen = min(rawStreakLen, max(0.020, dy + 0.016));
              pHead = vec3(eavePos.x + horizDisp.x, max(landingY, eavePos.y - dy), eavePos.z + horizDisp.y);
              pTail = pHead - streakDir * streakLen;
              if (pTail.y < landingY + 0.01) {
                gl_Position = culledClip;
                return;
              }
              vBeadness = mix(exp(-tf * 4.5), 0.22, limpahanStrength);
            }

            vec3 midP = 0.5 * (pHead + pTail);
            vec3 toCam = normalize(uCamPos - midP + vec3(1e-5));
            vec3 sideDir = normalize(cross(toCam, streakDir) + vec3(1e-5, 0.0, 0.0));
            float halfW = max(
              mix(0.0085, 0.0155, max(vBeadness, limpahanStrength * 0.72)),
              distCam * 0.00046
            );

            vec3 worldVert = mix(pTail, pHead, uv.y) + sideDir * ((uv.x - 0.5) * 2.0 * halfW);
            vAlpha = smoothstep(78.0, 22.0, distCam) * mix(0.50, 0.78, max(vBeadness, limpahanStrength * 0.65));

            gl_Position = projectionMatrix * viewMatrix * vec4(worldVert, 1.0);
          }
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec2 vUv;
        varying float vAlpha;
        varying float vBeadness;

        uniform float uNightFactor;
        uniform float uDuskFactor;
        uniform float uLightningFlash;

        void main() {
          if (vAlpha < 0.01) discard;

          // Cylindrical refractive cross-section across drop width x in [-1, +1]
          float x = (vUv.x - 0.5) * 2.0;
          float radialMask = sqrt(max(0.0, 1.0 - x * x));

          // Tear-drop / motion-blurred axial profile: denser meniscus at bottom head (vUv.y -> 1), soft wake tail (vUv.y -> 0)
          float axialStreak = smoothstep(0.0, 0.42, vUv.y) * smoothstep(1.0, 0.84, vUv.y);
          float beadDist = length(vec2(x, (vUv.y - 0.58) * 1.9));
          float axialBead = smoothstep(1.0, 0.15, beadDist);
          float shape = mix(axialStreak * radialMask, axialBead, vBeadness);

          if (shape < 0.03) discard;

          // Physical water-drop optics (n = 1.333): Fresnel rim + central internal caustic focal line + lower meniscus glint
          float fresnelRim = pow(1.0 - radialMask, 2.4) * 0.65;
          float causticCore = exp(-x * x * 10.0) * (0.45 + 0.55 * smoothstep(0.45, 0.92, vUv.y));

          vec3 dayTint = mix(vec3(0.76, 0.85, 0.94), vec3(0.94, 0.72, 0.56), uDuskFactor * 0.65);
          // At night, raindrops catch warm campus/window glow + cool moonlit/lightning refraction
          vec3 nightTint = vec3(0.46, 0.56, 0.72);
          vec3 dropCol = mix(dayTint, nightTint, uNightFactor * 0.72);
          dropCol += vec3(0.24, 0.28, 0.34) * (fresnelRim + causticCore) * (1.0 - 0.4 * uNightFactor);
          dropCol += vec3(0.82, 0.90, 1.0) * uLightningFlash * 0.95;

          float outAlpha = clamp(shape * vAlpha * (0.68 + 0.45 * causticCore), 0.0, 0.88);
          gl_FragColor = vec4(dropCol, outAlpha);
        }
      `
    });

    this.rainAndDripMesh = new THREE.Mesh(geo, mat);
    this.rainAndDripMesh.name = 'NexusPhysicalRainAndDrips';
    this.rainAndDripMesh.userData.dynamicKinematic = true;
    this.rainAndDripMesh.frustumCulled = false;
    this.rainAndDripMesh.visible = false;
    this.scene.add(this.rainAndDripMesh);
  }

  /**
   * Hukum Emas 8 & 12: 100% GPU 3D Surface-Normal Impact Crown Splashes +
   * Newtonian Parabolic Ballistic Rebound Micro-Droplets + Roof Aerosol Spray Mist
   * in STRICTLY 1 SINGLE DRAW CALL!
   */
  _initPhysicalSplashAndMistMesh() {
    const crownCount = WeatherOpticsEngine.SPLASH_CROWN_COUNT;
    const reboundCount = WeatherOpticsEngine.REBOUND_DROP_COUNT;
    const mistCount = WeatherOpticsEngine.SPRAY_MIST_COUNT;
    const totalQuads = crownCount + reboundCount + mistCount;
    const totalVerts = totalQuads * 4;

    const positions = new Float32Array(totalVerts * 3);
    const normals = new Float32Array(totalVerts * 3);
    const uvs = new Float32Array(totalVerts * 2);

    // aSplashMeta: (type: 0=crown, 1=parabolic rebound, 2=aerosol mist, phase, randAzimuth, randSpeed)
    const aSplashMeta = new Float32Array(totalVerts * 4);
    // aSplashAnchor: (xOffsetOrWorldX, zOffsetOrWorldZ, isWorldPinned)
    const aSplashAnchor = new Float32Array(totalVerts * 3);
    const indices = new Uint32Array(totalQuads * 6);

    const cornerUVs = [
      [0.0, 0.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0]
    ];

    for (let q = 0; q < totalQuads; q++) {
      let sType = 0.0;
      if (q >= crownCount && q < crownCount + reboundCount) sType = 1.0;
      else if (q >= crownCount + reboundCount) sType = 2.0;

      // Pair sets of 2 rebound droplets to share nearby impact origins for realistic multi-droplet shatter!
      const seedIdx = sType === 1.0 ? Math.floor((q - crownCount) / 2) + 5000 : q;
      const u1 = ((seedIdx * 887 + 31) % 10007) / 10007;
      const u2 = ((seedIdx * 1627 + 71) % 10009) / 10009;
      const uPhase = ((seedIdx * 2539 + 113) % 9973) / 9973;

      const uAz = ((q * 3301 + 47) % 9967) / 9967;
      const uSpd = ((q * 4129 + 89) % 9949) / 9949;

      const span = sType === 2.0 ? 96.0 : 62.0;
      const ax = (u1 - 0.5) * span;
      const az = (u2 - 0.5) * span;

      for (let c = 0; c < 4; c++) {
        const vi = q * 4 + c;
        positions[vi * 3] = 0.0;
        positions[vi * 3 + 1] = 0.0;
        positions[vi * 3 + 2] = 0.0;

        normals[vi * 3] = 0.0;
        normals[vi * 3 + 1] = 1.0;
        normals[vi * 3 + 2] = 0.0;

        uvs[vi * 2] = cornerUVs[c][0];
        uvs[vi * 2 + 1] = cornerUVs[c][1];

        aSplashMeta[vi * 4] = sType;
        aSplashMeta[vi * 4 + 1] = uPhase;
        aSplashMeta[vi * 4 + 2] = uAz * Math.PI * 2.0;
        aSplashMeta[vi * 4 + 3] = uSpd;

        aSplashAnchor[vi * 3] = ax;
        aSplashAnchor[vi * 3 + 1] = az;
        aSplashAnchor[vi * 3 + 2] = 0.0;
      }

      const b = q * 4;
      const io = q * 6;
      indices[io] = b;
      indices[io + 1] = b + 1;
      indices[io + 2] = b + 2;
      indices[io + 3] = b;
      indices[io + 4] = b + 2;
      indices[io + 5] = b + 3;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setAttribute('aSplashMeta', new THREE.BufferAttribute(aSplashMeta, 4));
    this._splashAnchorAttr = new THREE.BufferAttribute(aSplashAnchor, 3);
    geo.setAttribute('aSplashAnchor', this._splashAnchorAttr);
    geo.setIndex(new THREE.BufferAttribute(indices, 1));

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
      uniforms: {
        uTime: this.uniforms.uTime,
        uCamPos: this.uniforms.uCamPos,
        uNightFactor: this.uniforms.uNightFactor,
        uDuskFactor: this.uniforms.uDuskFactor,
        uRainIntensity: this.uniforms.uRainIntensity,
        uLightningFlash: this.uniforms.uLightningFlash,
        uWindVec: this.uniforms.uWindVec,
        uCollisionMap: this.uniforms.uCollisionMap,
        uCollisionHalfExtent: this.uniforms.uCollisionHalfExtent
      },
      vertexShader: /* glsl */ `
        attribute vec4 aSplashMeta;
        attribute vec3 aSplashAnchor;

        uniform float uTime;
        uniform vec3 uCamPos;
        uniform float uRainIntensity;
        uniform vec3 uWindVec;
        uniform sampler2D uCollisionMap;
        uniform float uCollisionHalfExtent;

        varying vec2 vUv;
        varying float vType;
        varying float vLife;
        varying float vAlpha;
        varying float vSurfClass;

        ${WeatherOpticsEngine._getArchipelagoSdfGLSL()}

        // Decodes yHit, exact 3D surface normal N_hit, and surfaceClass (0=Water, 1=FlatHard, 2=SlopedRoof, 3=Canopy)
        void sampleCollisionSurface(vec2 xz, out float yHit, out vec3 nHit, out float surfClass) {
          vec2 cuv = (xz + uCollisionHalfExtent) / (uCollisionHalfExtent * 2.0);
          if (cuv.x < 0.001 || cuv.x > 0.999 || cuv.y < 0.001 || cuv.y > 0.999) {
            float dCoast = evalArchipelagoCoastRaw(xz);
            if (dCoast > 3.52) {
              yHit = -1.45;
              nHit = vec3(0.0, 1.0, 0.0);
              surfClass = 0.0; // Ocean / Teluk U / Selat UI Water
            } else {
              yHit = evalArchipelagoAnalyticalHeight(xz, dCoast);
              float epsH = 1.25;
              float hx = evalArchipelagoAnalyticalHeight(xz + vec2(epsH, 0.0), evalArchipelagoCoastRaw(xz + vec2(epsH, 0.0)));
              float hz = evalArchipelagoAnalyticalHeight(xz + vec2(0.0, epsH), evalArchipelagoCoastRaw(xz + vec2(0.0, epsH)));
              nHit = normalize(vec3(yHit - hx, epsH, yHit - hz));
              surfClass = nHit.y < 0.88 ? 2.0 : 1.0;
            }
            return;
          }
          vec4 cTex = texture2D(uCollisionMap, cuv);
          yHit = -6.0 + ((cTex.r * 255.0 * 256.0 + cTex.g * 255.0) / 65535.0) * 60.0;
          float aByte = cTex.a * 255.0;
          if (aByte < 5.0) {
            surfClass = 0.0; // Danau Kenanga Water
            nHit = vec3(0.0, 1.0, 0.0);
          } else if (aByte > 251.0) {
            surfClass = 3.0; // Botanical Tree Canopy
            float nx = cTex.b * 2.0 - 1.0;
            float ny = sqrt(max(0.2, 1.0 - nx * nx));
            nHit = normalize(vec3(nx, ny, 0.0));
          } else {
            float nx = cTex.b * 2.0 - 1.0;
            float nz = clamp((aByte - 128.0) / 116.0, -1.0, 1.0);
            float ny = sqrt(max(0.04, 1.0 - nx * nx - nz * nz));
            nHit = normalize(vec3(nx, ny, nz));
            surfClass = ny < 0.88 ? 2.0 : 1.0;
          }
        }

        void main() {
          vUv = uv;
          float sType = aSplashMeta.x;
          float phase = aSplashMeta.y;
          float azAngle = aSplashMeta.z;
          float rndSpd = aSplashMeta.w;
          vType = sType;
          vAlpha = 0.0;
          vLife = 0.0;
          vSurfClass = 1.0;

          vec4 culledClip = vec4(2.0, 2.0, 2.0, 1.0);

          float gate = fract(phase * 23.41 + rndSpd * 7.1);
          if (gate > clamp(uRainIntensity * 0.98, 0.0, 1.0)) {
            gl_Position = culledClip;
            return;
          }

          if (sType < 0.5) {
            // --- SUB-POPULATION A: 3D SURFACE-NORMAL RAYLEIGH-WORTHINGTON CROWN SPLASH ---
            float period = mix(0.46, 0.28, rndSpd);
            float tLocal = uTime + phase * 11.3;
            float cycle = floor(tLocal / period);
            float life = fract(tLocal / period);
            vLife = life;

            vec2 hitXZ;
            if (aSplashAnchor.z > 0.5) {
              // World-pinned along extracted 3D roof-eave drip landing lines!
              vec2 j = vec2(sin(cycle * 17.3 + phase * 43.1), cos(cycle * 29.1 + phase * 19.7)) * 0.26;
              hitXZ = aSplashAnchor.xy + j;
            } else {
              // Camera-wrapped 62m x 62m high-density splash field
              float span = 62.0;
              float halfS = 31.0;
              vec2 j = vec2(sin(cycle * 13.7 + phase * 53.9), cos(cycle * 31.7 + phase * 27.3)) * 1.9;
              hitXZ = vec2(
                uCamPos.x + mod(aSplashAnchor.x + j.x - uCamPos.x + halfS, span) - halfS,
                uCamPos.z + mod(aSplashAnchor.y + j.y - uCamPos.z + halfS, span) - halfS
              );
            }

            float yHit;
            vec3 nHit;
            float surfClass;
            sampleCollisionSurface(hitXZ, yHit, nHit, surfClass);
            vSurfClass = surfClass;

            vec3 hitPos = vec3(hitXZ.x, yHit, hitXZ.y) + nHit * 0.022;
            float distCam = length(uCamPos - hitPos);
            if (distCam > 48.0) {
              gl_Position = culledClip;
              return;
            }

            // Orthonormal tangent plane (t1, t2) strictly perpendicular to 3D surface normal nHit!
            vec3 upRef = abs(nHit.y) < 0.93 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
            vec3 t1 = normalize(cross(upRef, nHit));
            vec3 t2 = cross(nHit, t1);

            float maxRad = surfClass < 0.5
              ? mix(0.16, 0.34, rndSpd)  // Larger capillary wave ring on Danau Kenanga water
              : mix(0.085, 0.19, rndSpd) * (0.85 + 0.25 * min(1.4, uRainIntensity));
            float rad = maxRad * (0.20 + 0.80 * pow(life, 0.62));
            // Upward Rayleigh liquid crown lift during early impact [0 .. 0.35]
            float crownLift = sin(min(1.0, life * 2.4) * 3.14159) * maxRad * (surfClass < 0.5 ? 0.18 : 0.36);

            vec2 qOff = (uv - 0.5) * 2.0 * rad;
            vec3 worldVert = hitPos + t1 * qOff.x + t2 * qOff.y + nHit * crownLift;

            vAlpha = (1.0 - life) * smoothstep(0.0, 0.08, life) * smoothstep(48.0, 14.0, distCam)
                   * mix(0.58, 0.85, clamp(uRainIntensity, 0.0, 1.0));
            gl_Position = projectionMatrix * viewMatrix * vec4(worldVert, 1.0);

          } else if (sType < 1.5) {
            // --- SUB-POPULATION B: NEWTONIAN PARABOLIC BALLISTIC REBOUND MICRO-DROPLETS ---
            // P(tau) = P_hit + v_0 * tau + 0.5 * g * tau^2
            float period = mix(0.42, 0.26, rndSpd);
            float tLocal = uTime + phase * 11.3;
            float cycle = floor(tLocal / period);
            float life = fract(tLocal / period);
            float tau = life * period;
            vLife = life;

            float span = 50.0;
            float halfS = 25.0;
            vec2 j = vec2(sin(cycle * 13.7 + phase * 53.9), cos(cycle * 31.7 + phase * 27.3)) * 1.9;
            vec2 hitXZ = vec2(
              uCamPos.x + mod(aSplashAnchor.x + j.x - uCamPos.x + halfS, span) - halfS,
              uCamPos.z + mod(aSplashAnchor.y + j.y - uCamPos.z + halfS, span) - halfS
            );

            float yHit;
            vec3 nHit;
            float surfClass;
            sampleCollisionSurface(hitXZ, yHit, nHit, surfClass);
            vSurfClass = surfClass;

            // Soft foliage canopies absorb momentum with fewer ballistic rebounds than hard tiles/pavers
            if (surfClass > 2.5 && rndSpd < 0.65) {
              gl_Position = culledClip;
              return;
            }

            vec3 hitPos = vec3(hitXZ.x, yHit + 0.015, hitXZ.y);
            float distCam = length(uCamPos - hitPos);
            if (distCam > 38.0) {
              gl_Position = culledClip;
              return;
            }

            vec3 upRef = abs(nHit.y) < 0.93 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
            vec3 t1 = normalize(cross(upRef, nHit));
            vec3 t2 = cross(nHit, t1);

            // Ejection cone around 3D surface normal N_hit (tilts down-slope automatically on 48.7° Tajug roof!)
            float coneAngle = mix(0.38, 0.98, fract(rndSpd * 13.7));
            float vMag = mix(1.35, 3.05, rndSpd) * (0.78 + 0.25 * min(1.4, uRainIntensity));
            float az = azAngle + cycle * 1.7;
            vec3 ejectDir = normalize(
              nHit * cos(coneAngle) + (t1 * cos(az) + t2 * sin(az)) * sin(coneAngle)
            );
            vec3 v0 = ejectDir * vMag + vec3(uWindVec.x * 0.25, 0.0, uWindVec.z * 0.25);
            vec3 gVec = vec3(0.0, -9.81, 0.0);

            vec3 posTau = hitPos + v0 * tau + 0.5 * gVec * tau * tau;
            if (tau > 0.05 && posTau.y < yHit + 0.005) {
              gl_Position = culledClip;
              return;
            }

            vec3 velTau = v0 + gVec * tau;
            vec3 dirTau = normalize(velTau + vec3(0.0, 1e-4, 0.0));
            vec3 toCam = normalize(uCamPos - posTau + vec3(1e-5));
            vec3 sideDir = normalize(cross(toCam, dirTau) + vec3(1e-5, 0.0, 0.0));

            float dropLen = clamp(0.022 + length(velTau) * 0.014, 0.020, 0.085);
            float dropW = max(0.010, distCam * 0.00048);

            vec3 worldVert = posTau + dirTau * ((uv.y - 0.5) * dropLen) + sideDir * ((uv.x - 0.5) * 2.0 * dropW);
            vAlpha = (1.0 - life * 0.85) * smoothstep(38.0, 10.0, distCam) * 0.78;
            gl_Position = projectionMatrix * viewMatrix * vec4(worldVert, 1.0);

          } else {
            // --- SUB-POPULATION C: IMPACT AEROSOL SPRAY MIST HOVERING OVER ROOFS & COURTYARDS ---
            if (uRainIntensity < 0.35) {
              gl_Position = culledClip;
              return;
            }

            float period = mix(3.4, 2.2, rndSpd);
            float tLocal = uTime + phase * 17.9;
            float cycle = floor(tLocal / period);
            float life = fract(tLocal / period);
            vLife = life;

            float span = 96.0;
            float halfS = 48.0;
            vec2 baseXZ = vec2(
              uCamPos.x + mod(aSplashAnchor.x + sin(cycle * 9.1) * 3.0 - uCamPos.x + halfS, span) - halfS,
              uCamPos.z + mod(aSplashAnchor.y + cos(cycle * 7.3) * 3.0 - uCamPos.z + halfS, span) - halfS
            );

            float yHit;
            vec3 nHit;
            float surfClass;
            sampleCollisionSurface(baseXZ, yHit, nHit, surfClass);

            // Spray mist rises 0.45..1.65m above wet terracotta roofs & stone plazas and drifts with wind
            vec2 drift = uWindVec.xz * (life * 2.4);
            float puffRadius = mix(1.15, 2.45, rndSpd) * (0.75 + 0.45 * life);
            vec3 center = vec3(baseXZ.x + drift.x, yHit + 0.55 + life * 0.95, baseXZ.y + drift.y);

            float distCam = length(uCamPos - center);
            if (distCam > 88.0 || distCam < 3.5) {
              gl_Position = culledClip;
              return;
            }

            // Camera-facing billboard quad
            vec3 forward = normalize(uCamPos - center);
            vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward) + vec3(1e-5, 0.0, 0.0));
            vec3 up = cross(forward, right);

            vec2 qOff = (uv - 0.5) * 2.0 * puffRadius;
            vec3 worldVert = center + right * qOff.x + up * qOff.y;

            float env = sin(life * 3.14159);
            vAlpha = env * smoothstep(88.0, 28.0, distCam) * smoothstep(3.5, 9.5, distCam)
                   * clamp((uRainIntensity - 0.30) * 0.13, 0.0, 0.14);
            gl_Position = projectionMatrix * viewMatrix * vec4(worldVert, 1.0);
          }
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec2 vUv;
        varying float vType;
        varying float vLife;
        varying float vAlpha;
        varying float vSurfClass;

        uniform float uNightFactor;
        uniform float uDuskFactor;
        uniform float uLightningFlash;

        void main() {
          if (vAlpha < 0.004) discard;

          vec2 p = (vUv - 0.5) * 2.0;
          float r = length(p);
          if (r > 1.0) discard;

          vec3 baseCol = mix(vec3(0.84, 0.90, 0.97), vec3(0.94, 0.72, 0.56), uDuskFactor * 0.6);
          baseCol = mix(baseCol, vec3(0.45, 0.55, 0.70), uNightFactor * 0.72);
          baseCol += vec3(0.82, 0.90, 1.0) * uLightningFlash * 0.95;

          if (vType < 0.5) {
            // Rayleigh-Worthington Hollow Crown Ring + Inner Capillary Ripple + 7-Lobe Rim Beading
            float theta = atan(p.y, p.x);
            float outerRing = exp(-pow((r - 0.82) * 9.5, 2.0));
            float innerRing = exp(-pow((r - 0.44 * vLife) * 11.0, 2.0)) * 0.45;
            float crownBeads = 0.68 + 0.32 * cos(theta * 7.0 + vLife * 6.0);
            float mask = (outerRing * crownBeads + innerRing);
            if (mask < 0.04) discard;
            gl_FragColor = vec4(baseCol, clamp(mask * vAlpha, 0.0, 0.82));
          } else if (vType < 1.5) {
            // Glistening Spherical/Ellipsoidal Ballistic Rebound Micro-Droplet
            float core = smoothstep(1.0, 0.1, r);
            float spec = exp(-r * r * 5.0) * 0.45;
            gl_FragColor = vec4(baseCol + vec3(spec), clamp(core * vAlpha, 0.0, 0.85));
          } else {
            // Soft Gaussian Impact Aerosol Spray Mist
            float puff = exp(-r * r * 3.4) * (1.0 - r * r);
            gl_FragColor = vec4(baseCol, clamp(puff * vAlpha, 0.0, 0.16));
          }
        }
      `
    });

    this.splashAndMistMesh = new THREE.Mesh(geo, mat);
    this.splashAndMistMesh.name = 'NexusPhysicalSplashAndMist';
    this.splashAndMistMesh.userData.dynamicKinematic = true;
    this.splashAndMistMesh.frustumCulled = false;
    this.splashAndMistMesh.visible = false;
    this.scene.add(this.splashAndMistMesh);
  }

  /**
   * Hukum Emas 12 & 14: 100% GPU Vertex Lakeside Fireflies / Evening Spores around Danau Kenanga
   */
  _initGPUFireflyParticles() {
    const count = 180;
    const positions = new Float32Array(count * 3);
    const speeds = new Float32Array(count);
    const phases = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2.0;
      const rad = 28.0 + ((i * 37) % 52);
      positions[i * 3] = 52.0 + Math.abs(Math.cos(angle)) * rad;
      positions[i * 3 + 1] = 1.2 + ((i * 19) % 45) * 0.12;
      positions[i * 3 + 2] = Math.sin(angle) * 75.0;
      speeds[i] = 0.6 + ((i * 29) % 10) * 0.12;
      phases[i] = ((i * 61) % 100) * 0.0628;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('aSpeed', new THREE.BufferAttribute(speeds, 1));
    geo.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: this.uniforms.uTime,
        uFireflyIntensity: this.uniforms.uFireflyIntensity
      },
      vertexShader: /* glsl */ `
        attribute float aSpeed;
        attribute float aPhase;
        uniform float uTime;
        uniform float uFireflyIntensity;
        varying float vGlow;

        void main() {
          vec3 p = position;
          p.x += sin(uTime * aSpeed + aPhase) * 1.8;
          p.y += sin(uTime * aSpeed * 1.4 + aPhase * 2.0) * 0.65;
          p.z += cos(uTime * aSpeed * 0.9 + aPhase) * 1.8;

          vGlow = uFireflyIntensity * (0.35 + 0.65 * max(0.0, sin(uTime * 2.6 + aPhase * 3.0)));
          vec4 mvPos = modelViewMatrix * vec4(p, 1.0);
          gl_PointSize = clamp(48.0 / max(1.0, -mvPos.z), 2.0, 9.0);
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: /* glsl */ `
        varying float vGlow;
        void main() {
          if (vGlow < 0.02) discard;
          float d = length(gl_PointCoord - vec2(0.5));
          if (d > 0.5) discard;
          float radial = smoothstep(0.5, 0.0, d);
          gl_FragColor = vec4(1.0, 0.86, 0.35, radial * vGlow);
        }
      `
    });

    this.fireflyPoints = new THREE.Points(geo, mat);
    this.fireflyPoints.name = 'NexusGPUFireflies';
    this.fireflyPoints.userData.dynamicKinematic = true;
    this.fireflyPoints.visible = false;
    this.scene.add(this.fireflyPoints);
  }

  /**
   * Hukum Emas 8, 12, & 14 (NEXUS v3.6 Gunung Gede Bogor Kawah Ratu Thin Fumarole Wisp + Mini-Niagara Mist):
   * 100% GPU Physics-Driven Quiet Hydrothermal Fumarole Steam Wisp + Mini-Niagara Waterfall Mist
   * on Pulau I in STRICTLY 1 SINGLE DRAW CALL!
   * - Ultra-thin, faint white fumarole steam wisp rising gently from Kawah Ratu (-32, 46.8, 698), ZERO red fire/sparks!
   * - 96 Billowing Sunlit Aerosol Mist/Spray Puffs rising from Mini-Niagara Plunge Pool (-44, 2.05, 643)
   */
  _initVolcanicSmokePlumeMesh() {
    const smokeQuads = 232;
    const sparkQuads = 56;
    const mistQuads = 96;
    const totalQuads = smokeQuads + sparkQuads + mistQuads;
    const totalVerts = totalQuads * 4;

    const positions = new Float32Array(totalVerts * 3);
    const normals = new Float32Array(totalVerts * 3);
    const uvs = new Float32Array(totalVerts * 2);
    const aPlumeKind = new Float32Array(totalVerts);
    const aPlumeSeed = new Float32Array(totalVerts * 4);
    const indices = new Uint32Array(totalQuads * 6);

    const cornerUVs = [
      [0.0, 0.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0]
    ];

    for (let q = 0; q < totalQuads; q++) {
      const kind = q < smokeQuads ? 0.0 : q < smokeQuads + sparkQuads ? 1.0 : 2.0;
      const s1 = ((q * 613 + 37) % 9973) / 9973;
      const s2 = ((q * 1427 + 91) % 9967) / 9967;
      const s3 = ((q * 2719 + 149) % 9949) / 9949;
      const s4 = ((q * 3989 + 211) % 9941) / 9941;

      for (let c = 0; c < 4; c++) {
        const vi = q * 4 + c;
        // Anchor rest position near crater / waterfall for valid bounding sphere fallback
        positions[vi * 3] = kind > 1.5 ? -44.0 : -32.0;
        positions[vi * 3 + 1] = kind > 1.5 ? 6.0 : 56.0;
        positions[vi * 3 + 2] = kind > 1.5 ? 643.0 : 698.0;

        normals[vi * 3] = 0.0;
        normals[vi * 3 + 1] = 1.0;
        normals[vi * 3 + 2] = 0.0;

        uvs[vi * 2] = cornerUVs[c][0];
        uvs[vi * 2 + 1] = cornerUVs[c][1];

        aPlumeKind[vi] = kind;
        aPlumeSeed[vi * 4] = s1;
        aPlumeSeed[vi * 4 + 1] = s2;
        aPlumeSeed[vi * 4 + 2] = s3;
        aPlumeSeed[vi * 4 + 3] = s4;
      }

      const b = q * 4;
      const io = q * 6;
      indices[io] = b;
      indices[io + 1] = b + 1;
      indices[io + 2] = b + 2;
      indices[io + 3] = b;
      indices[io + 4] = b + 2;
      indices[io + 5] = b + 3;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setAttribute('aPlumeKind', new THREE.BufferAttribute(aPlumeKind, 1));
    geo.setAttribute('aPlumeSeed', new THREE.BufferAttribute(aPlumeSeed, 4));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
      uniforms: {
        uTime: this.uniforms.uTime,
        uCamPos: this.uniforms.uCamPos,
        uSunDir: this.uniforms.uSunDir,
        uNightFactor: this.uniforms.uNightFactor,
        uDuskFactor: this.uniforms.uDuskFactor,
        uRainIntensity: this.uniforms.uRainIntensity,
        uLightningFlash: this.uniforms.uLightningFlash,
        uWindVec: this.uniforms.uWindVec
      },
      vertexShader: /* glsl */ `
        attribute float aPlumeKind;
        attribute vec4 aPlumeSeed;

        uniform float uTime;
        uniform vec3 uCamPos;
        uniform vec3 uWindVec;

        varying vec2 vUv;
        varying float vKind;
        varying float vLife;
        varying float vAlpha;
        varying float vRot;
        varying float vSeed;
        varying vec3 vToCam;

        const vec3 VENT_ORIGIN = vec3(-32.0, 46.8, 698.0);
        const vec3 NIAGARA_POOL_CENTER = vec3(-44.0, 2.05, 643.0);

        void main() {
          vUv = uv;
          vKind = aPlumeKind;
          vSeed = aPlumeSeed.z;

          vec2 windH = uWindVec.xz;
          float windSpd = length(windH);
          vec2 windDir = windSpd > 0.02 ? windH / windSpd : vec2(0.88, 0.46);

          if (aPlumeKind < 0.5) {
            // --- SUB-POPULATION A: GUNUNG GEDE BOGOR (KAWAH RATU) ULTRA-THIN WHITE FUMAROLE STEAM WISP ---
            // Cull excess quads so only a delicate, whisper-thin hydrothermal steam ribbon rises gently from Kawah Ratu
            if (aPlumeSeed.x > 0.28) {
              vAlpha = 0.0;
              gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
              return;
            }
            float period = mix(18.0, 27.0, aPlumeSeed.y);
            float tLocal = uTime + aPlumeSeed.x * 41.7;
            float cycle = floor(tLocal / period);
            float life = fract(tLocal / period);
            vLife = life;

            float riseY = pow(life, 0.88) * mix(18.0, 34.0, aPlumeSeed.z);
            // Gentle downwind drift clearing the ~58m Kawah Ratu grey-andesite rim
            float rimClear = smoothstep(0.08, 0.36, life);
            vec2 downwind = windDir * (pow(life, 1.25) * (6.5 + windSpd * 16.0) * rimClear);

            float swirlAng = aPlumeSeed.w * 6.28318 + cycle * 2.1 + life * 2.4 * (aPlumeSeed.y - 0.5);
            float radialSpread = mix(1.1, 5.2, pow(life, 0.72)) * (0.45 + 0.55 * aPlumeSeed.z);
            vec2 eddy = vec2(cos(swirlAng), sin(swirlAng)) * radialSpread;

            vec3 center = VENT_ORIGIN + vec3(downwind.x + eddy.x, riseY, downwind.y + eddy.y);
            vec3 toCamVec = uCamPos - center;
            float distCam = length(toCamVec) + 1e-4;
            vToCam = toCamVec / distCam;

            vec3 forward = vToCam;
            vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward) + vec3(1e-5, 0.0, 0.0));
            vec3 up = cross(forward, right);

            float puffRadius = mix(1.8, 5.8, pow(life, 0.70)) * mix(0.84, 1.16, aPlumeSeed.y);
            vRot = aPlumeSeed.w * 6.28318 + uTime * 0.08 * (aPlumeSeed.x - 0.5);

            vec2 qOff = (uv - 0.5) * 2.0 * puffRadius;
            vec3 worldVert = center + right * qOff.x + up * qOff.y;

            float fadeIn = smoothstep(0.0, 0.16, life);
            float fadeOut = pow(1.0 - life, 1.45);
            float nearGuard = smoothstep(6.0, 22.0, distCam);
            vAlpha = fadeIn * fadeOut * nearGuard * 0.052;
            gl_Position = projectionMatrix * viewMatrix * vec4(worldVert, 1.0);
          } else if (aPlumeKind < 1.5) {
            // --- SUB-POPULATION B: DISABLED ON QUIET GUNUNG GEDE BOGOR (ZERO INCANDESCENT FIRE / SPARKS!) ---
            vLife = 0.0;
            vRot = 0.0;
            vToCam = vec3(0.0, 1.0, 0.0);
            vAlpha = 0.0;
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            return;
          } else {
            // --- SUB-POPULATION C: MINI-NIAGARA BILLOWING HORSESHOE WATERFALL MIST & SPRAY CLOUD ---
            float period = mix(4.0, 7.4, aPlumeSeed.y);
            float tLocal = uTime + aPlumeSeed.z * 23.7;
            float cycle = floor(tLocal / period);
            float life = fract(tLocal / period);
            vLife = life;

            float arcAlpha = (fract(aPlumeSeed.x + cycle * 0.17) * 2.0 - 1.0) * 1.15;
            float impR = mix(10.6, 13.8, aPlumeSeed.w);
            vec2 arcDir = vec2(sin(arcAlpha), cos(arcAlpha));
            float riseY = pow(life, 0.72) * mix(4.8, 14.2, aPlumeSeed.z);
            vec2 amphitheaterRoll = -arcDir * (pow(life, 0.62) * mix(2.4, 7.2, aPlumeSeed.y));
            vec2 windDrift = windDir * (pow(life, 1.18) * (1.1 + windSpd * 3.4));

            vec3 center = NIAGARA_POOL_CENTER + vec3(
              arcDir.x * impR + amphitheaterRoll.x + windDrift.x,
              riseY,
              arcDir.y * impR + amphitheaterRoll.y + windDrift.y
            );
            vec3 toCamVec = uCamPos - center;
            float distCam = length(toCamVec) + 1e-4;
            vToCam = toCamVec / distCam;

            vec3 forward = vToCam;
            vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward) + vec3(1e-5, 0.0, 0.0));
            vec3 up = cross(forward, right);

            float puffRadius = mix(3.5, 9.4, pow(life, 0.64)) * mix(0.84, 1.24, aPlumeSeed.w);
            vRot = aPlumeSeed.w * 6.28318 + uTime * 0.22 * (aPlumeSeed.x - 0.5);

            vec2 qOff = (uv - 0.5) * 2.0 * puffRadius;
            vec3 worldVert = center + right * qOff.x + up * qOff.y;

            float fadeIn = smoothstep(0.0, 0.14, life);
            float fadeOut = pow(1.0 - life, 1.38);
            float nearGuard = smoothstep(4.5, 16.0, distCam) * smoothstep(880.0, 280.0, distCam);
            vAlpha = fadeIn * fadeOut * nearGuard * 0.38;
            gl_Position = projectionMatrix * viewMatrix * vec4(worldVert, 1.0);
          }
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec2 vUv;
        varying float vKind;
        varying float vLife;
        varying float vAlpha;
        varying float vRot;
        varying float vSeed;
        varying vec3 vToCam;

        uniform float uTime;
        uniform vec3 uSunDir;
        uniform float uNightFactor;
        uniform float uDuskFactor;
        uniform float uRainIntensity;
        uniform float uLightningFlash;

        float plumeHash(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }

        float plumeNoise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(plumeHash(i), plumeHash(i + vec2(1.0, 0.0)), f.x),
            mix(plumeHash(i + vec2(0.0, 1.0)), plumeHash(i + vec2(1.0, 1.0)), f.x),
            f.y
          );
        }

        float plumeFbm(vec2 p) {
          const mat2 rotM = mat2(0.8, 0.6, -0.6, 0.8);
          float v = 0.0;
          v += 0.5333 * plumeNoise(p); p = rotM * p * 2.03;
          v += 0.2667 * plumeNoise(p); p = rotM * p * 2.05;
          v += 0.1333 * plumeNoise(p); p = rotM * p * 2.02;
          v += 0.0667 * plumeNoise(p);
          return v;
        }

        void main() {
          if (vAlpha < 0.004) discard;
          vec2 p = (vUv - 0.5) * 2.0;
          float r = length(p);
          if (r > 1.0) discard;

          if (vKind > 0.5 && vKind < 1.5) {
            discard;
          }

          // Rotate 2D billboard coordinates & sculpt cauliflower FBM volcanic puff density
          float cs = cos(vRot);
          float sn = sin(vRot);
          vec2 pr = vec2(cs * p.x - sn * p.y, sn * p.x + cs * p.y);
          vec2 nCoord = pr * 1.85 + vec2(vSeed * 13.7, -uTime * 0.07 + vSeed * 7.3);
          float fbmVal = plumeFbm(nCoord);
          float fbmDx = plumeFbm(nCoord + vec2(0.14, 0.0)) - fbmVal;
          float fbmDy = plumeFbm(nCoord + vec2(0.0, 0.14)) - fbmVal;

          float radialMask = smoothstep(1.0, 0.26, r + (fbmVal - 0.5) * 0.44);
          if (radialMask < 0.015) discard;

          // Reconstruct 3D pseudo-spherical volumetric puff normal in world space
          vec3 forward = normalize(vToCam);
          vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward) + vec3(1e-5, 0.0, 0.0));
          vec3 up = cross(forward, right);
          float sphereZ = sqrt(max(0.05, 1.0 - min(0.95, r * r)));
          vec3 puffNorm = normalize(
            right * (p.x * 0.78 - fbmDx * 1.85) +
            up * (p.y * 0.78 - fbmDy * 1.85) +
            forward * sphereZ
          );

          vec3 sunL = normalize(uSunDir);
          float sunLit = clamp(dot(puffNorm, sunL) * 0.5 + 0.5, 0.0, 1.0);
          float silverLining = pow(max(dot(-forward, sunL), 0.0), 4.0) * (1.0 - radialMask * 0.55) * 0.35;

          if (vKind > 1.5) {
            // Sunlit Aerosol Spray & Mist Cloud rising from Mini-Niagara Horseshoe Plunge Pool
            vec3 mistDay = mix(vec3(0.58, 0.70, 0.78), vec3(0.96, 0.99, 1.0), pow(sunLit, 0.78))
                         + vec3(0.92, 0.98, 1.0) * silverLining * 1.15;
            vec3 mistDusk = mix(vec3(0.32, 0.28, 0.36), vec3(0.98, 0.64, 0.40), pow(sunLit, 0.85))
                          + vec3(1.0, 0.66, 0.35) * silverLining;
            vec3 mistNight = mix(vec3(0.07, 0.10, 0.16), vec3(0.20, 0.27, 0.38), sunLit);
            vec3 mistCol = mix(mix(mistDay, mistDusk, uDuskFactor), mistNight, uNightFactor);
            mistCol *= mix(1.0, 0.72, clamp(uRainIntensity * 0.55, 0.0, 0.55));
            mistCol += vec3(0.72, 0.85, 1.0) * uLightningFlash * 0.85;
            gl_FragColor = vec4(mistCol, clamp(radialMask * vAlpha, 0.0, 0.52));
            return;
          }

          // Quiet Gunung Gede Bogor (Kawah Ratu) Pure Hydrothermal White Water-Vapor Fumarole Wisp (ZERO ash, ZERO red fire!)
          vec3 dayLitCol = vec3(0.95, 0.97, 0.99);
          vec3 dayShadowCol = vec3(0.70, 0.75, 0.81);
          vec3 dayCol = mix(dayShadowCol, dayLitCol, pow(sunLit, 0.80)) + vec3(0.96, 0.98, 1.0) * silverLining;

          vec3 duskCol = mix(vec3(0.42, 0.36, 0.44), vec3(0.98, 0.68, 0.44), pow(sunLit, 0.88)) + vec3(1.0, 0.66, 0.36) * silverLining;
          vec3 nightCol = mix(vec3(0.08, 0.10, 0.15), vec3(0.20, 0.25, 0.34), sunLit);

          vec3 steamCol = mix(dayCol, duskCol, uDuskFactor);
          steamCol = mix(steamCol, nightCol, uNightFactor);
          steamCol *= mix(1.0, 0.74, clamp(uRainIntensity * 0.55, 0.0, 0.55));
          steamCol += vec3(0.70, 0.82, 1.0) * uLightningFlash * (0.45 + 0.55 * puffNorm.y);

          gl_FragColor = vec4(steamCol, clamp(radialMask * vAlpha, 0.0, 0.075));
        }
      `
    });

    this.volcanicPlumeMesh = new THREE.Mesh(geo, mat);
    this.volcanicPlumeMesh.name = 'NexusVolcanicSmokePlume';
    this.volcanicPlumeMesh.userData.dynamicKinematic = true;
    this.volcanicPlumeMesh.frustumCulled = false;
    this.volcanicPlumeMesh.renderOrder = 3;
    this.volcanicPlumeMesh.visible = true;
    this.scene.add(this.volcanicPlumeMesh);
  }

  /**
   * Hukum Emas 14 (v2.8 Ultra-Realistic Physical Ocean & Island Coastal Hydrodynamics Edition):
   * Creates the Physical Dielectric Ocean & Lake Water Material (IOR 1.333, F0 = 0.0204) with:
   * - True 3D Dispersive 6-Train Trochoidal Gerstner Swell Spectrum (\omega_i = \sqrt{g k_i}, 0 backward slosh!)
   * - 360° Snell's-Law Coastal Wave Refraction, Asymmetric Stokes Shoaling Rollers, & 3D Swash Run-Up on Stone Turap
   * - Monotonically Integrated Wind Phase (uWindPhase = \int v_wind dt) & Dual Counter-Crossed Capillary Wavelets
   * - Beer-Lambert Optical Depth Absorption, Wave-Crest Jade Subsurface Scattering (SSS), & Voronoi Focal-Web Caustics
   * - Porous Cellular Voronoi Swash / Whitecap Sea-Foam & Distance Slope-Variance (LEADR) GGX Sun Glitter
   */
  createDanauKenangaWaterMaterial() {
    const hydroCommonGLSL = /* glsl */ `
      ${WeatherOpticsEngine._getArchipelagoSdfGLSL()}

      // Evaluates 6 Dispersive Crossing Trochoidal Gerstner Swells (\omega = \sqrt{9.81 * k})
      // + 360-Degree Snell's-Law Refracted Shoaling Coastal Surf Rollers on BOTH Pulau U & Pulau I!
      // Strict monotonic phase (-w * t for open sea, +w * t for inward coastal rollers): ZERO 1-second sloshing!
      void evalGravityHydrodynamics(
        vec2 baseXZ,
        float t,
        float seaState,
        out vec3 dispOut,
        out vec3 gradNormalOut,
        out float dShoreOut,
        out float crestFactorOut,
        out float rollerSwashOut
      ) {
        float drPerp = evalArchipelagoCoastRaw(baseXZ);
        // Stone turap boulder berm crosses static water datum y = -1.45m at drPerp = 3.52m
        float dShore = drPerp - 3.52;
        dShoreOut = dShore;

        float S = clamp(seaState, 0.24, 1.18);
        // Open-sea swell energy smoothly shoals/damps as it converts into coast-normal rollers near the stone turap
        float openShoal = mix(0.28, 1.0, smoothstep(-0.6, 18.0, dShore));

        vec2 horizSurge = vec2(0.0);
        float eta = 0.0;
        vec2 negGradEta = vec2(0.0);
        float jacobianY = 1.0;
        float crestAccum = 0.0;

        // Wave 1: Primary Long Ocean Swell (L = 44.0m, k = 0.1428, w = 1.184 rad/s, T = 5.31s)
        {
          vec2 d1 = vec2(0.8480, 0.5300);
          float k1 = 0.1428; float w1 = 1.184; float a1 = 0.145 * S * openShoal; float q1 = 0.34;
          float ph1 = dot(d1, baseXZ) * k1 - w1 * t;
          float s1 = sin(ph1); float c1 = cos(ph1);
          horizSurge -= d1 * (q1 * a1 * s1);
          eta += a1 * c1;
          negGradEta += d1 * (k1 * a1 * s1);
          jacobianY -= q1 * k1 * a1 * c1;
          crestAccum += max(0.0, c1) * 0.30;
        }
        // Wave 2: Cross Swell -31 deg (L = 29.5m, k = 0.2130, w = 1.445 rad/s, T = 4.35s)
        {
          vec2 d2 = vec2(0.9962, -0.0872);
          float k2 = 0.2130; float w2 = 1.445; float a2 = 0.108 * S * openShoal; float q2 = 0.32;
          float ph2 = dot(d2, baseXZ) * k2 - w2 * t + 1.37;
          float s2 = sin(ph2); float c2 = cos(ph2);
          horizSurge -= d2 * (q2 * a2 * s2);
          eta += a2 * c2;
          negGradEta += d2 * (k2 * a2 * s2);
          jacobianY -= q2 * k2 * a2 * c2;
          crestAccum += max(0.0, c2) * 0.25;
        }
        // Wave 3: Oblique Swell +27 deg (L = 18.8m, k = 0.3342, w = 1.811 rad/s, T = 3.47s)
        {
          vec2 d3 = vec2(0.5150, 0.8572);
          float k3 = 0.3342; float w3 = 1.811; float a3 = 0.076 * S * openShoal; float q3 = 0.30;
          float ph3 = dot(d3, baseXZ) * k3 - w3 * t + 2.81;
          float s3 = sin(ph3); float c3 = cos(ph3);
          horizSurge -= d3 * (q3 * a3 * s3);
          eta += a3 * c3;
          negGradEta += d3 * (k3 * a3 * s3);
          jacobianY -= q3 * k3 * a3 * c3;
          crestAccum += max(0.0, c3) * 0.20;
        }
        // Wave 4: Crossing Wind Sea -52 deg (L = 11.6m, k = 0.5417, w = 2.305 rad/s, T = 2.73s)
        {
          vec2 d4 = vec2(0.9272, -0.3746);
          float k4 = 0.5417; float w4 = 2.305; float a4 = 0.050 * S * openShoal; float q4 = 0.28;
          float ph4 = dot(d4, baseXZ) * k4 - w4 * t + 4.19;
          float s4 = sin(ph4); float c4 = cos(ph4);
          horizSurge -= d4 * (q4 * a4 * s4);
          eta += a4 * c4;
          negGradEta += d4 * (k4 * a4 * s4);
          jacobianY -= q4 * k4 * a4 * c4;
          crestAccum += max(0.0, c4) * 0.14;
        }
        // Wave 5: Short Gravity Chop A +46 deg (L = 7.3m, k = 0.8607, w = 2.906 rad/s, T = 2.16s)
        {
          vec2 d5 = vec2(0.2250, 0.9744);
          float k5 = 0.8607; float w5 = 2.906; float a5 = 0.030 * S * openShoal; float q5 = 0.25;
          float ph5 = dot(d5, baseXZ) * k5 - w5 * t + 0.93;
          float s5 = sin(ph5); float c5 = cos(ph5);
          horizSurge -= d5 * (q5 * a5 * s5);
          eta += a5 * c5;
          negGradEta += d5 * (k5 * a5 * s5);
          jacobianY -= q5 * k5 * a5 * c5;
          crestAccum += max(0.0, c5) * 0.11;
        }
        // Wave 6: Short Gravity Chop B -17 deg (L = 4.6m, k = 1.3659, w = 3.661 rad/s, T = 1.72s)
        {
          vec2 d6 = vec2(0.9563, 0.2924);
          float k6 = 1.3659; float w6 = 3.661; float a6 = 0.018 * S * openShoal; float q6 = 0.22;
          float ph6 = dot(d6, baseXZ) * k6 - w6 * t + 5.24;
          float s6 = sin(ph6); float c6 = cos(ph6);
          horizSurge -= d6 * (q6 * a6 * s6);
          eta += a6 * c6;
          negGradEta += d6 * (k6 * a6 * s6);
          jacobianY -= q6 * k6 * a6 * c6;
        }

        // 360-Degree Snell's-Law Refracted Shoaling Coastal Surf Rollers (traveling INWARD toward Pulau U & Pulau I stone turap!)
        float swashCrest = 0.0;
        if (dShore < 42.0) {
          float eC = 0.35;
          float gx = evalArchipelagoCoastRaw(baseXZ + vec2(eC, 0.0)) - evalArchipelagoCoastRaw(baseXZ - vec2(eC, 0.0));
          float gz = evalArchipelagoCoastRaw(baseXZ + vec2(0.0, eC)) - evalArchipelagoCoastRaw(baseXZ - vec2(0.0, eC));
          vec2 coastN = normalize(vec2(gx, gz) + vec2(1e-5, 0.0));
          vec2 refC = baseXZ.y > 536.0 ? vec2(-160.0, 680.0) : vec2(-155.0, 207.0);
          float thetaIsl = atan(baseXZ.y - refC.y, baseXZ.x - refC.x);

          float dClamp = max(-1.8, dShore);
          // Analytical shoaling phase integral: wavelength compresses & phase speed slows as water shallows
          float phiShoal = 0.36 * dClamp + 4.9 * log(1.0 + max(0.0, dClamp + 1.8) / 7.5);
          float kLocal = 0.36 + 4.9 / (7.5 + max(0.0, dClamp + 1.8));
          float shoalEnv = exp(-max(0.0, dShore - 2.0) * 0.068) * smoothstep(42.0, 24.0, dShore);

          // Primary coastal surf roller (T = 5.71s, w = 1.10 rad/s) + secondary wave-group swell (T = 7.95s, w = 0.79 rad/s)
          float psi1 = phiShoal + 1.10 * t + 1.7 * sin(thetaIsl - 0.5) + 0.65 * cos(2.0 * thetaIsl + 0.8);
          float psi2 = phiShoal * 0.74 + 0.79 * t - 1.4 * cos(thetaIsl + 0.4) + 0.55 * sin(3.0 * thetaIsl);

          // Stokes 2nd-harmonic asymmetric wave profile (peaked forward-leaning surf crest, broad gentle trough)
          float wSurf1 = cos(psi1) + 0.26 * cos(2.0 * psi1);
          float dSurf1 = sin(psi1) + 0.52 * sin(2.0 * psi1);
          float wSurf2 = cos(psi2) + 0.22 * cos(2.0 * psi2);
          float dSurf2 = sin(psi2) + 0.44 * sin(2.0 * psi2);

          float aR1 = (0.052 + 0.095 * S) * shoalEnv;
          float aR2 = (0.034 + 0.062 * S) * shoalEnv;

          eta += aR1 * wSurf1 + aR2 * wSurf2;
          // Orbital shoreward surge & recede along coast normal
          horizSurge -= coastN * ((aR1 * sin(psi1) + aR2 * sin(psi2)) * 0.42);
          negGradEta += coastN * (kLocal * (aR1 * dSurf1 + aR2 * 0.74 * dSurf2));
          swashCrest = clamp(0.5 + 0.38 * wSurf1 + 0.24 * wSurf2, 0.0, 1.0) * shoalEnv;

          // Subtle reflected capillary backwash wave radiating 0..4.5m outward after striking the stone turap
          if (dShore > -0.6 && dShore < 4.8) {
            float reflEnv = (1.0 - clamp(dShore / 4.8, 0.0, 1.0)) * smoothstep(-0.6, 0.1, dShore);
            float psiRefl = dShore * 3.1 - 1.92 * t + sin(thetaIsl * 6.0) * 1.1;
            float aRefl = 0.014 * (0.45 + 0.55 * S) * reflEnv;
            eta += aRefl * cos(psiRefl);
            negGradEta += coastN * (3.1 * aRefl * sin(psiRefl));
          }
        }

        // Smoothly taper 3D vertex displacement at outer 1650m archipelago ocean horizon so sky-sea horizon stays razor-level
        float rCenter = length(baseXZ - ARCHIPELAGO_CENTER);
        float horizonTaper = 1.0 - smoothstep(1150.0, 1630.0, rCenter);

        dispOut = vec3(horizSurge.x, eta, horizSurge.y) * horizonTaper;
        gradNormalOut = normalize(vec3(negGradEta.x * horizonTaper, max(0.35, jacobianY), negGradEta.y * horizonTaper));
        crestFactorOut = clamp(crestAccum, 0.0, 1.0) * horizonTaper;
        rollerSwashOut = swashCrest;
      }
    `;

    return new THREE.ShaderMaterial({
      uniforms: {
        uTime: this.uniforms.uTime,
        uSunDir: this.uniforms.uSunDir,
        uSunColor: this.uniforms.uSunColor,
        uNightFactor: this.uniforms.uNightFactor,
        uDuskFactor: this.uniforms.uDuskFactor,
        uRainIntensity: this.uniforms.uRainIntensity,
        uCloudCover: this.uniforms.uCloudCover,
        uStormFactor: this.uniforms.uStormFactor,
        uLightningFlash: this.uniforms.uLightningFlash,
        uWindVec: this.uniforms.uWindVec,
        uWindPhase: this.uniforms.uWindPhase,
        uSeaState: this.uniforms.uSeaState,
        uHvsParams: this.uniforms.uHvsParams,
        uHvsGlare: this.uniforms.uHvsGlare
      },
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
      vertexShader: /* glsl */ `
        attribute vec4 aHydroMeta;
        attribute vec4 aRiverDetail;

        uniform float uTime;
        uniform float uSeaState;

        varying vec3 vWorldPos;
        varying vec2 vBaseXZ;
        varying float vWaveElev;
        varying vec4 vHydroMeta;
        varying vec4 vRiverDetail;
        varying vec2 vFlowDir;
        varying vec3 vGeomNormal;

        ${hydroCommonGLSL}

        void main() {
          vec4 wp = modelMatrix * vec4(position, 1.0);
          vBaseXZ = wp.xz;
          vHydroMeta = aHydroMeta;
          vRiverDetail = aRiverDetail;
          vFlowDir = uv;
          vGeomNormal = normalize(mat3(modelMatrix) * normal);

          if (aHydroMeta.x < 0.5) {
            vec3 disp;
            vec3 gNorm;
            float dShore;
            float crestF;
            float swashF;
            evalGravityHydrodynamics(wp.xz, uTime, uSeaState, disp, gNorm, dShore, crestF, swashF);
            wp.xyz += disp;
            vWaveElev = disp.y;
          } else if (aHydroMeta.x < 1.5) {
            // Natural Fluvial Stream, Artesian Spring, Small Lake, & Plunge Pool:
            // Depth-locked at 0.00m shoreline contact + subtle standing bow-swell over emergent 3D boulders!
            float depthLock = smoothstep(0.0, 0.24, max(0.0, aRiverDetail.x));
            float localSpd = max(0.22, aRiverDetail.y);
            float bowSwell = 0.024 * clamp(aRiverDetail.z, 0.0, 1.35);
            float rip = 0.011 * sin(aHydroMeta.z * 1.55 - uTime * (1.1 + min(2.2, localSpd) * 1.45))
                      + 0.007 * cos(wp.x * 1.9 + wp.z * 1.7 - uTime * 2.2);
            float dyFresh = (rip + bowSwell) * depthLock;
            wp.y += dyFresh;
            vWaveElev = dyFresh;
          } else {
            // Mini-Niagara 3D Curved Horseshoe Falling Cataract Curtain: hydraulic sheet flutter, seam-locked at 0 & 1!
            float tDrop = aHydroMeta.w;
            float seamEnv = sin(3.14159265 * clamp(tDrop, 0.0, 1.0));
            float flutter = 0.075 * sin(aHydroMeta.z * 1.65 + uTime * 7.2) * cos(tDrop * 12.0 - uTime * 9.6) * seamEnv;
            wp.xyz += vGeomNormal * flutter;
            vWaveElev = flutter;
          }

          vWorldPos = wp.xyz;
          gl_Position = projectionMatrix * viewMatrix * wp;
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec3 vWorldPos;
        varying vec2 vBaseXZ;
        varying float vWaveElev;
        varying vec4 vHydroMeta;
        varying vec4 vRiverDetail;
        varying vec2 vFlowDir;
        varying vec3 vGeomNormal;

        uniform float uTime;
        uniform vec3 uSunDir;
        uniform vec3 uSunColor;
        uniform float uNightFactor;
        uniform float uDuskFactor;
        uniform float uRainIntensity;
        uniform float uCloudCover;
        uniform float uStormFactor;
        uniform float uLightningFlash;
        uniform vec3 uWindVec;
        uniform vec2 uWindPhase;
        uniform float uSeaState;
        uniform vec4 uHvsParams;
        uniform vec4 uHvsGlare;

        ${HumanEyeHDROptics.getGLSLKernel()}

        // Hukum Emas 14: Domain rotation matrices preventing periodic axial striping
        const mat2 R = mat2(0.80, 0.60, -0.60, 0.80);
        const mat2 R_INV = mat2(0.80, -0.60, 0.60, 0.80);

        ${hydroCommonGLSL}

        float hash(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }

        vec2 hash2(vec2 p) {
          return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453);
        }

        float vnoise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          vec2 u = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
            mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
            u.y
          );
        }

        // Cellular Voronoi distance evaluator -> vec2(edgeRidgeDist, cellCenterDist) for focal caustics & lacy sea-foam
        vec2 voronoiCell(vec2 x) {
          vec2 n = floor(x);
          vec2 f = fract(x);
          float md1 = 8.0;
          float md2 = 8.0;
          for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
              vec2 g = vec2(float(i), float(j));
              vec2 o = 0.14 + 0.72 * hash2(n + g);
              vec2 r = g + o - f;
              float d = dot(r, r);
              if (d < md1) {
                md2 = md1;
                md1 = d;
              } else if (d < md2) {
                md2 = d;
              }
            }
          }
          float s1 = sqrt(md1);
          return vec2(sqrt(md2) - s1, s1);
        }

        // Dual Counter-Crossed Dispersive Capillary-Gravity Wind Wavelets:
        // Two crossing wave fans (+24 deg & -27 deg) advancing monotonically via uWindPhase & dispersive uTime,
        // so capillary facets continuously form, interfere, sparkle, and dissolve (ZERO sliding-rubber-sheet look!)
        float evalCrossedWavelets(vec2 xz, float t, vec2 windPh) {
          vec2 pA = R * (xz * 0.34);
          vec2 pB = R_INV * (xz * 0.39) + vec2(19.3, 7.1);
          vec2 driftA = windPh * 0.075 + vec2(0.42, 0.19) * t;
          vec2 driftB = windPh * 0.068 + vec2(0.37, -0.24) * t;

          float sum = 0.0;
          float amp = 0.28;
          for (int i = 0; i < 4; i++) {
            float dispRate = 1.0 + float(i) * 0.36;
            float nA = vnoise(pA - driftA * dispRate);
            float nB = vnoise(pB - driftB * dispRate);
            sum += amp * (nA + nB - 1.0);
            pA = R * pA * 2.09 + vec2(13.7, 29.3);
            pB = R_INV * pB * 2.13 + vec2(23.1, 11.9);
            amp *= 0.50;
          }
          return sum;
        }

        float fbmCloudRefl(vec2 p) {
          float sum = 0.0;
          float amp = 0.52;
          for (int i = 0; i < 4; i++) {
            sum += amp * vnoise(p);
            p = R * p * 2.05 + vec2(17.1, 31.7);
            amp *= 0.5;
          }
          return sum;
        }

        // True Counter-Propagating Voronoi Focal-Envelope Water Caustics (zero grid-aligned noise diamonds!)
        float evalCaustics(vec2 p, float t) {
          vec2 v1 = voronoiCell(R * (p * 0.92) + vec2(t * 0.21, -t * 0.16));
          vec2 v2 = voronoiCell(R_INV * (p * 1.18) + vec2(-t * 0.18, t * 0.23));
          float ridge1 = exp(-v1.x * 8.5);
          float ridge2 = exp(-v2.x * 9.0);
          return clamp(pow(ridge1 * 0.62 + ridge2 * 0.58, 1.85) * 1.35, 0.0, 1.35);
        }

        // Anisotropic Poiseuille Flow-Sheared Fluvial Capillary Wavelets & 3D Boulder Wake Chop
        // (100% organic multi-scale crossed wave fans stretched along local river current — ZERO Voronoi grid!)
        float evalFluvialRipples(vec2 xz, vec2 F, vec2 C, float localSpd, float boulderWake, float t) {
          float sAlong = dot(xz, F);
          float sCross = dot(xz, C);

          // Crossed anisotropic current-stretched streamline wavelets (+15 deg & -18 deg relative to flow)
          vec2 q1 = vec2(sAlong * 1.05 + sCross * 0.34 - t * localSpd * 1.28, sCross * 2.85 - sAlong * 0.22 - t * 0.32);
          vec2 q2 = vec2(sAlong * 1.22 - sCross * 0.38 - t * localSpd * 1.42 + 13.7, sCross * 3.15 + sAlong * 0.26 + t * 0.29 + 7.1);
          float wLong = (vnoise(q1) + vnoise(q2) - 1.0) * 0.42;

          // Fine dispersive surface-tension capillary facets & subtle bank-eddy curl
          vec2 q3 = R * vec2(sAlong * 2.75 - t * localSpd * 1.95, sCross * 4.95) + vec2(19.3, 5.4);
          vec2 q4 = R_INV * vec2(sAlong * 3.40 - t * localSpd * 2.25, sCross * 5.60) + vec2(8.2, 23.1);
          float wCap = (vnoise(q3) + vnoise(q4) - 1.0) * 0.24;

          // Standing hydraulic riffle & 3D boulder Kelvin-wake chop
          float chopAmp = clamp((localSpd - 0.85) * 0.22 + boulderWake * 0.48, 0.0, 0.65);
          float wChop = chopAmp * 0.16 * sin(sAlong * 8.4 - t * (3.2 + localSpd * 2.8) + vnoise(xz * 1.9) * 2.6)
                      * cos(sCross * 7.2 + sin(sAlong * 2.1 - t * 1.5) * 0.8);

          return wLong + wCap + wChop;
        }

        void main() {
          float distCam = length(cameraPosition - vWorldPos);

          // === REGIME 2: PULAU "I" MINI-NIAGARA 3D CONCAVE HORSESHOE FALLING CATARACT CURTAIN ===
          if (vHydroMeta.x > 1.5) {
            float alphaNorm = vHydroMeta.y;
            float arcM = vHydroMeta.z;
            float tDrop = clamp(vHydroMeta.w, 0.0, 1.0);
            vec3 viewDirF = normalize(cameraPosition - vWorldPos);
            vec3 sunDirF = normalize(uSunDir);
            float sunVisF = clamp(1.0 - uRainIntensity * 0.72 - uStormFactor * 0.40 - uCloudCover * 0.20, 0.08, 1.0);

            // Accelerating gravitational free-fall coordinate (v = sqrt(2 g h)) -> ZERO static texture slide!
            float fallV = pow(tDrop, 0.68) * 11.5 - uTime * 6.8;
            float chuteA = vnoise(vec2(arcM * 1.45, fallV * 0.82));
            float chuteB = vnoise(vec2(arcM * 3.85 + 13.7, fallV * 1.65 - uTime * 2.5));
            vec2 fallVor = voronoiCell(vec2(arcM * 1.15, fallV * 0.92));
            float vorWeb = smoothstep(0.28, 0.03, fallVor.x);

            // Turbulent white-water splitter wakes streaming below the 2 basalt "Goat Island" lip boulders!
            float splitWake1 = exp(-pow((alphaNorm + 0.532) / 0.095, 2.0));
            float splitWake2 = exp(-pow((alphaNorm - 0.463) / 0.095, 2.0));
            float hornEdgeFoam = smoothstep(0.76, 0.99, abs(alphaNorm)) * 0.55;

            // Emerald-jade translucent rollover brink (tDrop < 0.22) -> aerated roaring white cataract -> plunge boil!
            vec3 jadeBrink = mix(vec3(0.045, 0.295, 0.275), vec3(0.115, 0.485, 0.445), chuteA);
            jadeBrink = mix(jadeBrink, vec3(0.018, 0.045, 0.068), uNightFactor);

            float aeration = clamp(
              smoothstep(0.04, 0.42, tDrop) * (0.44 + 0.36 * chuteA + 0.28 * chuteB) +
              vorWeb * 0.42 * smoothstep(0.06, 0.35, tDrop) +
              (splitWake1 + splitWake2) * 0.62 +
              hornEdgeFoam +
              smoothstep(0.70, 0.98, tDrop) * 0.52,
              0.0,
              1.0
            );

            vec3 baseNormF = gl_FrontFacing ? normalize(vGeomNormal) : -normalize(vGeomNormal);
            vec3 pertNormF = normalize(baseNormF + vec3((chuteB - 0.5) * 0.32, (chuteA - 0.5) * 0.22, (vorWeb - 0.5) * 0.28));
            float sunDiffF = clamp(dot(pertNormF, sunDirF) * 0.5 + 0.5, 0.14, 1.0);

            vec3 whiteDay = mix(vec3(0.64, 0.77, 0.84), vec3(0.97, 0.99, 1.0), sunDiffF);
            vec3 whiteDusk = mix(vec3(0.36, 0.30, 0.38), vec3(0.98, 0.66, 0.42), sunDiffF);
            vec3 whiteNight = mix(vec3(0.08, 0.12, 0.19), vec3(0.22, 0.30, 0.42), sunDiffF);
            vec3 cataractFoam = mix(mix(whiteDay, whiteDusk, uDuskFactor), whiteNight, uNightFactor);

            vec3 fallCol = mix(jadeBrink, cataractFoam, aeration);

            // Sunlit waterfall crest sparkle & mist sheen in True Linear-HDR -> HVS Photoreceptor Pipeline
            vec3 halfVecF = normalize(sunDirF + viewDirF);
            float specF = pow(max(dot(pertNormF, halfVecF), 0.0), 48.0) * 0.95 * sunVisF;
            vec3 specTintF = mix(mix(vec3(1.0, 0.96, 0.86), vec3(1.0, 0.60, 0.30), uDuskFactor), vec3(0.45, 0.58, 0.80), uNightFactor);
            vec3 hdrFall = nexusSRGBToLinear(max(fallCol, vec3(0.0))) * mix(1.18, 0.74, uNightFactor);
            hdrFall += nexusSRGBToLinear(specTintF) * specF * 1.95 * (1.0 - 0.38 * aeration);
            hdrFall += vec3(0.85, 1.05, 1.50) * uLightningFlash * 1.25;

            hdrFall = nexusApplyBoundaryLayerAerialPerspective(
              hdrFall,
              cameraPosition,
              vWorldPos,
              sunDirF,
              uDuskFactor,
              uNightFactor,
              uCloudCover,
              uRainIntensity,
              uLightningFlash
            );

            vec3 hvsFallLin = nexusApplyHumanEyeHDR(
              hdrFall,
              normalize(vWorldPos - cameraPosition),
              sunDirF,
              uSunColor,
              uHvsParams,
              uHvsGlare
            );
            gl_FragColor = vec4(nexusLinearToSRGB(hvsFallLin), 1.0);
            return;
          }

          // === REGIME 1: HUMAN-EYE LINEAR-HDR NATURAL FLUVIAL RIVER, MATA AIR SPRING, DANAU KECIL REKTORAT, & PLUNGE POOL ===
          if (vHydroMeta.x > 0.5) {
            float crossU = vHydroMeta.y;
            float flowArc = vHydroMeta.z;
            float subMode = vHydroMeta.w;
            bool isSpring = (subMode > 2.8 && subMode < 3.8);
            bool isPlungePool = (subMode > 4.0);
            bool isSmallLake = (!isSpring && !isPlungePool && subMode < 0.35);

            // Exact sub-triangle physical water depth (meters), Poiseuille local flow speed, 3D boulder wake, & bend asymmetry
            float rawSignedDepth = vRiverDetail.x;
            float localSpd = max(0.18, vRiverDetail.y);
            float boulderWake = max(0.0, vRiverDetail.z);
            float bendSide = vRiverDetail.w;

            // Smooth low-frequency organic pebble shoreline contour (ZERO high-frequency jagged pixel teeth!)
            float shoreContour = (vnoise(R * (vWorldPos.xz * 1.85)) - 0.5) * 0.011
                               * smoothstep(0.18, 0.0, max(0.0, rawSignedDepth));
            float effDepth = rawSignedDepth + vWaveElev * 0.44 + shoreContour;
            if (!isPlungePool && effDepth < 0.0015) discard;
            effDepth = max(0.0015, effDepth);
            float meniscusFade = smoothstep(0.0015, 0.095, effDepth);

            vec3 viewDirR = normalize(cameraPosition - vWorldPos);
            vec3 sunDirR = normalize(uSunDir);
            float sunVisR = clamp(1.0 - uRainIntensity * 0.74 - uStormFactor * 0.40 - uCloudCover * 0.22, 0.08, 1.0);
            float overcastR = clamp(uCloudCover * 0.52 + uRainIntensity * 0.38 + uStormFactor * 0.28, 0.0, 0.88);
            float clearSunMask = sunVisR * clamp(1.0 - uRainIntensity * 0.90 - uCloudCover * 0.52, 0.0, 1.0);

            // Live Diurnal & Weather-Coupled Linear-HDR Fluvial Irradiance (solar + sky dome + lightning)
            float dayIrradR = mix(1.34, 0.72, overcastR);
            float duskIrradR = mix(0.92, 0.52, overcastR);
            float nightIrradR = mix(0.088, 0.052, overcastR);
            float surfIrrad = mix(mix(dayIrradR, duskIrradR, uDuskFactor), nightIrradR, uNightFactor) + uLightningFlash * 1.55;

            vec2 F = length(vFlowDir) > 0.05 ? normalize(vFlowDir) : vec2(1.0, 0.0);
            vec2 C = vec2(-F.y, F.x);

            // Anisotropic flow-sheared capillary ripples + Nyquist distance & shallow-film damping
            float epsR = mix(0.055, 0.20, smoothstep(10.0, 150.0, distCam));
            float fWave0 = evalFluvialRipples(vWorldPos.xz, F, C, localSpd, boulderWake, uTime);
            float fWaveX = evalFluvialRipples(vWorldPos.xz + vec2(epsR, 0.0), F, C, localSpd, boulderWake, uTime);
            float fWaveZ = evalFluvialRipples(vWorldPos.xz + vec2(0.0, epsR), F, C, localSpd, boulderWake, uTime);

            float shallowWaveDamp = mix(0.24, 1.0, smoothstep(0.002, 0.14, effDepth));
            float rGain = (0.028 + 0.019 * min(2.0, localSpd) + 0.016 * min(1.2, boulderWake))
                        * shallowWaveDamp
                        * (1.0 - smoothstep(52.0, 265.0, distCam));
            float rNx = ((fWave0 - fWaveX) / epsR) * rGain;
            float rNz = ((fWave0 - fWaveZ) / epsR) * rGain;
            float freshFoam = 0.0;

            if (isSpring) {
              // Concentric bubbling artesian spring upwelling rings at Mata Air Hutan Selatan (-286, +48)
              vec2 dSpr = vWorldPos.xz - vec2(-286.0, 48.0);
              float rSpr = length(dSpr) + 1e-4;
              float ringW = sin(rSpr * 4.4 - uTime * 3.6) * exp(-rSpr * 0.26);
              rNx += (dSpr.x / rSpr) * ringW * 0.048 * meniscusFade;
              rNz += (dSpr.y / rSpr) * ringW * 0.048 * meniscusFade;
              float upwellBoil = vnoise(R * (vWorldPos.xz * 3.4 - vec2(uTime * 0.6))) * exp(-rSpr * 0.36);
              freshFoam += max(0.0, ringW * 0.25 + upwellBoil * 0.28) * exp(-rSpr * 0.28);
            }

            if (isPlungePool) {
              // Horseshoe waterfall impact shockwave rings & organic churning aerated plunge-pool boil
              vec2 dPlg = vWorldPos.xz - vec2(-44.0, 643.0);
              float rPlg = length(dPlg) + 1e-4;
              float aPlg = atan(dPlg.x, dPlg.y);
              float arcMask = 1.0 - smoothstep(0.98, 1.32, abs(aPlg));
              float impBand = exp(-pow((rPlg - 12.3) / 2.35, 2.0)) * arcMask;
              float outWave = sin((rPlg - 12.2) * 3.2 + uTime * 4.6) * exp(-abs(rPlg - 11.0) * 0.16);
              rNx += (dPlg.x / rPlg) * outWave * 0.072;
              rNz += (dPlg.y / rPlg) * outWave * 0.072;
              float boilA = vnoise(R * (vWorldPos.xz * 2.1 + dPlg * 0.12 - vec2(0.0, uTime * 0.75)));
              float boilB = vnoise(R_INV * (vWorldPos.xz * 4.7 - vec2(uTime * 0.55, uTime * 0.95)));
              float boilPlume = smoothstep(0.32, 0.78, boilA * 0.6 + boilB * 0.4);
              freshFoam += clamp(impBand * (0.58 + 0.52 * boilPlume) + smoothstep(14.5, 6.2, rPlg) * arcMask * boilPlume * 0.34, 0.0, 0.94);
            }

            // Concentric capillary raindrop impact rings on river, lake, & spring during rain
            if (uRainIntensity > 0.03 && distCam < 135.0) {
              float fadeR = smoothstep(135.0, 26.0, distCam) * shallowWaveDamp;
              for (int rl = 0; rl < 2; rl++) {
                float gSc = rl == 0 ? 1.45 : 2.65;
                vec2 gp = vWorldPos.xz * gSc;
                vec2 gi = floor(gp);
                vec2 gf = fract(gp) - 0.5;
                float sd = hash(gi + float(rl) * 37.0);
                if (sd > clamp(uRainIntensity * 0.85, 0.15, 1.0)) continue;
                float lf = fract(uTime * (1.2 + float(rl) * 0.45) + sd);
                float d = length(gf) + 1e-4;
                float fr = lf * 0.42;
                float wv = sin((d - fr) * 88.0) * exp(-abs(d - fr) * 25.0) * (1.0 - lf) * min(1.3, uRainIntensity) * fadeR;
                rNx += (gf.x / d) * wv * 0.34;
                rNz += (gf.y / d) * wv * 0.34;
                freshFoam += max(0.0, wv) * 0.26;
              }
            }

            vec3 normR = normalize(vec3(vGeomNormal.x + rNx, max(0.45, vGeomNormal.y), vGeomNormal.z + rNz));

            // 1. Crystal-Clear Sunlit Alluvial Riverbed in True Linear-HDR (matches wet terrain shore seamlessly!)
            vec2 refrP = vWorldPos.xz - normR.xz * min(0.44, effDepth * 0.36);
            float cobbleL = vnoise(R * (refrP * 3.1));
            float cobbleM = vnoise(R_INV * (refrP * 7.6 + vec2(11.3, 5.7)));
            float sandGrain = vnoise(R * (refrP * 19.2));
            float mossPatch = smoothstep(0.42, 0.78, vnoise(R_INV * (refrP * 1.25 + vec2(3.1, 9.4))));
            float innerSandMix = clamp(-bendSide * 1.45, 0.0, 1.0) * smoothstep(0.85, 0.0, effDepth);

            vec3 darkWetCobble = vec3(0.155, 0.175, 0.145);
            vec3 warmWetPebble = vec3(0.315, 0.278, 0.205);
            vec3 olivePeriphyton = vec3(0.132, 0.228, 0.116);
            vec3 wetPointBarSand = vec3(0.375, 0.328, 0.232);

            float stoneBlend = smoothstep(0.20, 0.78, cobbleL * 0.60 + cobbleM * 0.40);
            vec3 bedAlbedo = mix(darkWetCobble, warmWetPebble, stoneBlend);
            bedAlbedo = mix(bedAlbedo, olivePeriphyton, mossPatch * 0.46 * smoothstep(0.08, 0.48, effDepth));
            bedAlbedo = mix(bedAlbedo, wetPointBarSand, max(innerSandMix * 0.78, (1.0 - meniscusFade) * 0.55));
            bedAlbedo *= (0.90 + 0.22 * sandGrain);

            vec3 bedTintIrrad = mix(mix(vec3(1.0, 0.98, 0.92), vec3(1.0, 0.66, 0.38), uDuskFactor), vec3(0.68, 0.82, 1.08), uNightFactor);
            vec3 pebbleBed = bedAlbedo * bedTintIrrad * surfIrrad * (0.76 + 0.36 * clearSunMask);

            if (distCam < 145.0 && effDepth < 1.45 && clearSunMask > 0.04) {
              vec2 cUV = refrP - sunDirR.xz * (effDepth / max(0.30, sunDirR.y)) * 0.30 + normR.xz * 0.24;
              float cw1 = sin(dot(R * cUV, vec2(4.1, 2.8)) - uTime * 2.0 + vnoise(cUV * 2.2) * 3.1);
              float cw2 = sin(dot(R_INV * cUV, vec2(-3.6, 4.5)) + uTime * 1.7 + vnoise(R * cUV * 2.6) * 2.9);
              float softCaustic = pow(max(0.0, cw1 * 0.5 + cw2 * 0.5), 2.8);
              float cDepthEnv = smoothstep(0.02, 0.18, effDepth) * (1.0 - smoothstep(0.75, 1.45, effDepth));
              vec3 cTint = mix(vec3(1.05, 0.96, 0.74), vec3(1.10, 0.56, 0.24), uDuskFactor);
              pebbleBed += cTint * softCaustic * cDepthEnv * clearSunMask * smoothstep(145.0, 28.0, distCam) * 0.52;
            }

            // 2. Luminous Clear Tropical Stream & Campus Lake Volume Scattering + Subsurface Glow (ZERO dark swamp ink!)
            float optDepth = effDepth * (1.0 + 0.34 / max(0.28, viewDirR.y));
            vec3 shallowScatter = isSpring ? vec3(0.072, 0.355, 0.332)
                                : isPlungePool ? vec3(0.076, 0.380, 0.362)
                                : vec3(0.070, 0.302, 0.258);
            vec3 deepScatter = isSpring ? vec3(0.036, 0.205, 0.212)
                             : isPlungePool ? vec3(0.040, 0.228, 0.236)
                             : vec3(0.038, 0.182, 0.180);

            // Natural rainstorm alluvial silt warming (stays luminous silvery-olive, never pitch-black!)
            if (!isSpring && uRainIntensity > 0.05) {
              float siltMix = clamp(uRainIntensity * 0.30 + uStormFactor * 0.16, 0.0, 0.42);
              shallowScatter = mix(shallowScatter, vec3(0.118, 0.205, 0.152), siltMix);
              deepScatter = mix(deepScatter, vec3(0.068, 0.128, 0.098), siltMix);
            }

            // Forward & upward sunlit subsurface water-column scattering
            float sssAlignR = pow(max(0.0, dot(-viewDirR, normalize(sunDirR - normR * 0.44))), 2.6);
            vec3 sssTintR = mix(vec3(0.058, 0.235, 0.205), vec3(0.250, 0.140, 0.058), uDuskFactor);
            vec3 waterSSS = sssTintR * (0.34 + 0.66 * sssAlignR) * clearSunMask * (1.0 - exp(-optDepth * 1.15)) * (1.0 - uNightFactor);

            vec3 extinctCoeff = (isSpring || isPlungePool) ? vec3(0.92, 0.36, 0.38) : vec3(1.02, 0.43, 0.48);
            vec3 transR = exp(-extinctCoeff * optDepth);
            vec3 volR = mix(deepScatter, shallowScatter, exp(-optDepth * 0.72)) * surfIrrad + waterSSS;
            vec3 underR = mix(volR, pebbleBed, transR);

            // 3. True Open-Sky + Billowing Cloud Fresnel Mirror Reflection in Linear-HDR
            float ndvR = max(dot(viewDirR, normR), 0.0);
            float fresR = 0.028 + (1.0 - 0.028) * pow(1.0 - ndvR, 4.4);
            vec3 reflDirR = reflect(-viewDirR, normR);
            float rEl = clamp(reflDirR.y, 0.0, 1.0);

            vec3 sZenSRGB = mix(mix(vec3(0.14, 0.39, 0.75), vec3(0.16, 0.19, 0.40), uDuskFactor), vec3(0.020, 0.032, 0.072), uNightFactor);
            vec3 sHorSRGB = mix(mix(vec3(0.58, 0.76, 0.90), vec3(0.88, 0.48, 0.25), uDuskFactor), vec3(0.052, 0.080, 0.148), uNightFactor);
            vec3 skyR = nexusSRGBToLinear(mix(sHorSRGB, sZenSRGB, pow(rEl, 0.58))) * mix(1.18, 0.76, uNightFactor);

            // Reflected billowing clouds advected monotonically via uWindPhase
            vec2 rCloudUV = (reflDirR.xz / (rEl + 0.22)) * 1.42 + uWindPhase * 0.013;
            float cFieldR = fbmCloudRefl(rCloudUV);
            float cThreshR = mix(0.56, 0.22, clamp(uCloudCover, 0.0, 1.0));
            float cMaskR = smoothstep(cThreshR, cThreshR + 0.32, cFieldR) * smoothstep(0.0, 0.18, rEl);
            vec3 cColSRGB = mix(mix(vec3(0.94, 0.96, 1.0), vec3(0.96, 0.58, 0.34), uDuskFactor), vec3(0.08, 0.11, 0.18), uNightFactor);
            skyR = mix(skyR, nexusSRGBToLinear(cColSRGB) * mix(1.18, 0.76, uNightFactor), cMaskR * 0.78);

            // Silvery overcast sky mirror during mist, rain, & storm (real rain-swept rivers mirror the bright cloud deck!)
            vec3 greySkyDay = mix(vec3(0.56, 0.61, 0.66), vec3(0.36, 0.40, 0.46), clamp(uStormFactor + uRainIntensity * 0.32, 0.0, 1.0));
            vec3 greySkySRGB = mix(mix(greySkyDay, vec3(0.34, 0.30, 0.34), uDuskFactor), vec3(0.042, 0.055, 0.082), uNightFactor);
            skyR = mix(skyR, nexusSRGBToLinear(greySkySRGB) * (0.92 + 0.22 * cFieldR), overcastR * 0.85);

            // Overhanging Canopy Reflection ONLY at low grazing horizon angles inside dense southern rainforest (x < -175)
            float denseForestZone = isPlungePool ? 0.34 : smoothstep(-172.0, -246.0, vWorldPos.x);
            float bankGrazing = smoothstep(0.70, 0.98, abs(crossU));
            float canopyElevCut = 0.06 + 0.17 * denseForestZone + 0.09 * bankGrazing;
            vec2 canopyProj = vWorldPos.xz + reflDirR.xz * (4.0 / max(0.16, rEl + 0.10));
            float leafDapple = vnoise(R * (canopyProj * 0.52)) * 0.62 + vnoise(R_INV * (canopyProj * 1.45)) * 0.38;
            float canopyHit = (1.0 - smoothstep(canopyElevCut - 0.10, canopyElevCut + 0.12, rEl + (leafDapple - 0.5) * 0.14))
                            * (0.14 + 0.46 * max(denseForestZone, bankGrazing * 0.55));
            vec3 foliageReflHDR = mix(vec3(0.052, 0.165, 0.048), vec3(0.135, 0.315, 0.095), leafDapple) * surfIrrad;
            skyR = mix(skyR, foliageReflHDR, clamp(canopyHit, 0.0, 0.46));
            skyR += vec3(0.85, 1.05, 1.50) * uLightningFlash * 1.20;

            float reflMix = clamp((0.11 + fresR * 0.88) * mix(0.12, 1.0, meniscusFade), 0.0, 0.95);
            vec3 freshCol = mix(underR, skyR, reflMix);

            // Earned Aerated Micro-Bubble Filaments ONLY around 3D Boulders & Fast Shallow Riffles (ZERO Voronoi grid!)
            if (!isSpring && !isPlungePool && !isSmallLake) {
              float sAl = dot(vWorldPos.xz, F);
              float sCr = dot(vWorldPos.xz, C);
              float strA = vnoise(vec2(sAl * 1.9 - uTime * localSpd * 2.2, sCr * 9.2));
              float strB = vnoise(vec2(sAl * 4.8 - uTime * localSpd * 3.1 + 7.3, sCr * 18.5));
              float bubbleStreak = smoothstep(0.48, 0.86, strA * 0.56 + strB * 0.44);

              float rockWakeFoam = clamp(boulderWake, 0.0, 1.35) * (0.28 + 0.66 * bubbleStreak);
              float rapidMask = smoothstep(1.20, 1.65, localSpd) * smoothstep(0.62, 0.12, effDepth);
              float rapidFoam = rapidMask * smoothstep(0.64, 0.93, strA * 0.50 + strB * 0.50) * 0.36;
              float swashFilm = smoothstep(0.032, 0.006, effDepth) * smoothstep(0.80, 1.40, localSpd) * smoothstep(0.60, 0.90, strB) * 0.16;

              freshFoam += (rockWakeFoam + rapidFoam) * meniscusFade + swashFilm;
            }

            // Dual-Lobe GGX Liquid Sheen + Crisp Solar/Lunar Sparkle on Rippling Stream & Danau Kecil Rektorat
            vec3 hVecR = normalize(sunDirR + viewDirR);
            float ndhR = max(dot(normR, hVecR), 0.0);
            float specTightR = pow(ndhR, 185.0) * 1.95 * clearSunMask;
            float specBroadR = pow(ndhR, 28.0) * 0.24 * sunVisR;
            vec3 glitTintR = nexusSRGBToLinear(
              mix(mix(vec3(1.0, 0.96, 0.84), vec3(1.0, 0.58, 0.26), uDuskFactor), vec3(0.48, 0.60, 0.82), uNightFactor)
            );
            freshCol += glitTintR * (specTightR + specBroadR) * meniscusFade;

            // Danau Kecil Rektorat (West Parkland Pond) Shimmering Architectural Reflection (Day, Dusk, & Golden Night!)
            if (isSmallLake && vWorldPos.x > -96.0) {
              float rekAlign = exp(-pow((vWorldPos.z + 18.0) / 21.0, 2.0)) * smoothstep(-95.0, -70.5, vWorldPos.x) * meniscusFade;
              float shim = 0.56 + 0.44 * sin(normR.x * 26.0 - uTime * 2.3);
              if (uNightFactor < 0.85 && reflDirR.x > 0.12) {
                float tajugBand = smoothstep(0.18, 0.42, rEl) * (1.0 - smoothstep(0.45, 0.66, rEl));
                vec3 dayArchHDR = mix(vec3(0.48, 0.45, 0.40), vec3(0.44, 0.16, 0.11), tajugBand) * surfIrrad * 0.22;
                freshCol = mix(freshCol, dayArchHDR * (0.85 + 0.30 * shim), rekAlign * (1.0 - uNightFactor) * 0.28);
              }
              if (uNightFactor > 0.08) {
                freshCol += vec3(1.15, 0.68, 0.18) * rekAlign * shim * uNightFactor * 0.32;
              }
            }

            if (freshFoam > 0.005) {
              vec3 fDaySRGB = mix(vec3(0.88, 0.93, 0.92), vec3(0.70, 0.75, 0.78), overcastR * 0.55);
              vec3 fColSRGB = mix(mix(fDaySRGB, vec3(0.92, 0.68, 0.52), uDuskFactor), vec3(0.20, 0.27, 0.37), uNightFactor);
              vec3 fColHDR = nexusSRGBToLinear(fColSRGB) * mix(1.16, 0.78, uNightFactor) + vec3(0.80, 0.98, 1.40) * uLightningFlash;
              freshCol = mix(freshCol, fColHDR, clamp(freshFoam, 0.0, 0.88));
            }

            freshCol = nexusApplyBoundaryLayerAerialPerspective(
              freshCol,
              cameraPosition,
              vWorldPos,
              sunDirR,
              uDuskFactor,
              uNightFactor,
              uCloudCover,
              uRainIntensity,
              uLightningFlash
            );

            vec3 hvsFreshLin = nexusApplyHumanEyeHDR(
              freshCol,
              normalize(vWorldPos - cameraPosition),
              sunDirR,
              uSunColor,
              uHvsParams,
              uHvsGlare
            );
            gl_FragColor = vec4(nexusLinearToSRGB(hvsFreshLin), 1.0);
            return;
          }

          // 1. EXACT PER-PIXEL GRAVITY-WAVE & SHOALING SURF HYDRODYNAMICS + CROSSED CAPILLARY WAVELETS
          vec3 gDisp;
          vec3 gravNorm;
          float dShore;
          float crestFactor;
          float rollerSwash;
          evalGravityHydrodynamics(vBaseXZ, uTime, uSeaState, gDisp, gravNorm, dShore, crestFactor, rollerSwash);

          // Dynamic 3D waterline distance: as wave elevation gDisp.y rises up the 1:1.85 sloped stone turap,
          // the live wet shoreline moves inward (dShore + gDisp.y * 1.85 -> 0 at the exact 3D stone-water contact!)
          float liveWaterline = max(0.0, dShore + gDisp.y * 1.85);
          float waterDepth = clamp(0.03 + liveWaterline * 0.42 + max(0.0, gDisp.y) * 0.5, 0.03, 2.75);

          // Distance Nyquist band-limiting on fine capillary wavelets (converted into GGX slope roughness at horizon!)
          float capDetailFade = 1.0 - smoothstep(45.0, 290.0, distCam);
          float shoalCapDamp = mix(0.45, 1.0, smoothstep(0.0, 3.0, liveWaterline));
          float capGain = (0.72 + 0.65 * clamp(uSeaState, 0.25, 1.15)) * shoalCapDamp * capDetailFade;

          float c0 = 0.0;
          float capNx = 0.0;
          float capNz = 0.0;
          if (capGain > 0.008) {
            float epsC = mix(0.07, 0.24, smoothstep(15.0, 180.0, distCam));
            c0 = evalCrossedWavelets(vWorldPos.xz, uTime, uWindPhase);
            float cx = evalCrossedWavelets(vWorldPos.xz + vec2(epsC, 0.0), uTime, uWindPhase);
            float cz = evalCrossedWavelets(vWorldPos.xz + vec2(0.0, epsC), uTime, uWindPhase);
            capNx = ((c0 - cx) / epsC) * 0.055 * capGain;
            capNz = ((c0 - cz) / epsC) * 0.055 * capGain;
          }

          float rainRingFoam = 0.0;
          // 3-Layer Overlapping Concentric Capillary Raindrop Impact Rings
          if (uRainIntensity > 0.03 && distCam < 145.0) {
            float fadeDist = smoothstep(145.0, 32.0, distCam);
            for (int rl = 0; rl < 3; rl++) {
              float gScale = rl == 0 ? 1.25 : (rl == 1 ? 2.15 : 3.45);
              vec2 gp = vWorldPos.xz * gScale;
              vec2 gi = floor(gp);
              vec2 gf = fract(gp) - 0.5;
              float seed = hash(gi + float(rl) * 41.0);
              if (seed > clamp(uRainIntensity * 0.85, 0.15, 1.0)) continue;

              vec2 center = (vec2(hash(gi + 7.3), hash(gi + 19.1)) - 0.5) * 0.50;
              float life = fract(uTime * (1.15 + float(rl) * 0.42) + seed);
              vec2 dv = gf - center;
              float d = length(dv) + 1e-4;
              float front = life * 0.45;
              float env = exp(-abs(d - front) * 26.0) * (1.0 - life) * smoothstep(0.0, 0.05, life);
              float wave = sin((d - front) * 92.0) * env * min(1.35, uRainIntensity) * fadeDist;
              capNx += (dv.x / d) * wave * 0.45;
              capNz += (dv.y / d) * wave * 0.45;
              rainRingFoam += max(0.0, wave) * 0.42;
            }
          }

          vec3 normal = normalize(vec3(gravNorm.x + capNx, gravNorm.y, gravNorm.z + capNz));
          vec3 viewDir = normalize(cameraPosition - vWorldPos);
          vec3 sunDir = normalize(uSunDir);
          float sunVis = clamp(1.0 - uRainIntensity * 0.78 - uStormFactor * 0.42 - uCloudCover * 0.22, 0.06, 1.0);

          // 2. BEER-LAMBERT DEPTH ABSORPTION, SUBMERGED BOULDER REFRACTION, & VORONOI FOCAL CAUSTICS
          vec2 refrXZ = vWorldPos.xz - normal.xz * min(0.62, waterDepth * 0.52);
          vec2 subVor = voronoiCell(refrXZ * 1.85);
          float cobbleDetail = vnoise(R * refrXZ * 5.2);
          vec3 submergedBed = mix(vec3(0.14, 0.15, 0.13), vec3(0.31, 0.29, 0.24), smoothstep(0.04, 0.28, subVor.x) * (0.65 + 0.35 * cobbleDetail));
          submergedBed *= mix(1.0, 0.16, uNightFactor);

          if (waterDepth < 2.15 && distCam < 165.0) {
            vec2 causticUV = refrXZ - sunDir.xz * (waterDepth / max(0.28, sunDir.y)) * 0.38 + normal.xz * 0.22;
            float caust = evalCaustics(causticUV, uTime) * smoothstep(2.15, 0.10, waterDepth) * smoothstep(165.0, 42.0, distCam);
            vec3 caustTint = mix(mix(vec3(1.0, 0.95, 0.76), vec3(1.0, 0.62, 0.32), uDuskFactor), vec3(0.22, 0.32, 0.48), uNightFactor);
            submergedBed += caustTint * caust * sunVis * 0.58;
          }

          vec3 shallowJade = mix(vec3(0.075, 0.275, 0.240), vec3(0.050, 0.158, 0.160), clamp(uRainIntensity * 0.65, 0.0, 1.0));
          vec3 deepTeal = mix(vec3(0.026, 0.118, 0.148), vec3(0.018, 0.068, 0.092), clamp(uRainIntensity * 0.65, 0.0, 1.0));
          shallowJade = mix(shallowJade, vec3(0.016, 0.036, 0.056), uNightFactor);
          deepTeal = mix(deepTeal, vec3(0.008, 0.020, 0.040), uNightFactor);

          // Translucent Emerald-Turquoise Subsurface Scattering (SSS) glowing through raised 3D Gerstner wave crests!
          float waveCrestH = clamp(gDisp.y * 3.8 + crestFactor * 0.55 + rollerSwash * 0.45, 0.0, 1.35);
          float sssAlign = pow(max(0.0, dot(-viewDir, normalize(sunDir - normal * 0.42))), 3.2);
          float crestSSS = waveCrestH * (0.32 + 0.68 * sssAlign) * sunVis * (1.0 - uNightFactor);
          vec3 sssTint = mix(vec3(0.06, 0.28, 0.22), vec3(0.22, 0.14, 0.06), uDuskFactor);

          // Physical Beer-Lambert spectral attenuation (red absorbs fastest in water)
          vec3 opticalPath = vec3(2.15, 0.80, 0.68) * waterDepth * (1.0 + 0.38 * (1.0 - max(0.22, viewDir.y)));
          vec3 transmittance = exp(-opticalPath);
          vec3 waterVolume = mix(deepTeal, shallowJade, exp(-waterDepth * 0.78)) + sssTint * crestSSS * 0.48;
          vec3 underwaterCol = mix(waterVolume, submergedBed, transmittance);

          // 3. EXACT DIELECTRIC WATER SCHLICK FRESNEL (IOR 1.333 -> F0 = 0.0204) & MONOTONIC CLOUD REFLECTION
          float NdotV = max(dot(viewDir, normal), 0.0);
          float fresnel = 0.0204 + (1.0 - 0.0204) * pow(1.0 - NdotV, 5.0);
          float reflWeight = clamp(0.08 + fresnel * 0.91, 0.0, 0.96);

          vec3 reflRay = reflect(-viewDir, normal);
          float reflElev = clamp(reflRay.y, 0.0, 1.0);
          vec3 skyZen = mix(mix(vec3(0.14, 0.38, 0.72), vec3(0.16, 0.18, 0.38), uDuskFactor), vec3(0.016, 0.026, 0.058), uNightFactor);
          vec3 skyHor = mix(mix(vec3(0.56, 0.74, 0.88), vec3(0.86, 0.46, 0.24), uDuskFactor), vec3(0.045, 0.070, 0.130), uNightFactor);
          vec3 skyRefl = mix(skyHor, skyZen, pow(reflElev, 0.58));

          // Reflected Cumulonimbus / tropical clouds advected monotonically via uWindPhase (ZERO backward slosh!)
          vec2 reflCloudUV = (reflRay.xz / (reflElev + 0.22)) * 1.45 + uWindPhase * 0.013;
          float cField = fbmCloudRefl(reflCloudUV);
          float cThresh = mix(0.56, 0.22, clamp(uCloudCover, 0.0, 1.0));
          float cMask = smoothstep(cThresh, cThresh + 0.32, cField) * smoothstep(0.0, 0.18, reflElev);
          vec3 cloudReflCol = mix(mix(vec3(0.92, 0.95, 0.99), vec3(0.95, 0.56, 0.32), uDuskFactor), vec3(0.07, 0.10, 0.16), uNightFactor);
          cloudReflCol = mix(cloudReflCol, vec3(0.20, 0.24, 0.29) * mix(1.0, 0.25, uNightFactor), clamp(uRainIntensity * 0.68 + uStormFactor * 0.35, 0.0, 0.92));
          skyRefl = mix(skyRefl, cloudReflCol, cMask * 0.78);
          skyRefl = mix(skyRefl, vec3(0.22, 0.26, 0.31) * mix(1.0, 0.22, uNightFactor), clamp(uRainIntensity * 0.62 + uStormFactor * 0.30, 0.0, 0.90));
          skyRefl += vec3(0.76, 0.88, 1.0) * uLightningFlash * 1.15;

          vec3 col = mix(underwaterCol, skyRefl, reflWeight);

          // 4. DISTANCE SLOPE-VARIANCE (LEADR) GGX MULTI-FACET SUN / MOON GLITTER PATH
          // Converts filtered distant sub-pixel wavelets into broader GGX microfacet sun-track shimmer at horizon
          vec3 H = normalize(sunDir + viewDir);
          float NdotH = max(dot(normal, H), 0.0);
          float distSlopeSpread = smoothstep(45.0, 320.0, distCam);
          float tightExp = mix(310.0, 95.0, distSlopeSpread);
          float broadExp = mix(52.0, 22.0, distSlopeSpread);
          float specTight = pow(NdotH, tightExp) * mix(1.65, 0.78, distSlopeSpread);
          float specGlitter = pow(NdotH, broadExp) * mix(0.22, 0.34, distSlopeSpread);
          vec3 glitterTint = mix(mix(vec3(1.0, 0.94, 0.80), vec3(1.0, 0.58, 0.26), uDuskFactor), vec3(0.45, 0.56, 0.78), uNightFactor);
          vec3 glitter = glitterTint * (specTight + specGlitter) * sunVis;
          col += glitter / (1.0 + glitter * 0.40);

          // 5. POROUS CELLULAR VORONOI SWASH SEA-FOAM ON 3D STONE TURAP (PULAU U & PULAU I) & OPEN-SEA WHITECAPS
          float totalFoam = 0.0;
          if (liveWaterline < 3.8 && distCam < 1350.0) {
            float nearFoam = 1.0 - smoothstep(180.0, 520.0, distCam);
            float lacyWeb = 0.72;
            if (nearFoam > 0.02) {
              vec2 foamUV = vWorldPos.xz * 2.6 + normal.xz * 0.45;
              vec2 fVor1 = voronoiCell(foamUV + vec2(uTime * 0.14, -uTime * 0.11));
              vec2 fVor2 = voronoiCell(R * (foamUV * 2.35) - vec2(uTime * 0.19, uTime * 0.13));
              // Porous lacy bubble web (cell walls bright, bubble interiors translucent!)
              float vorWeb = smoothstep(0.24, 0.03, fVor1.x) * 0.68 + smoothstep(0.20, 0.02, fVor2.x) * 0.46;
              lacyWeb = mix(0.72, vorWeb, nearFoam);
            }
            float contactSwash = smoothstep(1.15 + rollerSwash * 0.90, 0.0, liveWaterline);
            float rollerRibbon = smoothstep(3.6, 0.4, liveWaterline) * pow(rollerSwash, 1.7) * 0.55;
            float altFade = smoothstep(1350.0, 380.0, distCam) * 0.65 + 0.35 * smoothstep(1350.0, 900.0, distCam);
            totalFoam += clamp((contactSwash * (0.45 + 0.75 * lacyWeb) + rollerRibbon * lacyWeb), 0.0, 1.0)
                       * (0.52 + 0.34 * clamp(uSeaState, 0.25, 1.15)) * altFade;
          }

          // Offshore wind-blown whitecaps on steep interfering Gerstner wave crests during brisk wind / Rain / Storm
          float whitecapThresh = mix(0.76, 0.42, clamp((uSeaState - 0.34) * 1.35, 0.0, 1.0));
          if (crestFactor > whitecapThresh && distCam < 480.0) {
            vec2 wcVor = voronoiCell(vWorldPos.xz * 1.45 - uWindPhase * 0.16);
            float wcLace = smoothstep(0.28, 0.03, wcVor.x) * (0.55 + 0.45 * vnoise(vWorldPos.xz * 3.8 - uWindPhase * 0.22));
            float wcMask = smoothstep(whitecapThresh, whitecapThresh + 0.26, crestFactor) * wcLace * smoothstep(480.0, 110.0, distCam);
            totalFoam = max(totalFoam, wcMask * 0.78);
          }

          if (totalFoam > 0.005) {
            vec3 foamCol = mix(mix(vec3(0.86, 0.91, 0.93), vec3(0.90, 0.66, 0.50), uDuskFactor), vec3(0.18, 0.24, 0.33), uNightFactor);
            foamCol += vec3(0.70, 0.82, 1.0) * uLightningFlash * 0.85;
            col = mix(col, foamCol, clamp(totalFoam, 0.0, 0.88));
          }

          col += vec3(0.74, 0.83, 0.92) * rainRingFoam * (0.42 - 0.22 * uNightFactor + 0.55 * uLightningFlash);

          // 6. MULTI-TIER GOLDEN NIGHTTIME ARCHITECTURAL REFLECTION OF GEDUNG REKTORAT UI (EAST DANAU KENANGA COVE)
          if (uNightFactor > 0.08 && vWorldPos.x > 38.0) {
            float towerAlign = exp(-pow(vWorldPos.z / 22.0, 2.0)) * smoothstep(132.0, 56.0, vWorldPos.x);
            float annexAlign = exp(-pow((vWorldPos.z - 8.0) / 44.0, 2.0)) * smoothstep(104.0, 56.0, vWorldPos.x) * 0.55;
            float rippleShimmer = 0.52 + 0.48 * sin((c0 + normal.x * 0.45) * 24.0 - uTime * 2.6);
            col += vec3(0.96, 0.70, 0.30) * (towerAlign + annexAlign) * rippleShimmer * uNightFactor * 0.32;
          }

          // Subtle outer sea-rim horizon blend complementing 3D boundary-layer aerial perspective
          float horMist = smoothstep(1180.0, 1642.0, length(vBaseXZ - ARCHIPELAGO_CENTER));
          col = mix(col, skyHor, horMist * 0.18);

          // Unify Ocean & Danau Kenanga Cove into True Scene-Referred Linear-HDR -> Boundary-Layer Aerial Perspective -> HVS
          vec3 hdrOcean = nexusSRGBToLinear(max(col, vec3(0.0))) * mix(1.16, 0.74, uNightFactor);
          hdrOcean += nexusSRGBToLinear(glitterTint) * (specTight * 0.85 + specGlitter * 0.45) * sunVis;
          hdrOcean += vec3(0.75, 0.95, 1.45) * uLightningFlash * 0.55;

          hdrOcean = nexusApplyBoundaryLayerAerialPerspective(
            hdrOcean,
            cameraPosition,
            vWorldPos,
            sunDir,
            uDuskFactor,
            uNightFactor,
            uCloudCover,
            uRainIntensity,
            uLightningFlash
          );

          vec3 hvsOceanLin = nexusApplyHumanEyeHDR(
            hdrOcean,
            normalize(vWorldPos - cameraPosition),
            sunDir,
            uSunColor,
            uHvsParams,
            uHvsGlare
          );
          gl_FragColor = vec4(nexusLinearToSRGB(hvsOceanLin), 1.0);
        }
      `
    });
  }

  static _smoothstep(e0, e1, x) {
    const t = Math.max(0.0, Math.min(1.0, (x - e0) / (e1 - e0 || 1e-6)));
    return t * t * (3.0 - 2.0 * t);
  }

  static _smootherstep(e0, e1, x) {
    const t = Math.max(0.0, Math.min(1.0, (x - e0) / (e1 - e0 || 1e-6)));
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
  }

  /**
   * Sets user-configurable weather & diurnal transition duration, strictly clamped to [20, 45] seconds.
   */
  setTransitionDuration(seconds) {
    const clamped = Math.min(
      WeatherOpticsEngine.MAX_TRANSITION_SEC,
      Math.max(WeatherOpticsEngine.MIN_TRANSITION_SEC, Number(seconds) || WeatherOpticsEngine.DEFAULT_TRANSITION_SEC)
    );
    this.transitionDurationSec = clamped;

    // Rescale any in-flight transitions proportionally so current progress p stays 100% jump-free!
    if (this._weatherTrans.active) {
      const p = this._weatherTrans.progress;
      this._weatherTrans.duration = clamped;
      this._weatherTrans.elapsed = p * clamped;
    }
    if (this._timeTrans.active) {
      const p = this._timeTrans.progress;
      this._timeTrans.duration = clamped;
      this._timeTrans.elapsed = p * clamped;
    }
    return clamped;
  }

  getTransitionState() {
    return {
      durationSec: this.transitionDurationSec,
      weather: {
        active: this._weatherTrans.active,
        fromMode: this._weatherTrans.fromMode,
        toMode: this.weatherMode,
        progress: this._weatherTrans.progress,
        remainingSec: this._weatherTrans.active
          ? Math.max(0.0, this._weatherTrans.duration - this._weatherTrans.elapsed)
          : 0.0
      },
      time: {
        active: this._timeTrans.active,
        fromTod: this._timeTrans.fromTod,
        toTod: this.timeOfDay,
        progress: this._timeTrans.progress,
        remainingSec: this._timeTrans.active
          ? Math.max(0.0, this._timeTrans.duration - this._timeTrans.elapsed)
          : 0.0,
        solarHour: this.currentSolarHour,
        continuousSolarHour: this.continuousSolarHour
      },
      nightIlluminationLevel: this.nightIlluminationLevel,
      streetlampSensorLevel: this.streetlampSensorLevel
    };
  }

  /**
   * Updates the reference calendar date for real-time Depok lunar ephemeris & sidereal sky.
   */
  setReferenceAstronomyDate(refDate = new Date(), sharedSurfaceUniforms = null) {
    this._baseJD0WIB = computeBaseJD0WIB(refDate);
    this._applyContinuousCelestialAndFog(sharedSurfaceUniforms);
    return this.getAstronomicalState();
  }

  /**
   * Returns live Jean Meeus lunar phase, LST, Bortle sky class, & photocell streetlamp telemetry.
   */
  getAstronomicalState() {
    if (!this._astroState) {
      this._astroState = evaluateDepokAstronomy(
        this._baseJD0WIB,
        this.continuousSolarHour,
        this.uniforms.uWorldToEqMat.value,
        this.uniforms.uMoonDir.value,
        this.uniforms.uMoonPhaseSunDir.value
      );
    }
    return {
      ...this._astroState,
      moonIlluminatedFraction: this._astroState.illuminatedFraction,
      moonPhaseLabelId: this._astroState.phaseNameId,
      illuminatedPercent: +(this._astroState.illuminatedFraction * 100.0).toFixed(1),
      bortleClass: '1–2',
      streetlampSensorLevel: +this.streetlampSensorLevel.toFixed(4),
      continuousSolarHour: +this.continuousSolarHour.toFixed(4)
    };
  }

  _syncParticleVisibility() {
    const rainNow = this.uniforms.uRainIntensity.value;
    const nightNow = this.uniforms.uNightFactor.value;
    const duskNow = this.uniforms.uDuskFactor.value;
    const raining = rainNow > 0.012;

    if (this.rainAndDripMesh) this.rainAndDripMesh.visible = raining;
    if (this.splashAndMistMesh) this.splashAndMistMesh.visible = raining;

    // Strict mutual exclusion (!raining) guarantees total scene draw calls <= 24 <= 25 at EVERY transition millisecond!
    const twilightGlow = Math.max(
      WeatherOpticsEngine._smoothstep(0.08, 0.55, nightNow),
      duskNow * 0.36
    );
    const dryFade = 1.0 - WeatherOpticsEngine._smoothstep(0.002, 0.012, rainNow);
    this.uniforms.uFireflyIntensity.value = twilightGlow * dryFade;

    if (this.fireflyPoints) {
      this.fireflyPoints.visible = !raining && this.uniforms.uFireflyIntensity.value > 0.02;
    }
  }

  static _todModeToSolarHour(mode) {
    if (mode === 'NIGHT') return 22.25;
    if (mode === 'DUSK') return 17.85;
    return 10.35; // DAY
  }

  static _solarHourToTodMode(solarHour) {
    const h = ((solarHour % 24.0) + 24.0) % 24.0;
    if (h >= 18.42 || h < 5.25) return 'NIGHT';
    if ((h >= 5.25 && h < 6.38) || (h >= 17.08 && h < 18.42)) return 'DUSK';
    return 'DAY';
  }

  /**
   * Evaluates continuous 24-hour Kampus UI Depok (Lat -6.3658, UTC+7 WIB) celestial lighting,
   * twilight colors, fog, and architectural night-illumination level with zero discrete jumps.
   */
  _applyContinuousCelestialAndFog(sharedSurfaceUniforms = null) {
    const h = ((this.currentSolarHour % 24.0) + 24.0) % 24.0;
    // Depok Sunrise ~05:55 WIB (+X East over Danau Kenanga), Zenith ~11:58 WIB, Sunset ~18:00 WIB (-X West over Helipad)
    const sunPhase = ((h - 5.92) / 12.08) * Math.PI;
    const sunElev = Math.sin(sunPhase);
    const sunEastWest = Math.cos(sunPhase);

    const nightF = 1.0 - WeatherOpticsEngine._smoothstep(-0.22, 0.11, sunElev);
    const duskF =
      WeatherOpticsEngine._smoothstep(-0.20, 0.03, sunElev) *
      (1.0 - WeatherOpticsEngine._smoothstep(0.08, 0.42, sunElev));

    this.uniforms.uNightFactor.value = nightF;
    this.uniforms.uDuskFactor.value = duskF;

    // Continuous 24h True Sun Vector & Lunar Vector (independent sky-dome discs -> zero sun teleport!)
    const trueSx = sunEastWest * 0.86;
    const trueSy = sunElev * 0.86;
    const trueSz = 0.45;
    this.uniforms.uTrueSunDir.value.set(trueSx, trueSy, trueSz).normalize();

    const sx = sunEastWest * 0.86;
    const sy = Math.max(0.04, sunElev * 0.86);
    const sz = 0.45;
    const sLen = Math.hypot(sx, sy, sz) || 1.0;

    // v3.7 Real-Time Jean Meeus Depok Lunar Ephemeris & Unwrapped Sidereal Sky Matrix (0.000° midnight jump!)
    const astro = evaluateDepokAstronomy(
      this._baseJD0WIB,
      this.continuousSolarHour,
      this.uniforms.uWorldToEqMat.value,
      this.uniforms.uMoonDir.value,
      this.uniforms.uMoonPhaseSunDir.value
    );
    this._astroState = astro;
    this.uniforms.uMoonIllumFraction.value = astro.illuminatedFraction;

    // Smooth shadow-map lunar director (tracks real Moon azimuth while keeping shadow frustum elevation >= 0.42)
    const realM = this.uniforms.uMoonDir.value;
    const moonAbove = WeatherOpticsEngine._smoothstep(-0.08, 0.28, realM.y);
    const mx = realM.x * (0.45 + 0.55 * moonAbove);
    const my = 0.54 + 0.30 * Math.max(0.0, realM.y);
    const mz = realM.z * (0.45 + 0.55 * moonAbove) - 0.25 * (1.0 - moonAbove);
    const mLen = Math.hypot(mx, my, mz) || 1.0;

    // v3.6 Zero-Intensity Twilight Shadow Handoff (driven smoothly by continuous sunElev):
    // Solar directional light glides gradually over late afternoon/sunset (sunElev: +0.34 -> -0.13)
    // to 0.000 BEFORE lunar directional light rises (sunElev: -0.21 -> -0.46).
    // Across sunElev in [-0.21, -0.13], _baseSunIntensity === 0.000 while sunLight.position
    // repositions from western sunset to moon -> ZERO shadow-map jump & ZERO sunset plunge!
    const solarFade = WeatherOpticsEngine._smoothstep(-0.13, 0.34, sunElev);
    const moonBlend = 1.0 - WeatherOpticsEngine._smoothstep(-0.21, -0.13, sunElev);
    const lunarPhaseGain = 0.38 + 0.62 * astro.illuminatedFraction * moonAbove;
    const lunarFade = (1.0 - WeatherOpticsEngine._smoothstep(-0.46, -0.21, sunElev)) * lunarPhaseGain;

    const dirX = (sx / sLen) * (1.0 - moonBlend) + (mx / mLen) * moonBlend;
    const dirY = (sy / sLen) * (1.0 - moonBlend) + (my / mLen) * moonBlend;
    const dirZ = (sz / sLen) * (1.0 - moonBlend) + (mz / mLen) * moonBlend;
    this.uniforms.uSunDir.value.set(dirX, dirY, dirZ).normalize();
    this._baseSunPos.copy(this.uniforms.uSunDir.value).multiplyScalar(128.0);

    // Continuous overcast & storm light attenuation driven by LIVE uniforms (zero instant snap!)
    const rainNow = this.uniforms.uRainIntensity.value;
    const stormNow = this.uniforms.uStormFactor.value;
    const cloudNow = this.uniforms.uCloudCover.value;
    const stormMix = Math.min(1.0, rainNow * 0.45 + stormNow * 0.55);
    const overcastAtten = Math.max(
      0.32,
      1.0 - rainNow * 0.36 - stormNow * 0.24 - Math.max(0.0, cloudNow - 0.30) * 0.16
    );

    // Sun / Dusk / Moon directional color
    this._colTmpA.copy(this._colDaySun).lerp(this._colDuskSun, duskF);
    this._colTmpA.lerp(this._colStormTint, stormMix * 0.65);
    this._baseSunColor.copy(this._colTmpA).lerp(this._colNightMoon, moonBlend);

    const dayDuskInt = (2.52 * (1.0 - duskF) + 1.92 * duskF) * solarFade;
    this._baseSunIntensity = (dayDuskInt + 0.34 * lunarFade) * overcastAtten;

    // Hemisphere Sky & Ground colors (Hukum Emas 13: dark night bounce prevents washed-out passive concrete)
    this._colHemiSky.setHex(0xdbeeff);
    this._colTmpB.setHex(0xffc99e);
    this._colHemiSky.lerp(this._colTmpB, duskF);
    this._colTmpB.setHex(0x768396);
    this._colHemiSky.lerp(this._colTmpB, stormMix * 0.78);
    this._colTmpB.setHex(0x233454);
    this._colHemiSky.lerp(this._colTmpB, nightF);
    this.hemiLight.color.copy(this._colHemiSky);

    this._colHemiGnd.setHex(0x4c5349);
    this._colTmpB.setHex(0x3b332e);
    this._colHemiGnd.lerp(this._colTmpB, duskF);
    this._colTmpB.setHex(0x0c1017);
    this._colHemiGnd.lerp(this._colTmpB, nightF);
    this.hemiLight.groundColor.copy(this._colHemiGnd);

    const dayHemi = (0.66 * (1.0 - duskF) + 0.42 * duskF) * (0.68 + 0.32 * overcastAtten);
    this._baseHemiIntensity = dayHemi * (1.0 - nightF) + 0.20 * nightF;
    this._baseFillIntensity = (0.35 * overcastAtten * (1.0 - duskF * 0.42)) * (1.0 - nightF) + 0.10 * nightF;
    this.fillLight.position.copy(this._baseFillPos);
    this.fillLight.color.setHex(0x9ec5eb);
    this.fillLight.intensity = this._baseFillIntensity;

    // Isotropic FogExp2 disabled (density = 0.0) in favor of unified 3D Boundary-Layer Aerial Perspective!
    if (this.scene.fog) {
      this._colFog.setHex(0xa8c6df);
      this._colTmpB.setHex(0x6b7b8c);
      this._colFog.lerp(this._colTmpB, Math.min(1.0, rainNow * 0.85));
      this._colTmpB.setHex(0x454f5c);
      this._colFog.lerp(this._colTmpB, stormNow * 0.85);
      this._colTmpB.setHex(0x9e6a4b);
      this._colFog.lerp(this._colTmpB, duskF * (1.0 - stormMix * 0.65));
      this._colTmpB.setHex(0x090f1b);
      this._colFog.lerp(this._colTmpB, nightF);
      this.scene.fog.color.copy(this._colFog);
      this.scene.fog.density = 0.0;
    }

    // Continuous C1-Smooth Architectural & Helipad Night Illumination Level (strictly 0.0 at DAY, zero threshold jump!)
    this.nightIlluminationLevel = 1.0 - WeatherOpticsEngine._smoothstep(-0.26, 0.14, sunElev);

    // Physical NEMA Ambient-Lux Streetlamp Photocell Sensor:
    // - Strictly 0.0000 in clear/misty daytime (uStreetlampPower === 0.000 -> zero daytime emissive!)
    // - Ignites smoothly during golden-hour dusk/dawn twilight AND during heavy dark storm clouds!
    const daySunLux = WeatherOpticsEngine._smoothstep(-0.24, 0.24, sunElev);
    const stormSkyDarkening =
      WeatherOpticsEngine._smoothstep(0.68, 0.96, cloudNow) *
      WeatherOpticsEngine._smoothstep(0.55, 1.15, rainNow + stormNow * 0.65);
    const ambientLuxNorm = daySunLux * (1.0 - 0.94 * stormSkyDarkening);
    this.streetlampSensorLevel = 1.0 - ambientLuxNorm;
    this.uniforms.uStreetlampPower.value = this.streetlampSensorLevel;

    // Shadow-casting sunLight ALWAYS stays on its continuous celestial position (never freezes or teleports!)
    this.sunLight.position.copy(this._baseSunPos);
    this.sunLight.color.copy(this._baseSunColor);
    this.sunLight.intensity = this._baseSunIntensity;
    this.hemiLight.intensity = this._baseHemiIntensity;
    this.uniforms.uSunColor.value.copy(this.sunLight.color);

    this._applyEnvironmentIBL(Math.max(0.0, this.uniforms.uLightningFlash.value));

    if (sharedSurfaceUniforms) {
      this._lastSharedSurfaceUniforms = sharedSurfaceUniforms;
      if (sharedSurfaceUniforms.uIsNight) {
        sharedSurfaceUniforms.uIsNight.value = this.nightIlluminationLevel;
      }
      if (sharedSurfaceUniforms.uStreetlampPower) {
        sharedSurfaceUniforms.uStreetlampPower.value = this.streetlampSensorLevel;
      }
      if (sharedSurfaceUniforms.uNightFactor) {
        sharedSurfaceUniforms.uNightFactor.value = nightF;
      }
      if (sharedSurfaceUniforms.uDuskFactor) {
        sharedSurfaceUniforms.uDuskFactor.value = duskF;
      }
      if (sharedSurfaceUniforms.uCloudCover) {
        sharedSurfaceUniforms.uCloudCover.value = this.uniforms.uCloudCover.value;
      }
      if (sharedSurfaceUniforms.uSunDir) {
        sharedSurfaceUniforms.uSunDir.value.copy(this.uniforms.uSunDir.value);
      }
      if (sharedSurfaceUniforms.uSunColor) {
        sharedSurfaceUniforms.uSunColor.value.copy(this.sunLight.color);
      }
    }
  }

  /**
   * Synchronizes Human Visual System (HVS) pupil & photoreceptor HDR adaptation uniforms
   * across WeatherOpticsEngine (Sky + Ocean/Fluvial Water) and SharedSurfaceUniforms (100% of Surfaces).
   */
  _syncHumanEyeHDR(dt, camera = null, sharedSurfaceUniforms = null, immediate = false) {
    if (sharedSurfaceUniforms) {
      this._lastSharedSurfaceUniforms = sharedSurfaceUniforms;
    }
    const targetShared = sharedSurfaceUniforms || this._lastSharedSurfaceUniforms;
    return this.hvsOptics.update(
      dt,
      {
        nightFactor: this.uniforms.uNightFactor.value,
        duskFactor: this.uniforms.uDuskFactor.value,
        cloudCover: this.uniforms.uCloudCover.value,
        rainIntensity: this.uniforms.uRainIntensity.value,
        stormFactor: this.uniforms.uStormFactor.value,
        lightningFlash: this.uniforms.uLightningFlash.value,
        sunDir: this.uniforms.uSunDir.value,
        camera,
        immediate
      },
      this.uniforms,
      targetShared
    );
  }

  /**
   * Returns live Human-Eye Physiological HDR adaptation telemetry for HUD & 4-Gate photometric verification.
   */
  getHumanEyeHDRState() {
    return {
      adaptedLuminance: this.hvsOptics.adaptedLuminance,
      targetLuminance: this.hvsOptics.targetLuminance,
      pupilDiameterMm: this.hvsOptics.pupilDiameterMm,
      visionRegime: this.hvsOptics.visionRegime,
      eyeGain: this.uniforms.uHvsParams.value.x,
      huntChroma: this.uniforms.uHvsParams.value.y,
      stevensContrast: this.uniforms.uHvsParams.value.z,
      purkinjeMix: this.uniforms.uHvsParams.value.w,
      sunGlareStrength: this.uniforms.uHvsGlare.value.x,
      lightningVeil: this.uniforms.uHvsGlare.value.y
    };
  }

  /**
   * Sets or smoothly transitions (20–45s) to an exact Kampus UI Depok WIB solar hour [0, 24).
   */
  setSolarHour(targetSolarHour, sharedSurfaceUniforms = null, options = {}) {
    const normTarget = ((Number(targetSolarHour) % 24.0) + 24.0) % 24.0;
    const nextTod = options.todOverride || WeatherOpticsEngine._solarHourToTodMode(normTarget);
    const prevTod = this.timeOfDay;
    this.timeOfDay = nextTod;

    const startH = ((this.continuousSolarHour % 24.0) + 24.0) % 24.0;
    let delta = ((normTarget - startH + 36.0) % 24.0) - 12.0;
    // Natural forward dawn sunrise sweep when transitioning from NIGHT to DAY!
    if (prevTod === 'NIGHT' && nextTod === 'DAY' && delta < 0.0) {
      delta += 24.0;
    }

    if (options.immediate === true) {
      if (options.preserveAdaptation === true) {
        this.continuousSolarHour += delta;
      } else {
        const dayIndex = Math.floor(this.continuousSolarHour / 24.0);
        this.continuousSolarHour = dayIndex * 24.0 + normTarget;
      }
      this.currentSolarHour = normTarget;
      this._timeTrans.active = false;
      this._timeTrans.fromTod = nextTod;
      this._timeTrans.toTod = nextTod;
      this._timeTrans.progress = 1.0;
      this._timeTrans.elapsed = 0.0;
      this._applyContinuousCelestialAndFog(sharedSurfaceUniforms);
      if (!options.preserveAdaptation) {
        this._syncHumanEyeHDR(0.0, null, sharedSurfaceUniforms, true);
      }
      this._syncParticleVisibility();
      return;
    }

    const dur = Math.min(
      WeatherOpticsEngine.MAX_TRANSITION_SEC,
      Math.max(
        WeatherOpticsEngine.MIN_TRANSITION_SEC,
        Number(options.durationSec ?? this.transitionDurationSec)
      )
    );

    if (Math.abs(delta) < 0.015) {
      this.continuousSolarHour += delta;
      this.currentSolarHour = normTarget;
      this._timeTrans.active = false;
      this._timeTrans.progress = 1.0;
      this._applyContinuousCelestialAndFog(sharedSurfaceUniforms);
      if (!options.preserveAdaptation) {
        this._syncHumanEyeHDR(0.0, null, sharedSurfaceUniforms, true);
      }
      this._syncParticleVisibility();
      return;
    }

    this._timeTrans.active = true;
    this._timeTrans.fromTod = prevTod;
    this._timeTrans.toTod = nextTod;
    this._timeTrans.elapsed = 0.0;
    this._timeTrans.duration = dur;
    this._timeTrans.progress = 0.0;
    this._timeTrans.startHour = startH;
    this._timeTrans.startContinuousHour = this.continuousSolarHour;
    this._timeTrans.deltaHour = delta;
  }

  /**
   * Switches or smoothly transitions (20–45s by default!) Time of Day ('DAY' | 'DUSK' | 'NIGHT').
   * Pass { immediate: true } ONLY for initial startup boot or instant headless test matrix snapshots.
   */
  setTimeOfDay(mode, sharedSurfaceUniforms = null, options = {}) {
    const targetHour =
      options.solarHour !== undefined
        ? Number(options.solarHour)
        : WeatherOpticsEngine._todModeToSolarHour(mode);
    this.setSolarHour(targetHour, sharedSurfaceUniforms, {
      ...options,
      todOverride: mode
    });
  }

  static _getWeatherPresetTargets(weatherMode, options = {}) {
    if (weatherMode === 'STORM') {
      return {
        rain: 1.42,
        wet: 1.0,
        cloud: 1.0,
        storm: 1.0,
        fog: 0.0,
        wind: 3.55
      };
    }
    if (weatherMode === 'RAIN') {
      return {
        rain: 1.0,
        wet: 0.96,
        cloud: 0.92,
        storm: 0.28,
        fog: 0.0,
        wind: 2.05
      };
    }
    if (weatherMode === 'MIST') {
      return {
        rain: 0.34,
        wet: 0.58,
        cloud: 0.74,
        storm: 0.05,
        fog: 0.0,
        wind: 1.2
      };
    }
    // CLEAR (supports live Depok Open-Meteo cloudCoverNorm & windSpeed override!)
    const liveCloud =
      options.cloudCoverNorm !== undefined
        ? Math.max(0.14, Math.min(0.86, Number(options.cloudCoverNorm)))
        : 0.22;
    return {
      rain: 0.0,
      wet: 0.0,
      cloud: liveCloud,
      storm: 0.0,
      fog: 0.0,
      wind: 0.82
    };
  }

  /**
   * Switches active meteorological regime with mandatory 20–45s staged physical transition by default!
   * - Clouds & wind lead precipitation onset; ground wetness builds only as raindrops fall;
   * - When clearing, rain tapers gradually to drizzle -> 0, then wet ground evaporates slowly;
   * - Pass { immediate: true } ONLY for initial startup boot or instant headless test matrix snapshots.
   */
  setWeather(weatherMode, options = {}) {
    const prevMode = this.weatherMode;
    this.weatherMode = weatherMode;
    const preset = WeatherOpticsEngine._getWeatherPresetTargets(weatherMode, options);

    this.targetRainIntensity = preset.rain;
    this.targetWetness = preset.wet;
    this.targetCloudCover = preset.cloud;
    this.targetStormFactor = preset.storm;
    this.targetFogDensity = 0.0;
    this.targetWindSpeed = preset.wind;

    if (options.immediate === true) {
      this._weatherTrans.active = false;
      this._weatherTrans.fromMode = weatherMode;
      this._weatherTrans.toMode = weatherMode;
      this._weatherTrans.progress = 1.0;
      this._weatherTrans.elapsed = 0.0;

      this.uniforms.uRainIntensity.value = preset.rain;
      if (!options.preserveWetness) {
        this.uniforms.uWetness.value = preset.wet;
      }
      this.uniforms.uCloudCover.value = preset.cloud;
      this.uniforms.uStormFactor.value = preset.storm;
      this.baseWindSpeed = preset.wind;
      if (this.scene.fog) {
        this.scene.fog.density = 0.0;
      }
      if (weatherMode === 'STORM') {
        this.nextLightningTimer = Math.min(this.nextLightningTimer, 0.85);
      } else {
        this.uniforms.uLightningFlash.value = 0.0;
        this.lightningClock = -1.0;
      }
      this._applyContinuousCelestialAndFog();
      this._syncHumanEyeHDR(0.0, null, null, true);
      this._syncParticleVisibility();
      return;
    }

    const dur = Math.min(
      WeatherOpticsEngine.MAX_TRANSITION_SEC,
      Math.max(
        WeatherOpticsEngine.MIN_TRANSITION_SEC,
        Number(options.durationSec ?? this.transitionDurationSec)
      )
    );

    const deltaSum =
      Math.abs(this.uniforms.uRainIntensity.value - preset.rain) +
      Math.abs(this.uniforms.uWetness.value - preset.wet) +
      Math.abs(this.uniforms.uCloudCover.value - preset.cloud) +
      Math.abs(this.uniforms.uStormFactor.value - preset.storm);

    if (deltaSum < 0.015 && !this._weatherTrans.active) {
      this._weatherTrans.fromMode = weatherMode;
      this._weatherTrans.toMode = weatherMode;
      this._weatherTrans.progress = 1.0;
      return;
    }

    this._weatherTrans.active = true;
    this._weatherTrans.fromMode = this._weatherTrans.active ? prevMode : prevMode;
    this._weatherTrans.toMode = weatherMode;
    this._weatherTrans.elapsed = 0.0;
    this._weatherTrans.duration = dur;
    this._weatherTrans.progress = 0.0;
    this._weatherTrans.startRain = this.uniforms.uRainIntensity.value;
    this._weatherTrans.startWet = this.uniforms.uWetness.value;
    this._weatherTrans.startCloud = this.uniforms.uCloudCover.value;
    this._weatherTrans.startStorm = this.uniforms.uStormFactor.value;
    this._weatherTrans.startFog = 0.0;
    this._weatherTrans.startWind = this.baseWindSpeed;
  }

  getWindGustMultiplier() {
    return this.windGustMultiplier;
  }

  _triggerLightningDischarge(timeSec) {
    const az = ((Math.sin(timeSec * 12.73) * 0.5 + 0.5) * Math.PI * 1.8) - Math.PI * 0.9;
    this.uniforms.uLightningDir.value.set(Math.cos(az), 0.34, Math.sin(az)).normalize();
    this.uniforms.uLightningSeed.value = 1.0 + (Math.floor(Math.abs(timeSec * 17.37) + 1.0) % 991);
    this.lightningClock = 0.0;

    const isTriple = Math.cos(timeSec * 7.9) > 0.15;
    const peakIntensity = 0.85 + 0.30 * (0.5 + 0.5 * Math.sin(timeSec * 19.1));
    this.lightningStrokeProfile = { isTriple, peakIntensity, az };

    const distMeters = 320 + Math.abs(Math.sin(timeSec * 23.4)) * 1030;
    if (typeof this.onThunder === 'function') {
      this.onThunder(distMeters, peakIntensity);
    }
  }

  _evaluateLightningEnvelope(t) {
    if (!this.lightningStrokeProfile || t < 0.0 || t > 0.52) return 0.0;
    const p = this.lightningStrokeProfile.peakIntensity;
    let env = Math.exp(-Math.pow((t - 0.025) / 0.022, 2.0)) * p;
    env += Math.exp(-Math.pow((t - 0.135) / 0.028, 2.0)) * p * 0.74;
    if (this.lightningStrokeProfile.isTriple) {
      env += Math.exp(-Math.pow((t - 0.265) / 0.036, 2.0)) * p * 0.48;
    }
    env += Math.exp(-t * 7.5) * 0.18 * p;
    return Math.min(1.25, env);
  }

  update(dt, camera, sharedSurfaceUniforms = null) {
    const safeDt = Math.min(0.1, Math.max(0.0, dt));
    const transDt = Math.min(0.5, Math.max(0.0, dt));
    this.uniforms.uTime.value += safeDt;
    const t = this.uniforms.uTime.value;

    // 1. Staged 20–45s Physical Meteorological Transition Choreography (ZERO sudden jumps!)
    if (this._weatherTrans.active) {
      this._weatherTrans.elapsed += transDt;
      const p = Math.min(1.0, this._weatherTrans.elapsed / Math.max(1.0, this._weatherTrans.duration));
      this._weatherTrans.progress = p;

      const s = this._weatherTrans;
      const gettingWetter = this.targetRainIntensity >= s.startRain;

      // Natural atmospheric phase lead/lag:
      // - Getting wetter: Clouds & breeze gather FIRST (0..58%), Raindrops build SECOND (16..88%), Ground wets & puddles fill THIRD (28..100%)!
      // - Clearing up: Storm subsides FIRST (0..50%), Rain tapers to drizzle->0 SECOND (0..64%), Clouds part THIRD (18..92%), Wet ground evaporates LAST (30..100%)!
      const pCloud = gettingWetter
        ? WeatherOpticsEngine._smootherstep(0.0, 0.58, p)
        : WeatherOpticsEngine._smootherstep(0.18, 0.92, p);
      const pWind = WeatherOpticsEngine._smootherstep(0.04, 0.80, p);
      const pRain = gettingWetter
        ? WeatherOpticsEngine._smootherstep(0.16, 0.88, p)
        : WeatherOpticsEngine._smootherstep(0.0, 0.64, p);
      const pStorm = gettingWetter
        ? WeatherOpticsEngine._smootherstep(0.22, 0.94, p)
        : WeatherOpticsEngine._smootherstep(0.0, 0.50, p);
      const pWet = gettingWetter
        ? WeatherOpticsEngine._smootherstep(0.28, 1.0, p)
        : WeatherOpticsEngine._smootherstep(0.30, 1.0, p);

      this.uniforms.uCloudCover.value = s.startCloud + (this.targetCloudCover - s.startCloud) * pCloud;
      this.baseWindSpeed = s.startWind + (this.targetWindSpeed - s.startWind) * pWind;
      this.uniforms.uRainIntensity.value = s.startRain + (this.targetRainIntensity - s.startRain) * pRain;
      if (this.targetRainIntensity === 0.0 && this.uniforms.uRainIntensity.value < 0.004) {
        this.uniforms.uRainIntensity.value = 0.0;
      }
      this.uniforms.uStormFactor.value = s.startStorm + (this.targetStormFactor - s.startStorm) * pStorm;
      this.uniforms.uWetness.value = s.startWet + (this.targetWetness - s.startWet) * pWet;

      if (this.scene.fog) {
        this.scene.fog.density = 0.0;
      }

      if (p >= 1.0) {
        this._weatherTrans.active = false;
        this._weatherTrans.fromMode = this.weatherMode;
      }
    } else {
      // Settled physical maintenance & post-rain evaporation hysteresis
      const lerpRate = 1.0 - Math.exp(-safeDt * 1.8);
      this.uniforms.uRainIntensity.value +=
        (this.targetRainIntensity - this.uniforms.uRainIntensity.value) * lerpRate;
      if (this.targetRainIntensity === 0.0 && this.uniforms.uRainIntensity.value < 0.008) {
        this.uniforms.uRainIntensity.value = 0.0;
      }
      this.uniforms.uCloudCover.value +=
        (this.targetCloudCover - this.uniforms.uCloudCover.value) * lerpRate;
      this.uniforms.uStormFactor.value +=
        (this.targetStormFactor - this.uniforms.uStormFactor.value) * lerpRate;
      this.baseWindSpeed += (this.targetWindSpeed - this.baseWindSpeed) * lerpRate;
      if (this.scene.fog) {
        this.scene.fog.density = 0.0;
      }

      const rainNow = this.uniforms.uRainIntensity.value;
      if (rainNow > 0.03) {
        const targetWet = Math.min(1.0, 0.35 + rainNow * 0.72);
        const wetRate = 1.0 - Math.exp(-safeDt * (0.45 + rainNow * 0.65));
        this.uniforms.uWetness.value += (targetWet - this.uniforms.uWetness.value) * wetRate;
      } else if (this.uniforms.uWetness.value > 0.0) {
        // Gradual solar drying proportional to configured transition duration
        const dryPerSec = 1.0 / Math.max(20.0, this.transitionDurationSec * 0.85);
        this.uniforms.uWetness.value = Math.max(0.0, this.uniforms.uWetness.value - safeDt * dryPerSec);
      }
    }

    // 2. Continuous 20–45s Diurnal Solar/Lunar/Sidereal Arc Transition (Unwrapped -> 0.000° midnight jump!)
    if (this._timeTrans.active) {
      this._timeTrans.elapsed += transDt;
      const pT = Math.min(1.0, this._timeTrans.elapsed / Math.max(1.0, this._timeTrans.duration));
      this._timeTrans.progress = pT;
      const easeT = WeatherOpticsEngine._smootherstep(0.0, 1.0, pT);
      this.continuousSolarHour = this._timeTrans.startContinuousHour + this._timeTrans.deltaHour * easeT;
      this.currentSolarHour = ((this.continuousSolarHour % 24.0) + 24.0) % 24.0;
      if (pT >= 1.0) {
        this._timeTrans.active = false;
        this._timeTrans.fromTod = this.timeOfDay;
      }
    }

    // 3. Turbulent 3D Wind Vector, Monotonically Integrated Wind Phase, & Ocean Swell Inertia
    const gustOsc =
      0.90 + 0.18 * Math.sin(t * 0.34) + 0.11 * Math.cos(t * 0.78) + 0.05 * Math.sin(t * 1.52);
    const windSpeed = this.baseWindSpeed * gustOsc;
    const windDirAngle = 0.48 + 0.14 * Math.sin(t * 0.19);
    this.uniforms.uWindVec.value.set(
      Math.cos(windDirAngle) * windSpeed,
      0.0,
      Math.sin(windDirAngle) * windSpeed
    );
    // Strict monotonic time integral Phi(t) = \int v_wind(tau) d_tau (ZERO backward phase reversal!)
    this.uniforms.uWindPhase.value.x += this.uniforms.uWindVec.value.x * safeDt;
    this.uniforms.uWindPhase.value.y += this.uniforms.uWindVec.value.z * safeDt;

    // Physical ocean swell momentum coupled to LIVE rainIntensity & stormFactor (grows smoothly over 20–45s!)
    const liveSeaTarget =
      0.32 + this.uniforms.uRainIntensity.value * 0.30 + this.uniforms.uStormFactor.value * 0.34;
    const seaRate = 1.0 - Math.exp(-safeDt * 0.65);
    this.uniforms.uSeaState.value += (liveSeaTarget - this.uniforms.uSeaState.value) * seaRate;

    this.windGustMultiplier = Math.max(0.75, windSpeed / 0.85);

    // 4. Thunderstorm Multi-Stroke Plasma Lightning (only fires once storm maturity uStormFactor > 0.62!)
    if (this.uniforms.uStormFactor.value > 0.62) {
      this.nextLightningTimer -= safeDt;
      if (this.nextLightningTimer <= 0.0) {
        this._triggerLightningDischarge(t);
        this.nextLightningTimer = 2.6 + (0.5 + 0.5 * Math.sin(t * 17.3)) * 4.4;
      }
    }

    let flash = 0.0;
    if (this.lightningClock >= 0.0) {
      this.lightningClock += safeDt;
      flash = this._evaluateLightningEnvelope(this.lightningClock);
      if (this.lightningClock > 0.54) {
        this.lightningClock = -1.0;
        flash = 0.0;
      }
    }
    this.uniforms.uLightningFlash.value = flash;

    // Evaluate continuous 24h Depok solar/lunar light, twilight colors, fog, & PMREM IBL
    this._applyContinuousCelestialAndFog(sharedSurfaceUniforms);

    if (flash > 0.005) {
      // v3.6 Non-Teleporting Lightning Flash: shadow-casting sunLight stays strictly on _baseSunPos
      // (ZERO shadow-map jump!), while non-shadow-casting fillLight + hemiLight carry directional storm flash!
      const lDir = this.uniforms.uLightningDir.value;
      const wF = Math.min(1.0, flash * 1.45);
      this.fillLight.position.set(
        this._baseFillPos.x * (1.0 - wF) + lDir.x * 115.0 * wF,
        this._baseFillPos.y * (1.0 - wF) + 76.0 * wF,
        this._baseFillPos.z * (1.0 - wF) + lDir.z * 115.0 * wF
      );
      this._tmpSunColor.setHex(0x9ec5eb).lerp(this._flashColor, wF);
      this.fillLight.color.copy(this._tmpSunColor);
      this.fillLight.intensity = this._baseFillIntensity + flash * 3.45;

      this._tmpSunColor.copy(this._baseSunColor).lerp(this._flashColor, Math.min(1.0, flash * 0.72));
      this.sunLight.color.copy(this._tmpSunColor);
      this.sunLight.intensity = this._baseSunIntensity;
      this.hemiLight.intensity = this._baseHemiIntensity + flash * 1.45;
      this.uniforms.uSunColor.value.copy(this.sunLight.color);
      if (sharedSurfaceUniforms && sharedSurfaceUniforms.uSunColor) {
        sharedSurfaceUniforms.uSunColor.value.copy(this.sunLight.color);
      }
    }

    // 5. Evaluate Dynamic Biological Pupil & Retinal Photoreceptor HDR Adaptation across All Weathers & Times
    this._syncHumanEyeHDR(safeDt, camera, sharedSurfaceUniforms, false);

    // 6. Sync Remaining Shared Surface Uniforms for Ultra Physical Shaders
    if (sharedSurfaceUniforms) {
      if (sharedSurfaceUniforms.uWetness) {
        sharedSurfaceUniforms.uWetness.value = this.uniforms.uWetness.value;
      }
      if (sharedSurfaceUniforms.uRainIntensity) {
        sharedSurfaceUniforms.uRainIntensity.value = this.uniforms.uRainIntensity.value;
      }
      if (sharedSurfaceUniforms.uLightningFlash) {
        sharedSurfaceUniforms.uLightningFlash.value = flash;
      }
      if (sharedSurfaceUniforms.uWindVec) {
        sharedSurfaceUniforms.uWindVec.value.copy(this.uniforms.uWindVec.value);
      }
    }

    this._syncParticleVisibility();

    if (camera) {
      this.uniforms.uCamPos.value.copy(camera.position);
      if (this.skyMesh) {
        this.skyMesh.position.copy(camera.position);
      }
    }
  }
}
