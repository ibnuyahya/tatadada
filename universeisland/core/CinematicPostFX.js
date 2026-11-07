import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v6.40
 * ./core/CinematicPostFX.js
 *
 * View-Aware Dual-Perspective Optical Pipeline + Real Planar Water Mirror Reflections:
 *
 * 1. HUMAN-EYE PERSPECTIVE (uDroneCinemaBlend = 0.0 -> WALK_SIVITAS & All FPS_* Eye-Level Views):
 *    - Strictly True-Color Photorealistic Human-Eye Perception!
 *    - 0.0 Vignette, 0.0 Chromatic Dispersion, 0.0 Artificial LUT / Color Grading shift
 *      (100% faithful preservation of HumanEyeHDROptics photoreceptor colors).
 *    - Hardware 4x MSAA foveal acuity + physiological intraocular highlight scatter
 *      (strictly gated to intense highlights luma > 0.93, zero midtone haze).
 *
 * 2. DRONE / AERIAL CINEMA PERSPECTIVE (uDroneCinemaBlend = 1.0 -> DRONE_6DOF Mavic 3 Cine,
 *    DRONE_AVATA2_FPV, ORBIT, & AUTO_CINEMATIC):
 *    - Smoothly transitions (~0.6s) to DJI Mavic 3 Cine Hasselblad D-Log M Aerial Cinema Grade:
 *      CPL Circular-Polarizer sky/sea richness, filmic S-curve contrast, subtle warm-golden
 *      highlight & cool-cerulean shadow split-tone, lush Dual-Kawase cinema halation bloom,
 *      and delicate 24mm f/2.8 natural cos^4 optical vignette.
 *
 * 3. REAL PLANAR WATER MIRROR REFLECTIONS (renderPlanarWaterReflection):
 *    - Right-handed SO(3) mirrored virtual camera (det = +1, zero flipped triangle winding!)
 *      + Lengyel oblique near-plane clipping at WATER_LEVEL_Y = -1.45m, projecting real 3D
 *      reflections of Gedung Rektorat UI, Jembatan Teksas UI, trees, volcano, & clouds onto
 *      Danau Kenanga, Teluk U, Selat UI, and Kepulauan Seribu lagoons.
 *    - Motion-aware reflection cadence + shadow-map rebuild suppression + sky star-cull fast-path.
 *
 * 4. CSS3D YOUTUBE VIDEOTRON PORTAL GUARD:
 *    - Strictly preserves alpha < 0.99 punch-through windows so the live CSS3D YouTube Videotron
 *      renders cleanly with zero post-FX bleed.
 */
export class CinematicPostFX {
  constructor(renderer, width, height, options = {}) {
    this.renderer = renderer;
    this.width = Math.max(1, Math.floor(width || 1280));
    this.height = Math.max(1, Math.floor(height || 720));
    this.msaaSamples = options.msaaSamples !== undefined ? options.msaaSamples : 4;
    this.planarReflSize = options.planarReflSize || 512;
    this.planarEnabled = options.planarEnabled !== false;

    this.droneCinemaBlend = 0.0;
    this.targetDroneCinemaBlend = 0.0;
    this.activeModeLabel = 'HUMAN_EYE_TRUE_COLOR';
    this.lastWorldDrawCalls = 0;
    this.lastWorldTriangles = 0;
    this.lastPlanarReflRan = false;
    this.lastPlanarReflCalls = 0;
    this._reflFrameCount = 0;
    this._hasCachedPlanarRefl = false;
    this._cachedPlanarMatrix = new THREE.Matrix4();
    this._lastReflCamPos = new THREE.Vector3(1e9, 1e9, 1e9);
    this._lastReflCamBack = new THREE.Vector3(0, 0, 1);
    this._lastReflCamUp = new THREE.Vector3(0, 1, 0);

    // 1. Hardware 4x MSAA Main Scene RenderTarget (RGBA8 + Alpha preserved for CSS3D Videotron punch-through!)
    this.msaaTarget = new THREE.WebGLRenderTarget(this.width, this.height, {
      samples: this.msaaSamples,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
      stencilBuffer: false
    });

    // 2. Dual-Kawase Highlight / Halation Bloom Pyramid (Half-Res & Quarter-Res)
    const bw1 = Math.max(2, Math.floor(this.width / 2));
    const bh1 = Math.max(2, Math.floor(this.height / 2));
    const bw2 = Math.max(2, Math.floor(this.width / 4));
    const bh2 = Math.max(2, Math.floor(this.height / 4));

    const rtOpts = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: false,
      stencilBuffer: false
    };
    this.bloomHalfA = new THREE.WebGLRenderTarget(bw1, bh1, rtOpts);
    this.bloomQuarter = new THREE.WebGLRenderTarget(bw2, bh2, rtOpts);
    this.bloomHalfB = new THREE.WebGLRenderTarget(bw1, bh1, rtOpts);

    // 3. Real Planar Water Mirror Reflection RenderTarget & Virtual Right-Handed Camera
    this.planarReflTarget = new THREE.WebGLRenderTarget(
      this.planarReflSize,
      this.planarReflSize,
      {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
        depthBuffer: true,
        stencilBuffer: false
      }
    );
    this.reflCamera = new THREE.PerspectiveCamera(50, 1.0, 0.25, 5400);
    this.planarTextureMatrix = new THREE.Matrix4();
    this._biasMatrix = new THREE.Matrix4().set(
      0.5, 0.0, 0.0, 0.5,
      0.0, 0.5, 0.0, 0.5,
      0.0, 0.0, 0.5, 0.5,
      0.0, 0.0, 0.0, 1.0
    );
    this._camWorldPos = new THREE.Vector3();
    this._camRight = new THREE.Vector3();
    this._camUp = new THREE.Vector3();
    this._camBack = new THREE.Vector3();
    this._reflBasisMat = new THREE.Matrix4();

    // 4. Fullscreen PostFX Quad & Shaders
    this._fsScene = new THREE.Scene();
    this._fsCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const quadGeo = new THREE.PlaneGeometry(2, 2);

    this._bloomDownMat = this._createBloomDownMaterial();
    this._bloomUpMat = this._createBloomUpMaterial();
    this._compositeMat = this._createCompositeMaterial();

    this._fsQuad = new THREE.Mesh(quadGeo, this._compositeMat);
    this._fsQuad.frustumCulled = false;
    this._fsScene.add(this._fsQuad);
  }

  _createBloomDownMaterial() {
    return new THREE.ShaderMaterial({
      uniforms: {
        uTex: { value: null },
        uTexelSize: { value: new THREE.Vector2(1 / 640, 1 / 360) },
        uThreshold: { value: 0.90 },
        uIsFirstPass: { value: 1.0 }
      },
      depthTest: false,
      depthWrite: false,
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform sampler2D uTex;
        uniform vec2 uTexelSize;
        uniform float uThreshold;
        uniform float uIsFirstPass;
        varying vec2 vUv;

        vec3 extractBright(vec4 s) {
          if (s.a < 0.99) return vec3(0.0);
          vec3 lin = pow(max(s.rgb, vec3(0.0)), vec3(2.2));
          if (uIsFirstPass < 0.5) return lin;
          float luma = dot(lin, vec3(0.2126, 0.7152, 0.0722));
          float knee = 0.16;
          float soft = clamp(luma - uThreshold + knee, 0.0, 2.0 * knee);
          soft = (soft * soft) / (4.0 * knee + 1e-5);
          float contrib = max(soft, luma - uThreshold) / max(luma, 1e-4);
          return lin * clamp(contrib, 0.0, 2.5);
        }

        void main() {
          vec2 h = uTexelSize * 0.5;
          vec3 sum = extractBright(texture2D(uTex, vUv)) * 4.0;
          sum += extractBright(texture2D(uTex, vUv + vec2(-h.x, -h.y)));
          sum += extractBright(texture2D(uTex, vUv + vec2( h.x, -h.y)));
          sum += extractBright(texture2D(uTex, vUv + vec2(-h.x,  h.y)));
          sum += extractBright(texture2D(uTex, vUv + vec2( h.x,  h.y)));
          vec3 avg = sum / 8.0;
          gl_FragColor = vec4(pow(clamp(avg, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
        }
      `
    });
  }

  _createBloomUpMaterial() {
    return new THREE.ShaderMaterial({
      uniforms: {
        uTex: { value: null },
        uTexelSize: { value: new THREE.Vector2(1 / 320, 1 / 180) }
      },
      depthTest: false,
      depthWrite: false,
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform sampler2D uTex;
        uniform vec2 uTexelSize;
        varying vec2 vUv;

        vec3 toLin(vec2 p) {
          return pow(max(texture2D(uTex, p).rgb, vec3(0.0)), vec3(2.2));
        }

        void main() {
          vec2 d = uTexelSize;
          vec3 sum = toLin(vUv + vec2(-d.x * 2.0, 0.0));
          sum += toLin(vUv + vec2(-d.x, d.y)) * 2.0;
          sum += toLin(vUv + vec2(0.0, d.y * 2.0));
          sum += toLin(vUv + vec2(d.x, d.y)) * 2.0;
          sum += toLin(vUv + vec2(d.x * 2.0, 0.0));
          sum += toLin(vUv + vec2(d.x, -d.y)) * 2.0;
          sum += toLin(vUv + vec2(0.0, -d.y * 2.0));
          sum += toLin(vUv + vec2(-d.x, -d.y)) * 2.0;
          vec3 avg = sum / 12.0;
          gl_FragColor = vec4(pow(clamp(avg, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
        }
      `
    });
  }

  _createCompositeMaterial() {
    return new THREE.ShaderMaterial({
      uniforms: {
        uSceneTex: { value: null },
        uBloomTex: { value: null },
        uTexelSize: { value: new THREE.Vector2(1 / 1280, 1 / 720) },
        uDroneCinemaBlend: { value: 0.0 },
        uDuskFactor: { value: 0.0 },
        uNightFactor: { value: 0.0 }
      },
      depthTest: false,
      depthWrite: false,
      transparent: true,
      blending: THREE.NoBlending,
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform sampler2D uSceneTex;
        uniform sampler2D uBloomTex;
        uniform vec2 uTexelSize;
        uniform float uDroneCinemaBlend;
        uniform float uDuskFactor;
        uniform float uNightFactor;
        varying vec2 vUv;

        vec3 srgbToLin(vec3 c) {
          return pow(max(c, vec3(0.0)), vec3(2.2));
        }

        vec3 linToSrgb(vec3 c) {
          return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
        }

        void main() {
          vec4 sceneSample = texture2D(uSceneTex, vUv);

          // Preserve CSS3D YouTube Videotron punch-through window 100% untouched!
          if (sceneSample.a < 0.99) {
            gl_FragColor = sceneSample;
            return;
          }

          vec3 colLin = srgbToLin(sceneSample.rgb);

          // Subtle edge-preserving foveal acuity & micro-contact definition (never halos!)
          vec3 nL = srgbToLin(texture2D(uSceneTex, vUv + vec2(-uTexelSize.x, 0.0)).rgb);
          vec3 nR = srgbToLin(texture2D(uSceneTex, vUv + vec2( uTexelSize.x, 0.0)).rgb);
          vec3 nU = srgbToLin(texture2D(uSceneTex, vUv + vec2(0.0,  uTexelSize.y)).rgb);
          vec3 nD = srgbToLin(texture2D(uSceneTex, vUv + vec2(0.0, -uTexelSize.y)).rgb);
          vec3 blurCross = 0.25 * (nL + nR + nU + nD);
          float lumaCenter = dot(colLin, vec3(0.2126, 0.7152, 0.0722));
          float lumaBlur = dot(blurCross, vec3(0.2126, 0.7152, 0.0722));
          float localDiff = lumaCenter - lumaBlur;
          float haloGuard = 1.0 - smoothstep(0.08, 0.28, abs(localDiff));
          float acuityStrength = mix(0.11, 0.14, uDroneCinemaBlend) * haloGuard;
          colLin = max(vec3(0.0), colLin + colLin * (localDiff / max(lumaCenter, 0.04)) * acuityStrength);

          // Dual-Kawase Highlight Scatter:
          // - Human-Eye (blend = 0.0): subtle physiological ciliary scatter strictly on intense highlights (0.075)
          // - Drone Cinema (blend = 1.0): lush Hasselblad cinema halation bloom (0.22)
          vec3 bloomLin = srgbToLin(texture2D(uBloomTex, vUv).rgb);
          float bloomWeight = mix(0.075, 0.22, uDroneCinemaBlend);
          colLin += bloomLin * bloomWeight;

          // === DRONE CINEMA COLOR GRADE (Strictly 0.0 in Human-Eye Mode -> 100% True-Color Preserved!) ===
          if (uDroneCinemaBlend > 0.002) {
            vec3 droneLin = colLin;

            // 1. Aerial CPL Circular Polarizer richness on azure sky & turquoise-emerald ocean
            float lumaD = dot(droneLin, vec3(0.2126, 0.7152, 0.0722));
            float cyanBlueDom = clamp((max(droneLin.b, droneLin.g * 0.92) - droneLin.r) / max(lumaD + 0.04, 0.04), 0.0, 1.0);
            droneLin = mix(vec3(lumaD), droneLin, 1.0 + 0.11 * cyanBlueDom + 0.05);
            droneLin *= mix(1.0, 0.94, cyanBlueDom * 0.45 * (1.0 - uNightFactor));

            // 2. Hasselblad HNCS / DJI D-Log M Filmic S-Curve Contrast
            vec3 sCurve = droneLin * droneLin * (3.0 - 2.0 * clamp(droneLin, 0.0, 1.0));
            droneLin = mix(droneLin, sCurve, 0.22 * (1.0 - smoothstep(0.68, 1.0, lumaD)));

            // 3. Tasteful Aerial Cinema Split-Tone (cool cerulean shadow depth + warm golden-hour sunlight)
            float shadowMask = (1.0 - smoothstep(0.0, 0.28, lumaD)) * (1.0 - uNightFactor * 0.45);
            float highlightMask = smoothstep(0.26, 0.88, lumaD);
            vec3 shadowTint = vec3(0.93, 0.98, 1.07);
            vec3 hiTint = mix(vec3(1.035, 1.005, 0.955), vec3(1.065, 0.995, 0.915), uDuskFactor);
            droneLin *= mix(vec3(1.0), shadowTint, shadowMask * 0.55);
            droneLin *= mix(vec3(1.0), hiTint, highlightMask * 0.55);

            // 4. Hasselblad 24mm f/2.8 Natural Optical Corner Vignette (cos^4 falloff, 0.0 in Human-Eye!)
            vec2 centered = (vUv - 0.5) * vec2(1.18, 1.0);
            float r2 = dot(centered, centered);
            float cos4Vig = 1.0 / pow(1.0 + r2 * 0.145, 2.0);
            droneLin *= mix(1.0, cos4Vig, 0.48);

            colLin = mix(colLin, max(vec3(0.0), droneLin), uDroneCinemaBlend);
          }

          gl_FragColor = vec4(linToSrgb(colLin), 1.0);
        }
      `
    });
  }

  /**
   * Updates smooth optical mode transition:
   * - isDroneCinema = false -> Human-Eye True-Color Photorealistic Perception (uDroneCinemaBlend -> 0.0)
   * - isDroneCinema = true  -> Drone Aerial Cinema Color-Graded Perception (uDroneCinemaBlend -> 1.0)
   */
  updateMode(dt, isDroneCinema, weatherState = null) {
    this.targetDroneCinemaBlend = isDroneCinema ? 1.0 : 0.0;
    const rate = 1.0 - Math.exp(-Math.max(0.0, dt || 0.016) * 5.2);
    this.droneCinemaBlend += (this.targetDroneCinemaBlend - this.droneCinemaBlend) * rate;
    if (Math.abs(this.targetDroneCinemaBlend - this.droneCinemaBlend) < 0.001) {
      this.droneCinemaBlend = this.targetDroneCinemaBlend;
    }
    this.activeModeLabel =
      this.droneCinemaBlend > 0.5 ? 'DRONE_MAVIC3_CINEMA_GRADE' : 'HUMAN_EYE_TRUE_COLOR';

    this._compositeMat.uniforms.uDroneCinemaBlend.value = this.droneCinemaBlend;
    if (weatherState) {
      this._compositeMat.uniforms.uDuskFactor.value = weatherState.duskFactor || 0.0;
      this._compositeMat.uniforms.uNightFactor.value = weatherState.nightFactor || 0.0;
    }
  }

  /**
   * Renders the real-time planar mirror reflection of the above-water world into planarReflTarget
   * and binds uPlanarReflTex / uPlanarReflMatrix / uPlanarReflActive on weatherOptics.oceanUniforms.
   */
  renderPlanarWaterReflection(scene, camera, weatherOptics, waterY = -1.45, hiddenDuringRefl = []) {
    this.lastPlanarReflRan = false;
    this.lastPlanarReflCalls = 0;
    const oceanU = weatherOptics && (weatherOptics.uniforms || weatherOptics.oceanUniforms);
    if (!this.planarEnabled || !oceanU || !oceanU.uPlanarReflActive || !this.renderer) return;

    camera.updateMatrixWorld();
    this._camWorldPos.setFromMatrixPosition(camera.matrixWorld);

    // Disable planar pass if camera is below water / inside subsea underpass or above 540m altitude
    if (this._camWorldPos.y <= waterY + 0.14 || this._camWorldPos.y > 540.0) {
      oceanU.uPlanarReflActive.value = 0.0;
      this._hasCachedPlanarRefl = false;
      return;
    }

    // Extract main camera world basis [R, U, B] (det = +1)
    const me = camera.matrixWorld.elements;
    this._camRight.set(me[0], me[1], me[2]);
    this._camUp.set(me[4], me[5], me[6]);
    this._camBack.set(me[8], me[9], me[10]);

    // Motion-aware reflection cadence:
    // - Full 1:1 refresh whenever camera translates >= 2.2cm or rotates >= 0.07 deg (zero lag while flying/orbiting!)
    // - Stride-2 refresh when camera is stationary/hovering gently (halves idle planar GPU work with 0px projection drift!)
    const r = this.renderer;
    const prevShadowNeedsUpdate = Boolean(r.shadowMap && r.shadowMap.needsUpdate);
    this._reflFrameCount++;
    const posDeltaSq = this._camWorldPos.distanceToSquared(this._lastReflCamPos);
    const rotDelta =
      Math.abs(1.0 - this._camBack.dot(this._lastReflCamBack)) +
      Math.abs(1.0 - this._camUp.dot(this._lastReflCamUp));
    const cameraSteady = posDeltaSq < 0.00048 && rotDelta < 0.0000018;
    // Phase-interleave directional shadow map refresh and planar water reflection refresh during continuous motion (<2.5m/frame)
    // so both heavy render passes never collide on the same VSync frame!
    const deferOnShadowFrame = prevShadowNeedsUpdate && posDeltaSq < 6.25 && rotDelta < 0.02;
    if (
      this._hasCachedPlanarRefl &&
      (deferOnShadowFrame || (cameraSteady && this._reflFrameCount % 2 !== 0))
    ) {
      oceanU.uPlanarReflTex.value = this.planarReflTarget.texture;
      oceanU.uPlanarReflMatrix.value.copy(this._cachedPlanarMatrix);
      oceanU.uPlanarReflActive.value = 1.0;
      return;
    }

    // Build right-handed (det = +1) mirrored camera basis across horizontal plane y = waterY:
    // R' = (Rx, -Ry, Rz), U' = (-Ux, Uy, -Uz), B' = (Bx, -By, Bz)
    const rx = this._camRight.x;
    const ry = -this._camRight.y;
    const rz = this._camRight.z;
    const ux = -this._camUp.x;
    const uy = this._camUp.y;
    const uz = -this._camUp.z;
    const bx = this._camBack.x;
    const by = -this._camBack.y;
    const bz = this._camBack.z;
    const px = this._camWorldPos.x;
    const py = 2.0 * waterY - this._camWorldPos.y;
    const pz = this._camWorldPos.z;

    this._reflBasisMat.set(
      rx, ux, bx, px,
      ry, uy, by, py,
      rz, uz, bz, pz,
      0,  0,  0,  1
    );

    const rc = this.reflCamera;
    rc.fov = camera.fov;
    rc.aspect = camera.aspect;
    rc.near = camera.near;
    rc.far = camera.far;
    rc.updateProjectionMatrix();

    rc.position.set(px, py, pz);
    rc.quaternion.setFromRotationMatrix(this._reflBasisMat);
    rc.matrixWorld.copy(this._reflBasisMat);
    rc.matrixWorldInverse.copy(this._reflBasisMat).invert();

    // Build projective UV matrix: bias * P * V_refl
    this.planarTextureMatrix.copy(this._biasMatrix);
    this.planarTextureMatrix.multiply(rc.projectionMatrix);
    this.planarTextureMatrix.multiply(rc.matrixWorldInverse);

    // Apply Lengyel's oblique near-plane clip (world plane: y >= waterY - 0.14) to projectionMatrix row 2
    const d0 = -(waterY - 0.14);
    const cx = ry;
    const cy = uy;
    const cz = by;
    const cw = py + d0;
    const pe = rc.projectionMatrix.elements;
    const qx = ((cx >= 0.0 ? 1.0 : -1.0) + pe[8]) / pe[0];
    const qy = ((cy >= 0.0 ? 1.0 : -1.0) + pe[9]) + 0.0;
    const qyNorm = ((cy >= 0.0 ? 1.0 : -1.0) + pe[9]) / pe[5];
    void qy;
    const qz = -1.0;
    const qw = (1.0 + pe[10]) / pe[14];
    const denom = cx * qx + cy * qyNorm + cz * qz + cw * qw;
    if (Math.abs(denom) > 1e-5) {
      const scale = 2.0 / denom;
      pe[2] = cx * scale;
      pe[6] = cy * scale;
      pe[10] = cz * scale + 1.0;
      pe[14] = cw * scale;
    }

    const prevVis = [];
    for (let i = 0; i < hiddenDuringRefl.length; i++) {
      const obj = hiddenDuringRefl[i];
      prevVis.push(obj ? obj.visible : false);
      if (obj) obj.visible = false;
    }

    const prevTarget = r.getRenderTarget();
    const prevClearAlpha = r.getClearAlpha();
    // Never rebuild directional shadow maps inside the planar reflection pass; preserve pending main-pass flag!
    if (r.shadowMap) r.shadowMap.needsUpdate = false;
    if (oceanU.uRenderingPlanarRefl) oceanU.uRenderingPlanarRefl.value = 1.0;

    try {
      r.setRenderTarget(this.planarReflTarget);
      r.setClearAlpha(1.0);
      r.clear(true, true, false);
      r.render(scene, rc);
      this.lastPlanarReflRan = true;
      this.lastPlanarReflCalls = r.info?.render?.calls || 0;
    } finally {
      if (oceanU.uRenderingPlanarRefl) oceanU.uRenderingPlanarRefl.value = 0.0;
      if (r.shadowMap) r.shadowMap.needsUpdate = prevShadowNeedsUpdate;
      r.setClearAlpha(prevClearAlpha);
      r.setRenderTarget(prevTarget);
      for (let i = 0; i < hiddenDuringRefl.length; i++) {
        if (hiddenDuringRefl[i]) hiddenDuringRefl[i].visible = prevVis[i];
      }
    }

    this._lastReflCamPos.copy(this._camWorldPos);
    this._lastReflCamBack.copy(this._camBack);
    this._lastReflCamUp.copy(this._camUp);
    this._cachedPlanarMatrix.copy(this.planarTextureMatrix);
    this._hasCachedPlanarRefl = true;

    oceanU.uPlanarReflTex.value = this.planarReflTarget.texture;
    oceanU.uPlanarReflMatrix.value.copy(this.planarTextureMatrix);
    oceanU.uPlanarReflActive.value = 1.0;
  }

  /**
   * Renders main scene + optional companion scenes (e.g. TropicalGrassField) into 4x MSAA target,
   * runs Dual-Kawase highlight/halation pyramid, and composites to screen!
   */
  render(scene, camera, companionScene = null) {
    const r = this.renderer;
    const prevAutoClear = r.autoClear;
    const prevClearAlpha = r.getClearAlpha();

    r.setRenderTarget(this.msaaTarget);
    r.setClearAlpha(1.0);
    r.autoClear = true;
    r.clear(true, true, true);
    r.render(scene, camera);
    const mainCalls = r.info.render.calls;
    const mainTris = r.info.render.triangles;

    let compCalls = 0;
    let compTris = 0;
    if (companionScene) {
      r.autoClear = false;
      r.render(companionScene, camera);
      compCalls = r.info.render.calls;
      compTris = r.info.render.triangles;
    }

    this.lastWorldDrawCalls = mainCalls + compCalls;
    this.lastWorldTriangles = mainTris + compTris;

    // Pass 1: Downsample MSAA -> Half-Res bright highlights
    // Human-Eye uses higher threshold (0.93) for pure physiological highlight scatter;
    // Drone Cinema uses softer cinema halation threshold (0.77).
    const thresh = 0.93 - 0.16 * this.droneCinemaBlend;
    this._bloomDownMat.uniforms.uTex.value = this.msaaTarget.texture;
    this._bloomDownMat.uniforms.uTexelSize.value.set(1.0 / this.width, 1.0 / this.height);
    this._bloomDownMat.uniforms.uThreshold.value = thresh;
    this._bloomDownMat.uniforms.uIsFirstPass.value = 1.0;
    this._fsQuad.material = this._bloomDownMat;
    r.setRenderTarget(this.bloomHalfA);
    r.render(this._fsScene, this._fsCam);

    // Pass 2: Downsample Half -> Quarter
    this._bloomDownMat.uniforms.uTex.value = this.bloomHalfA.texture;
    this._bloomDownMat.uniforms.uTexelSize.value.set(
      1.0 / this.bloomHalfA.width,
      1.0 / this.bloomHalfA.height
    );
    this._bloomDownMat.uniforms.uIsFirstPass.value = 0.0;
    r.setRenderTarget(this.bloomQuarter);
    r.render(this._fsScene, this._fsCam);

    // Pass 3: Upsample Quarter -> HalfB
    this._bloomUpMat.uniforms.uTex.value = this.bloomQuarter.texture;
    this._bloomUpMat.uniforms.uTexelSize.value.set(
      1.0 / this.bloomQuarter.width,
      1.0 / this.bloomQuarter.height
    );
    this._fsQuad.material = this._bloomUpMat;
    r.setRenderTarget(this.bloomHalfB);
    r.render(this._fsScene, this._fsCam);

    // Pass 4: Final Dual-Mode Composite to Default Framebuffer (Screen)
    this._compositeMat.uniforms.uSceneTex.value = this.msaaTarget.texture;
    this._compositeMat.uniforms.uBloomTex.value = this.bloomHalfB.texture;
    this._compositeMat.uniforms.uTexelSize.value.set(
      1.0 / this.width,
      this.height > 0 ? 1.0 / this.height : 1.0 / 720
    );
    this._fsQuad.material = this._compositeMat;
    r.setRenderTarget(null);
    r.setClearAlpha(0.0);
    r.autoClear = true;
    r.clear(true, true, false);
    r.render(this._fsScene, this._fsCam);

    r.setClearAlpha(prevClearAlpha);
    r.autoClear = prevAutoClear;
    r.info.render.calls = this.lastWorldDrawCalls;
    r.info.render.triangles = this.lastWorldTriangles;
  }

  setSize(width, height) {
    this.width = Math.max(1, Math.floor(width || 1280));
    this.height = Math.max(1, Math.floor(height || 720));
    this.msaaTarget.setSize(this.width, this.height);
    this.bloomHalfA.setSize(Math.max(2, Math.floor(this.width / 2)), Math.max(2, Math.floor(this.height / 2)));
    this.bloomQuarter.setSize(Math.max(2, Math.floor(this.width / 4)), Math.max(2, Math.floor(this.height / 4)));
    this.bloomHalfB.setSize(Math.max(2, Math.floor(this.width / 2)), Math.max(2, Math.floor(this.height / 2)));
  }

  dispose() {
    this.msaaTarget.dispose();
    this.bloomHalfA.dispose();
    this.bloomQuarter.dispose();
    this.bloomHalfB.dispose();
    this.planarReflTarget.dispose();
    this._fsQuad.geometry.dispose();
    this._bloomDownMat.dispose();
    this._bloomUpMat.dispose();
    this._compositeMat.dispose();
  }
}
