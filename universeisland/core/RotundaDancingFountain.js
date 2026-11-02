import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v5.0
 * ./core/RotundaDancingFountain.js
 *
 * Physics-Accurate Compact Architectural Dancing Fountain at Center of Lapangan Rotunda UI (-2.5, 106.5):
 * - Strictly 1 Single Draw Call (indexed BufferGeometry + custom ShaderMaterial).
 * - Matches SplineCorridorBuilder.FOUNTAIN_SPEC (waterY = 0.46m, poolInnerR = 4.34m,
 *   1 central brass spire nozzle at y = 0.63m, 8 inner crown gimbal nozzles at R = 1.35m y = 0.52m,
 *   and 16 outer dancing gimbal nozzles at R = 2.85m y = 0.495m).
 * - True Newtonian Ballistic + Historical Gimbal Kinematics (g = 9.81 m/s^2 + aerodynamic drag +
 *   Rayleigh-Plateau droplet breakup dispersion): each fluid parcel evaluates the nozzle's historical
 *   velocity vector v0(t - tau) at emission time t_emit = t - tau, producing authentic traveling
 *   mid-air parabolic ribbons, spiral helices, pulsing crests, and live tracking water-surface splash rings.
 * - Daytime Mode: Crisp solar-lit aerated white/aqua architectural crown jets.
 * - Night Show Mode (sunElevationDeg <= 4 deg): 64-second 4-Act Choreographed RGBW Water & Light Show
 *   synchronized live with sharedSurfaceUniforms.uFountainNightGlow (submerged LED pods & Rotunda spill).
 */
export class RotundaDancingFountain {
  static CENTER_X = -2.5;
  static CENTER_Z = 106.5;
  static WATER_Y = 0.462;
  static POOL_INNER_R = 4.34;

  constructor() {
    this.elapsed = 0.0;
    this.nightBlend = 0.0;
    this.currentActName = 'Daytime Royal Crown';
    this._glowColor = new THREE.Vector3(0.0, 0.0, 0.0);

    const { geometry, triangleCount } = this._buildBatchedHydrodynamicGeometry();
    this.triangleCount = triangleCount;

    this.uniforms = {
      uTime: { value: 0.0 },
      uNightBlend: { value: 0.0 },
      uSunDir: { value: new THREE.Vector3(0.45, 0.78, 0.42) },
      uGlobalShowColor: { value: new THREE.Vector3(1.0, 0.76, 0.18) }
    };

    this.material = new THREE.ShaderMaterial({
      name: 'RotundaDancingFountainMat',
      uniforms: this.uniforms,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
      vertexShader: /* glsl */ `
        attribute vec3 aCross;
        attribute vec4 aNozzle;

        uniform float uTime;
        uniform float uNightBlend;

        varying float vS;
        varying float vIsSplash;
        varying float vSplashR;
        varying float vBreakup;
        varying float vTier;
        varying float vSubmerged;
        varying vec3 vJetTint;
        varying vec3 vWorldNormal;
        varying vec3 vWorldPos;

        vec3 safeNorm3(vec3 v, vec3 fallback) {
          float l2 = dot(v, v);
          return l2 > 1e-8 ? v * inversesqrt(l2) : fallback;
        }

        void evalLaunchState(
          float t,
          float tier,
          float idx,
          float baseAng,
          float nightBlend,
          out vec3 v0,
          out vec3 ledCol
        ) {
          float cA = cos(baseAng);
          float sA = sin(baseAng);
          vec3 eRad = vec3(cA, 0.0, sA);
          vec3 eTan = vec3(-sA, 0.0, cA);

          // 1. Daytime Calm Architectural Crown Kinematics
          vec3 dayVel = vec3(0.0, 4.2, 0.0);
          if (tier < 0.5) {
            float vy = 5.15 + 0.34 * sin(1.45 * t);
            float sway = 0.15;
            dayVel = vec3(cos(0.95 * t) * sway, vy, sin(0.95 * t) * sway);
          } else if (tier < 1.5) {
            float vy = 4.22 + 0.28 * sin(1.65 * t + idx * 0.785398);
            float vr = 0.58 + 0.24 * sin(0.92 * t);
            float vt = 0.12 * sin(0.75 * t);
            dayVel = eRad * vr + eTan * vt + vec3(0.0, vy, 0.0);
          } else {
            float vy = 3.72 + 0.26 * sin(1.90 * t + idx * 0.785398);
            float vr = -1.24 + 0.24 * cos(1.10 * t + idx * 0.392699);
            float vt = 0.16 * sin(0.85 * t);
            dayVel = eRad * vr + eTan * vt + vec3(0.0, vy, 0.0);
          }

          // 2. Night Show 4-Act Choreographed Kinematics (64-second musical loop)
          float tC = mod(t, 64.0);
          float fade = 1.8;
          float w1 = (1.0 - smoothstep(16.0 - fade, 16.0 + fade, tC)) + smoothstep(64.0 - fade, 64.0, tC);
          float w2 = smoothstep(16.0 - fade, 16.0 + fade, tC) * (1.0 - smoothstep(32.0 - fade, 32.0 + fade, tC));
          float w3 = smoothstep(32.0 - fade, 32.0 + fade, tC) * (1.0 - smoothstep(48.0 - fade, 48.0 + fade, tC));
          float w4 = smoothstep(48.0 - fade, 48.0 + fade, tC) * (1.0 - smoothstep(64.0 - fade, 64.0, tC));
          float wSum = max(1e-4, w1 + w2 + w3 + w4);
          w1 /= wSum; w2 /= wSum; w3 /= wSum; w4 /= wSum;

          float parity = mod(idx, 2.0) * 2.0 - 1.0;

          // Act 1: UI Makara Golden Crown Waltz
          vec3 vAct1;
          vec3 cAct1;
          float psi1 = sin(2.0 * baseAng - 2.6 * t);
          if (tier < 0.5) {
            vAct1 = vec3(0.28 * cos(1.8 * t), 6.50 + 0.65 * sin(2.4 * t), 0.28 * sin(1.8 * t));
            cAct1 = vec3(1.00, 0.76, 0.16);
          } else if (tier < 1.5) {
            float vy1 = 5.10 + 0.62 * sin(baseAng + 2.2 * t);
            vAct1 = eRad * (0.82 + 0.42 * sin(1.5 * t)) + eTan * (0.54 * cos(1.5 * t)) + vec3(0.0, vy1, 0.0);
            cAct1 = vec3(1.00, 0.58, 0.12);
          } else {
            float vy1 = 4.30 + 0.92 * psi1;
            vAct1 = eRad * (-1.52 - 0.34 * psi1) + eTan * (0.44 * sin(1.8 * t)) + vec3(0.0, vy1, 0.0);
            cAct1 = mix(vec3(1.00, 0.72, 0.12), vec3(1.00, 0.94, 0.76), 0.5 + 0.5 * psi1);
          }

          // Act 2: Emerald-Aqua Double Helix
          vec3 vAct2;
          vec3 cAct2;
          if (tier < 0.5) {
            vAct2 = vec3(0.50 * cos(3.5 * t), 6.18 + 0.52 * cos(2.9 * t), 0.50 * sin(3.5 * t));
            cAct2 = vec3(0.08, 0.94, 1.00);
          } else if (tier < 1.5) {
            float vy2 = 5.02 + 0.46 * sin(2.7 * t - baseAng);
            vAct2 = eRad * (0.34 * sin(2.0 * t)) + eTan * (1.22 * sin(2.35 * t)) + vec3(0.0, vy2, 0.0);
            cAct2 = vec3(0.08, 0.96, 0.62);
          } else {
            float vy2 = 4.52 + 0.54 * cos(2.9 * t + 3.0 * baseAng);
            vAct2 = eRad * (-1.34 + 0.34 * cos(2.0 * t)) + eTan * (-1.42 * sin(2.35 * t)) + vec3(0.0, vy2, 0.0);
            cAct2 = mix(vec3(0.04, 0.80, 1.00), vec3(0.14, 0.98, 0.74), 0.5 + 0.5 * sin(baseAng * 2.0 + t));
          }

          // Act 3: Amethyst-Magenta Chasing Cathedral Arches
          vec3 vAct3;
          vec3 cAct3;
          if (tier < 0.5) {
            vAct3 = vec3(0.22 * cos(2.4 * t), 6.75 + 0.82 * sin(3.3 * t), 0.22 * sin(2.4 * t));
            cAct3 = vec3(0.84, 0.60, 1.00);
          } else if (tier < 1.5) {
            float vy3 = 5.26 + 0.72 * sin(3.3 * t + parity * 1.5708);
            vAct3 = eRad * (-0.52 + 0.72 * sin(2.1 * t * parity)) + eTan * (0.34 * parity) + vec3(0.0, vy3, 0.0);
            cAct3 = vec3(0.42, 0.46, 1.00);
          } else {
            float pulse3 = max(-0.35, sin(3.3 * t + idx * 1.5708));
            float vy3 = 4.60 + 0.94 * pulse3;
            vAct3 = eRad * (-1.72 - 0.34 * sin(2.4 * t)) + eTan * (0.30 * cos(2.4 * t)) + vec3(0.0, vy3, 0.0);
            cAct3 = mix(vec3(0.96, 0.22, 0.82), vec3(0.54, 0.34, 1.00), 0.5 + 0.5 * parity);
          }

          // Act 4: Nusantara Merah-Putih & Gold Peony Finale
          vec3 vAct4;
          vec3 cAct4;
          float cresc = smoothstep(48.0, 56.5, tC) * (1.0 - 0.45 * smoothstep(60.5, 64.0, tC));
          if (tier < 0.5) {
            float vy4 = 6.35 + 1.42 * cresc + 0.38 * sin(4.1 * t);
            vAct4 = vec3(0.36 * cos(3.0 * t), vy4, 0.36 * sin(3.0 * t));
            cAct4 = vec3(1.00, 0.96, 0.86);
          } else if (tier < 1.5) {
            float vy4 = 5.30 + 0.92 * cresc + 0.38 * cos(3.5 * t + baseAng * 2.0);
            vAct4 = eRad * (0.72 + 0.52 * sin(2.5 * t)) + eTan * (0.62 * cos(2.5 * t)) + vec3(0.0, vy4, 0.0);
            cAct4 = vec3(1.00, 0.18, 0.32);
          } else {
            float vy4 = 4.55 + 0.82 * cresc + 0.52 * sin(3.9 * t - baseAng * 4.0);
            vAct4 = eRad * (-1.62 + 0.52 * sin(2.9 * t)) + eTan * (0.72 * sin(2.7 * t)) + vec3(0.0, vy4, 0.0);
            cAct4 = (parity > 0.0) ? vec3(0.98, 0.98, 1.00) : vec3(1.00, 0.76, 0.16);
          }

          vec3 nightVel = vAct1 * w1 + vAct2 * w2 + vAct3 * w3 + vAct4 * w4;
          ledCol = cAct1 * w1 + cAct2 * w2 + cAct3 * w3 + cAct4 * w4;
          v0 = mix(dayVel, nightVel, nightBlend);
        }

        vec3 evalBallisticParcel(
          float tau,
          float tNow,
          float tier,
          float idx,
          float baseAng,
          float nozY,
          float nightBlend,
          out vec3 ledCol
        ) {
          float rBase = (tier < 0.5) ? 0.0 : ((tier < 1.5) ? 1.35 : 2.85);
          vec3 pNoz = vec3(-2.5 + cos(baseAng) * rBase, nozY, 106.5 + sin(baseAng) * rBase);

          vec3 vHist;
          evalLaunchState(tNow - tau, tier, idx, baseAng, nightBlend, vHist, ledCol);

          // Quadratic aerodynamic drag envelope + Newtonian gravity (g = 9.81 m/s^2)
          float dragK = 0.085;
          float tauEff = tau * max(0.55, 1.0 - 0.5 * dragK * tau);
          vec3 p = pNoz + vHist * tauEff;
          p.y -= 0.5 * 9.81 * tau * tau;

          // Keep strictly inside granite fountain basin (R <= 4.14m < poolInnerR 4.34m)
          float dx = p.x - (-2.5);
          float dz = p.z - 106.5;
          float rHoriz = sqrt(dx * dx + dz * dz);
          if (rHoriz > 4.14) {
            float sc = 4.14 / rHoriz;
            p.x = -2.5 + dx * sc;
            p.z = 106.5 + dz * sc;
          }
          return p;
        }

        void main() {
          float s = uv.x;
          float isSplash = uv.y;
          float tier = aNozzle.x;
          float idx = aNozzle.y;
          float baseAng = aNozzle.z;
          float nozY = aNozzle.w;

          vS = s;
          vIsSplash = isSplash;
          vSplashR = aCross.z;
          vTier = tier;

          // Nominal flight duration to basin water surface (y = 0.462m)
          vec3 vNow;
          vec3 cNow;
          evalLaunchState(uTime, tier, idx, baseAng, uNightBlend, vNow, cNow);
          float waterY = 0.462;
          float dy0 = max(0.02, nozY - waterY);
          float vySafe = max(2.2, vNow.y);
          float tFlight = (vySafe + sqrt(vySafe * vySafe + 2.0 * 9.81 * dy0)) / 9.81;

          if (isSplash > 0.5) {
            // Live tracking splash foam & concentric expanding water ripple disc at impact point!
            vec3 cHit;
            vec3 pHit = evalBallisticParcel(tFlight * 0.98, uTime, tier, idx, baseAng, nozY, uNightBlend, cHit);
            float pulse = fract(uTime * 2.1 + idx * 0.19 + tier * 0.31);
            float splashMaxR = (tier < 0.5) ? 0.42 : 0.30;
            float rDisc = aCross.z * splashMaxR * (0.45 + 0.65 * pulse);

            vec3 worldP = vec3(
              pHit.x + aCross.x * rDisc,
              waterY + 0.008 + (1.0 - aCross.z) * 0.035 * (0.6 + 0.4 * sin(uTime * 11.0 + idx)),
              pHit.z + aCross.y * rDisc
            );
            vJetTint = cHit;
            vBreakup = 1.0;
            vSubmerged = 0.0;
            vWorldNormal = vec3(0.0, 1.0, 0.0);
            vWorldPos = worldP;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(worldP, 1.0);
            return;
          }

          float tau = s * tFlight * 1.04;
          vec3 ledTint;
          vec3 pCenter = evalBallisticParcel(tau, uTime, tier, idx, baseAng, nozY, uNightBlend, ledTint);

          // Finite-difference trajectory tangent for Frenet-stable stream tube cross-section
          float dTau = 0.018;
          vec3 dummyC;
          vec3 pPrev = evalBallisticParcel(max(0.0, tau - dTau), uTime, tier, idx, baseAng, nozY, uNightBlend, dummyC);
          vec3 pNext = evalBallisticParcel(tau + dTau, uTime, tier, idx, baseAng, nozY, uNightBlend, dummyC);
          vec3 tang = safeNorm3(pNext - pPrev, vec3(0.0, 1.0, 0.0));

          vec3 refUp = abs(tang.y) > 0.92 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
          vec3 sideA = safeNorm3(cross(refUp, tang), vec3(1.0, 0.0, 0.0));
          vec3 sideB = safeNorm3(cross(tang, sideA), vec3(0.0, 0.0, 1.0));

          // Rayleigh-Plateau jet dispersion: tight coherent orifice core -> aerated plume near apogee & descent
          float rOrifice = (tier < 0.5) ? 0.028 : ((tier < 1.5) ? 0.021 : 0.018);
          float rSpread = (tier < 0.5) ? 0.095 : ((tier < 1.5) ? 0.072 : 0.062);
          float breakup = pow(clamp(s, 0.0, 1.0), 1.18);
          float shimmer = 1.0 + 0.14 * breakup * sin(uTime * 16.0 - s * 22.0 + idx * 1.7);
          float tubeR = (rOrifice + rSpread * breakup) * shimmer;

          // Clamp below water surface cleanly so jets terminate right on pool plane
          vSubmerged = pCenter.y < (waterY - 0.015) ? 1.0 : 0.0;
          pCenter.y = max(waterY, pCenter.y);

          vec3 radialDir = sideA * aCross.x + sideB * aCross.y;
          vec3 worldP = pCenter + radialDir * tubeR;

          vJetTint = ledTint;
          vBreakup = breakup;
          vWorldNormal = safeNorm3(radialDir + tang * 0.18, vec3(0.0, 1.0, 0.0));
          vWorldPos = worldP;

          gl_Position = projectionMatrix * modelViewMatrix * vec4(worldP, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uTime;
        uniform float uNightBlend;
        uniform vec3 uSunDir;
        uniform vec3 uGlobalShowColor;

        varying float vS;
        varying float vIsSplash;
        varying float vSplashR;
        varying float vBreakup;
        varying float vTier;
        varying float vSubmerged;
        varying vec3 vJetTint;
        varying vec3 vWorldNormal;
        varying vec3 vWorldPos;

        float hash12(vec2 p) {
          vec3 p3 = fract(vec3(p.xyx) * 0.1031);
          p3 += dot(p3, p3.yzx + 33.33);
          return fract((p3.x + p3.y) * p3.z);
        }

        void main() {
          if (vSubmerged > 0.5) discard;

          vec3 V = normalize(cameraPosition - vWorldPos);
          vec3 N = normalize(vWorldNormal);
          float fresnel = pow(1.0 - abs(dot(N, V)), 2.2);

          if (vIsSplash > 0.5) {
            // Concentric aerated splash foam + expanding water surface ripple rings
            float ring1 = smoothstep(0.18, 0.0, abs(vSplashR - fract(uTime * 1.9 + vTier * 0.3)));
            float ring2 = smoothstep(0.16, 0.0, abs(vSplashR - fract(uTime * 1.9 + 0.5 + vTier * 0.3)));
            float coreFoam = (1.0 - smoothstep(0.0, 0.48, vSplashR));
            float foamMask = clamp(coreFoam * 0.85 + (ring1 + ring2) * 0.55 * (1.0 - vSplashR * 0.45), 0.0, 1.0);

            vec3 dayFoam = vec3(0.88, 0.96, 1.00);
            vec3 nightFoam = mix(vJetTint, vec3(1.0), 0.38) * 1.35;
            vec3 col = mix(dayFoam, nightFoam, uNightBlend);
            float alpha = foamMask * mix(0.62, 0.82, uNightBlend);
            if (alpha < 0.02) discard;
            gl_FragColor = vec4(col, alpha);
            return;
          }

          // Flowing longitudinal water striations & Rayleigh-Plateau droplet breakup texture
          float flowCoord = vS * 26.0 - uTime * 13.5;
          float streakA = 0.5 + 0.5 * sin(flowCoord * 3.14159 + vWorldPos.x * 7.0);
          float dropletNoise = hash12(floor(vec2(flowCoord * 3.2, (N.x + N.z) * 6.0)));
          float aeration = mix(0.92, 0.48 + 0.52 * (0.55 * streakA + 0.45 * dropletNoise), vBreakup);

          // Daylight optics: crisp sky-aqua water core + aerated white foam + solar specular sparkle
          vec3 L = normalize(uSunDir);
          vec3 H = normalize(L + V);
          float spec = pow(max(0.0, dot(N, H)), 36.0) * (1.0 - 0.5 * uNightBlend);
          vec3 dayWater = mix(vec3(0.58, 0.84, 0.96), vec3(0.96, 0.99, 1.00), clamp(0.35 + 0.55 * vBreakup + 0.35 * fresnel, 0.0, 1.0));
          dayWater += vec3(1.0, 0.99, 0.95) * spec * 0.65;

          // Night Show fiber-optic water column illumination: intensely lit base tapering into glowing mist crest
          float baseBoost = 1.35 - 0.38 * vS;
          vec3 nightWater = mix(vJetTint, vec3(1.0), 0.25 + 0.28 * fresnel + 0.18 * streakA) * baseBoost;
          nightWater += uGlobalShowColor * 0.18 * (1.0 - vS);

          vec3 finalCol = mix(dayWater, nightWater, uNightBlend);

          // Physical opacity profile: dense coherent jet near brass orifice, misty droplet veil at descent
          float tipFade = smoothstep(1.0, 0.88, vS);
          float orificeFade = smoothstep(0.0, 0.03, vS);
          float baseAlpha = mix(0.84, 0.44, vBreakup) * aeration * orificeFade * tipFade;
          float alpha = clamp(baseAlpha * mix(0.88, 1.05, uNightBlend), 0.0, 0.92);

          if (alpha < 0.02) discard;
          gl_FragColor = vec4(finalCol, alpha);
        }
      `
    });

    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.name = 'RotundaDancingFountain_SingleDrawCall';
    this.mesh.renderOrder = 14;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    this.mesh.userData = { skipRainBake: true, dynamicKinematic: true };
    // Keep frustum culling generous around Rotunda fountain basin so tall finale spire never clips
    const fCenter = new THREE.Vector3(RotundaDancingFountain.CENTER_X, 2.1, RotundaDancingFountain.CENTER_Z);
    geometry.boundingSphere = THREE.Sphere ? new THREE.Sphere(fCenter, 6.5) : { center: fCenter, radius: 6.5 };
  }

  _buildBatchedHydrodynamicGeometry() {
    const nozzles = [
      // Tier 0: Central Brass Spire Nozzle
      { tier: 0, idx: 0, baseAng: 0.0, rBase: 0.0, nozY: 0.63 }
    ];
    // Tier 1: 8 Inner Crown Gimbal Nozzles (R = 1.35m, y = 0.52m)
    for (let k = 0; k < 8; k++) {
      nozzles.push({
        tier: 1,
        idx: k,
        baseAng: (k / 8) * Math.PI * 2.0,
        rBase: 1.35,
        nozY: 0.52
      });
    }
    // Tier 2: 16 Outer Dancing Ring Gimbal Nozzles (R = 2.85m, y = 0.495m)
    for (let k = 0; k < 16; k++) {
      nozzles.push({
        tier: 2,
        idx: k,
        baseAng: (k / 16) * Math.PI * 2.0,
        rBase: 2.85,
        nozY: 0.495
      });
    }

    const radialSides = 6;
    const axialSlices = 26;
    const splashSegs = 12;

    const pos = [];
    const norm = [];
    const uv = [];
    const cross = [];
    const nozAttr = [];
    const indices = [];

    let vCursor = 0;

    for (let n = 0; n < nozzles.length; n++) {
      const nz = nozzles[n];
      const baseX = RotundaDancingFountain.CENTER_X + Math.cos(nz.baseAng) * nz.rBase;
      const baseZ = RotundaDancingFountain.CENTER_Z + Math.sin(nz.baseAng) * nz.rBase;

      // 1. Volumetric 3D Stream Tube (radialSides x (axialSlices + 1))
      const tubeStart = vCursor;
      for (let i = 0; i <= axialSlices; i++) {
        const s = i / axialSlices;
        // Rest-pose world position inside Rotunda basin for valid static bounding box audits
        const restY = nz.nozY + Math.sin(s * Math.PI) * (nz.tier === 0 ? 1.8 : 1.1);
        for (let j = 0; j < radialSides; j++) {
          const phi = (j / radialSides) * Math.PI * 2.0;
          const cP = Math.cos(phi);
          const sP = Math.sin(phi);
          pos.push(baseX + cP * 0.04, restY, baseZ + sP * 0.04);
          norm.push(cP, 0.2, sP);
          uv.push(s, 0.0); // uv.y = 0.0 -> jet stream
          cross.push(cP, sP, 0.0);
          nozAttr.push(nz.tier, nz.idx, nz.baseAng, nz.nozY);
          vCursor++;
        }
      }

      for (let i = 0; i < axialSlices; i++) {
        const rowA = tubeStart + i * radialSides;
        const rowB = tubeStart + (i + 1) * radialSides;
        for (let j = 0; j < radialSides; j++) {
          const jNext = (j + 1) % radialSides;
          const a = rowA + j;
          const b = rowA + jNext;
          const c = rowB + jNext;
          const d = rowB + j;
          indices.push(a, b, c, a, c, d);
        }
      }

      // 2. Live Impact Splash Crown & Concentric Water Ripple Disc (center + 2 rings)
      const splashCenterIdx = vCursor;
      pos.push(baseX, RotundaDancingFountain.WATER_Y + 0.01, baseZ);
      norm.push(0, 1, 0);
      uv.push(1.0, 1.0); // uv.y = 1.0 -> splash/ripple disc
      cross.push(0.0, 0.0, 0.0);
      nozAttr.push(nz.tier, nz.idx, nz.baseAng, nz.nozY);
      vCursor++;

      const ringRadii = [0.45, 1.0];
      const ringStarts = [];
      for (let rIdx = 0; rIdx < ringRadii.length; rIdx++) {
        const rFrac = ringRadii[rIdx];
        ringStarts.push(vCursor);
        for (let j = 0; j < splashSegs; j++) {
          const phi = (j / splashSegs) * Math.PI * 2.0;
          const cP = Math.cos(phi);
          const sP = Math.sin(phi);
          pos.push(baseX + cP * 0.25 * rFrac, RotundaDancingFountain.WATER_Y + 0.008, baseZ + sP * 0.25 * rFrac);
          norm.push(0, 1, 0);
          uv.push(1.0, 1.0);
          cross.push(cP, sP, rFrac);
          nozAttr.push(nz.tier, nz.idx, nz.baseAng, nz.nozY);
          vCursor++;
        }
      }

      // Inner splash fan
      for (let j = 0; j < splashSegs; j++) {
        const jNext = (j + 1) % splashSegs;
        indices.push(splashCenterIdx, ringStarts[0] + j, ringStarts[0] + jNext);
      }
      // Outer ripple ring strip
      for (let j = 0; j < splashSegs; j++) {
        const jNext = (j + 1) % splashSegs;
        const a = ringStarts[0] + j;
        const b = ringStarts[0] + jNext;
        const c = ringStarts[1] + jNext;
        const d = ringStarts[1] + j;
        indices.push(a, d, c, a, c, b);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(norm), 3));
    geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uv), 2));
    geometry.setAttribute('aCross', new THREE.BufferAttribute(new Float32Array(cross), 3));
    geometry.setAttribute('aNozzle', new THREE.BufferAttribute(new Float32Array(nozAttr), 4));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));

    return { geometry, triangleCount: indices.length / 3 };
  }

  /**
   * Evaluates the live 4-Act Night Show RGB color & intensity on CPU so submerged LED pods
   * and Rotunda fountain basin surfaces (uFountainNightGlow) glow in 100% real-time sync.
   */
  _evalShowGlowColor(t) {
    const smoothstep = (e0, e1, x) => {
      const k = Math.max(0.0, Math.min(1.0, (x - e0) / (e1 - e0)));
      return k * k * (3.0 - 2.0 * k);
    };
    const tC = ((t % 64.0) + 64.0) % 64.0;
    const fade = 1.8;
    let w1 = (1.0 - smoothstep(16.0 - fade, 16.0 + fade, tC)) + smoothstep(64.0 - fade, 64.0, tC);
    let w2 = smoothstep(16.0 - fade, 16.0 + fade, tC) * (1.0 - smoothstep(32.0 - fade, 32.0 + fade, tC));
    let w3 = smoothstep(32.0 - fade, 32.0 + fade, tC) * (1.0 - smoothstep(48.0 - fade, 48.0 + fade, tC));
    let w4 = smoothstep(48.0 - fade, 48.0 + fade, tC) * (1.0 - smoothstep(64.0 - fade, 64.0, tC));
    const wSum = Math.max(1e-4, w1 + w2 + w3 + w4);
    w1 /= wSum;
    w2 /= wSum;
    w3 /= wSum;
    w4 /= wSum;

    if (tC < 16.0) this.currentActName = 'Act I: UI Makara Golden Crown Waltz';
    else if (tC < 32.0) this.currentActName = 'Act II: Emerald-Aqua Double Helix';
    else if (tC < 48.0) this.currentActName = 'Act III: Amethyst-Magenta Cathedral Arches';
    else this.currentActName = 'Act IV: Nusantara Peony Finale';

    // Palette matching vertex shader 4-Act LED choreography
    const r = 1.00 * w1 + 0.07 * w2 + 0.78 * w3 + (0.85 + 0.15 * Math.sin(t * 3.5)) * w4;
    const g = 0.70 * w1 + 0.92 * w2 + 0.34 * w3 + (0.48 + 0.35 * Math.cos(t * 2.8)) * w4;
    const b = 0.14 * w1 + 0.88 * w2 + 0.96 * w3 + (0.46 + 0.35 * Math.sin(t * 2.8)) * w4;

    this._glowColor.set(r, g, b);
    return this._glowColor;
  }

  update(elapsed, sunElevationDeg = 45.0, sharedSurfaceUniforms = null, sunDirection = null) {
    this.elapsed = elapsed;
    this.uniforms.uTime.value = elapsed;

    // Smooth dusk-to-night activation: starts fading in below +5.5 deg solar elevation, full show by -1.5 deg
    const rawNight = Math.max(0.0, Math.min(1.0, (5.5 - sunElevationDeg) / 7.0));
    this.nightBlend = rawNight * rawNight * (3.0 - 2.0 * rawNight);
    this.uniforms.uNightBlend.value = this.nightBlend;

    if (sunDirection) {
      this.uniforms.uSunDir.value.copy(sunDirection);
    }

    if (this.nightBlend > 0.005) {
      const col = this._evalShowGlowColor(elapsed);
      this.uniforms.uGlobalShowColor.value.copy(col);
      const pulse = 0.86 + 0.14 * Math.sin(elapsed * 3.1);
      if (sharedSurfaceUniforms && sharedSurfaceUniforms.uFountainNightGlow) {
        sharedSurfaceUniforms.uFountainNightGlow.value.set(
          col.x,
          col.y,
          col.z,
          this.nightBlend * pulse
        );
      }
    } else {
      this.currentActName = 'Daytime Royal Crown';
      if (sharedSurfaceUniforms && sharedSurfaceUniforms.uFountainNightGlow) {
        sharedSurfaceUniforms.uFountainNightGlow.value.set(0.0, 0.0, 0.0, 0.0);
      }
    }
  }
}
