import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (True-Color Photo-Realistic Edition)
 * ./core/WeatherOpticsEngine.js
 *
 * Calibrated 14:00 WIB (2:00 PM) UI Depok Tropical Daylight & Night GI Illumination Controller:
 * - Default: Equirectangular afternoon sky dome with soft cumulus clouds & horizon haze
 * - Warm tropical sun + turf-to-sky hemisphere bounce + opposite sky fill (zero black coal shadows)
 * - Night GI Toggle: Smoothly transitions sky, hemisphere bounce, and passive #000000 window emissiveMap
 */
export class WeatherOpticsEngine {
  constructor(scene, renderer, materialBank) {
    this.scene = scene;
    this.renderer = renderer;
    this.materials = materialBank;
    this.mode = 'afternoon_1400';

    this.bgAfternoonColor = new THREE.Color(0xcfe2f5);
    this.bgNightColor = new THREE.Color(0x0b1324);
    this.skyTexture = scene.environment?.userData?.skyBackgroundTexture || null;

    this.scene.background = this.skyTexture || this.bgAfternoonColor.clone();
    if ('environmentIntensity' in this.scene) {
      this.scene.environmentIntensity = 0.68;
    }
    this.scene.fog = new THREE.FogExp2(0xd4e4f4, 0.00095);
    this.renderer.toneMappingExposure = 0.96;

    // 1. Balanced Hemisphere Sky-to-Turf Bounce Light (preserves rich shadow contrast while preventing pitch-black crevices)
    this.hemiLight = new THREE.HemisphereLight(0xdcebff, 0x586847, 0.78);
    this.hemiLight.position.set(0, 120, 0);
    this.scene.add(this.hemiLight);

    // 2. Realistic Afternoon Sun (~39.5° elevation from WNW, casting crisp dimensional shadows across roads, curbs, trees & facades)
    this.sunOffset = new THREE.Vector3(-98, 92, 56);
    this.sunLight = new THREE.DirectionalLight(0xfff5de, 2.38);
    this.sunLight.position.copy(this.sunOffset);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.set(4096, 4096);
    this.sunLight.shadow.camera.near = 24;
    this.sunLight.shadow.camera.far = 265;
    const d = 78;
    this.sunLight.shadow.camera.left = -d;
    this.sunLight.shadow.camera.right = d;
    this.sunLight.shadow.camera.top = d;
    this.sunLight.shadow.camera.bottom = -d;
    this.sunLight.shadow.bias = -0.00014;
    this.sunLight.shadow.normalBias = 0.011;
    this.sunLight.shadow.radius = 1.25;
    this.scene.add(this.sunLight);
    this.scene.add(this.sunLight.target);

    // 3. Subtle Opposite Sky Fill Light (cool skylight fill in shadow zones)
    this.skyFillLight = new THREE.DirectionalLight(0xc8e0fb, 0.30);
    this.skyFillLight.position.set(78, 72, -58);
    this.scene.add(this.skyFillLight);

    // 4. Subtle GPU Atmospheric Pollen/Light Motes (1 single Points draw call)
    this.motes = this._createGpuAtmosphericMotes(180);
    this.scene.add(this.motes);
  }

  _createGpuAtmosphericMotes(count) {
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(count * 3);
    const seed = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const a = (i * 2.399963) % (Math.PI * 2);
      const r = 14 + ((i * 17.3) % 68);
      pos[i * 3] = Math.cos(a) * r;
      pos[i * 3 + 1] = 2.0 + ((i * 7.1) % 24.0);
      pos[i * 3 + 2] = Math.sin(a) * r;
      seed[i] = (i * 0.618034) % 1.0;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));

    this.moteUniforms = {
      uTime: { value: 0 },
      uOpacity: { value: 0.18 }
    };

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: this.moteUniforms,
      vertexShader: `
        attribute float aSeed;
        uniform float uTime;
        void main() {
          vec3 p = position;
          p.x += sin(uTime * 0.35 + aSeed * 18.0) * 1.4;
          p.y += cos(uTime * 0.28 + aSeed * 12.0) * 0.6;
          p.z += cos(uTime * 0.31 + aSeed * 22.0) * 1.4;
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_PointSize = clamp(16.0 / max(1.0, -mv.z), 1.1, 3.0);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        uniform float uOpacity;
        void main() {
          vec2 c = gl_PointCoord - vec2(0.5);
          float d = dot(c, c);
          if (d > 0.25) discard;
          float alpha = (1.0 - d * 4.0) * uOpacity;
          gl_FragColor = vec4(1.0, 0.98, 0.92, alpha);
        }
      `
    });
    const pts = new THREE.Points(geo, mat);
    pts.name = 'NexusAtmosphericMotes';
    pts.frustumCulled = false;
    return pts;
  }

  setLightingMode(mode) {
    this.mode = mode;
    if (mode === 'night_gi') {
      this.scene.background = this.bgNightColor.clone();
      this.scene.fog.color.copy(this.bgNightColor);
      this.scene.fog.density = 0.0022;
      this.hemiLight.color.setHex(0x3b5998);
      this.hemiLight.groundColor.setHex(0x18221e);
      this.hemiLight.intensity = 0.52;
      this.sunLight.color.setHex(0x9ec5ff);
      this.sunLight.intensity = 0.38;
      this.sunLight.position.set(55, 115, -45);
      this.skyFillLight.intensity = 0.22;
      this.renderer.toneMappingExposure = 0.96;
      if (this.materials?.windowGlass) {
        this.materials.windowGlass.emissiveIntensity = 1.45;
        this.materials.windowGlass.envMapIntensity = 0.35;
      }
      if (this.moteUniforms) this.moteUniforms.uOpacity.value = 0.14;
    } else {
      this.scene.background = this.skyTexture || this.bgAfternoonColor.clone();
      if ('environmentIntensity' in this.scene) {
        this.scene.environmentIntensity = 0.68;
      }
      this.scene.fog.color.setHex(0xd4e4f4);
      this.scene.fog.density = 0.00095;
      this.hemiLight.color.setHex(0xdcebff);
      this.hemiLight.groundColor.setHex(0x586847);
      this.hemiLight.intensity = 0.78;
      this.sunLight.color.setHex(0xfff5de);
      this.sunLight.intensity = 2.38;
      this.sunLight.position.copy(this.sunOffset);
      this.skyFillLight.intensity = 0.30;
      this.renderer.toneMappingExposure = 0.96;
      if (this.materials?.windowGlass) {
        this.materials.windowGlass.emissiveIntensity = 0.0;
        this.materials.windowGlass.envMapIntensity = 1.85;
      }
      if (this.moteUniforms) this.moteUniforms.uOpacity.value = 0.16;
    }
  }

  update(timeSec, camera = null) {
    if (this.moteUniforms) {
      this.moteUniforms.uTime.value = timeSec;
    }
    if (camera && this.mode !== 'night_gi') {
      // Snap shadow camera target in 4.0m increments toward the active ground-level sector for razor-sharp close-up shadows
      const isLow = camera.position.y < 36.0;
      const cx = isLow ? Math.round((camera.position.x * 0.36) / 4.0) * 4.0 : 0.0;
      const cz = isLow ? Math.round((camera.position.z * 0.36) / 4.0) * 4.0 : 0.0;
      if (this.sunLight.target.position.x !== cx || this.sunLight.target.position.z !== cz) {
        this.sunLight.target.position.set(cx, 0.0, cz);
        this.sunLight.position.set(cx + this.sunOffset.x, this.sunOffset.y, cz + this.sunOffset.z);
        this.sunLight.target.updateMatrixWorld();
      }
    }
  }
}
