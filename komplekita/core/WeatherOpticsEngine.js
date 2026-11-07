import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./core/WeatherOpticsEngine.js
 *
 * Calibrated 14:00 WIB (2:00 PM) UI Depok Tropical Daylight & Night GI Illumination Controller:
 * - Default: Bright, balanced 14:00 WIB afternoon light without harsh glare or crushed black shadows
 * - Night GI Toggle: Smoothly transitions sky, hemisphere bounce, and passive #000000 window emissiveMap
 * - 100% GPU Vertex-Animated Atmospheric Motes (1 single Points draw call)
 */
export class WeatherOpticsEngine {
  constructor(scene, renderer, materialBank) {
    this.scene = scene;
    this.renderer = renderer;
    this.materials = materialBank;
    this.mode = 'afternoon_1400'; // 'afternoon_1400' | 'night_gi'

    this.bgAfternoon = new THREE.Color(0xd6e6f7);
    this.bgNight = new THREE.Color(0x0b1324);

    this.scene.background = this.bgAfternoon.clone();
    this.scene.fog = new THREE.FogExp2(0xd6e6f7, 0.0014);
    this.renderer.toneMappingExposure = 1.02;

    // 1. Hemisphere Sky-to-Turf Bounce Light (eliminates pitch-black underexposed loggias!)
    this.hemiLight = new THREE.HemisphereLight(0xf0f7ff, 0x7c876b, 1.34);
    this.hemiLight.position.set(0, 120, 0);
    this.scene.add(this.hemiLight);

    // 2. 14:00 WIB Sun (slightly WNW of zenith, casting natural 2 PM Depok shadows)
    this.sunLight = new THREE.DirectionalLight(0xfff9ed, 1.48);
    this.sunLight.position.set(-62, 138, 32);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.set(2048, 2048);
    this.sunLight.shadow.camera.near = 20;
    this.sunLight.shadow.camera.far = 290;
    const d = 96;
    this.sunLight.shadow.camera.left = -d;
    this.sunLight.shadow.camera.right = d;
    this.sunLight.shadow.camera.top = d;
    this.sunLight.shadow.camera.bottom = -d;
    this.sunLight.shadow.bias = -0.00035;
    this.sunLight.shadow.normalBias = 0.022;
    this.scene.add(this.sunLight);

    // 3. Soft Opposite Sky Fill Light (prevents black coal shadows on East/SE walls)
    this.skyFillLight = new THREE.DirectionalLight(0xdbeafe, 0.56);
    this.skyFillLight.position.set(78, 84, -58);
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
      uOpacity: { value: 0.22 }
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
          gl_PointSize = clamp(18.0 / max(1.0, -mv.z), 1.2, 3.5);
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
      this.scene.background.copy(this.bgNight);
      this.scene.fog.color.copy(this.bgNight);
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
      // Calibrated 14:00 WIB UI Depok Afternoon
      this.scene.background.copy(this.bgAfternoon);
      this.scene.fog.color.copy(this.bgAfternoon);
      this.scene.fog.density = 0.0014;
      this.hemiLight.color.setHex(0xf0f7ff);
      this.hemiLight.groundColor.setHex(0x7c876b);
      this.hemiLight.intensity = 1.34;
      this.sunLight.color.setHex(0xfff9ed);
      this.sunLight.intensity = 1.48;
      this.sunLight.position.set(-62, 138, 32);
      this.skyFillLight.intensity = 0.56;
      this.renderer.toneMappingExposure = 1.02;
      if (this.materials?.windowGlass) {
        this.materials.windowGlass.emissiveIntensity = 0.0;
        this.materials.windowGlass.envMapIntensity = 1.35;
      }
      if (this.moteUniforms) this.moteUniforms.uOpacity.value = 0.22;
    }
  }

  update(timeSec) {
    if (this.moteUniforms) {
      this.moteUniforms.uTime.value = timeSec;
    }
  }
}
