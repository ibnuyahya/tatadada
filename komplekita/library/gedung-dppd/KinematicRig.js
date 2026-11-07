import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./library/gedung-dppd/KinematicRig.js
 *
 * Stage 5 Kinematic Sub-Rig for Gedung DPPD UI:
 * - 11.4m Tapered Silver Flagpole on Circular Concrete Drum Base at Front Center of Forecourt (ref_084, ref_175)
 * - Dynamic Breeze-Animated Merah-Putih Indonesian Flag (dynamicKinematic = true so NexusBatchOptimizer keeps it live)
 */
export class DPPDKinematicRig {
  static build(localGroup, materials) {
    // Static Concrete Drum Base + Tapered Pole (batched into static materials = 0 extra draw calls)
    const drum = new THREE.Mesh(new THREE.CylinderGeometry(0.58, 0.68, 0.42, 16), materials.architecturalWhite);
    drum.position.set(0, 0.32, 66.35);
    drum.castShadow = true;
    drum.receiveShadow = true;
    localGroup.add(drum);

    const poleH = 11.4;
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.105, poleH, 10), materials.silverLouver);
    pole.position.set(0, 0.5 + poleH * 0.5, 66.35);
    pole.castShadow = true;
    localGroup.add(pole);

    // Dynamic Merah-Putih Flag with GPU Shader Breeze Wave (1 single dynamic Mesh)
    const flagUniforms = {
      uTime: { value: 0.0 },
      uWind: { value: 1.0 }
    };

    const flagGeo = new THREE.PlaneGeometry(2.1, 1.36, 16, 8);
    flagGeo.translate(1.05, 0, 0); // hoist attached at x = 0 along flagpole!

    const flagMat = new THREE.ShaderMaterial({
      side: THREE.DoubleSide,
      uniforms: flagUniforms,
      vertexShader: `
        uniform float uTime;
        uniform float uWind;
        varying vec2 vUv;
        varying float vWave;
        void main() {
          vUv = uv;
          vec3 p = position;
          float pinFactor = smoothstep(0.0, 0.18, uv.x);
          float w = sin(uv.x * 7.5 - uTime * 3.8) * 0.14 + cos(uv.x * 13.0 + uv.y * 3.0 - uTime * 5.2) * 0.05;
          p.z += w * pinFactor * uWind;
          p.y -= 0.08 * pinFactor;
          vWave = w * pinFactor;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 vUv;
        varying float vWave;
        void main() {
          vec3 red = vec3(0.82, 0.10, 0.12);
          vec3 white = vec3(0.96, 0.97, 0.98);
          vec3 col = vUv.y >= 0.5 ? red : white;
          float shade = 0.88 + 0.24 * clamp(vWave * 3.2 + 0.5, 0.0, 1.0);
          gl_FragColor = vec4(col * shade, 1.0);
        }
      `
    });

    const flagMesh = new THREE.Mesh(flagGeo, flagMat);
    flagMesh.name = 'DPPD_MerahPutihFlag';
    flagMesh.position.set(0.05, 0.5 + poleH - 0.82, 66.35);
    flagMesh.rotation.y = 0.35;
    flagMesh.userData.dynamicKinematic = true;
    localGroup.add(flagMesh);

    return {
      update(dt, telemetry = {}) {
        const t = telemetry.timeSec ?? (performance.now() * 0.001);
        flagUniforms.uTime.value = t;
        flagUniforms.uWind.value = THREE.MathUtils.clamp((telemetry.windSpeed ?? 3.5) / 3.5, 0.35, 2.0);
      }
    };
  }
}
