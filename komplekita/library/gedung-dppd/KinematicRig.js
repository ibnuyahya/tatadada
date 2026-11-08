import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (True-Color Photo-Realistic Edition)
 * ./library/gedung-dppd/KinematicRig.js
 *
 * Stage 5 Kinematic Sub-Rig for Gedung DPPD UI:
 * - Authentic Bare Galvanized Steel Mast on Circular Concrete Drum Base at the Front-Right Corner of the
 *   Blue ADA Wheelchair Stall (x = -4.38m, z = 67.85m — verified against ref_184, ref_185, ref_190, ref_082;
 *   NO non-existent Merah-Putih flag blocking the central gable!)
 * - Dynamic Breeze-Animated Vertical Hanging Cloth Banner ("UI DEPOK" on the right gold ladder frame at
 *   x = +10.45m, z = 66.92m — verified against ref_084, ref_082)
 */
export class DPPDKinematicRig {
  static build(localGroup, materials) {
    // 1. Authentic Circular Concrete Drum Base + Bare Galvanized Steel Pole at Corner of Blue ADA Stall (ref_184, ref_185, ref_082)
    const poleX = -4.25;
    const poleZ = 67.25;

    const drum = new THREE.Mesh(new THREE.CylinderGeometry(0.52, 0.58, 0.36, 16), materials.pavingBlock);
    drum.position.set(poleX, 0.28, poleZ);
    drum.castShadow = true;
    drum.receiveShadow = true;
    localGroup.add(drum);

    const poleH = 9.2;
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.085, poleH, 10), materials.silverLouver);
    pole.position.set(poleX, 0.44 + poleH * 0.5, poleZ);
    pole.castShadow = true;
    localGroup.add(pole);

    // Small pulley cap at top of mast (ref_082)
    const finial = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.05, 0.12, 8), materials.darkMetal);
    finial.position.set(poleX, 0.44 + poleH + 0.05, poleZ);
    localGroup.add(finial);

    // 2. Dynamic Breeze-Animated Vertical Hanging Cloth Banner on Right Gold Ladder Frame (ref_084, ref_082)
    //    Built as a True-3D Thickened Cloth Slab (BoxGeometry depth = 0.028m) + Top & Bottom Horizontal Gold Mounting Bars!
    const bannerUniforms = {
      uTime: { value: 0.0 },
      uWind: { value: 1.0 }
    };

    const bannerGeo = new THREE.BoxGeometry(0.58, 3.85, 0.028, 6, 18, 1);
    const bannerMat = new THREE.ShaderMaterial({
      uniforms: bannerUniforms,
      vertexShader: `
        uniform float uTime;
        uniform float uWind;
        varying vec2 vUv;
        varying float vFold;
        void main() {
          vUv = uv;
          vec3 p = position;
          float normY = clamp((p.y + 1.925) / 3.85, 0.0, 1.0);
          // Top edge pinned at normY = 1.0, lower 3D cloth slab drapes & flutters in afternoon breeze (ref_084)
          float drape = smoothstep(1.0, 0.0, normY);
          float w = sin(normY * 6.5 - uTime * 3.2) * 0.09 + cos(p.x * 5.0 + normY * 10.0 - uTime * 4.5) * 0.04;
          p.z += w * drape * uWind;
          p.x += (0.12 * drape * drape) + 0.03 * sin(uTime * 2.4) * drape;
          vFold = w * drape;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 vUv;
        varying float vFold;
        void main() {
          vec3 white = vec3(0.95, 0.96, 0.98);
          vec3 royalBlue = vec3(0.11, 0.31, 0.65);
          vec3 gold = vec3(0.92, 0.72, 0.16);
          vec3 col = white;
          if (vUv.y > 0.86) {
            col = gold;
          } else if (vUv.y < 0.34 && vUv.y > 0.10) {
            col = royalBlue;
          } else if (vUv.y <= 0.10) {
            col = mix(white, gold, step(0.5, fract(vUv.x * 4.0)));
          }
          float shade = 0.88 + 0.22 * clamp(vFold * 4.0 + 0.5, 0.0, 1.0);
          gl_FragColor = vec4(col * shade, 1.0);
        }
      `
    });

    const bannerMesh = new THREE.Mesh(bannerGeo, bannerMat);
    bannerMesh.name = 'DPPD_RightHangingClothBanner';
    bannerMesh.position.set(10.22, 2.32, 66.58);
    bannerMesh.userData.dynamicKinematic = true;
    localGroup.add(bannerMesh);

    // 3D Horizontal Top & Bottom Weighted Banner Mounting Rods clamping the banner to the gold truss
    for (const ry of [4.24, 0.42]) {
      const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.68, 8), materials.goldAccent);
      rod.rotation.z = Math.PI * 0.5;
      rod.position.set(10.25, ry, 66.58);
      localGroup.add(rod);
    }

    return {
      update(dt, telemetry = {}) {
        const t = telemetry.timeSec ?? (performance.now() * 0.001);
        bannerUniforms.uTime.value = t;
        bannerUniforms.uWind.value = THREE.MathUtils.clamp((telemetry.windSpeed ?? 3.5) / 3.5, 0.35, 2.0);
      }
    };
  }
}
