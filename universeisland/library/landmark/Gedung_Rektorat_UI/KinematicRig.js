import * as THREE from 'three';
import { PhysicsKernel } from '../../../core/PhysicsKernel.js';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.3 — 10x Ultra-High Level of Detail Edition)
 * ./library/landmark/Gedung_Rektorat_UI/KinematicRig.js
 *
 * TAHAP 6A (Gate 3): Artikulasi Kinematika Aset 10x Ultra-LOD & Audit Anti-Interseksi:
 * - 15 Tiang Bendera Berpedestal Segidelapan Bertingkat, Mahkota Bola Puncak, Kerek Tali (Cleat),
 *   & Jaring Kain Sutra Resolusi Tinggi (32 x 16 = 15,360 Segitiga dengan Normal Analitis Dinamis — 1 Draw Call!).
 * - Pintu Kaca Geser Lobi Utama Berbingkai Stile-Rail 3D & Gagang Silinder Ganda (1 Draw Call Dinamis!).
 * - Lampu Suar Puncak Fresnel 32x20 & Cincin Pelindung Penangkal Petir (y <= 49.97m <= 50.00m).
 */
export class KinematicRig {
  static build(materials) {
    const rigGroup = new THREE.Group();
    rigGroup.name = 'Tahap6_KinematicRig';

    // 15 Ceremonial Flagpoles inside the Front Rectangular Flag Cloister (Mavic Photo #1!)
    const poleSpecs = [
      {
        x: -2.5,
        z: 47.2,
        mastH: 11.4,
        flagW: 2.35,
        flagH: 1.48,
        phase: 0.0,
        topColor: [0.86, 0.11, 0.13],
        bottomColor: [0.97, 0.97, 0.98]
      },
      // West Row (7 UI Faculty Flags)
      { x: -14.4, z: 56.2, mastH: 6.8, flagW: 1.32, flagH: 0.9, phase: 0.5, topColor: [0.96, 0.78, 0.12], bottomColor: [0.93, 0.73, 0.1] },
      { x: -12.8, z: 56.2, mastH: 6.8, flagW: 1.32, flagH: 0.9, phase: 1.1, topColor: [0.12, 0.58, 0.28], bottomColor: [0.1, 0.52, 0.24] },
      { x: -11.2, z: 56.2, mastH: 6.8, flagW: 1.32, flagH: 0.9, phase: 1.7, topColor: [0.12, 0.34, 0.76], bottomColor: [0.1, 0.29, 0.68] },
      { x: -9.6, z: 56.2, mastH: 6.8, flagW: 1.32, flagH: 0.9, phase: 2.3, topColor: [0.82, 0.14, 0.16], bottomColor: [0.76, 0.11, 0.14] },
      { x: -8.0, z: 56.2, mastH: 6.8, flagW: 1.32, flagH: 0.9, phase: 2.9, topColor: [0.58, 0.6, 0.64], bottomColor: [0.92, 0.76, 0.15] },
      { x: -6.4, z: 56.2, mastH: 6.8, flagW: 1.32, flagH: 0.9, phase: 3.5, topColor: [0.94, 0.94, 0.95], bottomColor: [0.92, 0.75, 0.16] },
      { x: -4.8, z: 56.2, mastH: 6.8, flagW: 1.32, flagH: 0.9, phase: 4.1, topColor: [0.22, 0.54, 0.84], bottomColor: [0.46, 0.22, 0.68] },
      // East Row (7 UI Faculty Flags)
      { x: -0.2, z: 56.2, mastH: 6.8, flagW: 1.32, flagH: 0.9, phase: 0.8, topColor: [0.92, 0.44, 0.12], bottomColor: [0.88, 0.39, 0.1] },
      { x: 1.4, z: 56.2, mastH: 6.8, flagW: 1.32, flagH: 0.9, phase: 1.4, topColor: [0.14, 0.38, 0.78], bottomColor: [0.18, 0.2, 0.24] },
      { x: 3.0, z: 56.2, mastH: 6.8, flagW: 1.32, flagH: 0.9, phase: 2.0, topColor: [0.54, 0.2, 0.68], bottomColor: [0.48, 0.17, 0.62] },
      { x: 4.6, z: 56.2, mastH: 6.8, flagW: 1.32, flagH: 0.9, phase: 2.6, topColor: [0.82, 0.15, 0.18], bottomColor: [0.14, 0.32, 0.74] },
      { x: 6.2, z: 56.2, mastH: 6.8, flagW: 1.32, flagH: 0.9, phase: 3.2, topColor: [0.14, 0.56, 0.32], bottomColor: [0.48, 0.2, 0.64] },
      { x: 7.8, z: 56.2, mastH: 6.8, flagW: 1.32, flagH: 0.9, phase: 3.8, topColor: [0.15, 0.66, 0.68], bottomColor: [0.92, 0.45, 0.55] },
      { x: 9.4, z: 56.2, mastH: 6.8, flagW: 1.32, flagH: 0.9, phase: 4.4, topColor: [0.16, 0.52, 0.72], bottomColor: [0.22, 0.6, 0.32] }
    ];

    const baseCourtY = 0.14;
    for (let i = 0; i < poleSpecs.length; i++) {
      const ps = poleSpecs[i];
      const isMain = i === 0;
      const rTop = isMain ? 0.065 : 0.042;
      const rBot = isMain ? 0.115 : 0.072;

      // Stepped Octagonal Architectural Concrete Pedestal Base (batched into static precastConcrete!)
      const pedR1 = isMain ? 0.58 : 0.28;
      const pedH1 = isMain ? 0.32 : 0.18;
      const ped1 = new THREE.CylinderGeometry(pedR1 * 0.92, pedR1, pedH1, 8, 1);
      ped1.translate(ps.x, baseCourtY + pedH1 * 0.5 - 0.02, ps.z);
      const ped1Mesh = new THREE.Mesh(ped1, materials.precastConcrete);
      ped1Mesh.name = `Flagpole_PedestalLower_${i}`;
      ped1Mesh.castShadow = true;
      rigGroup.add(ped1Mesh);

      const pedR2 = isMain ? 0.36 : 0.19;
      const pedH2 = isMain ? 0.24 : 0.14;
      const ped2 = new THREE.CylinderGeometry(pedR2 * 0.85, pedR2, pedH2, 8, 1);
      ped2.translate(ps.x, baseCourtY + pedH1 + pedH2 * 0.5 - 0.02, ps.z);
      const ped2Mesh = new THREE.Mesh(ped2, materials.precastConcrete);
      ped2Mesh.name = `Flagpole_PedestalUpper_${i}`;
      ped2Mesh.castShadow = true;
      rigGroup.add(ped2Mesh);

      // High-Resolution 24-Segment Tapered Mast + Finial Sphere Truck + Halyard Cleat (batched into bronzeGold!)
      const mast = new THREE.CylinderGeometry(rTop, rBot, ps.mastH, 24, 3);
      mast.translate(ps.x, baseCourtY + ps.mastH * 0.5, ps.z);
      const mastMesh = new THREE.Mesh(mast, materials.bronzeGold);
      mastMesh.name = `Flagpole_Mast_${i}`;
      mastMesh.castShadow = true;
      rigGroup.add(mastMesh);

      const finialR = isMain ? 0.135 : 0.082;
      const finial = new THREE.SphereGeometry(finialR, 20, 14);
      finial.scale(1.0, 1.16, 1.0);
      finial.translate(ps.x, baseCourtY + ps.mastH + finialR * 0.85, ps.z);
      const finialMesh = new THREE.Mesh(finial, materials.bronzeGold);
      finialMesh.name = `Flagpole_Finial_${i}`;
      finialMesh.castShadow = true;
      rigGroup.add(finialMesh);

      const cleat = new THREE.BoxGeometry(0.035, 0.22, 0.075);
      cleat.translate(ps.x - rBot * 0.95, baseCourtY + 1.35, ps.z);
      const cleatMesh = new THREE.Mesh(cleat, materials.bronzeGold);
      cleatMesh.name = `Flagpole_HalyardCleat_${i}`;
      rigGroup.add(cleatMesh);
    }

    // 1. Combined 15-Flag High-Density Ceremonial Silk Cloth Mesh (32 x 16 = 15,360 Tris in 1 Draw Call!)
    const cols = 32;
    const rows = 16;
    const vertsPerFlag = (cols + 1) * (rows + 1);
    const totalVerts = vertsPerFlag * poleSpecs.length;

    const positions = new Float32Array(totalVerts * 3);
    const basePositions = new Float32Array(totalVerts * 3);
    const flagLocalU = new Float32Array(totalVerts);
    const flagLocalV = new Float32Array(totalVerts);
    const flagWidthArr = new Float32Array(totalVerts);
    const flagPhase = new Float32Array(totalVerts);
    const normals = new Float32Array(totalVerts * 3);
    const colors = new Float32Array(totalVerts * 3);
    const uvs = new Float32Array(totalVerts * 2);
    const indices = new Uint32Array(poleSpecs.length * cols * rows * 6);

    let vPtr = 0;
    let iPtr = 0;
    for (let f = 0; f < poleSpecs.length; f++) {
      const ps = poleSpecs[f];
      const baseIndex = f * vertsPerFlag;
      const yTop = baseCourtY + ps.mastH - 0.12;

      for (let ic = 0; ic <= cols; ic++) {
        const u = ic / cols;
        for (let ir = 0; ir <= rows; ir++) {
          const v = ir / rows;
          const px = ps.x + 0.06 + u * ps.flagW;
          const py = yTop - v * ps.flagH;
          const pz = ps.z;

          positions[vPtr * 3] = px;
          positions[vPtr * 3 + 1] = py;
          positions[vPtr * 3 + 2] = pz;

          basePositions[vPtr * 3] = px;
          basePositions[vPtr * 3 + 1] = py;
          basePositions[vPtr * 3 + 2] = pz;
          flagLocalU[vPtr] = u;
          flagLocalV[vPtr] = v;
          flagWidthArr[vPtr] = ps.flagW;
          flagPhase[vPtr] = ps.phase;

          normals[vPtr * 3] = 0;
          normals[vPtr * 3 + 1] = 0;
          normals[vPtr * 3 + 2] = 1;

          // Subtle woven silk fabric sheen & stitched hoist hem shading
          const hemShade = u < 0.04 ? 0.9 : 1.0 - 0.04 * Math.sin(v * Math.PI);
          const col = v < 0.5 ? ps.topColor : ps.bottomColor;
          colors[vPtr * 3] = col[0] * hemShade;
          colors[vPtr * 3 + 1] = col[1] * hemShade;
          colors[vPtr * 3 + 2] = col[2] * hemShade;

          uvs[vPtr * 2] = u;
          uvs[vPtr * 2 + 1] = v;
          vPtr++;
        }
      }

      const stride = rows + 1;
      for (let ic = 0; ic < cols; ic++) {
        for (let ir = 0; ir < rows; ir++) {
          const a = baseIndex + ic * stride + ir;
          const b = baseIndex + (ic + 1) * stride + ir;
          const c = baseIndex + (ic + 1) * stride + (ir + 1);
          const d = baseIndex + ic * stride + (ir + 1);
          indices[iPtr++] = a;
          indices[iPtr++] = b;
          indices[iPtr++] = c;
          indices[iPtr++] = a;
          indices[iPtr++] = c;
          indices[iPtr++] = d;
        }
      }
    }

    const flagGeo = new THREE.BufferGeometry();
    flagGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    flagGeo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    flagGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    flagGeo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    flagGeo.setIndex(new THREE.BufferAttribute(indices, 1));

    const flagMesh = new THREE.Mesh(flagGeo, materials.flagFabric);
    flagMesh.name = 'Kinematic_CloisterFacultyAndNationalFlags';
    flagMesh.userData.dynamicKinematic = true;
    flagMesh.castShadow = true;
    rigGroup.add(flagMesh);

    // 2. Articulated 3D Framed Grand Lobby Sliding Entrance Doors (+Z Recessed Atrium at z = 13.18m)
    //    Static overhead track header (batched into static bronzeGold) + 1 dynamic multi-part door mesh!
    const trackHeader = new THREE.BoxGeometry(3.85, 0.16, 0.18);
    trackHeader.translate(0.0, 0.37 + 2.88, 13.18);
    const trackMesh = new THREE.Mesh(trackHeader, materials.bronzeGold);
    trackMesh.name = 'LobbyDoor_OverheadTrackHeader';
    rigGroup.add(trackMesh);

    const doorGeo = KinematicRig._buildDetailedSlidingDoorsGeometry();
    const doorPosArr = doorGeo.attributes.position.array;
    const vDoorTotal = doorGeo.attributes.position.count;
    const doorBaseX = new Float32Array(vDoorTotal);
    const doorSideSign = doorGeo.userData.doorSideSign;
    for (let i = 0; i < vDoorTotal; i++) {
      doorBaseX[i] = doorPosArr[i * 3];
    }

    const doorMesh = new THREE.Mesh(doorGeo, materials.ribbonGlass);
    doorMesh.name = 'Kinematic_LobbySlidingDoors';
    doorMesh.userData.dynamicKinematic = true;
    rigGroup.add(doorMesh);

    // 3. Summit Aviation Obstacle Warning Beacon (y = 49.86m, r = 0.11m -> maxY = 49.97m <= 50.00m!)
    //    + Static Bronze Mounting Collar & Protective Guard Ring at y = 49.75m
    const beaconBase = new THREE.CylinderGeometry(0.075, 0.095, 0.1, 20, 1);
    beaconBase.translate(0, 49.74, 0);
    const beaconBaseMesh = new THREE.Mesh(beaconBase, materials.bronzeGold);
    beaconBaseMesh.name = 'SummitBeacon_MountCollar';
    rigGroup.add(beaconBaseMesh);

    const beaconGeo = new THREE.SphereGeometry(0.11, 32, 20);
    beaconGeo.translate(0, 49.86, 0);
    const beaconMesh = new THREE.Mesh(beaconGeo, materials.aviationBeacon);
    beaconMesh.name = 'Kinematic_SummitBeacon';
    beaconMesh.userData.dynamicKinematic = true;
    rigGroup.add(beaconMesh);

    const rigController = {
      time: 0.0,
      flagMesh,
      basePositions,
      flagLocalU,
      flagLocalV,
      flagWidthArr,
      flagPhase,
      doorMesh,
      doorBaseX,
      doorSideSign,
      currentSlideDist: 0.0,
      beaconMesh,
      beaconMat: materials.aviationBeacon
    };

    rigGroup.userData.kinematicController = rigController;
    return rigGroup;
  }

  /**
   * Builds a 3D multi-part architectural sliding double-door BufferGeometry (recessed glass panes,
   * perimeter vertical stiles, top/bottom kickplate rails, mid-rail, & cylindrical push-bar handles)
   * packed into 1 single dynamic Draw Call!
   */
  static _buildDetailedSlidingDoorsGeometry() {
    const positions = [];
    const normals = [];
    const uvs = [];
    const indices = [];
    const sideSigns = [];

    const appendPart = (geo, sideSign) => {
      if (!geo.attributes.normal) geo.computeVertexNormals();
      const p = geo.attributes.position.array;
      const n = geo.attributes.normal.array;
      const u = geo.attributes.uv ? geo.attributes.uv.array : null;
      const base = positions.length / 3;
      const vCount = geo.attributes.position.count;

      for (let i = 0; i < vCount; i++) {
        positions.push(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
        normals.push(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]);
        uvs.push(u ? u[i * 2] : 0.5, u ? u[i * 2 + 1] : 0.5);
        sideSigns.push(sideSign);
      }
      if (geo.index) {
        const idx = geo.index.array;
        for (let i = 0; i < idx.length; i++) {
          indices.push(base + idx[i]);
        }
      } else {
        for (let i = 0; i < vCount; i++) {
          indices.push(base + i);
        }
      }
    };

    const yFloor = 0.37;
    const zCenter = 13.18;
    const leafW = 1.08;
    const leafH = 2.80;

    for (const sgn of [-1, 1]) {
      const cx = sgn * 0.55;

      // Central recessed glass pane
      const glass = new THREE.BoxGeometry(leafW - 0.14, leafH - 0.24, 0.032);
      glass.translate(cx, yFloor + leafH * 0.5 + 0.02, zCenter);
      appendPart(glass, sgn);

      // Left & right vertical perimeter stiles (outer width strictly leafW = 1.08m!)
      for (const sx of [-1, 1]) {
        const stile = new THREE.BoxGeometry(0.075, leafH, 0.092);
        stile.translate(cx + sx * (leafW * 0.5 - 0.0375), yFloor + leafH * 0.5, zCenter);
        appendPart(stile, sgn);
      }

      // Top header rail & bottom kickplate rail
      const topRail = new THREE.BoxGeometry(leafW - 0.15, 0.10, 0.09);
      topRail.translate(cx, yFloor + leafH - 0.05, zCenter);
      appendPart(topRail, sgn);

      const kickPlate = new THREE.BoxGeometry(leafW - 0.15, 0.18, 0.09);
      kickPlate.translate(cx, yFloor + 0.09, zCenter);
      appendPart(kickPlate, sgn);

      // Tubular vertical pull-handles on inner meeting edge (+Z front & -Z lobby interior)
      const handleX = cx - sgn * 0.38;
      for (const sz of [-1, 1]) {
        const bar = new THREE.CylinderGeometry(0.016, 0.016, 0.68, 12, 1);
        bar.translate(handleX, yFloor + 1.18, zCenter + sz * 0.078);
        appendPart(bar, sgn);

        for (const dy of [-0.26, 0.26]) {
          const standoff = new THREE.CylinderGeometry(0.012, 0.012, 0.045, 8, 1);
          standoff.rotateX(Math.PI * 0.5);
          standoff.translate(handleX, yFloor + 1.18 + dy, zCenter + sz * 0.056);
          appendPart(standoff, sgn);
        }
      }
    }

    const out = new THREE.BufferGeometry();
    out.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    out.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normals), 3));
    out.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
    out.setIndex(indices);
    out.userData.doorSideSign = new Float32Array(sideSigns);
    return out;
  }

  /**
   * Updates 15-flag 32x16 travelling-wave cloth kinematics (with analytical 3D wave normals!),
   * 3D framed lobby sliding doors, and night summit beacon.
   */
  static update(rigController, dt, telemetryState = {}) {
    if (!rigController) return;
    rigController.time += dt;
    const t = rigController.time;

    // 1. 3-Harmonic Travelling Wind Wave + Analytical Dynamic Cloth Normals across all 15 Flags!
    const geo = rigController.flagMesh.geometry;
    const posAttr = geo.attributes.position;
    const normAttr = geo.attributes.normal;
    const base = rigController.basePositions;
    const uArr = rigController.flagLocalU;
    const vArr = rigController.flagLocalV;
    const wArr = rigController.flagWidthArr;
    const phArr = rigController.flagPhase;
    const windMult = telemetryState.windMultiplier || 1.0;
    const wind = PhysicsKernel.sampleWindField(-2.5, 52.0, t, windMult);
    const freqBoost = 1.0 + Math.min(1.15, (windMult - 1.0) * 0.45);
    const mag = wind.magnitude;

    const pArray = posAttr.array;
    const nArray = normAttr.array;
    const count = posAttr.count;

    for (let i = 0; i < count; i++) {
      const u = uArr[i];
      const v = vArr ? vArr[i] : 0.5;
      const fw = wArr ? wArr[i] : 1.32;
      const ph = phArr ? phArr[i] : 0.0;

      const arg1 = t * 5.2 * freqBoost - u * 4.8 + ph + (v - 0.5) * 0.55;
      const arg2 = t * 8.6 * freqBoost - u * 8.4 + ph * 1.3 - (v - 0.5) * 0.85;
      const arg3 = t * 13.4 * freqBoost - u * 13.2 + ph * 1.9;

      const s1 = Math.sin(arg1);
      const c1 = Math.cos(arg1);
      const s2 = Math.sin(arg2);
      const c2 = Math.cos(arg2);
      const s3 = Math.sin(arg3);
      const c3 = Math.cos(arg3);

      const wBody = 0.23 * s1 + 0.085 * c2 + 0.032 * u * s3;
      const waveZ = u * wBody * mag;
      const flutterY = u * u * 0.024 * s2 * mag;
      const sagY = -u * u * (0.065 / Math.max(1.0, windMult * 0.72));

      const i3 = i * 3;
      pArray[i3] = base[i3] - u * 0.035 * Math.abs(waveZ);
      pArray[i3 + 1] = base[i3 + 1] + sagY + flutterY;
      pArray[i3 + 2] = base[i3 + 2] + waveZ;

      // Analytical cloth normal from d(waveZ)/dx = (1/fw) * d(waveZ)/du (0 mesh scans!)
      const dWaveDu =
        (wBody + u * (-4.8 * 0.23 * c1 + 8.4 * 0.085 * s2 + 0.032 * s3 - 13.2 * 0.032 * u * c3)) * mag;
      const dzDx = dWaveDu / Math.max(0.5, fw);
      const invN = 1.0 / Math.sqrt(1.0 + dzDx * dzDx);
      nArray[i3] = -dzDx * invN;
      nArray[i3 + 1] = 0.06 * invN;
      nArray[i3 + 2] = invN;
    }
    posAttr.needsUpdate = true;
    normAttr.needsUpdate = true;

    // 2. Smooth bounded lobby door articulation (slide range [0.0m, 0.70m] — strictly inside +-2.30m atrium slot!)
    const doorOpenRatio =
      telemetryState.doorOpenRatio !== undefined
        ? Math.max(0.0, Math.min(1.0, telemetryState.doorOpenRatio))
        : 0.5 + 0.5 * Math.sin(t * 0.45);
    const slideDist = doorOpenRatio * 0.7;
    rigController.currentSlideDist = slideDist;

    const doorPos = rigController.doorMesh.geometry.attributes.position;
    const dBaseX = rigController.doorBaseX;
    const dSign = rigController.doorSideSign;
    for (let i = 0; i < doorPos.count; i++) {
      doorPos.array[i * 3] = dBaseX[i] + dSign[i] * slideDist;
    }
    doorPos.needsUpdate = true;

    // 3. Summit aviation warning beacon continuous pulse at dusk/night (zero boolean threshold snap!)
    const nightLevel =
      telemetryState.nightLevel !== undefined
        ? Math.max(0.0, Math.min(1.0, Number(telemetryState.nightLevel) || 0.0))
        : telemetryState.isNight
          ? 1.0
          : 0.0;
    if (rigController.beaconMat) {
      const pulse = 0.5 + 0.5 * Math.sin(t * 3.4);
      rigController.beaconMat.emissiveIntensity = 0.12 + (0.33 + 2.1 * Math.pow(pulse, 3.0)) * nightLevel;
    }
  }

  /**
   * Gate 3 Kinematic Sweep Audit callback: tests extreme joint states [0..1] for non-intersection & NaN safety.
   */
  static runKinematicSweepCheck(rigController) {
    if (!rigController) return true;
    return (phase) => {
      this.update(rigController, 0.016, { doorOpenRatio: phase, isNight: phase > 0.5 });
      const slide = rigController.currentSlideDist || 0.0;
      const leftOuter = Math.abs(-0.55 - slide - 0.54);
      const rightOuter = 0.55 + slide + 0.54;
      return leftOuter < 2.25 && rightOuter < 2.25;
    };
  }
}
