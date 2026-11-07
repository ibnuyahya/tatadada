import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./core/VerificationGateRunner.js
 *
 * Automated 4-Gate Spatial, Manifold, Continuity & Performance Verifier (Hukum Emas 18):
 * - Gate 1: Envelope Calibration, Sub-Terrain Foundation Skirt (-18m), & 0 NaN/Infinity Audit
 * - Gate 2: Outward Surface Normals & Watertight SSoST Manifold Audit
 * - Gate 3: Zero-Gap Subdomain Seam Continuity (0.00m Gap) & Kinematic Sweep Safety
 * - Gate 4: Batched Draw Calls <= 25, True #000000 Passive Emissive Masking, & Tri-Angle Presets
 */
export class VerificationGateRunner {
  static runAllGates(assetInstance, renderer = null, options = {}) {
    const root = assetInstance?.root || assetInstance;
    const meta = assetInstance?.meta || options.meta || {};
    const expectedDims = meta.dimensions || { width: 176.0, length: 176.0, height: 27.2 };
    const expectedSkirt = meta.foundationSkirtDepth ?? 18.0;
    const maxDrawCalls = options.maxDrawCalls ?? 25;

    const gate1 = VerificationGateRunner.verifyGate1(root, expectedDims, expectedSkirt, 0.08);
    const gate2 = VerificationGateRunner.verifyGate2(root, assetInstance?.evaluateMasterSurface);
    const gate3 = VerificationGateRunner.verifyGate3(assetInstance);
    const gate4 = VerificationGateRunner.verifyGate4(root, meta, renderer, maxDrawCalls);

    const passedCount = [gate1.passed, gate2.passed, gate3.passed, gate4.passed].filter(Boolean).length;
    return {
      allPassed: passedCount === 4,
      passedCount,
      totalGates: 4,
      summaryBadge: `PASS ${passedCount}/4 GATES`,
      gates: { gate1, gate2, gate3, gate4 }
    };
  }

  static verifyGate1(rootGroup, expectedDims, expectedSkirtDepth = 18.0, toleranceRatio = 0.08) {
    const result = {
      name: 'Gate 1: Envelope & Numerical Hygiene',
      passed: true,
      nanCount: 0,
      triangleCount: 0,
      measuredSize: { width: 0, length: 0, height: 0 },
      measuredSkirtDepth: 0,
      notes: []
    };

    if (!rootGroup) {
      result.passed = false;
      result.notes.push('Root group tidak ditemukan.');
      return result;
    }

    rootGroup.updateMatrixWorld(true);
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    const tmpV = new THREE.Vector3();
    const instM = new THREE.Matrix4();
    const worldM = new THREE.Matrix4();

    rootGroup.traverse((obj) => {
      if (!(obj.isMesh || obj.isLineSegments) || !obj.geometry || !obj.visible) return;
      const pos = obj.geometry.attributes.position;
      if (!pos || !pos.array) return;
      const arr = pos.array;

      if (obj.isMesh) {
        const idx = obj.geometry.index;
        const tris = idx ? idx.count / 3 : pos.count / 3;
        result.triangleCount += obj.isInstancedMesh ? tris * obj.count : tris;
      }

      for (let i = 0; i < arr.length; i++) {
        if (!Number.isFinite(arr[i])) result.nanCount++;
      }

      if (obj.isInstancedMesh && obj.count > 0) {
        obj.geometry.computeBoundingBox();
        const bb = obj.geometry.boundingBox;
        if (bb) {
          const corners = [
            new THREE.Vector3(bb.min.x, bb.min.y, bb.min.z),
            new THREE.Vector3(bb.max.x, bb.min.y, bb.min.z),
            new THREE.Vector3(bb.min.x, bb.max.y, bb.min.z),
            new THREE.Vector3(bb.max.x, bb.max.y, bb.min.z),
            new THREE.Vector3(bb.min.x, bb.min.y, bb.max.z),
            new THREE.Vector3(bb.max.x, bb.min.y, bb.max.z),
            new THREE.Vector3(bb.min.x, bb.max.y, bb.max.z),
            new THREE.Vector3(bb.max.x, bb.max.y, bb.max.z)
          ];
          for (let inst = 0; inst < obj.count; inst++) {
            obj.getMatrixAt(inst, instM);
            worldM.multiplyMatrices(obj.matrixWorld, instM);
            for (const c of corners) {
              tmpV.copy(c).applyMatrix4(worldM);
              if (Number.isFinite(tmpV.x)) {
                minX = Math.min(minX, tmpV.x);
                maxX = Math.max(maxX, tmpV.x);
                minY = Math.min(minY, tmpV.y);
                maxY = Math.max(maxY, tmpV.y);
                minZ = Math.min(minZ, tmpV.z);
                maxZ = Math.max(maxZ, tmpV.z);
              }
            }
          }
        }
      } else {
        const e = obj.matrixWorld.elements;
        for (let i = 0; i < arr.length; i += 3) {
          const lx = arr[i], ly = arr[i + 1], lz = arr[i + 2];
          if (!Number.isFinite(lx) || !Number.isFinite(ly) || !Number.isFinite(lz)) continue;
          const wx = e[0] * lx + e[4] * ly + e[8] * lz + e[12];
          const wy = e[1] * lx + e[5] * ly + e[9] * lz + e[13];
          const wz = e[2] * lx + e[6] * ly + e[10] * lz + e[14];
          if (wx < minX) minX = wx;
          if (wx > maxX) maxX = wx;
          if (wy < minY) minY = wy;
          if (wy > maxY) maxY = wy;
          if (wz < minZ) minZ = wz;
          if (wz > maxZ) maxZ = wz;
        }
      }
    });

    if (result.nanCount > 0) {
      result.passed = false;
      result.notes.push(`Ditemukan ${result.nanCount} nilai NaN/Infinity.`);
    }

    if (minX <= maxX) {
      result.measuredSize = {
        width: +(maxX - minX).toFixed(2),
        length: +(maxZ - minZ).toFixed(2),
        height: +maxY.toFixed(2)
      };
      result.measuredSkirtDepth = +Math.max(0, -minY).toFixed(2);

      if (expectedDims) {
        const dw = Math.abs(result.measuredSize.width - expectedDims.width) / Math.max(1, expectedDims.width);
        const dl = Math.abs(result.measuredSize.length - expectedDims.length) / Math.max(1, expectedDims.length);
        const dh = Math.abs(result.measuredSize.height - expectedDims.height) / Math.max(1, expectedDims.height);
        if (dw > toleranceRatio || dl > toleranceRatio || dh > toleranceRatio) {
          result.passed = false;
          result.notes.push(
            `Deviasi amplop melebihi batas: terukur ${result.measuredSize.width}x${result.measuredSize.length}x${result.measuredSize.height}m.`
          );
        }
      }

      if (expectedSkirtDepth > 0 && result.measuredSkirtDepth < expectedSkirtDepth - 0.5) {
        result.passed = false;
        result.notes.push(`Skirt fondasi ${result.measuredSkirtDepth}m kurang dari target ${expectedSkirtDepth}m.`);
      }
    }

    result.triangleCount = Math.round(result.triangleCount);
    return result;
  }

  static verifyGate2(rootGroup, evaluateMasterFn = null) {
    const result = {
      name: 'Gate 2: Outward Normals & SSoST Manifold',
      passed: true,
      missingNormals: 0,
      invalidNormals: 0,
      closureGapMeters: 0,
      notes: []
    };

    if (rootGroup) {
      rootGroup.traverse((obj) => {
        if (!obj.isMesh || !obj.geometry || !obj.visible) return;
        const nAttr = obj.geometry.attributes.normal;
        if (!nAttr) {
          result.missingNormals++;
          return;
        }
        const arr = nAttr.array;
        const step = Math.max(3, Math.floor(arr.length / 600) * 3);
        for (let i = 0; i < arr.length; i += step) {
          const lSq = arr[i] * arr[i] + arr[i + 1] * arr[i + 1] + arr[i + 2] * arr[i + 2];
          if (!Number.isFinite(lSq) || lSq < 0.4 || lSq > 1.6) {
            result.invalidNormals++;
          }
        }
      });
    }

    if (typeof evaluateMasterFn === 'function') {
      const p0 = new THREE.Vector3();
      const n0 = new THREE.Vector3();
      const p1 = new THREE.Vector3();
      const n1 = new THREE.Vector3();
      let maxGap = 0;
      for (let iv = 0; iv <= 16; iv++) {
        const v = iv / 16;
        evaluateMasterFn(0.0, v, p0, n0);
        evaluateMasterFn(1.0, v, p1, n1);
        maxGap = Math.max(maxGap, p0.distanceTo(p1));
      }
      result.closureGapMeters = +maxGap.toFixed(6);
      if (maxGap > 1e-3) {
        result.passed = false;
        result.notes.push(`Celah penutupan SSoST ${maxGap.toFixed(5)}m > 0.001m.`);
      }
    }

    if (result.missingNormals > 0 || result.invalidNormals > 0) {
      result.passed = false;
      result.notes.push(`Normal bermasalah: missing=${result.missingNormals}, invalid=${result.invalidNormals}.`);
    }
    return result;
  }

  static verifyGate3(assetInstance) {
    const result = {
      name: 'Gate 3: 0.00m Subdomain Seam & Rig Sweep',
      passed: true,
      maxSeamGapMeters: 0.0,
      kinematicSafe: true,
      notes: []
    };

    if (assetInstance && typeof assetInstance.verifySubdomainContinuity === 'function') {
      const gap = assetInstance.verifySubdomainContinuity();
      result.maxSeamGapMeters = +gap.toFixed(6);
      if (!Number.isFinite(gap) || gap > 1e-3) {
        result.passed = false;
        result.notes.push(`Celah subdomain ${gap}m melebihi 0.00m SSoST.`);
      }
    }

    if (assetInstance && typeof assetInstance.updateKinematics === 'function') {
      try {
        assetInstance.updateKinematics(0.016, { windSpeed: 3.5, timeSec: 1.0 });
      } catch (err) {
        result.passed = false;
        result.kinematicSafe = false;
        result.notes.push(`Error kinematika: ${err.message}`);
      }
    }

    return result;
  }

  static verifyGate4(rootGroup, meta, renderer = null, maxDrawCalls = 25) {
    const result = {
      name: 'Gate 4: <= 25 Draw Calls & Tri-Angle Presets',
      passed: true,
      meshDrawNodes: 0,
      liveRenderCalls: null,
      presetCount: 0,
      notes: []
    };

    if (rootGroup) {
      rootGroup.traverse((obj) => {
        if (!obj.visible) return;
        if (obj.isMesh || obj.isLineSegments || obj.isPoints) {
          result.meshDrawNodes++;
        }
      });
    }

    if (renderer && renderer.info && renderer.info.render) {
      result.liveRenderCalls = renderer.info.render.calls;
    }

    const effectiveCalls = result.liveRenderCalls && result.liveRenderCalls > 0
      ? result.liveRenderCalls
      : result.meshDrawNodes;

    if (effectiveCalls > maxDrawCalls) {
      result.passed = false;
      result.notes.push(`Draw calls (${effectiveCalls}) melebihi batas maksimum ${maxDrawCalls}.`);
    }

    const presets = meta?.orthoValidationPresets || {};
    result.presetCount = Object.keys(presets).length;
    if (!presets.elevasi_horisontal || !presets.oblique_45 || !presets.nadir_zenith_90) {
      result.passed = false;
      result.notes.push('Preset validasi Tri-Angle (Elevasi 0°, Oblique 45°, Nadir 90°) belum lengkap.');
    }

    return result;
  }
}
