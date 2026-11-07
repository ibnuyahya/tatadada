import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./core/BatchRenderer.js
 *
 * NexusBatchOptimizer — Single-Pass Hierarchy Merger (<= 25 Draw Calls, 120 FPS Ultra — Hukum Emas 12).
 * Merges all static LineSegments into 1 Draw Call and all static Meshes sharing a Material
 * into 1 BufferGeometry per Material while leaving dynamicKinematic and InstancedMesh nodes intact.
 */
export class NexusBatchOptimizer {
  static optimizeHierarchy(rootGroup) {
    // Compute transforms strictly relative to rootGroup so moving rootGroup later works cleanly (RTC safe)
    const savedPos = rootGroup.position.clone();
    const savedQuat = rootGroup.quaternion.clone();
    const savedScale = rootGroup.scale.clone();

    rootGroup.position.set(0, 0, 0);
    rootGroup.quaternion.identity();
    rootGroup.scale.set(1, 1, 1);
    rootGroup.updateMatrixWorld(true);

    const meshesByMaterial = new Map();
    const edgeLines = [];
    const meshesToRemove = [];
    const linesToRemove = [];

    rootGroup.traverse((obj) => {
      if (obj === rootGroup) return;
      if (!obj.visible || obj.userData.dynamicKinematic || obj.isInstancedMesh || obj.isLOD) return;

      let p = obj.parent;
      while (p && p !== rootGroup) {
        if (!p.visible || p.userData.dynamicKinematic || p.isLOD) return;
        p = p.parent;
      }

      if (obj.isLineSegments && obj.geometry) {
        edgeLines.push(obj);
        linesToRemove.push(obj);
      } else if (obj.isMesh && obj.geometry && obj.material && !Array.isArray(obj.material)) {
        if (!meshesByMaterial.has(obj.material)) {
          meshesByMaterial.set(obj.material, []);
        }
        meshesByMaterial.get(obj.material).push(obj);
      }
    });

    // 1. Merge all static LineSegments into 1 single Draw Call
    if (edgeLines.length > 1) {
      let totalFloats = 0;
      const chunks = [];
      const sharedLineMat = edgeLines[0].material;

      edgeLines.forEach((line) => {
        const g = line.geometry.index ? line.geometry.toNonIndexed() : line.geometry.clone();
        g.applyMatrix4(line.matrixWorld);
        if (g.attributes.position) {
          chunks.push(g.attributes.position.array);
          totalFloats += g.attributes.position.array.length;
        }
      });

      if (totalFloats > 0) {
        const mergedPos = new Float32Array(totalFloats);
        let offset = 0;
        chunks.forEach((arr) => {
          mergedPos.set(arr, offset);
          offset += arr.length;
        });
        const lineGeo = new THREE.BufferGeometry();
        lineGeo.setAttribute('position', new THREE.BufferAttribute(mergedPos, 3));
        const batchedLines = new THREE.LineSegments(lineGeo, sharedLineMat);
        batchedLines.name = 'NexusBatchedLines';
        rootGroup.add(batchedLines);
        linesToRemove.forEach((l) => l.parent && l.parent.remove(l));
      }
    }

    // 2. Merge all static Meshes sharing the same Material into 1 Indexed BufferGeometry per Material
    //    Zero-clone direct TypedArray pass (identity matrixWorld fast-path + inline Matrix3 normal transform)
    const normalMat = new THREE.Matrix3();
    meshesByMaterial.forEach((meshList, mat) => {
      if (meshList.length <= 1) return;

      let hasAnyColor = false;
      let totalVerts = 0;
      let totalIndices = 0;
      const validMeshes = [];

      for (let mi = 0, mLen = meshList.length; mi < mLen; mi++) {
        const m = meshList[mi];
        const g = m.geometry;
        const pos = g && g.attributes.position;
        if (!pos || pos.count === 0) continue;
        if (!g.attributes.normal) g.computeVertexNormals();
        if (g.attributes.color) hasAnyColor = true;
        const vCount = pos.count;
        const idxAttr = g.index;
        totalVerts += vCount;
        totalIndices += idxAttr ? idxAttr.count : vCount;
        validMeshes.push(m);
        meshesToRemove.push(m);
      }

      if (totalVerts === 0 || totalIndices === 0) return;

      const mergedPos = new Float32Array(totalVerts * 3);
      const mergedNorm = new Float32Array(totalVerts * 3);
      const mergedUv = new Float32Array(totalVerts * 2);
      const mergedColor = hasAnyColor ? new Float32Array(totalVerts * 3) : null;
      const mergedIndex =
        totalVerts > 65535 ? new Uint32Array(totalIndices) : new Uint16Array(totalIndices);

      let pOff = 0;
      let uOff = 0;
      let vOff = 0;
      let iOff = 0;

      for (let i = 0, mLen = validMeshes.length; i < mLen; i++) {
        const m = validMeshes[i];
        const g = m.geometry;
        const srcPos = g.attributes.position.array;
        const srcNorm = g.attributes.normal.array;
        const vCount = g.attributes.position.count;
        const pLen = vCount * 3;
        const uLen = vCount * 2;

        const e = m.matrixWorld.elements;
        const m0 = e[0], m1 = e[1], m2 = e[2];
        const m4 = e[4], m5 = e[5], m6 = e[6];
        const m8 = e[8], m9 = e[9], m10 = e[10];
        const m12 = e[12], m13 = e[13], m14 = e[14];
        const isIdentity =
          m0 === 1 && m5 === 1 && m10 === 1 && e[15] === 1 &&
          m1 === 0 && m2 === 0 && m4 === 0 && m6 === 0 &&
          m8 === 0 && m9 === 0 && m12 === 0 && m13 === 0 && m14 === 0;

        if (isIdentity) {
          mergedPos.set(srcPos.length === pLen ? srcPos : srcPos.subarray(0, pLen), pOff);
          mergedNorm.set(srcNorm.length === pLen ? srcNorm : srcNorm.subarray(0, pLen), pOff);
        } else {
          normalMat.getNormalMatrix(m.matrixWorld);
          const ne = normalMat.elements;
          const n0 = ne[0], n1 = ne[1], n2 = ne[2];
          const n3 = ne[3], n4 = ne[4], n5 = ne[5];
          const n6 = ne[6], n7 = ne[7], n8 = ne[8];
          for (let k = 0; k < pLen; k += 3) {
            const lx = srcPos[k], ly = srcPos[k + 1], lz = srcPos[k + 2];
            const dst = pOff + k;
            mergedPos[dst] = m0 * lx + m4 * ly + m8 * lz + m12;
            mergedPos[dst + 1] = m1 * lx + m5 * ly + m9 * lz + m13;
            mergedPos[dst + 2] = m2 * lx + m6 * ly + m10 * lz + m14;

            const nx = srcNorm[k], ny = srcNorm[k + 1], nz = srcNorm[k + 2];
            const tx = n0 * nx + n3 * ny + n6 * nz;
            const ty = n1 * nx + n4 * ny + n7 * nz;
            const tz = n2 * nx + n5 * ny + n8 * nz;
            const invL = 1.0 / (Math.sqrt(tx * tx + ty * ty + tz * tz) || 1.0);
            mergedNorm[dst] = tx * invL;
            mergedNorm[dst + 1] = ty * invL;
            mergedNorm[dst + 2] = tz * invL;
          }
        }

        if (g.attributes.uv) {
          const srcUv = g.attributes.uv.array;
          mergedUv.set(srcUv.length === uLen ? srcUv : srcUv.subarray(0, uLen), uOff);
        }

        if (mergedColor) {
          if (g.attributes.color) {
            const srcCol = g.attributes.color.array;
            mergedColor.set(srcCol.length === pLen ? srcCol : srcCol.subarray(0, pLen), pOff);
          } else {
            mergedColor.fill(1.0, pOff, pOff + pLen);
          }
        }

        const idxAttr = g.index;
        if (idxAttr) {
          const srcIdx = idxAttr.array;
          const iLen = idxAttr.count;
          for (let k = 0; k < iLen; k++) {
            mergedIndex[iOff + k] = vOff + srcIdx[k];
          }
          iOff += iLen;
        } else {
          for (let k = 0; k < vCount; k++) {
            mergedIndex[iOff + k] = vOff + k;
          }
          iOff += vCount;
        }

        pOff += pLen;
        uOff += uLen;
        vOff += vCount;
      }

      const batchedGeo = new THREE.BufferGeometry();
      batchedGeo.setAttribute('position', new THREE.BufferAttribute(mergedPos, 3));
      batchedGeo.setAttribute('normal', new THREE.BufferAttribute(mergedNorm, 3));
      batchedGeo.setAttribute('uv', new THREE.BufferAttribute(mergedUv, 2));
      if (mergedColor) {
        batchedGeo.setAttribute('color', new THREE.BufferAttribute(mergedColor, 3));
      }
      batchedGeo.setIndex(new THREE.BufferAttribute(mergedIndex, 1));
      batchedGeo.computeBoundingBox();
      batchedGeo.computeBoundingSphere();

      const batchedMesh = new THREE.Mesh(batchedGeo, mat);
      batchedMesh.name = `NexusBatched_${mat.name || 'Material'}`;
      batchedMesh.castShadow = true;
      batchedMesh.receiveShadow = true;
      rootGroup.add(batchedMesh);
    });

    meshesToRemove.forEach((m) => m.parent && m.parent.remove(m));

    // Prune empty non-kinematic intermediate groups to keep scene graph lean
    const emptyGroups = [];
    rootGroup.traverse((obj) => {
      if (
        obj !== rootGroup &&
        obj.type === 'Group' &&
        !obj.userData.dynamicKinematic &&
        obj.children.length === 0
      ) {
        emptyGroups.push(obj);
      }
    });
    emptyGroups.forEach((g) => g.parent && g.parent.remove(g));

    // Restore rootGroup transform
    rootGroup.position.copy(savedPos);
    rootGroup.quaternion.copy(savedQuat);
    rootGroup.scale.copy(savedScale);
    rootGroup.updateMatrixWorld(true);
  }
}
