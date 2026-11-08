import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0
 * ./core/BatchRenderer.js
 *
 * NexusBatchOptimizer — Single-Pass Hierarchy Merger (<= 25 Draw Calls, 120 FPS Ultra — Hukum Emas 12).
 * Merges static LineSegments into 1 Draw Call and static Meshes sharing a Material into 1 Indexed
 * BufferGeometry per Material, sanitizes NaN/Infinity values, and disposes temporary child geometries
 * to prevent VRAM leaks while keeping dynamicKinematic and InstancedMesh nodes untouched.
 */
export class NexusBatchOptimizer {
  static optimizeHierarchy(rootGroup, options = {}) {
    const { disposeSourceGeometries = true } = options;

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
    const keptGeometries = new Set();

    rootGroup.traverse((obj) => {
      if (obj === rootGroup) return;

      let blockedByParent = false;
      let p = obj.parent;
      while (p && p !== rootGroup) {
        if (!p.visible || p.userData.dynamicKinematic || p.isLOD) {
          blockedByParent = true;
          break;
        }
        p = p.parent;
      }

      if (blockedByParent || !obj.visible || obj.userData.dynamicKinematic || obj.isInstancedMesh || obj.isLOD) {
        if (obj.isMesh || obj.isInstancedMesh) {
          if (!obj.userData.noShadow) {
            obj.castShadow = true;
            obj.receiveShadow = true;
          }
        }
        if (obj.geometry) keptGeometries.add(obj.geometry);
        return;
      }

      if (obj.isLineSegments && obj.geometry) {
        edgeLines.push(obj);
        linesToRemove.push(obj);
      } else if (obj.isMesh && obj.geometry && obj.material && !Array.isArray(obj.material)) {
        if (!meshesByMaterial.has(obj.material)) {
          meshesByMaterial.set(obj.material, []);
        }
        meshesByMaterial.get(obj.material).push(obj);
      } else if (obj.geometry) {
        keptGeometries.add(obj.geometry);
      }
    });

    const geometriesToDispose = new Set();

    // 1. Merge all static LineSegments into 1 single Draw Call
    if (edgeLines.length > 1) {
      let totalFloats = 0;
      const chunks = [];
      const sharedLineMat = edgeLines[0].material;

      for (const line of edgeLines) {
        const g = line.geometry.index ? line.geometry.toNonIndexed() : line.geometry.clone();
        g.applyMatrix4(line.matrixWorld);
        if (g.attributes.position) {
          const arr = g.attributes.position.array;
          for (let i = 0; i < arr.length; i++) {
            if (!Number.isFinite(arr[i])) arr[i] = 0.0;
          }
          chunks.push(arr);
          totalFloats += arr.length;
        }
        g.dispose();
        if (!keptGeometries.has(line.geometry)) geometriesToDispose.add(line.geometry);
      }

      if (totalFloats > 0) {
        const mergedPos = new Float32Array(totalFloats);
        let offset = 0;
        for (const arr of chunks) {
          mergedPos.set(arr, offset);
          offset += arr.length;
        }
        const lineGeo = new THREE.BufferGeometry();
        lineGeo.setAttribute('position', new THREE.BufferAttribute(mergedPos, 3));
        lineGeo.computeBoundingBox();
        lineGeo.computeBoundingSphere();
        const batchedLines = new THREE.LineSegments(lineGeo, sharedLineMat);
        batchedLines.name = 'NexusBatchedLines';
        rootGroup.add(batchedLines);
        for (const l of linesToRemove) {
          if (l.parent) l.parent.remove(l);
        }
      }
    } else if (edgeLines.length === 1) {
      keptGeometries.add(edgeLines[0].geometry);
    }

    // 2. Merge all static Meshes sharing the same Material into 1 Indexed BufferGeometry per Material
    const normalMat = new THREE.Matrix3();

    meshesByMaterial.forEach((meshList, mat) => {
      if (meshList.length === 1) {
        const single = meshList[0];
        if (single.geometry && !single.geometry.attributes.normal) {
          single.geometry.computeVertexNormals();
        }
        single.castShadow = true;
        single.receiveShadow = true;
        keptGeometries.add(single.geometry);
        return;
      }

      let hasAnyColor = false;
      let totalVerts = 0;
      let totalIndices = 0;
      let anyCastShadow = false;
      let anyReceiveShadow = false;
      const validMeshes = [];

      for (let mi = 0, mLen = meshList.length; mi < mLen; mi++) {
        const m = meshList[mi];
        const g = m.geometry;
        const pos = g && g.attributes.position;
        if (!pos || pos.count === 0) continue;
        if (!g.attributes.normal) g.computeVertexNormals();
        if (g.attributes.color) hasAnyColor = true;
        if (m.castShadow) anyCastShadow = true;
        if (m.receiveShadow) anyReceiveShadow = true;

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
      const mergedIndex = totalVerts > 65535 ? new Uint32Array(totalIndices) : new Uint16Array(totalIndices);

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
          for (let k = 0; k < pLen; k++) {
            mergedPos[pOff + k] = Number.isFinite(srcPos[k]) ? srcPos[k] : 0.0;
            mergedNorm[pOff + k] = Number.isFinite(srcNorm[k]) ? srcNorm[k] : (k % 3 === 1 ? 1.0 : 0.0);
          }
        } else {
          normalMat.getNormalMatrix(m.matrixWorld);
          const ne = normalMat.elements;
          const n0 = ne[0], n1 = ne[1], n2 = ne[2];
          const n3 = ne[3], n4 = ne[4], n5 = ne[5];
          const n6 = ne[6], n7 = ne[7], n8 = ne[8];

          for (let k = 0; k < pLen; k += 3) {
            const lx = Number.isFinite(srcPos[k]) ? srcPos[k] : 0.0;
            const ly = Number.isFinite(srcPos[k + 1]) ? srcPos[k + 1] : 0.0;
            const lz = Number.isFinite(srcPos[k + 2]) ? srcPos[k + 2] : 0.0;
            const dst = pOff + k;

            const wx = m0 * lx + m4 * ly + m8 * lz + m12;
            const wy = m1 * lx + m5 * ly + m9 * lz + m13;
            const wz = m2 * lx + m6 * ly + m10 * lz + m14;
            mergedPos[dst] = Number.isFinite(wx) ? wx : 0.0;
            mergedPos[dst + 1] = Number.isFinite(wy) ? wy : 0.0;
            mergedPos[dst + 2] = Number.isFinite(wz) ? wz : 0.0;

            const nx = Number.isFinite(srcNorm[k]) ? srcNorm[k] : 0.0;
            const ny = Number.isFinite(srcNorm[k + 1]) ? srcNorm[k + 1] : 1.0;
            const nz = Number.isFinite(srcNorm[k + 2]) ? srcNorm[k + 2] : 0.0;
            const tx = n0 * nx + n3 * ny + n6 * nz;
            const ty = n1 * nx + n4 * ny + n7 * nz;
            const tz = n2 * nx + n5 * ny + n8 * nz;
            const len = Math.hypot(tx, ty, tz);
            if (len > 1e-6 && Number.isFinite(len)) {
              mergedNorm[dst] = tx / len;
              mergedNorm[dst + 1] = ty / len;
              mergedNorm[dst + 2] = tz / len;
            } else {
              mergedNorm[dst] = 0.0;
              mergedNorm[dst + 1] = 1.0;
              mergedNorm[dst + 2] = 0.0;
            }
          }
        }

        if (g.attributes.uv) {
          const srcUv = g.attributes.uv.array;
          for (let k = 0; k < uLen; k++) {
            mergedUv[uOff + k] = Number.isFinite(srcUv[k]) ? srcUv[k] : 0.0;
          }
        }

        if (mergedColor) {
          if (g.attributes.color) {
            const srcCol = g.attributes.color.array;
            const itemSize = g.attributes.color.itemSize || 3;
            for (let v = 0; v < vCount; v++) {
              mergedColor[pOff + v * 3] = Number.isFinite(srcCol[v * itemSize]) ? srcCol[v * itemSize] : 1.0;
              mergedColor[pOff + v * 3 + 1] = Number.isFinite(srcCol[v * itemSize + 1]) ? srcCol[v * itemSize + 1] : 1.0;
              mergedColor[pOff + v * 3 + 2] = Number.isFinite(srcCol[v * itemSize + 2]) ? srcCol[v * itemSize + 2] : 1.0;
            }
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

        if (!keptGeometries.has(g)) {
          geometriesToDispose.add(g);
        }
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

    for (const m of meshesToRemove) {
      if (m.parent) m.parent.remove(m);
    }

    if (disposeSourceGeometries) {
      for (const g of geometriesToDispose) {
        if (!keptGeometries.has(g)) {
          try { g.dispose(); } catch {}
        }
      }
    }

    // Prune empty non-kinematic intermediate groups
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
    for (const g of emptyGroups) {
      if (g.parent) g.parent.remove(g);
    }

    rootGroup.position.copy(savedPos);
    rootGroup.quaternion.copy(savedQuat);
    rootGroup.scale.copy(savedScale);
    rootGroup.updateMatrixWorld(true);
  }
}
