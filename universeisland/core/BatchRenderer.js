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
    //    Preserves shared vertex deduplication (Uint16/Uint32 index buffer) instead of 6x toNonIndexed() bloat!
    meshesByMaterial.forEach((meshList, mat) => {
      if (meshList.length <= 1) return;

      const posChunks = [];
      const normChunks = [];
      const uvChunks = [];
      const colorChunks = [];
      const indexChunks = [];
      const vertCounts = [];
      let hasAnyColor = false;
      let totalVerts = 0;
      let totalIndices = 0;

      meshList.forEach((m) => {
        if (m.geometry.attributes.color) hasAnyColor = true;
      });

      meshList.forEach((m) => {
        const g = m.geometry.clone();
        const pos = g.attributes.position;
        if (!pos || pos.count === 0) return;
        if (!g.attributes.normal) g.computeVertexNormals();
        g.applyMatrix4(m.matrixWorld);

        const vCount = pos.count;
        posChunks.push(g.attributes.position.array);
        normChunks.push(g.attributes.normal.array);
        uvChunks.push(g.attributes.uv ? g.attributes.uv.array : new Float32Array(vCount * 2));

        if (hasAnyColor) {
          if (g.attributes.color) {
            colorChunks.push(g.attributes.color.array);
          } else {
            const white = new Float32Array(vCount * 3);
            white.fill(1.0);
            colorChunks.push(white);
          }
        }

        const idxAttr = g.index;
        indexChunks.push(idxAttr ? idxAttr.array : null);
        vertCounts.push(vCount);
        totalVerts += vCount;
        totalIndices += idxAttr ? idxAttr.count : vCount;
        meshesToRemove.push(m);
      });

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
      for (let i = 0; i < posChunks.length; i++) {
        const vCount = vertCounts[i];
        mergedPos.set(posChunks[i], pOff);
        mergedNorm.set(normChunks[i], pOff);
        mergedUv.set(uvChunks[i], uOff);
        if (mergedColor) mergedColor.set(colorChunks[i], pOff);

        const srcIdx = indexChunks[i];
        if (srcIdx) {
          const iLen = srcIdx.length;
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

        pOff += posChunks[i].length;
        uOff += uvChunks[i].length;
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
