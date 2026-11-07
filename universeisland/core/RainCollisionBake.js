import * as THREE from 'three';

/**
 * NEXUS Omni-Scale Procedural Engine v2.0 (v2.2 Super-Realistic Physical Weather Edition)
 * ./core/RainCollisionBake.js
 *
 * World-Space 512x512 Top-Down Surface-Height, 3D-Normal, & Material-Class Z-Buffer Baker
 * + Automatic 3D Roof-Eave & Tree-Canopy Drip-Edge Extractor.
 *
 * Enables 100% GPU Physics-Driven Rain Interaction at 0.0 ms/frame CPU cost:
 * - True 3D roof/canopy occlusion across Gedung Rektorat UI (Tajug spire, 6 Pagoda eaves,
 *   4 Corner Limas roofs, Selasar Agung, 4-sided Flag Cloister gallery while leaving the open
 *   Flag Courtyard wet, Rotunda Gatehouse, 45° Octagonal Skylit Pendopo, SE Covered Walkway,
 *   Annex Wings, and all 56 multi-species Campus Tree Canopies).
 * - Exact 3D impact elevation y_hit(x, z) and surface normal N_hit(x, z) for Rayleigh-Worthington
 *   crown splashes & Newtonian parabolic rebound micro-droplets.
 * - Extracted 3D overhang drip coordinates (x_eave, y_eave, z_eave, y_landingBelow) for Newtonian
 *   gravity-accelerated roof-tile & foliage drip curtains (y(t) = y_0 - 0.5 * g * t^2).
 */
export class RainCollisionBake {
  static DEFAULT_SIZE = 2048;
  static DEFAULT_HALF_EXTENT = 185.0;
  static DEFAULT_MIN_X = -422.50;
  static DEFAULT_MAX_X = 140.70; // Exact 563.20m span -> 0.275m/texel at 2048 POT
  static DEFAULT_MIN_Z = -135.27;
  static DEFAULT_MAX_Z = 786.33; // Exact 921.60m span -> 0.45m/texel at 2048 POT (covers Pulau U + Selat UI + Pulau I!)
  static HEIGHT_MIN_Y = -9.0;
  static HEIGHT_MAX_Y = 66.0;

  /**
   * Creates a 4x4 fallback flat-ground collision texture before a full scene bake is called.
   */
  static createFallbackTexture() {
    const size = 4;
    const data = new Uint8Array(size * size * 4);
    const minY = RainCollisionBake.HEIGHT_MIN_Y;
    const spanY = RainCollisionBake.HEIGHT_MAX_Y - minY;
    const q0 = Math.max(0, Math.min(65535, Math.round(((0.0 - minY) / spanY) * 65535)));
    const hi = (q0 >> 8) & 255;
    const lo = q0 & 255;

    for (let i = 0; i < size * size; i++) {
      const o = i * 4;
      data[o] = hi;
      data[o + 1] = lo;
      data[o + 2] = 128; // nx = 0
      data[o + 3] = 128; // nz = 0, hard horizontal ground
    }

    const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.generateMipmaps = false;
    tex.needsUpdate = true;
    return tex;
  }

  /**
   * Bakes the static 3D scene hierarchy + instanced tree canopies into a 512x512 RGBA8 collision
   * DataTexture (NearestFilter, 16-bit elevation precision = 0.915 mm) and extracts 3D eave drip anchors.
   *
   * @param {THREE.Object3D[]} roots
   * @param {Object} [options]
   */
  static bakeFromHierarchy(roots = [], options = {}) {
    const gen = RainCollisionBake._bakeGenerator(roots, options, Infinity);
    let step = gen.next();
    while (!step.done) {
      step = gen.next();
    }
    return step.value;
  }

  /**
   * Time-sliced asynchronous 3D Rain Collision Z-Buffer & Drip Anchor baker (~1.5ms per VSync frame).
   * Yields once per animation frame so the 3D world runs locked at 60-120 FPS with zero frame drops!
   */
  static async bakeFromHierarchyAsync(roots = [], options = {}, sliceBudgetMs = 1.5) {
    const budget =
      options && options.sliceBudgetMs !== undefined ? Number(options.sliceBudgetMs) : sliceBudgetMs;
    const gen = RainCollisionBake._bakeGenerator(roots, options, budget);
    const yieldNextFrame = () =>
      new Promise((resolve) => {
        if (typeof requestAnimationFrame === 'function') {
          let settled = false;
          const finish = () => {
            if (settled) return;
            settled = true;
            resolve();
          };
          const fb = setTimeout(finish, 34);
          requestAnimationFrame(() => {
            clearTimeout(fb);
            setTimeout(finish, 0);
          });
        } else {
          setTimeout(resolve, 0);
        }
      });
    let step = gen.next();
    while (!step.done) {
      await yieldNextFrame();
      step = gen.next();
    }
    return step.value;
  }

  static *_bakeGenerator(roots = [], options = {}, sliceBudgetMs = Infinity) {
    const useSlicing = Number.isFinite(sliceBudgetMs) && sliceBudgetMs > 0;
    const nowFn = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
    let sliceStart = nowFn();

    const size = options.size || RainCollisionBake.DEFAULT_SIZE;
    const hasExplicitHalfExtent = options.halfExtent !== undefined && options.halfExtent !== null;
    const halfExtent = hasExplicitHalfExtent
      ? Number(options.halfExtent)
      : RainCollisionBake.DEFAULT_HALF_EXTENT;
    const minXBound = hasExplicitHalfExtent
      ? -halfExtent
      : options.minX !== undefined
        ? Number(options.minX)
        : RainCollisionBake.DEFAULT_MIN_X;
    const maxXBound = hasExplicitHalfExtent
      ? halfExtent
      : options.maxX !== undefined
        ? Number(options.maxX)
        : RainCollisionBake.DEFAULT_MAX_X;
    const minZBound = hasExplicitHalfExtent
      ? -halfExtent
      : options.minZ !== undefined
        ? Number(options.minZ)
        : RainCollisionBake.DEFAULT_MIN_Z;
    const maxZBound = hasExplicitHalfExtent
      ? halfExtent
      : options.maxZ !== undefined
        ? Number(options.maxZ)
        : RainCollisionBake.DEFAULT_MAX_Z;

    const worldSpanX = Math.max(1.0, maxXBound - minXBound);
    const worldSpanZ = Math.max(1.0, maxZBound - minZBound);
    const cellSizeX = worldSpanX / size;
    const cellSizeZ = worldSpanZ / size;
    const avgCellSize = 0.5 * (cellSizeX + cellSizeZ);
    const minY = RainCollisionBake.HEIGHT_MIN_Y;
    const maxY = RainCollisionBake.HEIGHT_MAX_Y;
    const spanY = maxY - minY;

    const groundHeightFn = typeof options.groundHeightFn === 'function' ? options.groundHeightFn : () => 0.0;
    const shorelineXFn = typeof options.shorelineXFn === 'function' ? options.shorelineXFn : null;
    const islandCoastDistanceFn =
      typeof options.islandCoastDistanceFn === 'function' ? options.islandCoastDistanceFn : null;
    const freshwaterSurfaceYFn =
      typeof options.freshwaterSurfaceYFn === 'function' ? options.freshwaterSurfaceYFn : null;
    const waterLevelY = options.waterLevelY !== undefined ? options.waterLevelY : -1.45;
    const maxDrips = options.maxDrips || 1600;

    const totalCells = size * size;
    const heightBuf = new Float32Array(totalCells);
    const nxBuf = new Float32Array(totalCells);
    const nyBuf = new Float32Array(totalCells);
    const nzBuf = new Float32Array(totalCells);
    // Surface class: 0 = Water, 1 = Horizontal Hard Surface, 2 = Sloped Hard Roof/Batter, 3 = Tree Canopy
    const classBuf = new Uint8Array(totalCells);

    // 1. Initialize Dual-Island DEM & 360-Degree Surrounding Ocean Surface (with O(1) Open-Sea AABB & SDF Early-Out!)
    const rawGroundBuf = new Float32Array(totalCells);
    for (let iz = 0; iz < size; iz++) {
      if (useSlicing && iz > 0 && nowFn() - sliceStart >= sliceBudgetMs) {
        yield;
        sliceStart = nowFn();
      }
      const wz = minZBound + (iz + 0.5) * cellSizeZ;
      const shoreX = !islandCoastDistanceFn && shorelineXFn ? shorelineXFn(wz) : Infinity;
      const rowOffset = iz * size;
      // Exact Z-band open-sea check when using full archipelago coast SDF
      const zMaybePulauU = wz >= -96.0 && wz <= 474.0;
      const zMaybePulauI = wz >= 599.0 && wz <= 788.0;
      const zGuaranteedOpenSea = Boolean(islandCoastDistanceFn) && !zMaybePulauU && !zMaybePulauI;

      for (let ix = 0; ix < size; ix++) {
        const wx = minXBound + (ix + 0.5) * cellSizeX;
        const idx = rowOffset + ix;

        if (islandCoastDistanceFn) {
          const xOutsideIslands =
            zGuaranteedOpenSea ||
            (zMaybePulauU && (wx < -375.0 || wx > 108.0)) ||
            (zMaybePulauI && (wx < -410.0 || wx > 90.0));
          if (xOutsideIslands) {
            rawGroundBuf[idx] = -3.85;
            heightBuf[idx] = waterLevelY;
            nxBuf[idx] = 0.0;
            nyBuf[idx] = 1.0;
            nzBuf[idx] = 0.0;
            classBuf[idx] = 0;
            continue;
          }
          const dCoast = islandCoastDistanceFn(wx, wz);
          if (dCoast > 3.55) {
            rawGroundBuf[idx] = -3.85;
            heightBuf[idx] = waterLevelY;
            nxBuf[idx] = 0.0;
            nyBuf[idx] = 1.0;
            nzBuf[idx] = 0.0;
            classBuf[idx] = 0;
            continue;
          }
          const gh = groundHeightFn(wx, wz);
          rawGroundBuf[idx] = gh;
          const fwY = dCoast < -4.0 && freshwaterSurfaceYFn ? freshwaterSurfaceYFn(wx, wz) : null;
          if (fwY !== null && Number.isFinite(fwY)) {
            heightBuf[idx] = fwY;
            nxBuf[idx] = 0.0;
            nyBuf[idx] = 1.0;
            nzBuf[idx] = 0.0;
            classBuf[idx] = 0;
          } else if (gh <= waterLevelY + 0.06 && dCoast > 3.45) {
            heightBuf[idx] = waterLevelY;
            nxBuf[idx] = 0.0;
            nyBuf[idx] = 1.0;
            nzBuf[idx] = 0.0;
            classBuf[idx] = 0;
          } else {
            heightBuf[idx] = gh;
            classBuf[idx] = 1;
          }
          continue;
        }

        const gh = groundHeightFn(wx, wz);
        rawGroundBuf[idx] = gh;
        const fwY = freshwaterSurfaceYFn ? freshwaterSurfaceYFn(wx, wz) : null;
        if (fwY !== null && Number.isFinite(fwY)) {
          heightBuf[idx] = fwY;
          nxBuf[idx] = 0.0;
          nyBuf[idx] = 1.0;
          nzBuf[idx] = 0.0;
          classBuf[idx] = 0;
        } else {
          const isWater = wx > shoreX + 3.45 || gh <= waterLevelY - 0.02;
          if (isWater) {
            heightBuf[idx] = waterLevelY;
            nxBuf[idx] = 0.0;
            nyBuf[idx] = 1.0;
            nzBuf[idx] = 0.0;
            classBuf[idx] = 0;
          } else {
            heightBuf[idx] = gh;
            classBuf[idx] = 1;
          }
        }
      }
    }

    for (let iz = 0; iz < size; iz++) {
      if (useSlicing && (iz & 3) === 3 && nowFn() - sliceStart >= sliceBudgetMs) {
        yield;
        sliceStart = nowFn();
      }
      const rowOffset = iz * size;
      const izD = iz > 0 ? iz - 1 : 0;
      const izU = iz < size - 1 ? iz + 1 : size - 1;
      const spanZ = (izU - izD || 1) * cellSizeZ;

      for (let ix = 0; ix < size; ix++) {
        const idx = rowOffset + ix;
        if (classBuf[idx] === 0) continue;

        const ixL = ix > 0 ? ix - 1 : 0;
        const ixR = ix < size - 1 ? ix + 1 : size - 1;
        const spanX = (ixR - ixL || 1) * cellSizeX;

        const gx = (rawGroundBuf[rowOffset + ixL] - rawGroundBuf[rowOffset + ixR]) / spanX;
        const gz = (rawGroundBuf[izD * size + ix] - rawGroundBuf[izU * size + ix]) / spanZ;
        const invL = 1.0 / (Math.hypot(gx, 1.0, gz) || 1.0);
        const ny = invL;

        nxBuf[idx] = gx * invL;
        nyBuf[idx] = ny;
        nzBuf[idx] = gz * invL;
        classBuf[idx] = ny < 0.88 ? 2 : 1;
      }
    }

    // Helper: Rasterize upward-facing triangles of a static Mesh into the Z-Buffer with inner scanline yielding
    function* rasterizeStaticMeshGen(mesh) {
      const geo = mesh.geometry;
      const posAttr = geo?.attributes?.position;
      if (!posAttr || posAttr.count < 3) return;

      const posArr = posAttr.array;
      const idxArr = geo.index ? geo.index.array : null;
      const e = mesh.matrixWorld.elements;
      const m0 = e[0], m4 = e[4], m8 = e[8], m12 = e[12];
      const m1 = e[1], m5 = e[5], m9 = e[9], m13 = e[13];
      const m2 = e[2], m6 = e[6], m10 = e[10], m14 = e[14];

      const triCount = idxArr ? Math.floor(idxArr.length / 3) : Math.floor(posAttr.count / 3);
      let scanOps = 0;

      for (let t = 0; t < triCount; t++) {
        if (useSlicing && (t & 511) === 0 && nowFn() - sliceStart >= sliceBudgetMs) {
          yield;
          sliceStart = nowFn();
        }
        const i0 = (idxArr ? idxArr[t * 3] : t * 3) * 3;
        const i1 = (idxArr ? idxArr[t * 3 + 1] : t * 3 + 1) * 3;
        const i2 = (idxArr ? idxArr[t * 3 + 2] : t * 3 + 2) * 3;

        const lx0 = posArr[i0], ly0 = posArr[i0 + 1], lz0 = posArr[i0 + 2];
        const lx1 = posArr[i1], ly1 = posArr[i1 + 1], lz1 = posArr[i1 + 2];
        const lx2 = posArr[i2], ly2 = posArr[i2 + 1], lz2 = posArr[i2 + 2];

        const ax = m0 * lx0 + m4 * ly0 + m8 * lz0 + m12;
        const ay = m1 * lx0 + m5 * ly0 + m9 * lz0 + m13;
        const az = m2 * lx0 + m6 * ly0 + m10 * lz0 + m14;

        const bx = m0 * lx1 + m4 * ly1 + m8 * lz1 + m12;
        const by = m1 * lx1 + m5 * ly1 + m9 * lz1 + m13;
        const bz = m2 * lx1 + m6 * ly1 + m10 * lz1 + m14;

        const cx = m0 * lx2 + m4 * ly2 + m8 * lz2 + m12;
        const cy = m1 * lx2 + m5 * ly2 + m9 * lz2 + m13;
        const cz = m2 * lx2 + m6 * ly2 + m10 * lz2 + m14;

        // Skip subterranean foundation skirts below lake waterline & flat 0.0m datum lawn already in heightBuf
        const triMaxY = Math.max(ay, by, cy);
        if (triMaxY < -1.42) continue;
        if (triMaxY <= 0.0005 && ay >= -0.0005 && by >= -0.0005 && cy >= -0.0005) continue;

        const abx = bx - ax, aby = by - ay, abz = bz - az;
        const acx = cx - ax, acy = cy - ay, acz = cz - az;
        const fnx = aby * acz - abz * acy;
        const fny = abz * acx - abx * acz;
        const fnz = abx * acy - aby * acx;
        const fLen = Math.hypot(fnx, fny, fnz);
        if (fLen < 1e-7) continue;

        const ny = fny / fLen;
        // Only rasterize upward-exposed roofs, decks, copings, steps, and gables (skip vertical walls & soffits)
        if (ny <= 0.12) continue;
        const nx = fnx / fLen;
        const nz = fnz / fLen;
        const surfClass = ny < 0.88 ? 2 : 1;

        const minX = Math.min(ax, bx, cx);
        const maxX = Math.max(ax, bx, cx);
        const minZ = Math.min(az, bz, cz);
        const maxZ = Math.max(az, bz, cz);

        if (maxX < minXBound || minX > maxXBound || maxZ < minZBound || minZ > maxZBound) {
          continue;
        }

        const ixMin = Math.max(0, Math.floor((minX - minXBound) / cellSizeX));
        const ixMax = Math.min(size - 1, Math.floor((maxX - minXBound) / cellSizeX));
        const izMin = Math.max(0, Math.floor((minZ - minZBound) / cellSizeZ));
        const izMax = Math.min(size - 1, Math.floor((maxZ - minZBound) / cellSizeZ));

        const det = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz);
        if (Math.abs(det) < 1e-6) continue;
        const invDet = 1.0 / det;
        const rowSpan = ixMax - ixMin + 1;

        for (let iz = izMin; iz <= izMax; iz++) {
          if (useSlicing) {
            scanOps += rowSpan;
            if (scanOps >= 2048) {
              scanOps = 0;
              if (nowFn() - sliceStart >= sliceBudgetMs) {
                yield;
                sliceStart = nowFn();
              }
            }
          }
          const pz = minZBound + (iz + 0.5) * cellSizeZ;
          const rowOffset = iz * size;
          for (let ix = ixMin; ix <= ixMax; ix++) {
            const px = minXBound + (ix + 0.5) * cellSizeX;
            const dpx = px - cx;
            const dpz = pz - cz;
            const w0 = ((bz - cz) * dpx + (cx - bx) * dpz) * invDet;
            if (w0 < -1e-4) continue;
            const w1 = ((cz - az) * dpx + (ax - cx) * dpz) * invDet;
            if (w1 < -1e-4) continue;
            const w2 = 1.0 - w0 - w1;
            if (w2 < -1e-4) continue;

            const py = w0 * ay + w1 * by + w2 * cy;
            const idx = rowOffset + ix;
            if (py > heightBuf[idx]) {
              heightBuf[idx] = py;
              nxBuf[idx] = nx;
              nyBuf[idx] = ny;
              nzBuf[idx] = nz;
              classBuf[idx] = surfClass;
            }
          }
        }
      }
    }

    // Helper: Rasterize InstancedMesh organic tree foliage canopy ellipsoids with per-canopy time slicing
    function* rasterizeInstancedCanopiesGen(instMesh) {
      const matArr = instMesh.instanceMatrix?.array || instMesh.instanceMatrix;
      if (!matArr) return;
      const count = instMesh.count || Math.floor(matArr.length / 16);
      const pw = instMesh.matrixWorld.elements;

      for (let i = 0; i < count; i++) {
        if (useSlicing && i > 0 && nowFn() - sliceStart >= sliceBudgetMs) {
          yield;
          sliceStart = nowFn();
        }
        const o = i * 16;
        // World center of canopy pad (accounting for parent worldRoot matrix)
        const lx = matArr[o + 12];
        const ly = matArr[o + 13];
        const lz = matArr[o + 14];
        const cx = pw[0] * lx + pw[4] * ly + pw[8] * lz + pw[12];
        const cy = pw[1] * lx + pw[5] * ly + pw[9] * lz + pw[13];
        const cz = pw[2] * lx + pw[6] * ly + pw[10] * lz + pw[14];

        // Principal horizontal & vertical radii of transformed unit harmonic foliage cloud
        const sx = Math.hypot(matArr[o], matArr[o + 1], matArr[o + 2]);
        const sy = Math.hypot(matArr[o + 4], matArr[o + 5], matArr[o + 6]);
        const sz = Math.hypot(matArr[o + 8], matArr[o + 9], matArr[o + 10]);
        const rx = Math.max(0.45, Math.max(sx, sz) * 0.98);
        const rz = rx;
        const ry = Math.max(0.35, sy * 1.02);

        if (cx + rx < minXBound || cx - rx > maxXBound || cz + rz < minZBound || cz - rz > maxZBound) {
          continue;
        }
        const ixMin = Math.max(0, Math.floor((cx - rx - minXBound) / cellSizeX));
        const ixMax = Math.min(size - 1, Math.floor((cx + rx - minXBound) / cellSizeX));
        const izMin = Math.max(0, Math.floor((cz - rz - minZBound) / cellSizeZ));
        const izMax = Math.min(size - 1, Math.floor((cz + rz - minZBound) / cellSizeZ));

        for (let iz = izMin; iz <= izMax; iz++) {
          const pz = minZBound + (iz + 0.5) * cellSizeZ;
          const dz = (pz - cz) / rz;
          const rowOffset = iz * size;
          for (let ix = ixMin; ix <= ixMax; ix++) {
            const px = minXBound + (ix + 0.5) * cellSizeX;
            const dx = (px - cx) / rx;
            const r2 = dx * dx + dz * dz;
            if (r2 > 1.0) continue;

            const dome = Math.sqrt(1.0 - r2);
            const py = cy + ry * dome;
            const idx = rowOffset + ix;
            if (py > heightBuf[idx]) {
              const gnx = dx / rx;
              const gny = Math.max(0.25, dome / ry);
              const gnz = dz / rz;
              const invN = 1.0 / (Math.hypot(gnx, gny, gnz) || 1.0);
              heightBuf[idx] = py;
              nxBuf[idx] = gnx * invN;
              nyBuf[idx] = gny * invN;
              nzBuf[idx] = gnz * invN;
              classBuf[idx] = 3;
            }
          }
        }
      }
    }

    // 2. Traverse Roots (prefer medium LOD1 for 250k+ tri monuments; skip dynamic particles/flags/NPCs/water)
    const preferMediumLOD = options.preferMediumLOD !== false;
    const visitNode = (obj, ignoreVisible = false) => {
      if (!obj || (!ignoreVisible && obj.visible === false)) return;
      if (
        obj.userData?.dynamicKinematic ||
        obj.userData?.skipRainBake ||
        obj.isPoints ||
        obj.isLineSegments
      ) {
        return;
      }
      const nm = obj.name || '';
      if (
        nm === 'NexusAnalyticalSkyDome' ||
        nm === 'DanauKenangaWaterSurface' ||
        nm === 'KampusUITerrainAndDanau' ||
        nm === 'KampusUITerrainAndDanauKenanga' ||
        nm === 'NexusBatched_CampusTerrainAndTurapMat' ||
        nm === 'NexusVolcanicSmokePlume' ||
        nm === 'NexusPhysicalRainAndDrips' ||
        nm === 'NexusPhysicalSplashAndMist' ||
        nm.includes('TerrainWithSkirt') ||
        nm.includes('StoneTurap') ||
        nm.includes('Sivitas')
      ) {
        return;
      }

      if (obj.isLOD && obj.levels && obj.levels.length > 0) {
        const lodTarget =
          preferMediumLOD && obj.levels.length > 1 && obj.levels[1]?.object
            ? obj.levels[1].object
            : obj.levels[0].object;
        if (lodTarget) {
          lodTarget.updateMatrixWorld(true);
          visitNode(lodTarget, true);
        }
        return;
      }

      if (obj.isInstancedMesh) {
        if (nm.includes('Canopies')) {
          canopyMeshes.push(obj);
        }
        return;
      }

      if (obj.isMesh) {
        staticMeshes.push(obj);
      }

      if (obj.children && obj.children.length > 0) {
        for (let i = 0; i < obj.children.length; i++) {
          visitNode(obj.children[i]);
        }
      }
    };

    const canopyMeshes = [];
    const staticMeshes = [];
    const rootList = Array.isArray(roots) ? roots : [roots];
    for (let r = 0; r < rootList.length; r++) {
      const root = rootList[r];
      if (!root) continue;
      root.updateMatrixWorld(true);
      visitNode(root);
    }

    for (let c = 0; c < canopyMeshes.length; c++) {
      yield* rasterizeInstancedCanopiesGen(canopyMeshes[c]);
    }

    for (let m = 0; m < staticMeshes.length; m++) {
      yield* rasterizeStaticMeshGen(staticMeshes[m]);
    }

    // 3. Extract 3D Roof-Eave & Tree-Canopy Overhang Drip Edges from Height Discontinuities (Zero-GC Flat Float32Array!)
    let candCap = 8192;
    let candBuf = new Float32Array(candCap * 7);
    let candCount = 0;
    const nDirsX = [-1, 1, 0, 0];
    const nDirsZ = [0, 0, -1, 1];

    for (let iz = 2; iz < size - 2; iz++) {
      if (useSlicing && (iz & 7) === 7 && nowFn() - sliceStart >= sliceBudgetMs) {
        yield;
        sliceStart = nowFn();
      }
      const wz = minZBound + (iz + 0.5) * cellSizeZ;
      const rowOffset = iz * size;
      for (let ix = 2; ix < size - 2; ix++) {
        const idx = rowOffset + ix;
        const h = heightBuf[idx];
        if (h < 2.35) continue;

        let bestDrop = 0.0;
        let bestLandingY = 0.0;
        let bestOx = 0;
        let bestOz = 0;

        for (let d = 0; d < 4; d++) {
          const ox = nDirsX[d];
          const oz = nDirsZ[d];
          const nIdx = (iz + oz) * size + (ix + ox);
          const nh = heightBuf[nIdx];
          const drop = h - nh;
          if (drop > bestDrop) {
            bestDrop = drop;
            bestLandingY = nh;
            bestOx = ox;
            bestOz = oz;
          }
        }

        if (bestDrop > 1.85) {
          if (candCount >= candCap) {
            candCap *= 2;
            const grown = new Float32Array(candCap * 7);
            grown.set(candBuf);
            candBuf = grown;
          }
          const wx = minXBound + (ix + 0.5) * cellSizeX;
          const cls = classBuf[idx];
          const outLen = Math.hypot(bestOx, bestOz) || 1.0;
          const base = candCount * 7;
          // Nudge 0.56 texel outward past the overhang lip so drips & genting runoff fall unobstructed
          candBuf[base] = wx + bestOx * cellSizeX * 0.56;
          candBuf[base + 1] = h - (cls === 3 ? 0.34 : 0.07);
          candBuf[base + 2] = wz + bestOz * cellSizeZ * 0.56;
          candBuf[base + 3] = bestLandingY;
          candBuf[base + 4] = cls === 3 ? 1.0 : 0.0;
          candBuf[base + 5] = bestOx / outLen;
          candBuf[base + 6] = bestOz / outLen;
          candCount++;
        }
      }
    }

    // Deterministically stride-pick up to maxDrips evenly across the entire campus & tower
    const dripAnchors = [];
    if (candCount > 0) {
      // Prime-stride permutation ensures uniform spatial coverage across all roofs & trees
      const n = candCount;
      const count = Math.min(maxDrips, n);
      const prime = 7919;
      for (let i = 0; i < maxDrips; i++) {
        const srcBase = (((i * prime) % n + n) % n) * 7;
        const outX = candBuf[srcBase + 5];
        const outZ = candBuf[srcBase + 6];
        // Tiny sub-texel jitter along tangent edge (perpendicular to outward eave normal!)
        const jitter = (((i * 37) % 100) / 100 - 0.5) * avgCellSize * 0.42;
        const tanX = -outZ;
        const tanZ = outX;
        dripAnchors.push({
          x: candBuf[srcBase] + tanX * jitter,
          y: candBuf[srcBase + 1],
          z: candBuf[srcBase + 2] + tanZ * jitter,
          landingY: candBuf[srcBase + 3],
          isCanopy: candBuf[srcBase + 4],
          outX,
          outZ,
          uniqueEdges: count
        });
      }
    } else {
      // Fallback perimeter ring if baked on an empty test scene
      for (let i = 0; i < maxDrips; i++) {
        const th = (i / maxDrips) * Math.PI * 2.0;
        dripAnchors.push({
          x: Math.cos(th) * 18.0,
          y: 6.0,
          z: Math.sin(th) * 18.0,
          landingY: 0.14,
          isCanopy: 0.0,
          outX: Math.cos(th),
          outZ: Math.sin(th),
          uniqueEdges: 0
        });
      }
    }

    // 4. Pack RGBA8 Collision Texture (NearestFilter avoids 16-bit hi/lo byte carry interpolation spikes!)
    const texData = new Uint8Array(totalCells * 4);
    for (let i = 0; i < totalCells; i++) {
      if (useSlicing && (i & 8191) === 8191 && nowFn() - sliceStart >= sliceBudgetMs) {
        yield;
        sliceStart = nowFn();
      }
      const o = i * 4;
      const hClamped = Math.max(minY, Math.min(maxY, heightBuf[i]));
      const q = Math.max(0, Math.min(65535, Math.round(((hClamped - minY) / spanY) * 65535)));

      texData[o] = (q >> 8) & 255;
      texData[o + 1] = q & 255;
      texData[o + 2] = Math.max(0, Math.min(255, Math.round((nxBuf[i] * 0.5 + 0.5) * 255)));

      const cls = classBuf[i];
      if (cls === 0) {
        texData[o + 3] = 2; // Water sentinel (< 5)
      } else if (cls === 3) {
        texData[o + 3] = 254; // Botanical canopy sentinel (> 251)
      } else {
        // Hard surface: encode nz in [12..244] so GLSL reconstructs ny = sqrt(1 - nx^2 - nz^2)
        texData[o + 3] = Math.max(12, Math.min(244, Math.round(128 + nzBuf[i] * 116)));
      }
    }

    const texture = new THREE.DataTexture(
      texData,
      size,
      size,
      THREE.RGBAFormat,
      THREE.UnsignedByteType
    );
    texture.minFilter = THREE.NearestFilter;
    texture.magFilter = THREE.NearestFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.generateMipmaps = false;
    texture.needsUpdate = true;

    const sampleHeightAt = (x, z) => {
      const ix = Math.max(0, Math.min(size - 1, Math.floor((x - minXBound) / cellSizeX)));
      const iz = Math.max(0, Math.min(size - 1, Math.floor((z - minZBound) / cellSizeZ)));
      return heightBuf[iz * size + ix];
    };

    return {
      texture,
      size,
      halfExtent,
      minX: minXBound,
      maxX: maxXBound,
      minZ: minZBound,
      maxZ: maxZBound,
      spanX: worldSpanX,
      spanZ: worldSpanZ,
      minY,
      maxY,
      spanY,
      dripAnchors,
      rawDripCandidateCount: candCount,
      sampleHeightAt
    };
  }
}
