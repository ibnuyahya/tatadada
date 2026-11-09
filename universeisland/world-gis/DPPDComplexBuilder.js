import * as THREE from 'three';
import { TerrainDEMLoader } from './TerrainDEMLoader.js';

/**
 * NEXUS Omni-Scale Procedural Engine v5.1 — Kepulauan UI Edition
 * ./world-gis/DPPDComplexBuilder.js
 *
 * Class-A Reference-Calibrated Procedural Digital Twin of Kompleks Inovasi & Pendidikan UI on Pulau U:
 * 1. Gedung DPPD UI (2 Lantai — Front Nusantara Heritage Terracotta-Roofed Pavilion facing Gedung Rektorat UI)
 * 2. Gedung ILRC UI (6 Lantai — Integrated Laboratory and Research Center with 6-Story Curved Concrete Sunshades
 *    & Symmetrical Terracotta-Brick Twin-Pier Glazed Atrium Entrance Portal)
 * 3. Gedung STP-UI (8 Lantai — Science Techno Park Universitas Indonesia with L1 Exhibition Lobby, L2-L3 Copper-ACP
 *    Diagrid Podium, L4-L7 White Spandrel & Staggered Dark-Frame Floors, L8 Glazed Crown Frieze + 3D Signage,
 *    Wrapping Copper-Terracotta Frame, & Rooftop Solar PV Array)
 * 4. Plaza Taman Inovasi Tengah (Open Landscaped Central Courtyard, Covered Link Pergolas, Benches, & Lamps)
 * 5. Seamless Pulau U Integration (100% Identical Linear-RGB Cambered Asphalt, UI-Yellow Dashed Centerline,
 *    White Shoulder Lines, Flared Spur Bell-Mouth Aprons, 360° Outer Sidewalk Promenade, & Halte Bikun DPPD-STP)
 *
 * Batched directly into SplineCorridorBuilder's { horizontalGeo, verticalGeo } pair (+0 extra Draw Calls!).
 */
export class DPPDComplexBuilder {
  // Polar layout constants in root local space (note: entranceGroup has rotation.y = -pi/4 -> th_root = 135° - alpha)
  // Wide 40° open landscaped avenue separates STP-UI (-44°) and Gedung ILRC (-4°) so both stand cleanly apart!
  static DPPD_ALPHA_MIN_DEG = -44.0;
  static DPPD_ALPHA_MAX_DEG = 44.0;
  static STP_START_DEG = -134.0;
  static STP_END_DEG = -44.0;
  static ILRC_START_DEG = -4.0;
  static ILRC_END_DEG = 74.0;

  static _sliceSubdomain(evaluateFn, uMin, uMax, vMin, vMax, uSegs = 24, vSegs = 12) {
    const vertCount = (uSegs + 1) * (vSegs + 1);
    const positions = new Float32Array(vertCount * 3);
    const normals = new Float32Array(vertCount * 3);
    const uvs = new Float32Array(vertCount * 2);
    const indices = [];
    const p = new THREE.Vector3();
    const n = new THREE.Vector3();
    let vIdx = 0;
    let uvIdx = 0;

    for (let iv = 0; iv <= vSegs; iv++) {
      const v = vMin + (vMax - vMin) * (iv / vSegs);
      for (let iu = 0; iu <= uSegs; iu++) {
        const u = uMin + (uMax - uMin) * (iu / uSegs);
        evaluateFn(u, v, p, n);
        positions[vIdx] = p.x;
        positions[vIdx + 1] = p.y;
        positions[vIdx + 2] = p.z;
        normals[vIdx] = n.x;
        normals[vIdx + 1] = n.y;
        normals[vIdx + 2] = n.z;
        uvs[uvIdx] = u;
        uvs[uvIdx + 1] = v;
        vIdx += 3;
        uvIdx += 2;
      }
    }

    const stride = uSegs + 1;
    const vA = new THREE.Vector3();
    const vB = new THREE.Vector3();
    const vC = new THREE.Vector3();
    const e1 = new THREE.Vector3();
    const e2 = new THREE.Vector3();
    const faceNorm = new THREE.Vector3();
    const refNorm = new THREE.Vector3();

    for (let iv = 0; iv < vSegs; iv++) {
      for (let iu = 0; iu < uSegs; iu++) {
        const a = iv * stride + iu;
        const b = a + 1;
        const c = (iv + 1) * stride + iu;
        const d = c + 1;

        vA.fromArray(positions, a * 3);
        vB.fromArray(positions, b * 3);
        vC.fromArray(positions, c * 3);
        refNorm.fromArray(normals, a * 3);

        e1.subVectors(vB, vA);
        e2.subVectors(vC, vA);
        faceNorm.crossVectors(e1, e2);

        if (faceNorm.dot(refNorm) >= 0) {
          indices.push(a, b, c, b, d, c);
        } else {
          indices.push(a, c, b, b, c, d);
        }
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    return geo;
  }

  /**
   * Evaluates a watertight curved cylindrical-sector slab/wall band in polar coordinates (rIn..rOut, yBot..yTop, th0..th1).
   */
  static _addCurvedBand(parent, mat, th0, th1, rIn, rOut, yBot, yTop, segs = 32, opts = {}) {
    const { top = true, bottom = yBot > 0.25, outer = true, inner = true, caps = true } = opts;

    if (outer) {
      parent.add(
        new THREE.Mesh(
          DPPDComplexBuilder._sliceSubdomain(
            (u, v, p, n) => {
              const th = th0 + u * (th1 - th0);
              p.set(rOut * Math.cos(th), yBot + v * (yTop - yBot), rOut * Math.sin(th));
              n.set(Math.cos(th), 0, Math.sin(th));
            },
            0, 1, 0, 1, segs, 1
          ),
          mat
        )
      );
    }
    if (inner) {
      parent.add(
        new THREE.Mesh(
          DPPDComplexBuilder._sliceSubdomain(
            (u, v, p, n) => {
              const th = th0 + u * (th1 - th0);
              p.set(rIn * Math.cos(th), yBot + v * (yTop - yBot), rIn * Math.sin(th));
              n.set(-Math.cos(th), 0, -Math.sin(th));
            },
            0, 1, 0, 1, segs, 1
          ),
          mat
        )
      );
    }
    if (top) {
      parent.add(
        new THREE.Mesh(
          DPPDComplexBuilder._sliceSubdomain(
            (u, v, p, n) => {
              const th = th0 + u * (th1 - th0);
              const r = rIn + v * (rOut - rIn);
              p.set(r * Math.cos(th), yTop, r * Math.sin(th));
              n.set(0, 1, 0);
            },
            0, 1, 0, 1, segs, 1
          ),
          mat
        )
      );
    }
    if (bottom) {
      parent.add(
        new THREE.Mesh(
          DPPDComplexBuilder._sliceSubdomain(
            (u, v, p, n) => {
              const th = th0 + u * (th1 - th0);
              const r = rIn + v * (rOut - rIn);
              p.set(r * Math.cos(th), yBot, r * Math.sin(th));
              n.set(0, -1, 0);
            },
            0, 1, 0, 1, segs, 1
          ),
          mat
        )
      );
    }
    if (caps) {
      for (const [thCap, sgn] of [
        [th0, -1],
        [th1, 1]
      ]) {
        const nx = -Math.sin(thCap) * sgn;
        const nz = Math.cos(thCap) * sgn;
        parent.add(
          new THREE.Mesh(
            DPPDComplexBuilder._sliceSubdomain(
              (u, v, p, n) => {
                const r = rIn + u * (rOut - rIn);
                p.set(r * Math.cos(thCap), yBot + v * (yTop - yBot), r * Math.sin(thCap));
                n.set(nx, 0, nz);
              },
              0, 1, 0, 1, 2, 1
            ),
            mat
          )
        );
      }
    }
  }

  /**
   * SSoST 2-Story Gedung DPPD Nusantara Hipped Curved Wing Roof Surface (eave y = 11.75 -> ridge y = 16.55).
   */
  static evaluateWingRoofLocalSurface(u, v, targetPos, targetNorm) {
    const alphaMin = (DPPDComplexBuilder.DPPD_ALPHA_MIN_DEG * Math.PI) / 180.0;
    const alphaMax = (DPPDComplexBuilder.DPPD_ALPHA_MAX_DEG * Math.PI) / 180.0;
    const alphaSpan = alphaMax - alphaMin;

    const sectorCount = 6.0;
    const sIdx = Math.floor(Math.min(sectorCount - 0.0001, u * sectorCount));
    const sFrac = u * sectorCount - sIdx;
    const a0 = alphaMin + (sIdx / sectorCount) * alphaSpan;
    const a1 = alphaMin + ((sIdx + 1) / sectorCount) * alphaSpan;

    const rInner = 27.2;
    const rRidge = 36.6;
    const rOuter = 46.0;
    const r = rInner + v * (rOuter - rInner);

    const x0 = r * Math.sin(a0);
    const z0 = r * Math.cos(a0);
    const x1 = r * Math.sin(a1);
    const z1 = r * Math.cos(a1);
    const x = x0 + (x1 - x0) * sFrac;
    const z = z0 + (z1 - z0) * sFrac;

    const distFromRidge = Math.abs(r - rRidge) / ((rOuter - rInner) * 0.5);
    const endHipTaper = Math.min(1.0, Math.min(u, 1.0 - u) * 6.4);
    const eaveY = 11.75;
    const ridgeH = 4.80 * endHipTaper;
    const y = eaveY + Math.max(0.0, (1.0 - distFromRidge) * ridgeH);

    targetPos.set(x, y, z);
    const alphaMid = (a0 + a1) * 0.5;
    const radialSign = r >= rRidge ? 0.36 : -0.36;
    targetNorm.set(Math.sin(alphaMid) * radialSign, 0.93, Math.cos(alphaMid) * radialSign).normalize();
  }

  /**
   * Builds a watertight extruded 3D prism along +Z (z in [0, depth]) from a 2D polygon.
   */
  static _buildPolygonPrismGeo(pts2D, depth) {
    const n = pts2D.length;
    let area2 = 0.0;
    for (let i = 0; i < n; i++) {
      const p0 = pts2D[i];
      const p1 = pts2D[(i + 1) % n];
      area2 += p0[0] * p1[1] - p1[0] * p0[1];
    }
    const ordered = area2 >= 0 ? pts2D : [...pts2D].reverse();

    const pos = [];
    const norm = [];
    const uvs = [];
    const idx = [];
    let vCount = 0;

    let cx = 0.0;
    let cy = 0.0;
    for (let i = 0; i < n; i++) {
      cx += ordered[i][0];
      cy += ordered[i][1];
    }
    cx /= n;
    cy /= n;

    const fCenter = vCount++;
    pos.push(cx, cy, depth);
    norm.push(0, 0, 1);
    uvs.push(0.5, 0.5);
    for (let i = 0; i < n; i++) {
      pos.push(ordered[i][0], ordered[i][1], depth);
      norm.push(0, 0, 1);
      uvs.push(0.5, 0.5);
      vCount++;
    }
    for (let i = 0; i < n; i++) {
      idx.push(fCenter, fCenter + 1 + i, fCenter + 1 + ((i + 1) % n));
    }

    const bCenter = vCount++;
    pos.push(cx, cy, 0.0);
    norm.push(0, 0, -1);
    uvs.push(0.5, 0.5);
    for (let i = 0; i < n; i++) {
      pos.push(ordered[i][0], ordered[i][1], 0.0);
      norm.push(0, 0, -1);
      uvs.push(0.5, 0.5);
      vCount++;
    }
    for (let i = 0; i < n; i++) {
      idx.push(bCenter, bCenter + 1 + ((i + 1) % n), bCenter + 1 + i);
    }

    for (let i = 0; i < n; i++) {
      const p0 = ordered[i];
      const p1 = ordered[(i + 1) % n];
      const ex = p1[0] - p0[0];
      const ey = p1[1] - p0[1];
      const el = Math.hypot(ex, ey) || 1.0;
      const nx = ey / el;
      const ny = -ex / el;

      const base = vCount;
      pos.push(
        p0[0], p0[1], 0.0,
        p1[0], p1[1], 0.0,
        p1[0], p1[1], depth,
        p0[0], p0[1], depth
      );
      for (let k = 0; k < 4; k++) {
        norm.push(nx, ny, 0.0);
        uvs.push(k === 1 || k === 2 ? 1.0 : 0.0, k >= 2 ? 1.0 : 0.0);
      }
      idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
      vCount += 4;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(norm), 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
    geo.setIndex(idx);
    return geo;
  }

  static _orientBeamAlongZ(mesh, pA, pB, liftY = 0.0) {
    mesh.position.set(
      (pA.x + pB.x) * 0.5,
      (pA.y + pB.y) * 0.5 + liftY,
      (pA.z + pB.z) * 0.5
    );
    const dx = pB.x - pA.x;
    const dy = pB.y - pA.y;
    const dz = pB.z - pA.z;
    const len = Math.hypot(dx, dy, dz) || 1.0;
    const yaw = Math.atan2(dx, dz);
    const pitch = -Math.asin(Math.max(-1.0, Math.min(1.0, dy / len)));
    mesh.rotation.set(pitch, yaw, 0.0, 'YXZ');
  }

  static _createPalette() {
    const mk = (hex, kind = 'H', uvSentinel = null) => {
      const c = new THREE.Color(hex);
      const m = new THREE.MeshBasicMaterial({ color: c });
      m.userData.batchKind = kind;
      m.userData.rgb = [c.r, c.g, c.b];
      m.userData.uvSentinel = uvSentinel;
      return m;
    };
    // Direct linear-RGB material matching SplineCorridorBuilder & TerrainDEMLoader 100.0%!
    const mkLin = (rgb, kind = 'H', uvSentinel = null) => {
      const m = new THREE.MeshBasicMaterial({ color: 0xffffff });
      m.userData.batchKind = kind;
      m.userData.rgb = rgb;
      m.userData.uvSentinel = uvSentinel;
      return m;
    };

    return {
      // Exact Pulau U Road & Landscape Linear RGBs ("warna aspal sama" & calibrated tropical Zoysia grass!)
      asphaltCrown: mkLin([0.195, 0.205, 0.225], 'H', [0.0, -420.0]),
      asphaltGutter: mkLin([0.155, 0.165, 0.185], 'H', [0.0, -419.15]),
      uiYellowLine: mkLin([0.86, 0.64, 0.10]),
      whiteEdgeLine: mkLin([0.89, 0.90, 0.92]),
      concreteCurb: mkLin([0.88, 0.86, 0.82]),
      campusGrass: mkLin([0.148, 0.312, 0.080]),
      sidewalkPaver: mkLin([0.45, 0.44, 0.41]),
      courtyardGranite: mkLin([0.52, 0.51, 0.48]),
      curbStriped: mkLin([0.92, 0.93, 0.95], 'Z'),

      // Calibrated Macbeth/Hasselblad Architectural & 206-Photo Ground-Truth Bundaran Palette
      terracottaRoof: mk(0x8a3424),
      terracottaRidge: mk(0x662216),
      terracottaFloor: mk(0x9e4634),
      salmonBrick: mk(0xb84a39),
      warmTaupeSoffit: mk(0xb88972),
      stoneRetaining: mk(0x52504c),
      darkChannel: mk(0x272a2e),
      koralSikat: mk(0x735844),
      bambooStake: mk(0x9c824c),
      dryLeafMound: mk(0x6b4c35),
      silverLouver: mk(0xc4cbd4),
      kanBlue: mk(0x1c4f9c),
      creamWall: mk(0xece6d6),
      warmStonePanel: mk(0xe0dbd0),
      facadeWhite: mk(0xeeede8),
      ilrcConcrete: mk(0xdad6ce),
      ilrcBrickPortal: mk(0x913b2b),
      concreteTrim: mk(0x868179),
      granitePlinth: mk(0x5e6166),
      darkWoodFascia: mk(0x3a2318),
      teakBench: mk(0x825428),
      LushShrub: mk(0x305820),
      canopyLeaf: mk(0x386626),
      treeBark: mk(0x584232),
      mulchSoil: mk(0x5c4432),
      windowGlass: mk(0x111e2b, 'H', [0.5, -720.0]),
      darkFriezeGlass: mk(0x0f1924),
      warmLuminaire: mk(0xfff1c4, 'H', [0.5, -890.0]),
      whiteMullion: mk(0xeeede8),
      darkAnodized: mk(0x23272f),
      darkIron: mk(0x1f232a),
      copperACP: mk(0xb56d45),
      copperDark: mk(0x8f5230),
      brassSign: mk(0xd4ac38),
      uiYellowFascia: mk(0xdca81c),
      solarPanel: mk(0x142244),
      galvSteel: mk(0x929aa3),
      disabilityBlue: mk(0x2253b8),
      hydrantRed: mk(0xb82824),
      carWhite: mk(0xeceff2),
      carSilver: mk(0xc2cbd6),
      carDarkGrey: mk(0x323740),
      carNavy: mk(0x1d3452)
    };
  }

  /**
   * Procedural 5x7 Block-Matrix 3D Extruded Architectural Signage Generator
   * (Supports both flat planar headers and curved cylindrical facade friezes!).
   */
  static _GLYPHS_5X7 = {
    A: ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
    B: ['11110', '10001', '10001', '11110', '10001', '10001', '11110'],
    C: ['01111', '10000', '10000', '10000', '10000', '10000', '01111'],
    D: ['11110', '10001', '10001', '10001', '10001', '10001', '11110'],
    E: ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
    F: ['11111', '10000', '10000', '11110', '10000', '10000', '10000'],
    G: ['01111', '10000', '10000', '10111', '10001', '10001', '01111'],
    H: ['10001', '10001', '10001', '11111', '10001', '10001', '10001'],
    I: ['01110', '00100', '00100', '00100', '00100', '00100', '01110'],
    J: ['00111', '00010', '00010', '00010', '10010', '10010', '01100'],
    K: ['10001', '10010', '10100', '11000', '10100', '10010', '10001'],
    L: ['10000', '10000', '10000', '10000', '10000', '10000', '11111'],
    M: ['10001', '11011', '10101', '10101', '10001', '10001', '10001'],
    N: ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
    O: ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
    P: ['11110', '10001', '10001', '11110', '10000', '10000', '10000'],
    Q: ['01110', '10001', '10001', '10001', '10101', '10010', '01101'],
    R: ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
    S: ['01111', '10000', '10000', '01110', '00001', '00001', '11110'],
    T: ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
    U: ['10001', '10001', '10001', '10001', '10001', '10001', '01110'],
    V: ['10001', '10001', '10001', '10001', '10001', '01010', '00100'],
    W: ['10001', '10001', '10001', '10101', '10101', '11011', '10001'],
    X: ['10001', '10001', '01010', '00100', '01010', '10001', '10001'],
    Y: ['10001', '10001', '01010', '00100', '00100', '00100', '00100'],
    Z: ['11111', '00001', '00010', '00100', '01000', '10000', '11111'],
    '-': ['00000', '00000', '00000', '01110', '00000', '00000', '00000'],
    '/': ['00001', '00010', '00010', '00100', '01000', '01000', '10000'],
    '.': ['00000', '00000', '00000', '00000', '00000', '01100', '01100']
  };

  /**
   * Builds a polar sloped ribbon surface between (rIn, yIn) and (rOut, yOut) across [th0, th1].
   */
  static _buildPolarSlopedRibbonGeo(th0, th1, rIn, yIn, rOut, yOut, segs = 28) {
    return DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const th = th0 + u * (th1 - th0);
        const r = rIn + v * (rOut - rIn);
        const y = yIn + v * (yOut - yIn);
        const c = Math.cos(th);
        const s = Math.sin(th);
        p.set(r * c, y, r * s);
        const dr = rOut - rIn;
        const dy = yOut - yIn;
        const len = Math.hypot(dr, dy) || 1.0;
        const nr = -dy / len;
        const ny = dr / len;
        if (ny >= 0) {
          n.set(nr * c, ny, nr * s).normalize();
        } else {
          n.set(-nr * c, -ny, -nr * s).normalize();
        }
      },
      0, 1, 0, 1, segs, 1
    );
  }

  /**
   * Builds a watertight 4-wing square courtyard Limasan / Hipped roof + eave soffit + 12 ridge/hip/valley line pairs
   * for ITTC Perpustakaan Lama (ref_174, ref_203).
   */
  static _buildSquareCourtyardLimasanRoof(halfOut, halfIn, eaveY, ridgeY) {
    const rMid = 0.5 * (halfOut + halfIn);
    const pos = [];
    const norm = [];
    const uvs = [];
    const idx = [];
    let vCount = 0;

    const addQuad = (a, b, c, d) => {
      const vA = new THREE.Vector3(...a);
      const vB = new THREE.Vector3(...b);
      const vC = new THREE.Vector3(...c);
      const fn = new THREE.Vector3().subVectors(vB, vA).cross(new THREE.Vector3().subVectors(vC, vA)).normalize();
      if (fn.y < 0) fn.negate();
      const base = vCount;
      for (const pt of [a, b, c, d]) {
        pos.push(pt[0], pt[1], pt[2]);
        norm.push(fn.x, fn.y, fn.z);
        uvs.push(0.5, 0.5);
      }
      // Ensure upward normal winding
      const rawCrossY =
        (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
      if (rawCrossY >= 0) {
        idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
      } else {
        idx.push(base, base + 2, base + 1, base, base + 3, base + 2);
      }
      vCount += 4;
    };

    // 4 Outer Slopes (S, E, N, W) + 4 Inner Courtyard Slopes (S, E, N, W)
    const oSW = [-halfOut, eaveY, halfOut];
    const oSE = [halfOut, eaveY, halfOut];
    const oNE = [halfOut, eaveY, -halfOut];
    const oNW = [-halfOut, eaveY, -halfOut];

    const rSW = [-rMid, ridgeY, rMid];
    const rSE = [rMid, ridgeY, rMid];
    const rNE = [rMid, ridgeY, -rMid];
    const rNW = [-rMid, ridgeY, -rMid];

    const iSW = [-halfIn, eaveY, halfIn];
    const iSE = [halfIn, eaveY, halfIn];
    const iNE = [halfIn, eaveY, -halfIn];
    const iNW = [-halfIn, eaveY, -halfIn];

    // Outer 4 trapezoidal slopes
    addQuad(oSW, oSE, rSE, rSW);
    addQuad(oSE, oNE, rNE, rSE);
    addQuad(oNE, oNW, rNW, rNE);
    addQuad(oNW, oSW, rSW, rNW);

    // Inner 4 trapezoidal slopes pitching down toward the central concrete deck
    addQuad(rSW, rSE, iSE, iSW);
    addQuad(rSE, rNE, iNE, iSE);
    addQuad(rNE, rNW, iNW, iNE);
    addQuad(rNW, rSW, iSW, iNW);

    const roofGeo = new THREE.BufferGeometry();
    roofGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    roofGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(norm), 3));
    roofGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(uvs), 2));
    roofGeo.setIndex(idx);

    const ridgePairs = [
      // 4 main horizontal ridges
      [new THREE.Vector3(...rSW), new THREE.Vector3(...rSE)],
      [new THREE.Vector3(...rSE), new THREE.Vector3(...rNE)],
      [new THREE.Vector3(...rNE), new THREE.Vector3(...rNW)],
      [new THREE.Vector3(...rNW), new THREE.Vector3(...rSW)],
      // 4 outer corner hips
      [new THREE.Vector3(...oSW), new THREE.Vector3(...rSW)],
      [new THREE.Vector3(...oSE), new THREE.Vector3(...rSE)],
      [new THREE.Vector3(...oNE), new THREE.Vector3(...rNE)],
      [new THREE.Vector3(...oNW), new THREE.Vector3(...rNW)],
      // 4 inner corner valleys
      [new THREE.Vector3(...iSW), new THREE.Vector3(...rSW)],
      [new THREE.Vector3(...iSE), new THREE.Vector3(...rSE)],
      [new THREE.Vector3(...iNE), new THREE.Vector3(...rNE)],
      [new THREE.Vector3(...iNW), new THREE.Vector3(...rNW)]
    ];

    return { roofGeo, ridgePairs };
  }

  static _addCurved3DText(parent, text, radius, thStart, yBase, charHeight, depth, mat, dirSign = -1.0) {
    const pxH = charHeight / 7.0;
    const pxW = pxH * 0.86;
    const dThPx = (pxW / radius) * dirSign;
    const boxGeo = new THREE.BoxGeometry(depth, pxH * 1.06, pxW * 1.06);
    let cursorTh = thStart;

    for (let ci = 0; ci < text.length; ci++) {
      const ch = text[ci];
      if (ch === ' ') {
        cursorTh += dThPx * 3.6;
        continue;
      }
      const rows = DPPDComplexBuilder._GLYPHS_5X7[ch];
      if (!rows) {
        cursorTh += dThPx * 4.0;
        continue;
      }
      for (let r = 0; r < 7; r++) {
        const rowStr = rows[r];
        let c = 0;
        while (c < 5) {
          if (rowStr[c] !== '1') {
            c++;
            continue;
          }
          let cEnd = c;
          while (cEnd + 1 < 5 && rowStr[cEnd + 1] === '1') cEnd++;
          const runLen = cEnd - c + 1;
          const midCol = (c + cEnd) * 0.5;
          const th = cursorTh + midCol * dThPx;
          const py = yBase + (6 - r + 0.5) * pxH;
          const m = new THREE.Mesh(
            runLen === 1 ? boxGeo : new THREE.BoxGeometry(depth, pxH * 1.06, pxW * runLen * 1.04),
            mat
          );
          m.position.set(radius * Math.cos(th), py, radius * Math.sin(th));
          m.rotation.y = -th;
          parent.add(m);
          c = cEnd + 1;
        }
      }
      cursorTh += dThPx * 6.3;
    }
  }

  static _addPlanar3DText(parent, text, cx, yBase, zFront, charHeight, depth, mat) {
    const pxH = charHeight / 7.0;
    const pxW = pxH * 0.84;
    let totalCols = 0;
    for (let i = 0; i < text.length; i++) {
      totalCols += text[i] === ' ' ? 3.5 : 6.2;
    }
    let curX = cx - (totalCols - 1.2) * pxW * 0.5;
    for (let ci = 0; ci < text.length; ci++) {
      const ch = text[ci];
      if (ch === ' ') {
        curX += pxW * 3.5;
        continue;
      }
      const rows = DPPDComplexBuilder._GLYPHS_5X7[ch];
      if (!rows) {
        curX += pxW * 4.0;
        continue;
      }
      for (let r = 0; r < 7; r++) {
        const rowStr = rows[r];
        let c = 0;
        while (c < 5) {
          if (rowStr[c] !== '1') {
            c++;
            continue;
          }
          let cEnd = c;
          while (cEnd + 1 < 5 && rowStr[cEnd + 1] === '1') cEnd++;
          const runLen = cEnd - c + 1;
          const midCol = (c + cEnd) * 0.5;
          const m = new THREE.Mesh(
            new THREE.BoxGeometry(pxW * runLen * 1.04, pxH * 1.06, depth),
            mat
          );
          m.position.set(curX + midCol * pxW, yBase + (6 - r + 0.5) * pxH, zFront);
          parent.add(m);
          c = cEnd + 1;
        }
      }
      curX += pxW * 6.2;
    }
  }

  /**
   * Builds a sculpted 3D Golden UI Makara Emblem (backplate disc + golden rim + Makara flame crown & open book).
   */
  static _buildUIMakaraEmblem(mats, radius = 1.15) {
    const g = new THREE.Group();
    const back = new THREE.Mesh(
      new THREE.CylinderGeometry(radius, radius, 0.14, 24),
      mats.darkAnodized
    );
    back.rotation.x = Math.PI * 0.5;
    g.add(back);

    const goldRing = new THREE.Mesh(
      new THREE.CylinderGeometry(radius * 0.90, radius * 0.90, 0.18, 24),
      mats.brassSign
    );
    goldRing.rotation.x = Math.PI * 0.5;
    g.add(goldRing);

    const innerPlate = new THREE.Mesh(
      new THREE.CylinderGeometry(radius * 0.72, radius * 0.72, 0.20, 20),
      mats.darkAnodized
    );
    innerPlate.rotation.x = Math.PI * 0.5;
    g.add(innerPlate);

    // Stylized Makara Kala-Tree rays & open book relief in brass-gold
    const trunk = new THREE.Mesh(
      new THREE.BoxGeometry(radius * 0.22, radius * 0.92, 0.25),
      mats.brassSign
    );
    trunk.position.y = radius * 0.08;
    g.add(trunk);

    for (const sgn of [-1, 1]) {
      for (let b = 0; b < 3; b++) {
        const branch = new THREE.Mesh(
          new THREE.BoxGeometry(radius * (0.48 - b * 0.08), radius * 0.11, 0.24),
          mats.brassSign
        );
        branch.position.set(sgn * radius * (0.26 - b * 0.03), radius * (-0.08 + b * 0.24), 0);
        branch.rotation.z = sgn * (0.28 + b * 0.12);
        g.add(branch);
      }
      const bookPage = new THREE.Mesh(
        new THREE.BoxGeometry(radius * 0.36, radius * 0.20, 0.25),
        mats.whiteMullion
      );
      bookPage.position.set(sgn * radius * 0.20, -radius * 0.40, 0);
      bookPage.rotation.z = -sgn * 0.14;
      g.add(bookPage);
    }
    return g;
  }

  static _buildParkedCar(mats, bodyMat, x, y, z, rotY) {
    const g = new THREE.Group();
    g.position.set(x, y, z);
    g.rotation.y = rotY;

    const lowerBody = new THREE.Mesh(new THREE.BoxGeometry(1.82, 0.56, 4.35), bodyMat);
    lowerBody.position.set(0, 0.50, 0);
    g.add(lowerBody);

    const frontBumper = new THREE.Mesh(new THREE.BoxGeometry(1.84, 0.28, 0.24), mats.darkIron);
    frontBumper.position.set(0, 0.34, 2.15);
    g.add(frontBumper);

    const rearBumper = new THREE.Mesh(new THREE.BoxGeometry(1.84, 0.30, 0.24), mats.darkIron);
    rearBumper.position.set(0, 0.35, -2.15);
    g.add(rearBumper);

    const cabinRoof = new THREE.Mesh(new THREE.BoxGeometry(1.52, 0.10, 2.05), bodyMat);
    cabinRoof.position.set(0, 1.32, -0.18);
    g.add(cabinRoof);

    const cabinGlass = new THREE.Mesh(new THREE.BoxGeometry(1.56, 0.52, 2.18), mats.windowGlass);
    cabinGlass.position.set(0, 1.02, -0.16);
    g.add(cabinGlass);

    for (const sx of [-0.77, 0.77]) {
      const bPillar = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.54, 0.14), mats.darkIron);
      bPillar.position.set(sx, 1.02, -0.12);
      g.add(bPillar);

      const mirror = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.11, 0.12), bodyMat);
      mirror.position.set(sx + Math.sign(sx) * 0.11, 0.86, 0.72);
      g.add(mirror);
    }

    const grille = new THREE.Mesh(new THREE.BoxGeometry(1.06, 0.22, 0.08), mats.darkIron);
    grille.position.set(0, 0.56, 2.16);
    g.add(grille);

    for (const sx of [-0.62, 0.62]) {
      const headLamp = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.14, 0.08), mats.whiteMullion);
      headLamp.position.set(sx, 0.63, 2.16);
      g.add(headLamp);

      const tailLamp = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.14, 0.08), mats.hydrantRed);
      tailLamp.position.set(sx, 0.66, -2.16);
      g.add(tailLamp);
    }

    const tireGeo = new THREE.CylinderGeometry(0.32, 0.32, 0.22, 14);
    tireGeo.rotateZ(Math.PI * 0.5);
    const rimGeo = new THREE.CylinderGeometry(0.19, 0.19, 0.23, 10);
    rimGeo.rotateZ(Math.PI * 0.5);
    for (const sx of [-0.84, 0.84]) {
      for (const sz of [-1.32, 1.34]) {
        const tire = new THREE.Mesh(tireGeo, mats.darkIron);
        tire.position.set(sx, 0.32, sz);
        g.add(tire);

        const rim = new THREE.Mesh(rimGeo, mats.carSilver);
        rim.position.set(sx, 0.32, sz);
        g.add(rim);
      }
    }
    return g;
  }

  static _buildCourtyardTree(mats, x, y, z, scale = 1.0) {
    const g = new THREE.Group();
    g.position.set(x, y, z);
    g.scale.set(scale, scale, scale);

    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.40, 3.6, 10), mats.treeBark);
    trunk.position.y = 1.8;
    g.add(trunk);

    const crownSpecs = [
      [0.0, 4.3, 0.0, 2.35, 1.55],
      [-1.15, 3.85, 0.65, 1.65, 1.25],
      [1.15, 3.90, -0.55, 1.65, 1.25],
      [0.25, 4.95, 0.35, 1.55, 1.15]
    ];
    for (const [cx, cy, cz, rx, ry] of crownSpecs) {
      const puff = new THREE.Mesh(new THREE.SphereGeometry(rx, 12, 9), mats.canopyLeaf);
      puff.scale.set(1.0, ry / rx, 1.0);
      puff.position.set(cx, cy, cz);
      g.add(puff);
    }
    return g;
  }

  static _buildRoyalPalm(mats, x, y, z, h = 10.5) {
    const g = new THREE.Group();
    g.position.set(x, y, z);

    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.38, h - 1.4, 10), mats.concreteTrim);
    trunk.position.y = (h - 1.4) * 0.5;
    g.add(trunk);

    const crownShaft = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.25, 1.5, 10), mats.LushShrub);
    crownShaft.position.y = h - 0.75;
    g.add(crownShaft);

    for (let f = 0; f < 9; f++) {
      const ang = (f / 9) * Math.PI * 2.0;
      const frondHolder = new THREE.Group();
      frondHolder.position.y = h - 0.1;
      frondHolder.rotation.y = ang;
      const frond = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.08, 3.2), mats.canopyLeaf);
      frond.position.set(0, -0.32, 1.55);
      frond.rotation.x = 0.34;
      frondHolder.add(frond);
      g.add(frondHolder);
    }
    return g;
  }

  /**
   * Builds and bakes the entire Kompleks DPPD (2 Lantai) – ILRC (6 Lantai) – STP-UI (8 Lantai)
   * into { horizontalGeo, verticalGeo } for SplineCorridorBuilder (+0 extra Draw Calls!).
   */
  static buildForCorridor() {
    const mats = DPPDComplexBuilder._createPalette();
    const root = new THREE.Group();
    root.position.set(
      TerrainDEMLoader.DPPD_CX,
      TerrainDEMLoader.DPPD_PLATEAU_Y,
      TerrainDEMLoader.DPPD_CZ
    );
    root.rotation.y = TerrainDEMLoader.DPPD_Y_ROT;
    root.scale.set(
      TerrainDEMLoader.DPPD_SCALE,
      TerrainDEMLoader.DPPD_SCALE,
      TerrainDEMLoader.DPPD_SCALE
    );

    const dppdGroup = new THREE.Group();
    const contextGroup = new THREE.Group();
    root.add(dppdGroup, contextGroup);

    const entranceGroup = new THREE.Group();
    entranceGroup.rotation.y = -0.25 * Math.PI;
    dppdGroup.add(entranceGroup);

    // =========================================================================
    // 1. SEAMLESS PULAU U PLATEAU LAWN, 4-STRIP CAMBERED RING ROAD (IDENTICAL ASPHALT COLOR!),
    //    FLARED SPUR BELL-MOUTH APRONS, & 360° OUTER CAMPUS SIDEWALK PROMENADE
    // =========================================================================
    const campusLawnPad = new THREE.Mesh(
      new THREE.CylinderGeometry(52.90, 53.10, 0.80, 96),
      mats.campusGrass
    );
    campusLawnPad.position.y = -0.39;
    dppdGroup.add(campusLawnPad);

    // Grass-colored sloped embankment skirt blending smoothly into Pulau U's +4.85m DEM plateau (ZERO grey tin-can wall!)
    const outerGrassSkirt = new THREE.Mesh(
      new THREE.CylinderGeometry(68.05, 71.80, 2.80, 96, 1, true),
      mats.campusGrass
    );
    outerGrassSkirt.position.y = -1.36;
    dppdGroup.add(outerGrassSkirt);

    // Smooth curb-cut factor [0..1] dropping outer ring-road curb & outer sidewalk curb flush across South & East Pulau U Spur Mouths!
    const evalSpurMouthFactor = (rLocal, th, halfMouthM = 4.9, fadeM = 2.6) => {
      const wp = DPPDComplexBuilder.localToWorld(rLocal * Math.cos(th), 0.0, rLocal * Math.sin(th), 0.0);
      const dSouth = Math.hypot(wp.x - -292.1, wp.z - 336.8);
      const dEast = Math.hypot(wp.x - -218.3, wp.z - 338.4);
      const dMin = Math.min(dSouth, dEast);
      const t = Math.max(0.0, Math.min(1.0, (dMin - halfMouthM) / fadeM));
      return t * t * (3.0 - 2.0 * t);
    };

    // Inner driveway curb-cut factor for East Visitor Parking entrance (-156°) & STP/ILRC drop-off throats
    const evalInnerMouthFactor = (th) => {
      const deg = ((th * 180.0) / Math.PI + 540.0) % 360.0 - 180.0;
      const dPark = Math.abs((((deg - -156.0) + 540.0) % 360.0) - 180.0);
      const t = Math.max(0.0, Math.min(1.0, (dPark - 7.5) / 3.5));
      return t * t * (3.0 - 2.0 * t);
    };

    // 4-Strip Cambered Circular Ring Road (r = 53.20 .. 64.80) using EXACT Pulau U asphaltCrown & asphaltGutter!
    const ringStrips = [
      { rA: 53.20, rB: 55.10, yA: 0.065, yB: 0.082, mat: mats.asphaltGutter },
      { rA: 55.10, rB: 59.00, yA: 0.082, yB: 0.092, mat: mats.asphaltCrown },
      { rA: 59.00, rB: 62.90, yA: 0.092, yB: 0.082, mat: mats.asphaltCrown },
      { rA: 62.90, rB: 64.80, yA: 0.082, yB: 0.065, mat: mats.asphaltGutter }
    ];
    for (const st of ringStrips) {
      const stripGeo = DPPDComplexBuilder._sliceSubdomain(
        (u, v, p, n) => {
          const th = u * Math.PI * 2.0;
          const r = st.rA + v * (st.rB - st.rA);
          const y = st.yA + v * (st.yB - st.yA);
          p.set(r * Math.cos(th), y, r * Math.sin(th));
          n.set(0, 1, 0);
        },
        0, 1, 0, 1, 128, 1
      );
      dppdGroup.add(new THREE.Mesh(stripGeo, st.mat));
    }

    // Flared Asphalt Bell-Mouth Junction Aprons at South Spur (-4.6°) & East Spur (-158.3°)
    // Seamlessly bridges r = 64.6 .. 69.8 across the outer sidewalk throat at exact Pulau U asphalt elevation!
    for (const spurDeg of [-4.6, -158.3]) {
      const thC = (spurDeg * Math.PI) / 180.0;
      const halfSpan = (11.5 * Math.PI) / 180.0;
      const apronGeo = DPPDComplexBuilder._sliceSubdomain(
        (u, v, p, n) => {
          const flare = 1.0 + 0.38 * (1.0 - v);
          const th = thC + (u - 0.5) * 2.0 * halfSpan * flare;
          const r = 64.55 + v * (69.60 - 64.55);
          const edgeDrop = Math.abs(u - 0.5) > 0.36 ? -0.010 : 0.0;
          p.set(r * Math.cos(th), 0.072 + edgeDrop, r * Math.sin(th));
          n.set(0, 1, 0);
        },
        0, 1, 0, 1, 20, 4
      );
      dppdGroup.add(new THREE.Mesh(apronGeo, mats.asphaltCrown));

      // White Stop Bar & Spur Crosswalk across each spur throat so outer sidewalk promenade continues safely!
      for (let cw = -3; cw <= 3; cw++) {
        const thS = thC + cw * 0.024;
        const stripe = new THREE.Mesh(new THREE.BoxGeometry(2.15, 0.032, 0.56), mats.whiteEdgeLine);
        stripe.position.set(66.6 * Math.cos(thS), 0.082, 66.6 * Math.sin(thS));
        stripe.rotation.y = -thS;
        dppdGroup.add(stripe);
      }
    }

    // 360° Paved Outer Campus Sidewalk & Jogging Promenade (r = 65.16 .. 68.10, dropped flush across spur throats!)
    const outerSidewalkGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const th = u * Math.PI * 2.0;
        const r = 65.16 + v * (68.10 - 65.16);
        const cf = evalSpurMouthFactor(r, th, 5.2, 2.4);
        const y = 0.055 + 0.135 * cf;
        p.set(r * Math.cos(th), y, r * Math.sin(th));
        n.set(0, 1, 0);
      },
      0, 1, 0, 1, 144, 1
    );
    dppdGroup.add(new THREE.Mesh(outerSidewalkGeo, mats.sidewalkPaver));

    // Outer Sidewalk Concrete Rim Curb (r = 68.05 .. 68.38)
    const outerRimGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => {
        const th = u * Math.PI * 2.0;
        const r = 68.05 + v * 0.34;
        const cf = evalSpurMouthFactor(r, th, 5.4, 2.4);
        p.set(r * Math.cos(th), 0.052 + 0.155 * cf, r * Math.sin(th));
        n.set(0, 1, 0);
      },
      0, 1, 0, 1, 144, 1
    );
    dppdGroup.add(new THREE.Mesh(outerRimGeo, mats.concreteCurb));

    // Inner & Outer Striped Zebra Curbs around Ring Road
    const buildRingZebraCurb = (rIn, rOut, yTop, mode) => {
      const segs = 144;
      const circum = Math.PI * (rIn + rOut) * TerrainDEMLoader.DPPD_SCALE;
      const zebraRepeats = Math.max(16, Math.round(circum * 0.26));
      const evalYTop = (th) => {
        let cf = 1.0;
        if (mode === 'OUTER') cf = evalSpurMouthFactor(0.5 * (rIn + rOut), th, 5.1, 2.3);
        else if (mode === 'INNER') cf = evalInnerMouthFactor(th);
        return 0.068 + (yTop - 0.068) * cf;
      };

      const geos = [
        DPPDComplexBuilder._sliceSubdomain(
          (u, v, p, n) => {
            const th = u * Math.PI * 2.0;
            const r = rIn + v * (rOut - rIn);
            p.set(r * Math.cos(th), evalYTop(th), r * Math.sin(th));
            n.set(0, 1, 0);
          },
          0, 1, 0, 1, segs, 1
        ),
        DPPDComplexBuilder._sliceSubdomain(
          (u, v, p, n) => {
            const th = u * Math.PI * 2.0;
            const yt = evalYTop(th);
            p.set(rIn * Math.cos(th), 0.06 + v * (yt - 0.06), rIn * Math.sin(th));
            n.set(-Math.cos(th), 0, -Math.sin(th));
          },
          0, 1, 0, 1, segs, 1
        ),
        DPPDComplexBuilder._sliceSubdomain(
          (u, v, p, n) => {
            const th = u * Math.PI * 2.0;
            const yt = evalYTop(th);
            p.set(rOut * Math.cos(th), -0.25 + v * (yt + 0.25), rOut * Math.sin(th));
            n.set(Math.cos(th), 0, Math.sin(th));
          },
          0, 1, 0, 1, segs, 1
        )
      ];

      for (const g of geos) {
        const uv = g.attributes.uv;
        for (let i = 0; i < uv.count; i++) {
          uv.array[i * uv.itemSize] = uv.getX(i) * zebraRepeats;
        }
        const m = new THREE.Mesh(g, mats.curbStriped);
        m.userData.keepCurbUV = true;
        dppdGroup.add(m);
      }
    };

    buildRingZebraCurb(52.84, 53.22, 0.22, 'INNER');
    buildRingZebraCurb(64.78, 65.16, 0.22, 'OUTER');

    // Continuous White Edge Shoulder Lines + UI Golden-Yellow Dashed Centerline (100% matching Pulau U road standard!)
    for (const [rLine, mode] of [
      [53.52, 'INNER'],
      [64.30, 'OUTER']
    ]) {
      const edgeLineGeo = DPPDComplexBuilder._sliceSubdomain(
        (u, v, p, n) => {
          const th = u * Math.PI * 2.0;
          const cf =
            mode === 'OUTER'
              ? evalSpurMouthFactor(rLine, th, 5.0, 1.8)
              : evalInnerMouthFactor(th);
          const r = rLine + v * 0.18;
          p.set(r * Math.cos(th), cf > 0.45 ? 0.088 : 0.030, r * Math.sin(th));
          n.set(0, 1, 0);
        },
        0, 1, 0, 1, 144, 1
      );
      dppdGroup.add(new THREE.Mesh(edgeLineGeo, mats.whiteEdgeLine));
    }

    const crossAnglesDeg = [135.0, 35.0, -89.0, -156.0];
    for (let d = 0; d < 56; d++) {
      const deg = (d / 56) * 360.0 - 180.0;
      const nearCross = crossAnglesDeg.some(
        (cd) => Math.abs((((deg - cd) + 540.0) % 360.0) - 180.0) < 4.8
      );
      if (nearCross) continue;
      const th = (deg * Math.PI) / 180.0;
      const dash = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.032, 2.35), mats.uiYellowLine);
      dash.position.set(59.0 * Math.cos(th), 0.104, 59.0 * Math.sin(th));
      dash.rotation.y = -th;
      dppdGroup.add(dash);
    }

    // 4 Crisp Zebra Crosswalks directly in front of Gedung DPPD (+135°), Gedung ILRC (+35°), Gedung STP-UI (-89°), & East Plaza (-156°)
    for (const crossDeg of crossAnglesDeg) {
      const thC = (crossDeg * Math.PI) / 180.0;
      for (let st = -4; st <= 4; st++) {
        const rSt = 59.0 + st * 1.12;
        const stripe = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.034, 3.30), mats.whiteEdgeLine);
        stripe.position.set(rSt * Math.cos(thC), 0.105, rSt * Math.sin(thC));
        stripe.rotation.y = -thC;
        dppdGroup.add(stripe);
      }
    }

    // 1B. Open Rubble-Stone Perimeter Drainage Channel ("Saluran Batu Kali", r = 51.05 .. 51.95)
    //     + 7 Ground-Truth Driveway/Pedestrian Bridges & "DILARANG PARKIR" Warning Sign (ref_164, ref_184, ref_202)
    {
      const ditchSectorsDeg = [
        [-138.0, -93.0],
        [-78.0, -9.0],
        [-1.0, 29.5],
        [40.5, 86.0],
        [93.0, 127.0],
        [143.0, 198.0]
      ];
      for (const [d0, d1] of ditchSectorsDeg) {
        const t0 = (d0 * Math.PI) / 180.0;
        const t1 = (d1 * Math.PI) / 180.0;
        const segs = Math.max(6, Math.round((d1 - d0) / 4.5));
        DPPDComplexBuilder._addCurvedBand(dppdGroup, mats.stoneRetaining, t0, t1, 50.88, 51.12, 0.02, 0.24, segs);
        DPPDComplexBuilder._addCurvedBand(dppdGroup, mats.stoneRetaining, t0, t1, 51.84, 52.08, 0.02, 0.24, segs);
        DPPDComplexBuilder._addCurvedBand(dppdGroup, mats.darkChannel, t0, t1, 51.10, 51.86, 0.01, 0.06, segs);
      }

      // 6 Concrete / Steel-Grate / Pebble-Grid Bridges across the Perimeter Ditch (7th is DPPD main forecourt bridge)
      const bridgeSpecs = [
        [-122.0, 4.5, mats.concreteCurb],   // STP-UI NW Exit Bridge
        [-84.0, 6.2, mats.courtyardGranite], // STP-UI Main Portico Bridge
        [-89.5, 2.6, mats.darkIron],        // STP-UI Pedestrian Steel Grate Bridge (ref_164)
        [-4.5, 4.4, mats.concreteCurb],     // ILRC East Portico Bridge
        [24.0, 4.6, mats.concreteCurb],     // DRPM Breezeway Loop Gate Bridge (ref_012)
        [35.0, 5.2, mats.koralSikat]        // ILRC Atrium Koral-Sikat Pebble-Grid Ramp Bridge (ref_024)
      ];
      for (const [bDeg, bSpanDeg, bMat] of bridgeSpecs) {
        const bt0 = ((bDeg - bSpanDeg * 0.5) * Math.PI) / 180.0;
        const bt1 = ((bDeg + bSpanDeg * 0.5) * Math.PI) / 180.0;
        DPPDComplexBuilder._addCurvedBand(dppdGroup, bMat, bt0, bt1, 50.65, 52.86, 0.06, 0.22, 4);
      }

      // Yellow/Black "DILARANG PARKIR DI SEPANJANG JALAN INI" Signboard at West/East Gate (ref_202, ref_203)
      const thWarn = (-142.5 * Math.PI) / 180.0;
      const warnGrp = new THREE.Group();
      warnGrp.position.set(50.2 * Math.cos(thWarn), 0.0, 50.2 * Math.sin(thWarn));
      warnGrp.rotation.y = -thWarn;
      for (const pz of [-0.42, 0.42]) {
        const post = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 1.55, 6), mats.darkIron);
        post.position.set(0, 0.78, pz);
        warnGrp.add(post);
      }
      const warnBoard = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.68, 1.18), mats.uiYellowFascia);
      warnBoard.position.set(0.04, 1.32, 0);
      const warnHeader = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.20, 1.14), mats.darkIron);
      warnHeader.position.set(0.045, 1.54, 0);
      warnGrp.add(warnBoard, warnHeader);
      dppdGroup.add(warnGrp);
    }

    // =========================================================================
    // 2. GEDUNG DPPD UI (eks-KSDP) — 206-PHOTO GROUND-TRUTH 2-STORY HERITAGE PAVILION
    //    Facing Gedung Rektorat UI (0, 0) across Teluk U:
    //    - Two-tier sloped green grass berm ("Talud Rumput") & Batu-Kali rubble-stone toe retaining wall
    //    - NW Asymmetrical Salmon-Brick Triangular Gable End Wall w/ white frame & sloped cheek-wall (ref_144, ref_202)
    //    - SE Hipped Roof End + Watertight Central Steep A-Frame Gable w/ 11-Bay Ribbon Windows, Dish & AC
    //    - L1 Stepped-Tile Portico Canopy, Authentic Stair/Grey-Step/Twin-Curb ADA Ramp Sequence,
    //      UI–Huawei Brass Monolith Totem, Gold Box-Truss Banner Frames, Campus Map Plaque & 3 SE Royal Palms
    // =========================================================================
    const alphaMin = (DPPDComplexBuilder.DPPD_ALPHA_MIN_DEG * Math.PI) / 180.0;
    const alphaMax = (DPPDComplexBuilder.DPPD_ALPHA_MAX_DEG * Math.PI) / 180.0;
    // Note: in entranceGroup, x = r * sin(alpha), z = r * cos(alpha) -> polar angle in entranceGroup XZ is thE = pi/2 - alpha
    const dppdTh0 = Math.PI * 0.5 - alphaMax;
    const dppdTh1 = Math.PI * 0.5 - alphaMin;

    // 2A. Raised Granite & Terracotta Verandah Plinth (y = -0.35 .. 0.68)
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.granitePlinth, dppdTh0, dppdTh1, 28.4, 44.8, -0.35, 0.55, 44);
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.terracottaFloor, dppdTh0, dppdTh1, 28.5, 44.7, 0.55, 0.68, 44);

    // Ground-Truth Two-Tier Sloped Green Grass Berm ("Talud Rumput" 32° Slope) & Batu-Kali Toe Retaining Wall
    // flanking the central forecourt (alpha = -43.8° .. -13.2° on NW and +13.2° .. +43.8° on SE, ref_084, ref_184, ref_185!)
    for (const [degA, degB] of [
      [-43.8, -13.2],
      [13.2, 43.8]
    ]) {
      const tA = Math.PI * 0.5 - (degB * Math.PI) / 180.0;
      const tB = Math.PI * 0.5 - (degA * Math.PI) / 180.0;

      // Upper sloped grass berm (r = 44.35, y = 3.25 -> r = 47.05, y = 1.48)
      const upperBerm = new THREE.Mesh(
        DPPDComplexBuilder._buildPolarSlopedRibbonGeo(tA, tB, 44.35, 3.25, 47.05, 1.48, 22),
        mats.campusGrass
      );
      entranceGroup.add(upperBerm);

      // Mid-slope grey stone terrace curb step (r = 46.95 .. 47.25, y = 1.22 .. 1.54)
      DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.stoneRetaining, tA, tB, 46.95, 47.25, 1.22, 1.54, 22);

      // Lower sloped grass apron (r = 47.20, y = 1.34 -> r = 49.65, y = 0.72)
      const lowerBerm = new THREE.Mesh(
        DPPDComplexBuilder._buildPolarSlopedRibbonGeo(tA, tB, 47.20, 1.34, 49.65, 0.72, 22),
        mats.campusGrass
      );
      entranceGroup.add(lowerBerm);

      // Vertical Batu-Kali rubble-stone toe retaining wall (r = 49.50 .. 49.88, y = 0.04 .. 0.74) + coping cap
      DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.stoneRetaining, tA, tB, 49.50, 49.88, 0.04, 0.74, 22);
      DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.concreteTrim, tA, tB, 49.46, 49.92, 0.72, 0.80, 22);

      // Ornamental shrub clusters along the mid-berm terrace (ref_084)
      for (let sh = 0; sh < 6; sh++) {
        const thS = tA + ((sh + 0.5) / 6) * (tB - tA);
        const bush = new THREE.Mesh(new THREE.SphereGeometry(0.68, 8, 6), mats.LushShrub);
        bush.scale.set(1.22, 0.78, 1.05);
        bush.position.set(46.25 * Math.cos(thS), 2.15, 46.25 * Math.sin(thS));
        entranceGroup.add(bush);
      }
    }

    // Sloped White Concrete Berm Cheek-Walls at NW (-44°) & SE (+44°) Ends + Plastered Forecourt Wing Walls ("Pipi Tangga")
    for (const sgn of [-1, 1]) {
      const endA = sgn * alphaMax;
      const cheekGrp = new THREE.Group();
      cheekGrp.rotation.y = endA;
      const cheekGeo = DPPDComplexBuilder._buildPolygonPrismGeo(
        [
          [44.2, 0.15],
          [49.8, 0.15],
          [49.8, 0.78],
          [44.2, 3.35]
        ],
        0.34
      );
      const cheekMesh = new THREE.Mesh(cheekGeo, mats.whiteMullion);
      cheekMesh.rotation.y = Math.PI * 0.5;
      cheekMesh.position.x = -0.17;
      cheekGrp.add(cheekMesh);
      entranceGroup.add(cheekGrp);
    }

    // 2B. LANTAI 1 (y = 0.68 .. 6.25) & LANTAI 2 LOGGIA COLONNADE (y = 6.25 .. 11.75) Curved Masonry & Glazed Walls
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.creamWall, dppdTh0, dppdTh1, 29.0, 43.0, 0.68, 1.85, 44);
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.windowGlass, dppdTh0, dppdTh1, 29.15, 42.85, 1.85, 5.35, 44);
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.creamWall, dppdTh0, dppdTh1, 29.0, 43.2, 5.35, 6.10, 44);

    // Projecting L1/L2 Inter-Story White Spandrel Slab & Tropical Sunshade Drip Cornice (y = 6.10 .. 6.65)
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.whiteMullion, dppdTh0, dppdTh1, 28.55, 44.85, 6.10, 6.65, 48);

    // LANTAI 2 Recessed Loggia Wall, Mullioned Ribbon Windows, & Top Eave Ring Beam (y = 6.65 .. 11.75)
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.creamWall, dppdTh0, dppdTh1, 28.85, 43.65, 6.65, 7.65, 44);
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.windowGlass, dppdTh0, dppdTh1, 29.05, 43.45, 7.65, 10.85, 44);
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.whiteMullion, dppdTh0, dppdTh1, 28.55, 44.65, 10.85, 11.75, 48);

    // 2C. Rhythmic L1 & L2 Loggia Square Columns, Window Mullions, & 3D Split-AC Condenser Units w/ Dark Fan Grilles (ref_084, ref_184)
    const dppdBays = 22;
    for (let i = 0; i <= dppdBays; i++) {
      const alpha = alphaMin + (i / dppdBays) * (alphaMax - alphaMin);
      const deg = (alpha * 180.0) / Math.PI;
      const inCenterGable = Math.abs(deg) < 10.8;
      const sA = Math.sin(alpha);
      const cA = Math.cos(alpha);

      if (!inCenterGable) {
        // Outer L1 Nusantara Verandah Square Column + Stone Base
        const col = new THREE.Mesh(new THREE.BoxGeometry(0.58, 5.45, 0.58), mats.creamWall);
        col.position.set(44.25 * sA, 3.38, 44.25 * cA);
        col.rotation.y = alpha;
        entranceGroup.add(col);

        const colBase = new THREE.Mesh(new THREE.BoxGeometry(0.76, 0.55, 0.76), mats.granitePlinth);
        colBase.position.set(44.25 * sA, 0.95, 44.25 * cA);
        colBase.rotation.y = alpha;
        entranceGroup.add(colBase);

        // Outer L2 Recessed Loggia Square Cream Column (ref_084, ref_184)
        const colL2 = new THREE.Mesh(new THREE.BoxGeometry(0.46, 4.25, 0.46), mats.creamWall);
        colL2.position.set(44.25 * sA, 8.78, 44.25 * cA);
        colL2.rotation.y = alpha;
        entranceGroup.add(colL2);

        // Outer L1 & L2 Masonry Pier
        const pierL1 = new THREE.Mesh(new THREE.BoxGeometry(0.52, 4.65, 0.42), mats.creamWall);
        pierL1.position.set(42.92 * sA, 3.55, 42.92 * cA);
        pierL1.rotation.y = alpha;
        entranceGroup.add(pierL1);
      }

      // Inner Courtyard L1 & L2 Pilasters (facing ITTC Perpustakaan Lama!)
      const inPier = new THREE.Mesh(new THREE.BoxGeometry(0.54, 10.45, 0.46), mats.creamWall);
      inPier.position.set(28.92 * sA, 5.90, 28.92 * cA);
      inPier.rotation.y = alpha;
      entranceGroup.add(inPier);

      // Sub-mullions & 3D outdoor split-AC condenser units with dark circular fan grilles inside each bay
      if (i < dppdBays) {
        const aMid = alpha + (0.5 / dppdBays) * (alphaMax - alphaMin);
        const degMid = (aMid * 180.0) / Math.PI;
        const sm = Math.sin(aMid);
        const cm = Math.cos(aMid);

        if (Math.abs(degMid) >= 10.8) {
          for (const sub of [-0.18, 0.18]) {
            const aSub = aMid + sub * ((alphaMax - alphaMin) / dppdBays);
            const m1 = new THREE.Mesh(new THREE.BoxGeometry(0.14, 3.50, 0.22), mats.whiteMullion);
            m1.position.set(42.90 * Math.sin(aSub), 3.60, 42.90 * Math.cos(aSub));
            m1.rotation.y = aSub;
            entranceGroup.add(m1);

            const m2 = new THREE.Mesh(new THREE.BoxGeometry(0.14, 3.20, 0.24), mats.whiteMullion);
            m2.position.set(43.52 * Math.sin(aSub), 9.25, 43.52 * Math.cos(aSub));
            m2.rotation.y = aSub;
            entranceGroup.add(m2);
          }

          if (i % 2 === 0) {
            const acBox = new THREE.Mesh(new THREE.BoxGeometry(1.08, 0.74, 0.48), mats.whiteMullion);
            acBox.position.set(44.12 * sm, 7.12, 44.12 * cm);
            acBox.rotation.y = aMid;
            entranceGroup.add(acBox);

            const acGrille = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 0.52, 10), mats.darkAnodized);
            acGrille.rotation.x = Math.PI * 0.5;
            acGrille.position.set(44.14 * sm, 7.12, 44.14 * cm);
            acGrille.rotation.y = aMid;
            entranceGroup.add(acGrille);
          }
        }

        const mIn1 = new THREE.Mesh(new THREE.BoxGeometry(0.14, 3.50, 0.22), mats.whiteMullion);
        mIn1.position.set(29.08 * sm, 3.60, 29.08 * cm);
        mIn1.rotation.y = aMid;
        entranceGroup.add(mIn1);

        const mIn2 = new THREE.Mesh(new THREE.BoxGeometry(0.14, 3.20, 0.22), mats.whiteMullion);
        mIn2.position.set(29.08 * sm, 9.25, 29.08 * cm);
        mIn2.rotation.y = aMid;
        entranceGroup.add(mIn2);
      }
    }

    // NW Asymmetrical Salmon-Terracotta Brick Triangular Gable End Wall at alphaMin (-44°, ref_144, ref_202!)
    // + SE End-Wall Fenestration at alphaMax (+44°, ref_071, ref_079)
    {
      const nwGroup = new THREE.Group();
      nwGroup.rotation.y = alphaMin;
      entranceGroup.add(nwGroup);

      // Salmon-terracotta brick lower wall (r = 28.8 .. 44.6, y = 0.55 .. 11.75)
      const nwBrickWall = new THREE.Mesh(new THREE.BoxGeometry(0.42, 11.20, 15.8), mats.salmonBrick);
      nwBrickWall.position.set(-0.08, 6.15, 36.7);
      nwGroup.add(nwBrickWall);

      // White reinforced-concrete horizontal spandrel beams & 4 vertical pilasters framing the salmon brick (ref_144, ref_202)
      for (const by of [6.35, 11.55]) {
        const beam = new THREE.Mesh(new THREE.BoxGeometry(0.50, 0.48, 16.0), mats.whiteMullion);
        beam.position.set(-0.10, by, 36.7);
        nwGroup.add(beam);
      }
      for (let p = 0; p <= 3; p++) {
        const pz = 29.0 + (p / 3) * 15.4;
        const pil = new THREE.Mesh(new THREE.BoxGeometry(0.50, 11.20, 0.48), mats.whiteMullion);
        pil.position.set(-0.10, 6.15, pz);
        nwGroup.add(pil);
      }
      // High horizontal clerestory window slots & yellow MEP conduit pipes (ref_144, ref_202)
      for (const wz of [33.2, 40.2]) {
        const cw = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.95, 3.8), mats.windowGlass);
        cw.position.set(-0.10, 5.15, wz);
        nwGroup.add(cw);
        const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 11.0, 6), mats.uiYellowFascia);
        pipe.position.set(-0.28, 6.0, wz + 2.1);
        nwGroup.add(pipe);
      }

      // SE End Wall (+44°) clean fenestration
      const seGroup = new THREE.Group();
      seGroup.rotation.y = alphaMax;
      entranceGroup.add(seGroup);
      for (const rW of [33.0, 39.2]) {
        for (const yW of [3.55, 9.20]) {
          const frame = new THREE.Mesh(new THREE.BoxGeometry(0.24, 3.10, 3.80), mats.whiteMullion);
          frame.position.set(0.08, yW, rW);
          seGroup.add(frame);
          const pane = new THREE.Mesh(new THREE.BoxGeometry(0.28, 2.75, 3.45), mats.windowGlass);
          pane.position.set(0.10, yW, rW);
          seGroup.add(pane);
        }
      }
    }

    // Rear Courtyard Portal Canopy on Gedung DPPD (alpha = 0°, r = 26.2 .. 29.0) facing ITTC Perpustakaan Lama
    const rearCanopy = new THREE.Mesh(new THREE.BoxGeometry(7.6, 0.42, 3.2), mats.whiteMullion);
    rearCanopy.position.set(0, 5.85, 27.4);
    entranceGroup.add(rearCanopy);
    for (const sx of [-3.1, 3.1]) {
      const rCol = new THREE.Mesh(new THREE.BoxGeometry(0.48, 5.20, 0.48), mats.whiteMullion);
      rCol.position.set(sx, 3.25, 26.2);
      entranceGroup.add(rCol);
    }

    // 2D. Proportional 2-Story Nusantara Terracotta Hipped Roof, Fascia, & Ridge Tiles (Bubungan)
    const wingRoofGeo = DPPDComplexBuilder._sliceSubdomain(
      (u, v, p, n) => DPPDComplexBuilder.evaluateWingRoofLocalSurface(u, v, p, n),
      0, 1, 0, 1, 60, 18
    );
    entranceGroup.add(new THREE.Mesh(wingRoofGeo, mats.terracottaRoof));

    // Soffit & dark-teak eave fascia along outer (r = 46.0) & inner (r = 27.2) eaves
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.darkWoodFascia, dppdTh0, dppdTh1, 45.72, 46.04, 11.42, 11.82, 48);
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.darkWoodFascia, dppdTh0, dppdTh1, 27.16, 27.48, 11.42, 11.82, 48);
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.whiteMullion, dppdTh0, dppdTh1, 27.35, 45.85, 11.38, 11.45, 48, {
      top: false,
      bottom: true,
      outer: false,
      inner: false,
      caps: false
    });

    const pStart = new THREE.Vector3();
    const pEnd = new THREE.Vector3();
    const dummyN = new THREE.Vector3();
    for (let s = 0; s < 6; s++) {
      const u0 = s === 0 ? 0.14 : s / 6;
      const u1 = s === 5 ? 0.86 : (s + 1) / 6;
      DPPDComplexBuilder.evaluateWingRoofLocalSurface(u0, 0.5, pStart, dummyN);
      DPPDComplexBuilder.evaluateWingRoofLocalSurface(u1, 0.5, pEnd, dummyN);
      const ridgeLen = pStart.distanceTo(pEnd);
      const ridgeMesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.46, 0.34, ridgeLen + 0.25),
        mats.terracottaRidge
      );
      DPPDComplexBuilder._orientBeamAlongZ(ridgeMesh, pStart, pEnd, 0.12);
      entranceGroup.add(ridgeMesh);

      if (s === 1 || s === 4) {
        const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.06, 1.8, 6), mats.galvSteel);
        rod.position.set(pStart.x, pStart.y + 0.95, pStart.z);
        entranceGroup.add(rod);
      }
    }
    for (const [uCorner, vCorner] of [
      [0.002, 0.0],
      [0.002, 1.0],
      [0.998, 0.0],
      [0.998, 1.0]
    ]) {
      const uTip = uCorner < 0.5 ? 0.14 : 0.86;
      DPPDComplexBuilder.evaluateWingRoofLocalSurface(uCorner, vCorner, pStart, dummyN);
      DPPDComplexBuilder.evaluateWingRoofLocalSurface(uTip, 0.5, pEnd, dummyN);
      const hipBeam = new THREE.Mesh(
        new THREE.BoxGeometry(0.40, 0.30, pStart.distanceTo(pEnd) + 0.2),
        mats.terracottaRidge
      );
      DPPDComplexBuilder._orientBeamAlongZ(hipBeam, pStart, pEnd, 0.10);
      entranceGroup.add(hipBeam);
    }

    // 2E. Central 2-Story Front Nusantara Gable Pavilion (11-Bay Ribbon Windows + Mini Satellite Dish + Split-AC),
    //     L1 Stepped-Tile Portico Canopy, Authentic Stair/Grey-Step/Twin-Curb ADA Ramp Sequence,
    //     UI–Huawei Brass Monolith Totem, Gold Box-Truss Banner Frames, & Campus Map Plaque (ref_084, ref_184, ref_185!)
    const gHalfW = 8.4;
    const gZBack = 35.8;
    const gZFront = 45.4;
    const gEaveY = 11.75;
    const gPeakY = 18.05;

    // 2-Story Front Central Projection Block
    const centerBlock = new THREE.Mesh(
      new THREE.BoxGeometry(gHalfW * 2.0, 11.05, gZFront - gZBack),
      mats.creamWall
    );
    centerBlock.position.set(0, 6.20, (gZBack + gZFront) * 0.5);
    entranceGroup.add(centerBlock);

    const centerSpandrel = new THREE.Mesh(
      new THREE.BoxGeometry(gHalfW * 2.0 + 0.8, 0.56, gZFront - gZBack + 0.9),
      mats.whiteMullion
    );
    centerSpandrel.position.set(0, 6.38, (gZBack + gZFront) * 0.5 + 0.15);
    entranceGroup.add(centerSpandrel);

    const centerTopBeam = new THREE.Mesh(
      new THREE.BoxGeometry(gHalfW * 2.0 + 0.8, 0.78, gZFront - gZBack + 0.9),
      mats.whiteMullion
    );
    centerTopBeam.position.set(0, 11.36, (gZBack + gZFront) * 0.5 + 0.15);
    entranceGroup.add(centerTopBeam);

    // Front L1 Grand Glazed Entrance Lobby & L2 11-Bay Dark-Framed Ribbon Windows w/ Horizontal Transom Bar (ref_184!)
    const l1LobbyGlass = new THREE.Mesh(new THREE.BoxGeometry(11.6, 4.55, 0.30), mats.windowGlass);
    l1LobbyGlass.position.set(0, 3.15, gZFront + 0.04);
    entranceGroup.add(l1LobbyGlass);

    const l2RibbonGlass = new THREE.Mesh(new THREE.BoxGeometry(14.2, 3.25, 0.30), mats.windowGlass);
    l2RibbonGlass.position.set(0, 9.22, gZFront + 0.04);
    entranceGroup.add(l2RibbonGlass);

    // Horizontal mid-height transom bar across the 11-bay L2 ribbon windows (ref_184)
    const l2Transom = new THREE.Mesh(new THREE.BoxGeometry(14.3, 0.12, 0.36), mats.darkAnodized);
    l2Transom.position.set(0, 9.35, gZFront + 0.06);
    entranceGroup.add(l2Transom);

    // 12 vertical mullions forming the 11-bay L2 ribbon window
    for (let m = 0; m <= 11; m++) {
      const mx = -7.1 + (m / 11) * 14.2;
      const mul2 = new THREE.Mesh(new THREE.BoxGeometry(0.14, 3.35, 0.36), mats.darkAnodized);
      mul2.position.set(mx, 9.22, gZFront + 0.07);
      entranceGroup.add(mul2);
    }
    for (let m = -2; m <= 2; m++) {
      const mul1 = new THREE.Mesh(new THREE.BoxGeometry(0.22, 4.65, 0.36), mats.whiteMullion);
      mul1.position.set(m * 2.85, 3.15, gZFront + 0.07);
      entranceGroup.add(mul1);
    }

    // White Mini-Satellite Dish on Left L2 Pier & White Split-AC Condenser on Right L2 Pier (ref_184!)
    const satDish = new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.14, 0.14, 10), mats.whiteMullion);
    satDish.rotation.x = Math.PI * 0.42;
    satDish.position.set(-7.65, 9.45, gZFront + 0.28);
    entranceGroup.add(satDish);

    const rightPierAc = new THREE.Mesh(new THREE.BoxGeometry(0.92, 0.66, 0.42), mats.whiteMullion);
    rightPierAc.position.set(7.65, 7.65, gZFront + 0.24);
    const rightPierFan = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.45, 10), mats.darkAnodized);
    rightPierFan.rotation.x = Math.PI * 0.5;
    rightPierFan.position.set(7.65, 7.65, gZFront + 0.25);
    entranceGroup.add(rightPierAc, rightPierFan);

    // Eye-level Wall Signage flanking L1 Entrance Doors ("GEDUNG KSDP UI" right & "GEDUNG DPPD UI" left, ref_184!)
    const ksdpBoard = new THREE.Mesh(new THREE.BoxGeometry(2.35, 0.68, 0.12), mats.whiteMullion);
    ksdpBoard.position.set(3.85, 2.55, gZFront + 0.18);
    entranceGroup.add(ksdpBoard);
    DPPDComplexBuilder._addPlanar3DText(entranceGroup, 'KSDP UI', 3.85, 2.38, gZFront + 0.26, 0.34, 0.08, mats.darkAnodized);

    const dppdBoard = new THREE.Mesh(new THREE.BoxGeometry(2.35, 0.68, 0.12), mats.darkAnodized);
    dppdBoard.position.set(-3.85, 2.55, gZFront + 0.18);
    entranceGroup.add(dppdBoard);
    DPPDComplexBuilder._addPlanar3DText(entranceGroup, 'DPPD UI', -3.85, 2.38, gZFront + 0.26, 0.34, 0.08, mats.brassSign);

    // Proportional Triangular Gable Pediment + Overhanging Sloped Terracotta Roof Planes + Timber Bargeboards
    const gableWallGeo = DPPDComplexBuilder._buildPolygonPrismGeo(
      [
        [-gHalfW, gEaveY],
        [gHalfW, gEaveY],
        [0.0, gPeakY - 0.35]
      ],
      gZFront - gZBack
    );
    const gableWall = new THREE.Mesh(gableWallGeo, mats.creamWall);
    gableWall.position.z = gZBack;
    entranceGroup.add(gableWall);

    // Recessed Pediment Accent Frame + Golden UI Makara Emblem + 3D "GEDUNG DPPD UI" Signage!
    const pedimentInsetGeo = DPPDComplexBuilder._buildPolygonPrismGeo(
      [
        [-gHalfW + 1.45, gEaveY + 0.45],
        [gHalfW - 1.45, gEaveY + 0.45],
        [0.0, gPeakY - 1.25]
      ],
      0.22
    );
    const pedimentInset = new THREE.Mesh(pedimentInsetGeo, mats.warmStonePanel);
    pedimentInset.position.z = gZFront - 0.04;
    entranceGroup.add(pedimentInset);

    const dppdMakara = DPPDComplexBuilder._buildUIMakaraEmblem(mats, 1.32);
    dppdMakara.position.set(0, 14.95, gZFront + 0.22);
    entranceGroup.add(dppdMakara);

    const dppdSignPlate = new THREE.Mesh(new THREE.BoxGeometry(11.4, 1.15, 0.20), mats.darkAnodized);
    dppdSignPlate.position.set(0, 12.65, gZFront + 0.18);
    entranceGroup.add(dppdSignPlate);
    DPPDComplexBuilder._addPlanar3DText(
      entranceGroup,
      'GEDUNG DPPD UI',
      0.0,
      12.32,
      gZFront + 0.32,
      0.66,
      0.16,
      mats.brassSign
    );

    const roofOverZ = 1.45;
    const roofOverX = 1.15;
    const slopeLen = Math.hypot(gHalfW + roofOverX, gPeakY - gEaveY + 0.45);
    const slopeAngle = Math.atan2(gPeakY - gEaveY, gHalfW);
    for (const sgn of [-1, 1]) {
      const roofSlope = new THREE.Mesh(
        new THREE.BoxGeometry(slopeLen, 0.32, gZFront - gZBack + roofOverZ * 2.0),
        mats.terracottaRoof
      );
      roofSlope.position.set(
        sgn * (gHalfW + roofOverX) * 0.48,
        (gEaveY + gPeakY) * 0.5 - 0.05,
        (gZBack + gZFront) * 0.5 + 0.35
      );
      roofSlope.rotation.z = -sgn * slopeAngle;
      entranceGroup.add(roofSlope);

      const barge = new THREE.Mesh(new THREE.BoxGeometry(slopeLen + 0.1, 0.44, 0.24), mats.darkWoodFascia);
      barge.position.set(
        sgn * (gHalfW + roofOverX) * 0.48,
        (gEaveY + gPeakY) * 0.5 - 0.12,
        gZFront + roofOverZ + 0.25
      );
      barge.rotation.z = -sgn * slopeAngle;
      entranceGroup.add(barge);
    }
    const mainGableRidge = new THREE.Mesh(
      new THREE.BoxGeometry(0.52, 0.40, gZFront - gZBack + roofOverZ * 2.0 + 0.3),
      mats.terracottaRidge
    );
    mainGableRidge.position.set(0, gPeakY + 0.14, (gZBack + gZFront) * 0.5 + 0.35);
    entranceGroup.add(mainGableRidge);

    // 1-Story Continuous Stepped-Tile Terracotta Portico Canopy, 4 Square Cream Columns, Plastered Wing Walls ("Pipi Tangga"),
    // & Authentic Left-to-Right Stair / Central Grey Stone Steps / Twin-Curb ADA Ramp Sequence (ref_184, ref_185!)
    const porchPodium = new THREE.Mesh(new THREE.BoxGeometry(16.8, 0.58, 3.8), mats.terracottaFloor);
    porchPodium.position.set(0, 0.30, 47.2);
    entranceGroup.add(porchPodium);

    // Forecourt weathered grey paving apron & driveway bridge with open radial concrete U-gutter ("Selokan Beton", ref_184)
    const forecourtApron = new THREE.Mesh(new THREE.BoxGeometry(17.2, 0.14, 3.4), mats.sidewalkPaver);
    forecourtApron.position.set(0, 0.09, 50.6);
    entranceGroup.add(forecourtApron);

    const dppdDriveBridge = new THREE.Mesh(new THREE.BoxGeometry(6.8, 0.16, 2.6), mats.concreteCurb);
    dppdDriveBridge.position.set(0.35, 0.10, 51.85);
    entranceGroup.add(dppdDriveBridge);

    // Open Radial Concrete U-Gutter ("Selokan Beton") with twin raised white concrete lips at left edge of bridge (ref_184)
    for (const dx of [-0.16, 0.16]) {
      const gLip = new THREE.Mesh(new THREE.BoxGeometry(0.10, 0.22, 2.5), mats.whiteMullion);
      gLip.position.set(-3.35 + dx, 0.15, 51.85);
      entranceGroup.add(gLip);
    }

    // Plastered masonry forecourt wing walls ("Pipi Tangga") with stone coping along x = ±8.55m (ref_184)
    for (const sgn of [-1, 1]) {
      const pipi = new THREE.Mesh(new THREE.BoxGeometry(0.34, 1.35, 4.8), mats.creamWall);
      pipi.position.set(sgn * 8.55, 0.72, 47.8);
      pipi.rotation.x = 0.18;
      const pipiCap = new THREE.Mesh(new THREE.BoxGeometry(0.40, 0.10, 4.9), mats.stoneRetaining);
      pipiCap.position.set(sgn * 8.55, 1.42, 47.8);
      pipiCap.rotation.x = 0.18;
      entranceGroup.add(pipi, pipiCap);
    }

    // Authentic Left-to-Right Stair, Central Grey Stone Steps, & Twin-Curb ADA Ramp Sequence (z = 49.0 .. 50.4, ref_184!):
    // A. Left Wide Terracotta Stairs (x = -8.3 .. -1.6)
    for (let st = 0; st < 4; st++) {
      const sh = 0.12 * (4 - st);
      const stepL = new THREE.Mesh(new THREE.BoxGeometry(6.6, sh, 0.36), mats.terracottaFloor);
      stepL.position.set(-4.95, 0.08 + sh * 0.5, 49.15 + st * 0.34);
      entranceGroup.add(stepL);
    }
    // B. Central Narrow Grey Stone Steps (x = -1.5 .. -0.3, ref_184 center!)
    for (let st = 0; st < 4; st++) {
      const sh = 0.12 * (4 - st);
      const stepC = new THREE.Mesh(new THREE.BoxGeometry(1.18, sh, 0.36), mats.stoneRetaining);
      stepC.position.set(-0.90, 0.08 + sh * 0.5, 49.15 + st * 0.34);
      entranceGroup.add(stepC);
    }
    // C. Central Gently Sloped Terracotta Approach Deck (x = -0.25 .. +2.65)
    const centerApproach = new THREE.Mesh(new THREE.BoxGeometry(2.85, 0.14, 1.45), mats.terracottaFloor);
    centerApproach.position.set(1.20, 0.30, 49.65);
    centerApproach.rotation.x = 0.18;
    entranceGroup.add(centerApproach);
    // D. Twin Parallel Sloped Red-Brick Ramp Walls w/ White Coping framing the ADA Wheelchair Ramp (x = +2.75 .. +4.15, ref_184!)
    for (const rx of [2.75, 4.15]) {
      const rCurb = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.32, 1.55), mats.salmonBrick);
      rCurb.position.set(rx, 0.38, 49.65);
      rCurb.rotation.x = 0.18;
      const rCap = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.06, 1.58), mats.whiteMullion);
      rCap.position.set(rx, 0.56, 49.65);
      rCap.rotation.x = 0.18;
      entranceGroup.add(rCurb, rCap);
    }
    const adaRamp = new THREE.Mesh(new THREE.BoxGeometry(1.22, 0.14, 1.50), mats.terracottaFloor);
    adaRamp.position.set(3.45, 0.30, 49.65);
    adaRamp.rotation.x = 0.18;
    entranceGroup.add(adaRamp);
    // E. Right Wide Terracotta Stairs (x = +4.30 .. +8.30)
    for (let st = 0; st < 4; st++) {
      const sh = 0.12 * (4 - st);
      const stepR = new THREE.Mesh(new THREE.BoxGeometry(3.95, sh, 0.36), mats.terracottaFloor);
      stepR.position.set(6.30, 0.08 + sh * 0.5, 49.15 + st * 0.34);
      entranceGroup.add(stepR);
    }

    // 4 Square Cream Portico Columns supporting the L1 Stepped-Tile Terracotta Canopy (ref_184)
    for (const px of [-6.6, -1.45, 1.45, 6.6]) {
      const pCol = new THREE.Mesh(new THREE.BoxGeometry(0.42, 4.85, 0.42), mats.creamWall);
      pCol.position.set(px, 3.02, 48.85);
      entranceGroup.add(pCol);
      const pBase = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.48, 0.58), mats.granitePlinth);
      pBase.position.set(px, 0.82, 48.85);
      entranceGroup.add(pBase);
    }

    // L1 Continuous Sloped Terracotta Portico Canopy w/ 6 Stepped Tile Courses & 28 Sculpted Eave Drip-Tile Nosings (ref_184)
    const canW = 16.8;
    const canSoffit = new THREE.Mesh(new THREE.BoxGeometry(canW, 0.18, 3.8), mats.creamWall);
    canSoffit.position.set(0, 5.42, 47.35);
    entranceGroup.add(canSoffit);

    const canRoof = new THREE.Mesh(new THREE.BoxGeometry(canW + 0.2, 0.18, 4.0), mats.terracottaRoof);
    canRoof.position.set(0, 5.72, 47.35);
    canRoof.rotation.x = 0.14;
    entranceGroup.add(canRoof);

    for (let k = 0; k < 6; k++) {
      const tk = k / 6;
      const courseBar = new THREE.Mesh(new THREE.BoxGeometry(canW + 0.24, 0.05, 0.68), mats.terracottaRoof);
      courseBar.position.set(0, 5.52 + tk * 0.46, 49.15 - tk * 3.6);
      courseBar.rotation.x = 0.14;
      entranceGroup.add(courseBar);
    }
    for (let i = 0; i < 28; i++) {
      const tx = -canW * 0.5 + 0.30 + (i / 27) * (canW - 0.60);
      const dripTile = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.08, 0.22), mats.terracottaRidge);
      dripTile.position.set(tx, 5.48, 49.36);
      dripTile.rotation.x = 0.18;
      entranceGroup.add(dripTile);
    }

    // Photocell warm portico downlights (v = -890.0)
    for (const lx of [-4.8, -1.6, 1.6, 4.8]) {
      const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.08, 0.48), mats.warmLuminaire);
      lamp.position.set(lx, 5.31, 47.5);
      entranceGroup.add(lamp);
    }

    // Left Forecourt: Sky-Blue ADA Wheelchair Stall + Orange Cone + Stone Pedestal + UI–Huawei Brass Monolith Totem (ref_184!)
    const blueStall = new THREE.Mesh(new THREE.BoxGeometry(3.8, 0.05, 2.2), mats.disabilityBlue);
    blueStall.position.set(-5.4, 0.16, 50.85);
    entranceGroup.add(blueStall);

    const huaweiPlinth = new THREE.Mesh(new THREE.BoxGeometry(1.18, 0.26, 0.38), mats.darkAnodized);
    huaweiPlinth.position.set(-6.15, 0.26, 51.65);
    const huaweiBody = new THREE.Mesh(new THREE.BoxGeometry(1.08, 2.15, 0.22), mats.brassSign);
    huaweiBody.position.set(-6.15, 1.46, 51.65);
    entranceGroup.add(huaweiPlinth, huaweiBody);
    DPPDComplexBuilder._addPlanar3DText(entranceGroup, 'HUAWEI', -6.15, 1.72, 51.78, 0.26, 0.06, mats.darkAnodized);
    DPPDComplexBuilder._addPlanar3DText(entranceGroup, 'UI', -6.15, 1.22, 51.78, 0.28, 0.06, mats.darkAnodized);

    // Right Forecourt: Twin Gold Box-Truss Banner Frames (perpendicular to facade) + Stainless Campus Map Plaque (ref_084, ref_184!)
    for (const bx of [4.45, 8.15]) {
      const tPad = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.22, 0.62), mats.darkAnodized);
      tPad.position.set(bx, 0.22, 51.25);
      entranceGroup.add(tPad);
      for (const lz of [-0.20, 0.20]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.06, 4.6, 0.06), mats.brassSign);
        leg.position.set(bx, 2.52, 51.25 + lz);
        entranceGroup.add(leg);
      }
      for (let r = 0; r < 8; r++) {
        const rung = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 0.44), mats.brassSign);
        rung.position.set(bx, 0.65 + r * 0.56, 51.25);
        entranceGroup.add(rung);
      }
    }
    const mapPlinth = new THREE.Mesh(new THREE.BoxGeometry(0.72, 0.22, 0.50), mats.darkAnodized);
    mapPlinth.position.set(6.30, 0.22, 51.15);
    const mapPost = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.10, 1.15, 8), mats.silverLouver);
    mapPost.position.set(6.30, 0.82, 51.15);
    const mapFrame = new THREE.Mesh(new THREE.BoxGeometry(1.42, 1.02, 0.14), mats.silverLouver);
    mapFrame.position.set(6.30, 1.75, 51.15);
    const mapBoard = new THREE.Mesh(new THREE.BoxGeometry(1.24, 0.84, 0.16), mats.darkFriezeGlass);
    mapBoard.position.set(6.30, 1.75, 51.16);
    entranceGroup.add(mapPlinth, mapPost, mapFrame, mapBoard);

    // Ceremonial Merah-Putih Flagpole on 2-Tier Concrete Drum Base & Framing Royal Palms (ref_071, ref_184)
    const flagPedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.82, 0.72, 12), mats.granitePlinth);
    flagPedestal.position.set(-9.85, 0.36, 49.6);
    entranceGroup.add(flagPedestal);
    const flagPole = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.13, 11.2, 8), mats.whiteMullion);
    flagPole.position.set(-9.85, 6.15, 49.6);
    entranceGroup.add(flagPole);
    const flagRed = new THREE.Mesh(new THREE.BoxGeometry(2.05, 0.62, 0.06), mats.hydrantRed);
    flagRed.position.set(-8.78, 11.25, 49.6);
    entranceGroup.add(flagRed);
    const flagWhite = new THREE.Mesh(new THREE.BoxGeometry(2.05, 0.62, 0.06), mats.whiteMullion);
    flagWhite.position.set(-8.78, 10.63, 49.6);
    entranceGroup.add(flagWhite);

    entranceGroup.add(DPPDComplexBuilder._buildRoyalPalm(mats, -13.6, 0.0, 48.0, 11.4));
    entranceGroup.add(DPPDComplexBuilder._buildRoyalPalm(mats, 13.6, 0.0, 48.0, 11.4));
    // 3rd SE Royal Palm along the SE corner lawn (ref_071)
    entranceGroup.add(DPPDComplexBuilder._buildRoyalPalm(mats, 21.5, 0.0, 45.2, 10.8));

    // =========================================================================
    // 3. GEDUNG ILRC & DRPM UI — 6 STORIES WITH 206-PHOTO GROUND-TRUTH FIDELITY
    //    Modeled directly after real Gedung ILRC & DRPM UI Depok photos (ref_025..ref_032, ref_081..ref_090):
    //    - 100.0% continuous, razor-level horizontal top white crown fascia & parapet (y = 36.25 from end to end)
    //    - 6 continuous curved precast-white horizontal spandrel & sunshade bands + projecting 3D silver aluminum louver blades
    //    - L4-L5 cantilevered 3-sided glass observation box on the East Wing (ref_026)
    //    - East Drop-Off Portico Canopy (+4.5°) + DRPM L1 Open Breezeway, 4 Manila Palms & Open Black Iron Picket Gate (+24°)
    //    - Central 6-story symmetrical terracotta-brick twin-pier glazed atrium portal (+35°) with warm taupe-terracotta soffit,
    //      3 open-air white horizontal corridor bridges at L3/L4/L5, KAN accreditation emblem, koral sikat pebble-grid ramp & open iron gate
    //    - South Wing bamboo-staked brown dry-leaf mound (+44°..+64°) & South-end burnt-brick tower with cantilevered SE sunshade fins
    //      and 3 stacked protruding hooded window boxes (+74°, ref_088-ref_090)
    // =========================================================================
    const ilrcTh0 = (DPPDComplexBuilder.ILRC_START_DEG * Math.PI) / 180.0;
    const ilrcTh1 = (DPPDComplexBuilder.ILRC_END_DEG * Math.PI) / 180.0;
    const ilrcCenterTh = 0.5 * (ilrcTh0 + ilrcTh1); // +35°
    const ilrcRIn = 31.4;
    const ilrcROut = 43.4;
    const ilrcMidR = 37.4;
    const ilrcFloorTops = [6.65, 12.15, 17.65, 23.15, 28.65, 34.15]; // Exact 6 stories!

    const pierHalfAng = 0.032;
    const portalHalfAng = 0.112;
    const rPierTh0 = ilrcCenterTh - portalHalfAng - pierHalfAng;
    const rPierTh1 = ilrcCenterTh - portalHalfAng + pierHalfAng;
    const lPierTh0 = ilrcCenterTh + portalHalfAng - pierHalfAng;
    const lPierTh1 = ilrcCenterTh + portalHalfAng + pierHalfAng;

    // Plinth, Ground Approach Apron, & L1 Glazed Research Commons Core
    DPPDComplexBuilder._addCurvedBand(contextGroup, mats.granitePlinth, ilrcTh0, ilrcTh1, ilrcRIn - 0.3, ilrcROut + 0.3, -0.35, 0.48, 48);
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.courtyardGranite,
      ilrcCenterTh - 0.17,
      ilrcCenterTh + 0.17,
      ilrcROut,
      50.85,
      0.02,
      0.16,
      16
    );
    DPPDComplexBuilder._addCurvedBand(contextGroup, mats.windowGlass, ilrcTh0, ilrcTh1, ilrcRIn + 0.45, ilrcROut - 0.55, 0.48, 5.80, 48);

    // Build all 6 clearly countable floor slabs, spandrel bands, projecting sunshade ledges & 3D silver aluminum louver blades (L1..L6)!
    for (let fl = 0; fl < 6; fl++) {
      const yTop = ilrcFloorTops[fl];
      const yBot = fl === 0 ? 0.48 : ilrcFloorTops[fl - 1];

      if (fl >= 1) {
        // Upper floors L2..L6: lower concrete sill wall (inner full-arc + outer right/left wings flanking central glass atrium)
        DPPDComplexBuilder._addCurvedBand(contextGroup, mats.ilrcConcrete, ilrcTh0, ilrcTh1, ilrcRIn, ilrcMidR, yBot, yBot + 1.35, 48);
        DPPDComplexBuilder._addCurvedBand(contextGroup, mats.ilrcConcrete, ilrcTh0, rPierTh0, ilrcMidR, ilrcROut, yBot, yBot + 1.35, 22);
        DPPDComplexBuilder._addCurvedBand(contextGroup, mats.ilrcConcrete, lPierTh1, ilrcTh1, ilrcMidR, ilrcROut, yBot, yBot + 1.35, 22);

        // Continuous dark-glazed laboratory ribbon windows across all bays
        DPPDComplexBuilder._addCurvedBand(
          contextGroup,
          mats.windowGlass,
          ilrcTh0,
          ilrcTh1,
          ilrcRIn + 0.22,
          ilrcROut - 0.22,
          yBot + 1.35,
          yTop - 0.85,
          48
        );

        // Projecting 3D silver-aluminum horizontal sunshade louver blades above L2..L6 ribbon windows (ref_025, ref_030, ref_085)
        for (const lvOff of [-1.28, -1.04]) {
          DPPDComplexBuilder._addCurvedBand(
            contextGroup,
            mats.silverLouver,
            ilrcTh0 + 0.015,
            rPierTh0 - 0.01,
            ilrcROut + 0.05,
            ilrcROut + 0.66,
            yTop + lvOff,
            yTop + lvOff + 0.08,
            20
          );
          DPPDComplexBuilder._addCurvedBand(
            contextGroup,
            mats.silverLouver,
            lPierTh1 + 0.01,
            ilrcTh1 - 0.015,
            ilrcROut + 0.05,
            ilrcROut + 0.66,
            yTop + lvOff,
            yTop + lvOff + 0.08,
            20
          );
        }
      }

      // Inner courtyard projecting sunshade ledge (continuous full arc)
      DPPDComplexBuilder._addCurvedBand(
        contextGroup,
        mats.facadeWhite,
        ilrcTh0,
        ilrcTh1,
        ilrcRIn - 0.52,
        ilrcMidR,
        yTop - 0.85,
        yTop,
        48
      );
      // Outer right & left wing projecting curved sunshade ledges (flanking the twin terracotta piers cleanly)
      DPPDComplexBuilder._addCurvedBand(
        contextGroup,
        mats.facadeWhite,
        ilrcTh0,
        rPierTh0,
        ilrcMidR,
        ilrcROut + 0.58,
        yTop - 0.85,
        yTop,
        24
      );
      DPPDComplexBuilder._addCurvedBand(
        contextGroup,
        mats.facadeWhite,
        lPierTh1,
        ilrcTh1,
        ilrcMidR,
        ilrcROut + 0.58,
        yTop - 0.85,
        yTop,
        24
      );

      // Subtle recessed dark-anodized horizontal curtain-wall transom inside central glass atrium (L1..L5)
      if (fl < 5) {
        DPPDComplexBuilder._addCurvedBand(
          contextGroup,
          mats.darkAnodized,
          rPierTh1,
          lPierTh0,
          ilrcROut - 0.52,
          ilrcROut + 0.10,
          yTop - 0.36,
          yTop - 0.06,
          10
        );
      }
    }

    // L4-L5 Cantilevered 3-Sided Glass Observation Box on East Wing (+7°..+14°, ref_026)
    {
      const cTh0 = (7.0 * Math.PI) / 180.0;
      const cTh1 = (14.0 * Math.PI) / 180.0;
      DPPDComplexBuilder._addCurvedBand(contextGroup, mats.facadeWhite, cTh0, cTh1, ilrcROut, ilrcROut + 1.65, 17.35, 17.85, 8);
      DPPDComplexBuilder._addCurvedBand(contextGroup, mats.facadeWhite, cTh0, cTh1, ilrcROut, ilrcROut + 1.65, 28.25, 28.75, 8);
      DPPDComplexBuilder._addCurvedBand(contextGroup, mats.windowGlass, cTh0 + 0.004, cTh1 - 0.004, ilrcROut + 0.15, ilrcROut + 1.52, 17.85, 28.25, 8);
      DPPDComplexBuilder._addCurvedBand(contextGroup, mats.darkAnodized, cTh0, cTh1, ilrcROut + 1.48, ilrcROut + 1.62, 23.00, 23.30, 8);
    }

    // 100.0% FLAT, CONTINUOUS, UNBROKEN FULL-ARC TOP ROOF CROWN FASCIA & COPING (y = 34.15 .. 36.25 across ENTIRE ilrcTh0..ilrcTh1!)
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.facadeWhite,
      ilrcTh0,
      ilrcTh1,
      ilrcRIn - 0.56,
      ilrcROut + 0.86,
      34.15,
      36.05,
      56
    );
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.whiteMullion,
      ilrcTh0,
      ilrcTh1,
      ilrcRIn - 0.64,
      ilrcROut + 0.94,
      36.05,
      36.25,
      56
    );
    // Flat recessed concrete roof membrane deck tucked cleanly inside the continuous 36.25 perimeter parapet coping
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.ilrcConcrete,
      ilrcTh0 + 0.006,
      ilrcTh1 - 0.006,
      ilrcRIn + 0.15,
      ilrcROut - 0.15,
      35.45,
      35.88,
      48
    );

    // Rhythmic L1 Structural Piloti Columns + L1..L6 Vertical Aluminum Sun-Breaker Louvers & Mullions (Outer + Inner!)
    const ilrcBays = 24;
    for (let i = 0; i <= ilrcBays; i++) {
      const th = ilrcTh0 + (i / ilrcBays) * (ilrcTh1 - ilrcTh0);
      const inPortal = th > rPierTh0 - 0.008 && th < lPierTh1 + 0.008;
      const cT = Math.cos(th);
      const sT = Math.sin(th);

      if (!inPortal) {
        // Outer L1 piloti round column + L2..L6 vertical sun-breaker fin (terminating cleanly at y = 34.15 under flat crown!)
        const piloti = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.40, 5.45, 12), mats.facadeWhite);
        piloti.position.set((ilrcROut - 0.15) * cT, 3.15, (ilrcROut - 0.15) * sT);
        contextGroup.add(piloti);

        const vFin = new THREE.Mesh(new THREE.BoxGeometry(0.55, 27.5, 0.18), mats.facadeWhite);
        vFin.position.set((ilrcROut + 0.18) * cT, 20.40, (ilrcROut + 0.18) * sT);
        vFin.rotation.y = -th;
        contextGroup.add(vFin);
      }

      // Inner courtyard L1 piloti + L2..L6 vertical mullion fin (zero blank inner wall!)
      const inPiloti = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.36, 5.45, 10), mats.facadeWhite);
      inPiloti.position.set((ilrcRIn + 0.15) * cT, 3.15, (ilrcRIn + 0.15) * sT);
      contextGroup.add(inPiloti);

      const inFin = new THREE.Mesh(new THREE.BoxGeometry(0.45, 27.5, 0.16), mats.facadeWhite);
      inFin.position.set((ilrcRIn - 0.14) * cT, 20.40, (ilrcRIn - 0.14) * sT);
      inFin.rotation.y = -th;
      contextGroup.add(inFin);

      if (i < ilrcBays) {
        const thMid = th + (0.5 / ilrcBays) * (ilrcTh1 - ilrcTh0);
        const midInPortal = thMid > rPierTh0 - 0.008 && thMid < lPierTh1 + 0.008;
        if (!midInPortal) {
          const mOut = new THREE.Mesh(new THREE.BoxGeometry(0.24, 33.0, 0.12), mats.darkAnodized);
          mOut.position.set((ilrcROut - 0.18) * Math.cos(thMid), 17.3, (ilrcROut - 0.18) * Math.sin(thMid));
          mOut.rotation.y = -thMid;
          contextGroup.add(mOut);
        }

        const mIn = new THREE.Mesh(new THREE.BoxGeometry(0.24, 33.0, 0.12), mats.darkAnodized);
        mIn.position.set((ilrcRIn + 0.18) * Math.cos(thMid), 17.3, (ilrcRIn + 0.18) * Math.sin(thMid));
        mIn.rotation.y = -thMid;
        contextGroup.add(mIn);
      }
    }

    // Both ILRC End Walls (ilrcTh0 & ilrcTh1):
    // - North-East End (ilrcTh0 = -4°): Precast-white stair-core frame + Ground-level grey metal Gardu utility box (ref_025)
    // - South End (ilrcTh1 = +74°): Tall burnt-brick end tower with cantilevered SE horizontal sunshade slab fins & 3 stacked protruding hooded window boxes (ref_088-ref_090)
    for (const [thEnd, sgn, isSouthEnd] of [
      [ilrcTh0, -1, false],
      [ilrcTh1, 1, true]
    ]) {
      const endGrp = new THREE.Group();
      endGrp.rotation.y = -thEnd;
      contextGroup.add(endGrp);

      const stairTrim = new THREE.Mesh(
        new THREE.BoxGeometry(8.8, 34.15, 0.42),
        isSouthEnd ? mats.ilrcBrickPortal : mats.facadeWhite
      );
      stairTrim.position.set(37.4, 17.075, sgn * 0.18);
      endGrp.add(stairTrim);

      const stairGlass = new THREE.Mesh(new THREE.BoxGeometry(3.0, 31.2, 0.48), mats.windowGlass);
      stairGlass.position.set(37.4, 17.6, sgn * 0.22);
      endGrp.add(stairGlass);

      for (let fl = 0; fl < 6; fl++) {
        const bar = new THREE.Mesh(
          new THREE.BoxGeometry(isSouthEnd ? 9.4 : 3.4, 0.36, isSouthEnd ? 1.15 : 0.50),
          mats.facadeWhite
        );
        bar.position.set(37.4, ilrcFloorTops[fl] - 0.42, sgn * (isSouthEnd ? 0.52 : 0.24));
        endGrp.add(bar);
      }

      if (isSouthEnd) {
        // 3 Stacked Protruding Hooded Window Boxes on South Burnt-Brick Tower (L3, L4, L5 — ref_088, ref_090)
        for (const flIdx of [2, 3, 4]) {
          const yC = ilrcFloorTops[flIdx] - 2.65;
          const hoodBox = new THREE.Mesh(new THREE.BoxGeometry(2.6, 2.2, 0.92), mats.facadeWhite);
          hoodBox.position.set(40.4, yC, sgn * 0.52);
          endGrp.add(hoodBox);
          const hoodGlass = new THREE.Mesh(new THREE.BoxGeometry(2.1, 1.7, 0.98), mats.windowGlass);
          hoodGlass.position.set(40.4, yC, sgn * 0.55);
          endGrp.add(hoodGlass);
        }
      } else {
        // Ground-level grey metal electrical utility cabinet (Gardu) at NE corner lawn (-6.5°, ref_025)
        const gardu = new THREE.Mesh(new THREE.BoxGeometry(1.6, 2.1, 1.2), mats.galvSteel);
        gardu.position.set(46.2, 1.05, -2.4);
        const garduPlinth = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.28, 1.5), mats.granitePlinth);
        garduPlinth.position.set(46.2, 0.14, -2.4);
        endGrp.add(gardu, garduPlinth);
      }
    }

    // East Wing Drop-Off Portico Canopy (+4.5°, ref_026) & DRPM L1 Breezeway + 4 Manila Palms + Open Black Iron Picket Gate (+24°, ref_027-ref_032)
    {
      const eastPortTh = (4.5 * Math.PI) / 180.0;
      DPPDComplexBuilder._addCurvedBand(
        contextGroup,
        mats.facadeWhite,
        eastPortTh - 0.055,
        eastPortTh + 0.055,
        ilrcROut + 0.1,
        ilrcROut + 3.4,
        4.65,
        5.15,
        8
      );
      for (const sgn of [-1, 1]) {
        const thC = eastPortTh + sgn * 0.042;
        const eCol = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.26, 4.65, 10), mats.facadeWhite);
        eCol.position.set((ilrcROut + 2.95) * Math.cos(thC), 2.32, (ilrcROut + 2.95) * Math.sin(thC));
        contextGroup.add(eCol);
      }

      // DRPM Entrance Approach Apron (+21°..+26°), Open Black Iron Picket Gate & "P PARKIR KHUSUS PIMPINAN" Signpost (ref_027-ref_032)
      const drpmTh = (23.2 * Math.PI) / 180.0;
      DPPDComplexBuilder._addCurvedBand(
        contextGroup,
        mats.courtyardGranite,
        drpmTh - 0.048,
        drpmTh + 0.048,
        ilrcROut,
        50.85,
        0.02,
        0.15,
        8
      );
      for (const sgn of [-1, 1]) {
        const gTh = drpmTh + sgn * 0.046;
        const gateLeaf = new THREE.Mesh(new THREE.BoxGeometry(1.6, 1.35, 0.10), mats.darkIron);
        gateLeaf.position.set(49.2 * Math.cos(gTh), 0.72, 49.2 * Math.sin(gTh));
        gateLeaf.rotation.y = -gTh + sgn * 0.35;
        contextGroup.add(gateLeaf);
      }
      const pSignTh = drpmTh - 0.058;
      const pPost = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 2.1, 8), mats.galvSteel);
      pPost.position.set(48.2 * Math.cos(pSignTh), 1.05, 48.2 * Math.sin(pSignTh));
      const pBoard = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.62, 0.48), mats.kanBlue);
      pBoard.position.set(48.2 * Math.cos(pSignTh), 1.95, 48.2 * Math.sin(pSignTh));
      pBoard.rotation.y = -pSignTh;
      contextGroup.add(pPost, pBoard);

      // 4 Clumping Manila Palms along the DRPM East-Central Berm (+15°, +19°, +26.5°, +29.5° — ref_028-ref_030)
      for (const palmDeg of [15.0, 19.0, 26.5, 29.5]) {
        const thP = (palmDeg * Math.PI) / 180.0;
        contextGroup.add(
          DPPDComplexBuilder._buildRoyalPalm(mats, 46.8 * Math.cos(thP), 0.0, 46.8 * Math.sin(thP), 6.8)
        );
      }
    }

    // ICONIC ILRC 6-STORY SYMMETRICAL TERRACOTTA-BRICK TWIN-PIER MONUMENTAL ATRIUM PORTAL (at +35°, ref_081-ref_086!)
    // Includes warm taupe-terracotta atrium soffit ceiling, 3 open-air white horizontal corridor bridges at L3/L4/L5,
    // KAN accreditation emblem plaque, koral sikat pebble-grid ramp & open black iron picket gate!
    for (const [pThA, pThB] of [
      [rPierTh0, rPierTh1],
      [lPierTh0, lPierTh1]
    ]) {
      DPPDComplexBuilder._addCurvedBand(
        contextGroup,
        mats.ilrcBrickPortal,
        pThA,
        pThB,
        ilrcRIn + 2.8,
        ilrcROut + 0.82,
        0.0,
        34.15,
        6
      );
    }
    // Warm Taupe-Terracotta Portal Header & Atrium Soffit Ceiling between the Twin Piers (ref_083)
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.warmTaupeSoffit,
      rPierTh0,
      lPierTh1,
      ilrcRIn + 2.6,
      ilrcROut + 0.86,
      32.95,
      34.15,
      12
    );
    // 3 Open-Air White Horizontal Atrium Corridor Bridges at L3, L4, L5 (ref_083)
    for (const flBridge of [2, 3, 4]) {
      const yBr = ilrcFloorTops[flBridge - 1];
      DPPDComplexBuilder._addCurvedBand(
        contextGroup,
        mats.facadeWhite,
        rPierTh1,
        lPierTh0,
        ilrcMidR - 1.2,
        ilrcROut - 0.65,
        yBr - 0.35,
        yBr + 1.05,
        10
      );
    }
    // Recessed 6-story central glass atrium curtain wall + vertical mullions between the twin terracotta piers
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.windowGlass,
      rPierTh1,
      lPierTh0,
      ilrcROut - 0.58,
      ilrcROut + 0.06,
      0.48,
      32.95,
      12
    );
    for (const frac of [-0.5, 0.0, 0.5]) {
      const thM = ilrcCenterTh + frac * (lPierTh0 - rPierTh1);
      const aMul = new THREE.Mesh(new THREE.BoxGeometry(0.18, 27.9, 0.18), mats.darkAnodized);
      aMul.position.set((ilrcROut + 0.08) * Math.cos(thM), 20.15, (ilrcROut + 0.08) * Math.sin(thM));
      aMul.rotation.y = -thM;
      contextGroup.add(aMul);
    }

    // Projecting ILRC Ground-Floor Drop-Off Canopy + Photocell Downlights + KAN Emblem Plaque + Koral Sikat Ramp!
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.ilrcBrickPortal,
      ilrcCenterTh - 0.14,
      ilrcCenterTh + 0.14,
      ilrcROut + 0.2,
      ilrcROut + 4.6,
      5.55,
      6.20,
      12
    );
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.facadeWhite,
      ilrcCenterTh - 0.144,
      ilrcCenterTh + 0.144,
      ilrcROut + 4.35,
      ilrcROut + 4.68,
      6.12,
      6.32,
      12
    );
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.warmLuminaire,
      ilrcCenterTh - 0.11,
      ilrcCenterTh + 0.11,
      ilrcROut + 1.2,
      ilrcROut + 3.8,
      5.48,
      5.54,
      8
    );
    // Koral Sikat (washed-pebble concrete grid) driveway apron & open black iron picket gate at +35° (ref_081, ref_086)
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.koralSikat,
      ilrcCenterTh - 0.12,
      ilrcCenterTh + 0.12,
      ilrcROut + 0.4,
      50.82,
      0.04,
      0.19,
      12
    );
    for (const sgn of [-1, 1]) {
      const thP = ilrcCenterTh + sgn * 0.11;
      const cCol = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.34, 5.55, 12), mats.ilrcBrickPortal);
      cCol.position.set((ilrcROut + 4.1) * Math.cos(thP), 2.78, (ilrcROut + 4.1) * Math.sin(thP));
      contextGroup.add(cCol);

      const gLeaf = new THREE.Mesh(new THREE.BoxGeometry(1.8, 1.35, 0.10), mats.darkIron);
      gLeaf.position.set(49.5 * Math.cos(thP), 0.72, 49.5 * Math.sin(thP));
      gLeaf.rotation.y = -thP + sgn * 0.42;
      contextGroup.add(gLeaf);
    }
    // KAN Accreditation Emblem Plaque on right terracotta portal pier (ref_084)
    {
      const kanTh = (rPierTh0 + rPierTh1) * 0.5;
      const kanPlaque = new THREE.Mesh(new THREE.BoxGeometry(0.14, 1.15, 1.55), mats.facadeWhite);
      kanPlaque.position.set((ilrcROut + 0.86) * Math.cos(kanTh), 9.2, (ilrcROut + 0.86) * Math.sin(kanTh));
      kanPlaque.rotation.y = -kanTh;
      const kanBadge = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.68, 0.98), mats.kanBlue);
      kanBadge.position.set((ilrcROut + 0.88) * Math.cos(kanTh), 9.2, (ilrcROut + 0.88) * Math.sin(kanTh));
      kanBadge.rotation.y = -kanTh;
      contextGroup.add(kanPlaque, kanBadge);
    }

    // South Wing Landscaped Berm with Bamboo-Staked Brown Dry-Leaf Composting Mound (+45°..+63°, ref_086-ref_089)
    {
      const leafTh0 = (46.0 * Math.PI) / 180.0;
      const leafTh1 = (62.0 * Math.PI) / 180.0;
      DPPDComplexBuilder._addCurvedBand(
        contextGroup,
        mats.dryLeafMound,
        leafTh0,
        leafTh1,
        45.2,
        48.8,
        0.04,
        0.52,
        14
      );
      for (let st = 0; st <= 10; st++) {
        const thSt = leafTh0 + (st / 10) * (leafTh1 - leafTh0);
        for (const rSt of [45.15, 48.85]) {
          const stake = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, 0.95, 6), mats.bambooStake);
          stake.position.set(rSt * Math.cos(thSt), 0.48, rSt * Math.sin(thSt));
          contextGroup.add(stake);
        }
      }
    }

    // UI Makara + Left-to-Right 3D "GEDUNG ILRC UI" Signage mounted directly on the continuous flat y = 34.15..36.05 crown fascia!
    const ilrcMakaraTh = ilrcCenterTh + 0.138;
    const ilrcMakara = DPPDComplexBuilder._buildUIMakaraEmblem(mats, 0.88);
    ilrcMakara.position.set(
      (ilrcROut + 0.94) * Math.cos(ilrcMakaraTh),
      35.12,
      (ilrcROut + 0.94) * Math.sin(ilrcMakaraTh)
    );
    ilrcMakara.rotation.y = Math.PI * 0.5 - ilrcMakaraTh;
    contextGroup.add(ilrcMakara);

    DPPDComplexBuilder._addCurved3DText(
      contextGroup,
      'GEDUNG ILRC UI',
      ilrcROut + 0.94,
      ilrcCenterTh + 0.092,
      34.68,
      0.92,
      0.20,
      mats.darkAnodized,
      -1.0
    );

    // =========================================================================
    // 4. GEDUNG STP-UI — 8 STORIES WITH 206-PHOTO GROUND-TRUTH FIDELITY
    //    Modeled directly after real Gedung Science Techno Park UI Depok photos (ref_008..ref_023, ref_076..ref_078):
    //    - Wrapping Copper-Terracotta ACP Vertical End Frames & Overhanging Top Roof Cornice + 8-Story Scissor-Stair Cores
    //    - L1 NW Dark Industrial Louver Wall (-131°..-92°) & Glazed Exhibition Lobby (-92°..-46°)
    //    - L2-L3 Angled Copper-ACP V-Diagrid Screen + Vertical Copper-Tube Louver Fins in alternating bays
    //    - L4-L7 4 Crisp Curved White Spandrel Floors with Staggered Dark Box-Window Accent Frames + L4-L8 Vertical Copper-Fin Louver Bays
    //    - L8 Dark-Glazed Penthouse Frieze bearing Golden UI Makara + High-Contrast 3D "SCIENCE TECHNO PARK / UNIVERSITAS INDONESIA"
    //    - Angled V-Strut Drop-Off Portico Canopy (-89°) with Warm Taupe Soffit & 4 Stainless Bollards
    //    - Sloped White Concrete "STP-UI" Wedge Planter (-86°), Black/Yellow Directory Totem (-76°), & Red Fire-Hydrant Pipeline (-128°..-76°)
    // =========================================================================
    const stpTh0 = (DPPDComplexBuilder.STP_START_DEG * Math.PI) / 180.0;
    const stpTh1 = (DPPDComplexBuilder.STP_END_DEG * Math.PI) / 180.0;
    const stpCenterTh = 0.5 * (stpTh0 + stpTh1); // -89°
    const stpRIn = 31.2;
    const stpROut = 43.8;
    const stpWingPad = 0.036; // Angular width of left & right copper-ACP framing pilasters
    const stpInnerTh0 = stpTh0 + stpWingPad;
    const stpInnerTh1 = stpTh1 - stpWingPad;
    // Exact 8 countable floor ceiling heights:
    const stpFloorTops = [6.85, 12.45, 18.05, 23.45, 28.85, 34.25, 39.65, 45.25];

    // Plinth & L1 Recessed Dark-Glazed Exhibition Lobby + L1 NW Dark Industrial Louver Wall (-131°..-94°, ref_014-ref_016)
    DPPDComplexBuilder._addCurvedBand(contextGroup, mats.granitePlinth, stpTh0, stpTh1, stpRIn - 0.3, stpROut + 0.3, -0.35, 0.48, 52);
    DPPDComplexBuilder._addCurvedBand(contextGroup, mats.windowGlass, stpInnerTh0, stpInnerTh1, stpRIn + 0.4, stpROut - 0.55, 0.48, 6.10, 48);
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.darkAnodized,
      stpInnerTh0 + 0.02,
      stpCenterTh - 0.14,
      stpROut - 0.42,
      stpROut + 0.04,
      0.85,
      5.95,
      18
    );
    DPPDComplexBuilder._addCurvedBand(contextGroup, mats.copperACP, stpInnerTh0, stpInnerTh1, stpRIn - 0.35, stpROut + 0.48, 6.10, 6.85, 48);

    // L2 & L3 Glazed Podium + Intermediate L2 Floor Slab + Top L3 Copper Podium Belt
    for (const fl of [1, 2]) {
      const yBot = stpFloorTops[fl - 1];
      const yTop = stpFloorTops[fl];
      DPPDComplexBuilder._addCurvedBand(contextGroup, mats.windowGlass, stpInnerTh0, stpInnerTh1, stpRIn + 0.25, stpROut - 0.25, yBot, yTop - 0.72, 48);
      DPPDComplexBuilder._addCurvedBand(
        contextGroup,
        fl === 2 ? mats.copperACP : mats.warmStonePanel,
        stpInnerTh0,
        stpInnerTh1,
        stpRIn - 0.42,
        stpROut + (fl === 2 ? 0.62 : 0.20),
        yTop - 0.72,
        yTop,
        48
      );
    }

    // Signature L2-L3 Angled Copper-Terracotta ACP V-Diagrid Structural Ribs + Vertical Copper-Tube Louver Fins (ref_014-ref_020)
    const diagBays = 16;
    for (let b = 0; b < diagBays; b++) {
      const tA = stpInnerTh0 + (b / diagBays) * (stpInnerTh1 - stpInnerTh0);
      const tB = stpInnerTh0 + ((b + 1) / diagBays) * (stpInnerTh1 - stpInnerTh0);
      const rRib = stpROut + 0.36;
      const thBot = b % 2 === 0 ? tA : tB;
      const thTop = b % 2 === 0 ? tB : tA;
      pStart.set(rRib * Math.cos(thBot), 6.80, rRib * Math.sin(thBot));
      pEnd.set(rRib * Math.cos(thTop), 17.45, rRib * Math.sin(thTop));
      const rib = new THREE.Mesh(
        new THREE.BoxGeometry(0.62, 0.56, pStart.distanceTo(pEnd) + 0.15),
        mats.copperACP
      );
      DPPDComplexBuilder._orientBeamAlongZ(rib, pStart, pEnd, 0.0);
      contextGroup.add(rib);

      // Vertical copper-tube louver fins inside alternating L2-L3 podium bays (ref_016, ref_018)
      if (b % 2 === 1) {
        for (const fSub of [0.25, 0.50, 0.75]) {
          const thL = tA + fSub * (tB - tA);
          const lFin = new THREE.Mesh(new THREE.BoxGeometry(0.32, 10.45, 0.14), mats.copperACP);
          lFin.position.set((stpROut + 0.16) * Math.cos(thL), 12.10, (stpROut + 0.16) * Math.sin(thL));
          lFin.rotation.y = -thL;
          contextGroup.add(lFin);
        }
      }
    }

    // L4, L5, L6, L7 (4 Crisp White-Stone Curved Stories with Staggered Dark Box-Window Accent Frames & Sunshade Fins!)
    for (let fl = 3; fl <= 6; fl++) {
      const yBot = stpFloorTops[fl - 1];
      const yTop = stpFloorTops[fl];

      // Lower white curved spandrel wall
      DPPDComplexBuilder._addCurvedBand(
        contextGroup,
        fl % 2 === 1 ? mats.facadeWhite : mats.warmStonePanel,
        stpInnerTh0,
        stpInnerTh1,
        stpRIn,
        stpROut,
        yBot,
        yBot + 1.55,
        52
      );
      // Recessed dark-tinted ribbon glazing (outer & inner courtyard!)
      DPPDComplexBuilder._addCurvedBand(
        contextGroup,
        mats.windowGlass,
        stpInnerTh0,
        stpInnerTh1,
        stpRIn + 0.22,
        stpROut - 0.22,
        yBot + 1.55,
        yTop - 0.68,
        52
      );
      // Continuous curved sunshade fin cleanly bounded inside [stpInnerTh0, stpInnerTh1]
      DPPDComplexBuilder._addCurvedBand(
        contextGroup,
        mats.facadeWhite,
        stpInnerTh0,
        stpInnerTh1,
        stpRIn - 0.48,
        stpROut + 0.54,
        yTop - 0.68,
        yTop,
        52
      );

      // Signature Staggered Dark-Charcoal Projected Box-Window Frames on Outer Facade (seen in real STP-UI photos!)
      const boxSlots = 14;
      for (let s = 0; s < boxSlots; s++) {
        if ((s + fl) % 2 !== 0) continue;
        const thBox = stpInnerTh0 + ((s + 0.5) / boxSlots) * (stpInnerTh1 - stpInnerTh0);
        const boxFrame = new THREE.Mesh(new THREE.BoxGeometry(0.55, 3.35, 2.35), mats.darkAnodized);
        boxFrame.position.set(
          (stpROut + 0.14) * Math.cos(thBox),
          yBot + 2.85,
          (stpROut + 0.14) * Math.sin(thBox)
        );
        boxFrame.rotation.y = -thBox;
        contextGroup.add(boxFrame);

        const boxPane = new THREE.Mesh(new THREE.BoxGeometry(0.62, 2.75, 1.75), mats.windowGlass);
        boxPane.position.set(
          (stpROut + 0.16) * Math.cos(thBox),
          yBot + 2.85,
          (stpROut + 0.16) * Math.sin(thBox)
        );
        boxPane.rotation.y = -thBox;
        contextGroup.add(boxPane);
      }
    }

    // L4-L8 Vertical Copper-Fin Louver Screen Bays adjacent to the NW & SW End Copper Framing Wings (ref_008, ref_014)
    for (const baseTh of [stpInnerTh0 + 0.012, stpInnerTh1 - 0.072]) {
      for (let f = 0; f < 5; f++) {
        const thFin = baseTh + f * 0.014;
        const cFin = new THREE.Mesh(new THREE.BoxGeometry(0.48, 27.10, 0.16), mats.copperACP);
        cFin.position.set((stpROut + 0.32) * Math.cos(thFin), 31.65, (stpROut + 0.32) * Math.sin(thFin));
        cFin.rotation.y = -thFin;
        contextGroup.add(cFin);
      }
    }

    // L8 (Top 8th-Floor Dark-Glazed Crown Frieze: y = 39.65 .. 45.25)
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.windowGlass,
      stpInnerTh0,
      stpInnerTh1,
      stpRIn + 0.22,
      stpROut - 0.35,
      39.65,
      45.25,
      52
    );
    // Dedicated dark architectural glass signage frieze on the left/upper L8 arc (matching real STP-UI Photo 1 & 3!)
    const signArc0 = stpInnerTh1 - 0.66;
    const signArc1 = stpInnerTh1 - 0.02;
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.darkFriezeGlass,
      signArc0,
      signArc1,
      stpROut - 0.32,
      stpROut - 0.04,
      39.85,
      45.15,
      28
    );

    // Wrapping Copper-Terracotta ACP Left & Right Vertical Framing Wings (y = 0 .. 47.10) + Overhanging Top Roof Cornice!
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.copperACP,
      stpTh0,
      stpInnerTh0,
      stpRIn - 0.38,
      stpROut + 0.88,
      0.0,
      47.10,
      6
    );
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.copperACP,
      stpInnerTh1,
      stpTh1,
      stpRIn - 0.38,
      stpROut + 0.88,
      0.0,
      47.10,
      6
    );
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.copperACP,
      stpTh0,
      stpTh1,
      stpRIn - 0.45,
      stpROut + 0.96,
      45.25,
      47.10,
      56
    );

    // Roof Deck, Safety Parapet, Mechanical Chiller Penthouse, & 16 Angled Rooftop Solar PV Panels on Galvanized Rails!
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.ilrcConcrete,
      stpInnerTh0,
      stpInnerTh1,
      stpRIn + 0.1,
      stpROut - 0.1,
      44.95,
      45.55,
      48
    );
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.warmStonePanel,
      stpInnerTh0 + 0.05,
      stpCenterTh - 0.36,
      34.0,
      40.5,
      45.55,
      48.35,
      14
    );

    const pvCount = 16;
    for (let p = 0; p < pvCount; p++) {
      const thPv = stpCenterTh - 0.28 + (p / pvCount) * (stpInnerTh1 - 0.08 - (stpCenterTh - 0.28));
      const pvRack = new THREE.Group();
      pvRack.position.set(37.8 * Math.cos(thPv), 46.25, 37.8 * Math.sin(thPv));
      pvRack.rotation.y = -thPv;

      const frame = new THREE.Mesh(new THREE.BoxGeometry(6.8, 0.16, 2.15), mats.galvSteel);
      frame.rotation.z = -0.24;
      pvRack.add(frame);

      const panel = new THREE.Mesh(new THREE.BoxGeometry(6.6, 0.12, 2.02), mats.solarPanel);
      panel.position.y = 0.09;
      panel.rotation.z = -0.24;
      pvRack.add(panel);

      for (const rx of [-2.5, 2.5]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.16, 1.05, 0.16), mats.galvSteel);
        leg.position.set(rx, -0.35, 0);
        pvRack.add(leg);
      }
      contextGroup.add(pvRack);
    }

    // Vertical Window Mullions & Inner Courtyard Pilasters across all 8 Stories of STP-UI (Outer + Inner + Both End Walls!)
    const stpMullions = 28;
    for (let i = 0; i <= stpMullions; i++) {
      const th = stpInnerTh0 + (i / stpMullions) * (stpInnerTh1 - stpInnerTh0);
      const cT = Math.cos(th);
      const sT = Math.sin(th);

      // L1 Piloti Columns (Outer & Inner Courtyard)
      if (i % 2 === 0) {
        const outCol = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.44, 6.10, 12), mats.warmStonePanel);
        outCol.position.set((stpROut - 0.15) * cT, 3.25, (stpROut - 0.15) * sT);
        contextGroup.add(outCol);

        const inCol = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.40, 6.10, 10), mats.warmStonePanel);
        inCol.position.set((stpRIn + 0.15) * cT, 3.25, (stpRIn + 0.15) * sT);
        contextGroup.add(inCol);
      }

      // Full-height 8-story inner courtyard vertical architectural mullions & fins (eliminates black inner wall!)
      const inMul = new THREE.Mesh(
        new THREE.BoxGeometry(i % 2 === 0 ? 0.44 : 0.20, 44.6, 0.18),
        i % 2 === 0 ? mats.facadeWhite : mats.darkAnodized
      );
      inMul.position.set((stpRIn + 0.08) * cT, 22.85, (stpRIn + 0.08) * sT);
      inMul.rotation.y = -th;
      contextGroup.add(inMul);

      // Outer vertical window mullions
      const outMul = new THREE.Mesh(new THREE.BoxGeometry(0.22, 44.6, 0.14), mats.darkAnodized);
      outMul.position.set((stpROut - 0.16) * cT, 22.85, (stpROut - 0.16) * sT);
      outMul.rotation.y = -th;
      contextGroup.add(outMul);
    }

    // Both STP-UI End Walls (stpTh0 & stpTh1): Articulated 8-Story Glazed Scissor-Stair / Elevator Core & Vertical Copper Louver Screen!
    for (const [thEnd, sgn] of [
      [stpTh0, -1],
      [stpTh1, 1]
    ]) {
      const endGrp = new THREE.Group();
      endGrp.rotation.y = -thEnd;
      contextGroup.add(endGrp);

      const coreWall = new THREE.Mesh(new THREE.BoxGeometry(8.6, 45.4, 0.36), mats.facadeWhite);
      coreWall.position.set(37.5, 22.9, sgn * 0.14);
      endGrp.add(coreWall);

      const coreGlass = new THREE.Mesh(new THREE.BoxGeometry(3.8, 42.4, 0.44), mats.windowGlass);
      coreGlass.position.set(37.5, 23.2, sgn * 0.18);
      endGrp.add(coreGlass);

      for (const xFin of [36.1, 37.5, 38.9]) {
        const endFin = new THREE.Mesh(new THREE.BoxGeometry(0.18, 42.4, 0.56), mats.copperACP);
        endFin.position.set(xFin, 23.2, sgn * 0.24);
        endGrp.add(endFin);
      }

      for (let fl = 0; fl < 8; fl++) {
        const bar = new THREE.Mesh(new THREE.BoxGeometry(8.8, 0.42, 0.50), mats.copperACP);
        bar.position.set(37.5, stpFloorTops[fl] - 0.32, sgn * 0.22);
        endGrp.add(bar);
      }
    }

    // Golden UI Makara Emblem + High-Contrast Left-to-Right 2-Line 3D Signage "SCIENCE TECHNO PARK" / "UNIVERSITAS INDONESIA" on L8 Dark Frieze!
    const stpMakaraTh = signArc1 - 0.055;
    const stpMakara = DPPDComplexBuilder._buildUIMakaraEmblem(mats, 1.50);
    stpMakara.position.set(
      (stpROut + 0.16) * Math.cos(stpMakaraTh),
      42.55,
      (stpROut + 0.16) * Math.sin(stpMakaraTh)
    );
    stpMakara.rotation.y = Math.PI * 0.5 - stpMakaraTh;
    contextGroup.add(stpMakara);

    DPPDComplexBuilder._addCurved3DText(
      contextGroup,
      'SCIENCE TECHNO PARK',
      stpROut + 0.14,
      signArc1 - 0.125,
      42.85,
      1.28,
      0.26,
      mats.whiteMullion,
      -1.0
    );
    DPPDComplexBuilder._addCurved3DText(
      contextGroup,
      'UNIVERSITAS INDONESIA',
      stpROut + 0.14,
      signArc1 - 0.125,
      40.88,
      0.98,
      0.22,
      mats.brassSign,
      -1.0
    );

    // STP-UI Grand L1 Drop-Off Canopy Portico at stpCenterTh (-89°) + Angled V-Strut Columns + Warm Taupe Soffit + 4 Stainless Bollards!
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.copperACP,
      stpCenterTh - 0.14,
      stpCenterTh + 0.14,
      stpROut + 0.2,
      stpROut + 5.4,
      5.85,
      6.60,
      14
    );
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.warmTaupeSoffit,
      stpCenterTh - 0.135,
      stpCenterTh + 0.135,
      stpROut + 0.3,
      stpROut + 5.25,
      5.70,
      5.85,
      14
    );
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.warmLuminaire,
      stpCenterTh - 0.11,
      stpCenterTh + 0.11,
      stpROut + 1.4,
      stpROut + 4.5,
      5.64,
      5.70,
      10
    );
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.courtyardGranite,
      stpCenterTh - 0.18,
      stpCenterTh + 0.18,
      stpROut,
      50.85,
      0.02,
      0.16,
      16
    );
    for (const sgn of [-1, 1]) {
      const thP = stpCenterTh + sgn * 0.115;
      const baseX = (stpROut + 4.7) * Math.cos(thP);
      const baseZ = (stpROut + 4.7) * Math.sin(thP);
      // Signature angled V-strut portico support columns (ref_020, ref_077)
      for (const vAng of [-0.022, 0.022]) {
        pStart.set(baseX, 0.12, baseZ);
        pEnd.set((stpROut + 4.55) * Math.cos(thP + vAng), 5.75, (stpROut + 4.55) * Math.sin(thP + vAng));
        const vCol = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.32, pStart.distanceTo(pEnd)), mats.darkAnodized);
        DPPDComplexBuilder._orientBeamAlongZ(vCol, pStart, pEnd, 0.0);
        contextGroup.add(vCol);
      }
    }
    // 4 Stainless Security Bollards flanking the STP-UI Lobby Drop-Off Apron (ref_020)
    for (const bOff of [-0.13, -0.045, 0.045, 0.13]) {
      const thB = stpCenterTh + bOff;
      const bol = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.12, 0.92, 10), mats.galvSteel);
      bol.position.set((stpROut + 2.2) * Math.cos(thB), 0.52, (stpROut + 2.2) * Math.sin(thB));
      contextGroup.add(bol);
    }
    DPPDComplexBuilder._addCurved3DText(
      contextGroup,
      'STP-UI',
      stpROut + 5.52,
      stpCenterTh + 0.042,
      6.72,
      0.92,
      0.20,
      mats.whiteMullion,
      -1.0
    );

    // Sloped White Concrete "STP-UI" Monumental Wedge Planter (-82°, ref_021) + Black/Yellow Directory Totem (-75°, ref_023)
    // + Exposed Red Fire-Hydrant Pipeline & Twin Red Pillar Hydrants (-126°..-76°, ref_014-ref_019)
    {
      const wedgeTh = (-81.5 * Math.PI) / 180.0;
      const wedgeGrp = new THREE.Group();
      wedgeGrp.position.set(49.2 * Math.cos(wedgeTh), 0.0, 49.2 * Math.sin(wedgeTh));
      wedgeGrp.rotation.y = -wedgeTh;
      const wBase = new THREE.Mesh(new THREE.BoxGeometry(1.35, 1.15, 4.2), mats.facadeWhite);
      wBase.position.y = 0.58;
      wBase.rotation.z = -0.14;
      const wCap = new THREE.Mesh(new THREE.BoxGeometry(1.42, 0.16, 4.3), mats.granitePlinth);
      wCap.position.y = 0.10;
      wedgeGrp.add(wBase, wCap);
      contextGroup.add(wedgeGrp);
      DPPDComplexBuilder._addCurved3DText(
        contextGroup,
        'STP-UI',
        49.94,
        wedgeTh + 0.025,
        0.42,
        0.52,
        0.10,
        mats.darkAnodized,
        -1.0
      );

      // Black & UI-Yellow Vertical Directory Signboard Totem at -75° (ref_023)
      const dirTh = (-75.0 * Math.PI) / 180.0;
      const dirTotem = new THREE.Group();
      dirTotem.position.set(49.4 * Math.cos(dirTh), 0.0, 49.4 * Math.sin(dirTh));
      dirTotem.rotation.y = -dirTh;
      const dPlinth = new THREE.Mesh(new THREE.BoxGeometry(0.65, 0.35, 1.45), mats.granitePlinth);
      dPlinth.position.y = 0.18;
      const dBody = new THREE.Mesh(new THREE.BoxGeometry(0.36, 2.55, 1.25), mats.darkAnodized);
      dBody.position.y = 1.60;
      const dHeader = new THREE.Mesh(new THREE.BoxGeometry(0.40, 0.42, 1.28), mats.uiYellowFascia);
      dHeader.position.y = 2.72;
      dirTotem.add(dPlinth, dBody, dHeader);
      contextGroup.add(dirTotem);

      // Exposed Red Fire-Hydrant Pipeline & 2 Red Pillar Hydrants along NW Perimeter Lawn (-125°..-96°, ref_014-ref_019)
      const hPipeTh0 = (-125.0 * Math.PI) / 180.0;
      const hPipeTh1 = (-96.0 * Math.PI) / 180.0;
      DPPDComplexBuilder._addCurvedBand(
        contextGroup,
        mats.flagRed,
        hPipeTh0,
        hPipeTh1,
        48.55,
        48.72,
        0.28,
        0.42,
        14
      );
      for (const hDeg of [-122.0, -98.0]) {
        const thH = (hDeg * Math.PI) / 180.0;
        const hyd = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.16, 0.95, 10), mats.flagRed);
        hyd.position.set(48.64 * Math.cos(thH), 0.50, 48.64 * Math.sin(thH));
        const hydCap = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.16, 0.16, 10), mats.brassSign);
        hydCap.position.set(48.64 * Math.cos(thH), 1.02, 48.64 * Math.sin(thH));
        contextGroup.add(hyd, hydCap);
      }
    }

    // =========================================================================
    // 5. GEDUNG ITTC / PERPUSTAKAAN LAMA UI (CENTRAL 4-STORY SQUARE COURTYARD HERITAGE BUILDING)
    //    Modeled directly after real Gedung ITTC / Perpustakaan Lama UI Depok photos (ref_004..ref_007, ref_057..ref_066)
    //    and aerial drone imagery:
    //    - Square courtyard footprint (halfOuter = 13.5m, halfInner = 5.0m, eaveY = 12.60m, ridgeY = 15.10m)
    //    - 4-wing pitched terracotta Limasan roof with 12 articulated ridge/hip/valley caps
    //    - High-level flat concrete central roof deck (Atap Coran) + raised silver-louvered clerestory skylight monitor
    //      crowned by a pyramidal terracotta roof cap (apex y = 15.85m, leaving 6.76m vertical clearance under Avata 2 FPV!)
    //    - 7-bay white reinforced-concrete column grid & horizontal spandrel beams across L1-L4 + L2/L3 split-AC condenser units
    //    - West entrance portico canopy & NW 1-story flat-roofed annex wing (w = 7.4m, d = 4.2m, h = 3.5m)
    //    - Surrounded by the walkable granite/terracotta courtyard ring (r = 16.5..31.2m), radial spokes, shade trees & benches
    // =========================================================================
    const courtyardOuterPad = new THREE.Mesh(
      new THREE.CylinderGeometry(31.2, 31.2, 0.16, 64),
      mats.courtyardGranite
    );
    courtyardOuterPad.position.y = 0.02;
    dppdGroup.add(courtyardOuterPad);

    const courtyardBrickRing = new THREE.Mesh(
      new THREE.CylinderGeometry(24.6, 24.6, 0.18, 56),
      mats.terracottaFloor
    );
    courtyardBrickRing.position.y = 0.03;
    dppdGroup.add(courtyardBrickRing);

    const courtyardInnerGranite = new THREE.Mesh(
      new THREE.CylinderGeometry(18.2, 18.2, 0.20, 56),
      mats.courtyardGranite
    );
    courtyardInnerGranite.position.y = 0.04;
    dppdGroup.add(courtyardInnerGranite);

    // 4-Story Square Courtyard Building: Gedung ITTC / Perpustakaan Lama UI
    {
      const ittcGroup = new THREE.Group();
      dppdGroup.add(ittcGroup);

      const hOut = 13.5;
      const hIn = 5.0;
      const ittcFlTops = [3.40, 6.50, 9.60, 12.60];

      // Stepped stone foundation plinth
      const ittcPlinth = new THREE.Mesh(new THREE.BoxGeometry(hOut * 2 + 0.9, 0.44, hOut * 2 + 0.9), mats.granitePlinth);
      ittcPlinth.position.y = 0.22;
      ittcGroup.add(ittcPlinth);

      // 4 Wings (North, South, East, West) forming the square courtyard massing
      const wingSpecs = [
        { cx: 0, cz: -9.25, sx: 27.0, sz: 8.5 },
        { cx: 0, cz: 9.25, sx: 27.0, sz: 8.5 },
        { cx: -9.25, cz: 0, sx: 8.5, sz: 10.0 },
        { cx: 9.25, cz: 0, sx: 8.5, sz: 10.0 }
      ];

      for (let fl = 0; fl < 4; fl++) {
        const yBot = fl === 0 ? 0.44 : ittcFlTops[fl - 1];
        const yTop = ittcFlTops[fl];
        const flH = yTop - yBot;

        for (const w of wingSpecs) {
          // Recessed dark window glazing core
          const glassCore = new THREE.Mesh(
            new THREE.BoxGeometry(w.sx - 0.36, flH - 0.95, w.sz - 0.36),
            mats.windowGlass
          );
          glassCore.position.set(w.cx, yBot + 0.65 + (flH - 0.95) * 0.5, w.cz);
          ittcGroup.add(glassCore);

          // White concrete sill spandrel band
          const sillBand = new THREE.Mesh(
            new THREE.BoxGeometry(w.sx + 0.08, 0.72, w.sz + 0.08),
            mats.facadeWhite
          );
          sillBand.position.set(w.cx, yBot + 0.36, w.cz);
          ittcGroup.add(sillBand);

          // Projecting white horizontal floor slab / sunshade cornice
          const sunSlab = new THREE.Mesh(
            new THREE.BoxGeometry(w.sx + 0.56, 0.34, w.sz + 0.56),
            mats.whiteMullion
          );
          sunSlab.position.set(w.cx, yTop - 0.17, w.cz);
          ittcGroup.add(sunSlab);
        }
      }

      // 7-Bay White Reinforced-Concrete Structural Column Grid & Dark Window Mullions on All 4 Outer Facades + Split-AC Units
      for (let b = 0; b <= 7; b++) {
        const t = -hOut + (b / 7) * (hOut * 2);
        for (const sgn of [-1, 1]) {
          const colNS = new THREE.Mesh(new THREE.BoxGeometry(0.42, 12.2, 0.46), mats.facadeWhite);
          colNS.position.set(t, 6.50, sgn * hOut);
          const colEW = new THREE.Mesh(new THREE.BoxGeometry(0.46, 12.2, 0.42), mats.facadeWhite);
          colEW.position.set(sgn * hOut, 6.50, t);
          ittcGroup.add(colNS, colEW);

          if (b < 7) {
            const tMid = t + hOut / 7;
            const mulNS = new THREE.Mesh(new THREE.BoxGeometry(0.16, 11.8, 0.22), mats.darkAnodized);
            mulNS.position.set(tMid, 6.50, sgn * (hOut - 0.08));
            const mulEW = new THREE.Mesh(new THREE.BoxGeometry(0.22, 11.8, 0.16), mats.darkAnodized);
            mulEW.position.set(sgn * (hOut - 0.08), 6.50, tMid);
            ittcGroup.add(mulNS, mulEW);

            // Exterior Split-AC Condenser Units mounted on L2 & L3 spandrel ledges (ref_057-ref_062)
            if (b % 2 === 1) {
              for (const yAc of [4.15, 7.25]) {
                const acBox = new THREE.Mesh(new THREE.BoxGeometry(0.68, 0.48, 0.34), mats.warmStonePanel);
                acBox.position.set(tMid, yAc, sgn * (hOut + 0.18));
                ittcGroup.add(acBox);
              }
            }
          }
        }
      }

      // 4-Wing Pitched Terracotta Limasan Roof with Overhanging Eaves (14.45m outer, 4.35m inner, eaveY = 12.60m, ridgeY = 15.10m)
      const eaveFascia = new THREE.Mesh(new THREE.BoxGeometry(28.9, 0.32, 28.9), mats.facadeWhite);
      eaveFascia.position.y = 12.52;
      ittcGroup.add(eaveFascia);

      const { roofGeo: limasanRoofGeo } = DPPDComplexBuilder._buildSquareCourtyardLimasanRoof(14.45, 4.35, 12.60, 15.10);
      const limasanRoofMesh = new THREE.Mesh(limasanRoofGeo, mats.terracottaRoof);
      ittcGroup.add(limasanRoofMesh);

      // 12 Articulated Terracotta Ridge, Outer Hip & Inner Valley Caps on the 4-Wing Limasan Roof
      for (const sgnX of [-1, 1]) {
        const rNS = new THREE.Mesh(new THREE.BoxGeometry(18.6, 0.26, 0.36), mats.terracottaRidge);
        rNS.position.set(0, 15.14, sgnX * 9.25);
        const rEW = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.26, 18.6), mats.terracottaRidge);
        rEW.position.set(sgnX * 9.25, 15.14, 0);
        ittcGroup.add(rNS, rEW);

        for (const sgnZ of [-1, 1]) {
          // Outer corner hip cap
          pStart.set(sgnX * 14.45, 12.62, sgnZ * 14.45);
          pEnd.set(sgnX * 9.25, 15.14, sgnZ * 9.25);
          const hipCap = new THREE.Mesh(
            new THREE.BoxGeometry(0.32, 0.24, pStart.distanceTo(pEnd) + 0.15),
            mats.terracottaRidge
          );
          DPPDComplexBuilder._orientBeamAlongZ(hipCap, pStart, pEnd, 0.0);
          ittcGroup.add(hipCap);

          // Inner courtyard valley cap
          pStart.set(sgnX * 4.35, 12.62, sgnZ * 4.35);
          pEnd.set(sgnX * 9.25, 15.14, sgnZ * 9.25);
          const valCap = new THREE.Mesh(
            new THREE.BoxGeometry(0.28, 0.22, pStart.distanceTo(pEnd) + 0.10),
            mats.terracottaRidge
          );
          DPPDComplexBuilder._orientBeamAlongZ(valCap, pStart, pEnd, 0.0);
          ittcGroup.add(valCap);
        }
      }

      // High-Level Flat Concrete Central Roof Deck (Atap Coran, y = 13.05m) + Raised Clerestory Skylight Monitor
      // & Pyramidal Terracotta Roof Cap (apex y = 15.85m — verified in aerial drone imagery!)
      const atapCoran = new THREE.Mesh(new THREE.BoxGeometry(9.6, 0.42, 9.6), mats.ilrcConcrete);
      atapCoran.position.y = 13.05;
      const monitorLouver = new THREE.Mesh(new THREE.BoxGeometry(5.6, 1.15, 5.6), mats.silverLouver);
      monitorLouver.position.y = 13.82;
      const monitorCap = new THREE.Mesh(new THREE.ConeGeometry(4.45, 1.45, 4), mats.terracottaRoof);
      monitorCap.position.y = 15.12;
      monitorCap.rotation.y = Math.PI * 0.25;
      ittcGroup.add(atapCoran, monitorLouver, monitorCap);

      // West Entrance Portico Canopy (facing STP-UI / West promenade)
      const westCanopy = new THREE.Mesh(new THREE.BoxGeometry(2.8, 0.34, 6.4), mats.facadeWhite);
      westCanopy.position.set(-14.6, 3.65, 0);
      const westLight = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.08, 4.8), mats.warmLuminaire);
      westLight.position.set(-14.5, 3.46, 0);
      ittcGroup.add(westCanopy, westLight);
      for (const czCol of [-2.6, 2.6]) {
        const wCol = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.24, 3.48, 10), mats.facadeWhite);
        wCol.position.set(-15.6, 1.74, czCol);
        ittcGroup.add(wCol);
      }

      // NW 1-Story Flat-Roofed Annex Wing (w = 7.4m, d = 4.2m, h = 3.50m — ref_004-ref_007, ref_057-ref_066)
      const nwAnnex = new THREE.Mesh(new THREE.BoxGeometry(7.4, 3.30, 4.2), mats.facadeWhite);
      nwAnnex.position.set(-9.6, 1.85, -15.4);
      const nwAnnexRoof = new THREE.Mesh(new THREE.BoxGeometry(7.9, 0.32, 4.7), mats.ilrcConcrete);
      nwAnnexRoof.position.set(-9.6, 3.56, -15.4);
      const nwAnnexGlass = new THREE.Mesh(new THREE.BoxGeometry(5.8, 1.75, 4.32), mats.windowGlass);
      nwAnnexGlass.position.set(-9.6, 2.05, -15.4);
      ittcGroup.add(nwAnnex, nwAnnexRoof, nwAnnexGlass);
    }

    // 3 Radial Granite Promenade Spokes linking Courtyard Ring (r = 18.2..28.5m) to DPPD (+135°), ILRC (+35°), & STP-UI (-89°)
    for (const spokeDeg of [135.0, 35.0, -89.0]) {
      const thS = (spokeDeg * Math.PI) / 180.0;
      const spoke = new THREE.Mesh(new THREE.BoxGeometry(11.6, 0.06, 4.2), mats.whiteMullion);
      spoke.position.set(23.2 * Math.cos(thS), 0.13, 23.2 * Math.sin(thS));
      spoke.rotation.y = -thS;
      dppdGroup.add(spoke);
    }

    // Open Landscaped Granite Courtyard Gateway Promenades between DPPD, ILRC, & STP-UI
    for (const [degA, degB, rOutProm] of [
      [74.5, 90.5, 44.2],
      [-36.0, -12.0, 50.85],
      [-151.5, -134.5, 35.2]
    ]) {
      const t0 = (Math.min(degA, degB) * Math.PI) / 180.0;
      const t1 = (Math.max(degA, degB) * Math.PI) / 180.0;
      DPPDComplexBuilder._addCurvedBand(contextGroup, mats.courtyardGranite, t0, t1, 30.6, rOutProm, 0.02, 0.15, 16);
    }

    // 4 Landscaped Courtyard Planter Islands with Shade Trees, Teak Benches, & Warm Photocell Bollards (at r = 22.8m around ITTC!)
    const courtyardTreeAnglesDeg = [82.0, -24.0, -138.0, 172.0];
    for (const cDeg of courtyardTreeAnglesDeg) {
      const th = (cDeg * Math.PI) / 180.0;
      const rIsle = 22.8;
      const px = rIsle * Math.cos(th);
      const pz = rIsle * Math.sin(th);

      const planter = new THREE.Mesh(new THREE.CylinderGeometry(2.35, 2.55, 0.52, 20), mats.granitePlinth);
      planter.position.set(px, 0.34, pz);
      contextGroup.add(planter);

      const turf = new THREE.Mesh(new THREE.CylinderGeometry(2.05, 2.05, 0.56, 20), mats.campusGrass);
      turf.position.set(px, 0.36, pz);
      contextGroup.add(turf);

      contextGroup.add(DPPDComplexBuilder._buildCourtyardTree(mats, px, 0.55, pz, 1.05));

      // Teak-and-concrete courtyard bench facing ITTC courtyard promenade
      const bench = new THREE.Group();
      bench.position.set((rIsle + 3.15) * Math.cos(th), 0.14, (rIsle + 3.15) * Math.sin(th));
      bench.rotation.y = -th - Math.PI * 0.5;
      const seat = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.12, 0.62), mats.teakBench);
      seat.position.y = 0.46;
      const back = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.46, 0.10), mats.teakBench);
      back.position.set(0, 0.74, -0.26);
      bench.add(seat, back);
      for (const lx of [-0.85, 0.85]) {
        const leg = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.44, 0.56), mats.concreteCurb);
        leg.position.set(lx, 0.22, 0);
        bench.add(leg);
      }
      contextGroup.add(bench);
    }

    // 8 Photocell Warm Courtyard Garden Bollard Lamps (v = -890.0)
    for (let b = 0; b < 8; b++) {
      const th = (b / 8) * Math.PI * 2.0 + 0.2;
      const bx = 25.4 * Math.cos(th);
      const bz = 25.4 * Math.sin(th);
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.16, 1.05, 10), mats.darkIron);
      post.position.set(bx, 0.62, bz);
      const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.24, 10), mats.warmLuminaire);
      cap.position.set(bx, 1.24, bz);
      contextGroup.add(post, cap);
    }

    // =========================================================================
    // 6. EAST VISITOR PARKING BAY (MATCHING ASPHALT!), HALTE BIKUN DPPD-STP,
    //    BELVEDERE OVERLOOK FACING REKTORAT, & RING-ROAD PHOTOCELL STREETLAMPS
    // =========================================================================
    // East Visitor Parking Bay (th = -170° .. -140°, r = 35.8 .. 50.8) — same Pulau U asphalt color!
    const parkTh0 = (-170.0 * Math.PI) / 180.0;
    const parkTh1 = (-140.0 * Math.PI) / 180.0;
    DPPDComplexBuilder._addCurvedBand(contextGroup, mats.asphaltCrown, parkTh0, parkTh1, 35.8, 50.82, 0.02, 0.085, 18);
    for (let st = 0; st <= 5; st++) {
      const thLine = parkTh0 + 0.05 + (st / 5) * (parkTh1 - parkTh0 - 0.10);
      const stallLine = new THREE.Mesh(new THREE.BoxGeometry(5.2, 0.03, 0.18), mats.whiteEdgeLine);
      stallLine.position.set(42.2 * Math.cos(thLine), 0.095, 42.2 * Math.sin(thLine));
      stallLine.rotation.y = -thLine;
      contextGroup.add(stallLine);
    }

    const carMats = [mats.carWhite, mats.carSilver, mats.carNavy, mats.carDarkGrey];
    for (let c = 0; c < 4; c++) {
      const thCar = parkTh0 + 0.09 + ((c + 0.5) / 5) * (parkTh1 - parkTh0 - 0.10);
      const cx = 42.2 * Math.cos(thCar);
      const cz = 42.2 * Math.sin(thCar);
      contextGroup.add(
        DPPDComplexBuilder._buildParkedCar(mats, carMats[c], cx, 0.09, cz, -thCar + Math.PI * 0.5)
      );
    }

    // Campus Entrance Security Post & Monolith Sign cleanly INSIDE inner lawn at th = -175° (ZERO obstruction on ring road!)
    {
      const thSign = (-175.0 * Math.PI) / 180.0;
      const sx = 49.2 * Math.cos(thSign);
      const sz = 49.2 * Math.sin(thSign);
      const totem = new THREE.Group();
      totem.position.set(sx, 0.0, sz);
      totem.rotation.y = -thSign;
      const base = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.55, 4.4), mats.granitePlinth);
      base.position.y = 0.28;
      const body = new THREE.Mesh(new THREE.BoxGeometry(0.75, 2.65, 3.8), mats.darkAnodized);
      body.position.y = 1.85;
      const goldCap = new THREE.Mesh(new THREE.BoxGeometry(0.84, 0.28, 3.95), mats.uiYellowFascia);
      goldCap.position.y = 3.28;
      totem.add(base, body, goldCap);
      contextGroup.add(totem);
    }

    // Halte Bus Kuning (Bikun) "HALTE DPPD - ILRC - STP" in Dedicated Outer Shelter Lay-By at th = -112° (r = 70.2)
    // Leaves the 360° Outer Sidewalk Promenade (r = 65.16..68.10) 100% clear for pedestrians & joggers!
    {
      const thHalte = (-112.0 * Math.PI) / 180.0;
      const halte = new THREE.Group();
      halte.position.set(70.20 * Math.cos(thHalte), 0.18, 70.20 * Math.sin(thHalte));
      halte.rotation.y = -thHalte - Math.PI * 0.5; // Faces inward across sidewalk toward ring road!

      const pad = new THREE.Mesh(new THREE.BoxGeometry(7.6, 1.10, 4.2), mats.granitePlinth);
      pad.position.set(0, -0.46, 0.25);
      halte.add(pad);
      const deck = new THREE.Mesh(new THREE.BoxGeometry(7.4, 0.08, 4.0), mats.courtyardGranite);
      deck.position.set(0, 0.10, 0.25);
      halte.add(deck);

      for (const px of [-3.1, 0.0, 3.1]) {
        const post = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 3.35, 10), mats.darkIron);
        post.position.set(px, 1.72, -0.95);
        halte.add(post);
      }
      const backGlass = new THREE.Mesh(new THREE.BoxGeometry(6.1, 2.35, 0.10), mats.windowGlass);
      backGlass.position.set(0, 1.55, -0.95);
      halte.add(backGlass);

      const roof = new THREE.Mesh(new THREE.BoxGeometry(7.6, 0.22, 3.1), mats.darkAnodized);
      roof.position.set(0, 3.45, -0.15);
      roof.rotation.x = 0.06;
      halte.add(roof);

      const yellowFascia = new THREE.Mesh(new THREE.BoxGeometry(7.68, 0.42, 0.16), mats.uiYellowFascia);
      yellowFascia.position.set(0, 3.55, 1.38);
      halte.add(yellowFascia);

      const lightStrip = new THREE.Mesh(new THREE.BoxGeometry(5.8, 0.06, 0.28), mats.warmLuminaire);
      lightStrip.position.set(0, 3.32, 0.25);
      halte.add(lightStrip);

      const benchSeat = new THREE.Mesh(new THREE.BoxGeometry(4.6, 0.12, 0.58), mats.teakBench);
      benchSeat.position.set(0, 0.62, -0.45);
      halte.add(benchSeat);
      for (const lx of [-1.8, 0.0, 1.8]) {
        const bLeg = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.54, 0.50), mats.darkIron);
        bLeg.position.set(lx, 0.32, -0.45);
        halte.add(bLeg);
      }
      contextGroup.add(halte);
    }

    // Scenic North-East Belvedere Overlook Terrace & Balustrade facing Gedung Rektorat UI across Teluk U (th = +135°)!
    {
      const belTh0 = (123.0 * Math.PI) / 180.0;
      const belTh1 = (147.0 * Math.PI) / 180.0;
      DPPDComplexBuilder._addCurvedBand(contextGroup, mats.courtyardGranite, belTh0, belTh1, 65.15, 68.35, 0.12, 0.24, 18);
      DPPDComplexBuilder._addCurvedBand(contextGroup, mats.whiteMullion, belTh0, belTh1, 68.02, 68.34, 1.15, 1.32, 18);
      for (let p = 0; p <= 10; p++) {
        const tp = belTh0 + (p / 10) * (belTh1 - belTh0);
        const bal = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.98, 0.32), mats.granitePlinth);
        bal.position.set(68.18 * Math.cos(tp), 0.70, 68.18 * Math.sin(tp));
        bal.rotation.y = -tp;
        contextGroup.add(bal);
      }
    }

    // 10 Ring-Road Campus Streetlamps along Inner Lawn Shoulder (r = 52.3, photocell warm luminaires v = -890.0)
    for (let k = 0; k < 10; k++) {
      const deg = -168.0 + k * 36.0;
      if (Math.abs(deg - 135.0) < 12.0 || Math.abs(deg - -156.0) < 12.0) continue;
      const th = (deg * Math.PI) / 180.0;
      const lamp = new THREE.Group();
      lamp.position.set(52.3 * Math.cos(th), 0.0, 52.3 * Math.sin(th));
      lamp.rotation.y = -th;

      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.17, 7.6, 8), mats.darkIron);
      pole.position.y = 3.8;
      const arm = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.12, 0.12), mats.darkIron);
      arm.position.set(0.82, 7.55, 0);
      const head = new THREE.Mesh(new THREE.BoxGeometry(0.72, 0.16, 0.34), mats.darkIron);
      head.position.set(1.58, 7.52, 0);
      const bulb = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.08, 0.26), mats.warmLuminaire);
      bulb.position.set(1.58, 7.42, 0);
      lamp.add(pole, arm, head, bulb);
      contextGroup.add(lamp);
    }

    // =========================================================================
    // 7. BAKE ENTIRE HIERARCHY INTO { horizontalGeo, verticalGeo } (+0 EXTRA DRAW CALLS!)
    // =========================================================================
    root.updateMatrixWorld(true);

    const hPos = [];
    const hNorm = [];
    const hUv = [];
    const hCol = [];
    const hIdx = [];
    let hCount = 0;

    const vPos = [];
    const vNorm = [];
    const vUv = [];
    const vCol = [];
    const vIdx = [];
    let vCount = 0;

    const tmpV = new THREE.Vector3();
    const tmpN = new THREE.Vector3();

    root.traverse((obj) => {
      if (!obj.isMesh || !obj.geometry || !obj.material) return;
      const mat = obj.material;
      const isZebra = mat.userData.batchKind === 'Z';
      const rgb = mat.userData.rgb || [0.85, 0.85, 0.85];
      const uvSentinel = mat.userData.uvSentinel;
      const keepCurbUV = Boolean(obj.userData.keepCurbUV);

      const g = obj.geometry;
      const posAttr = g.attributes.position;
      if (!posAttr) return;
      if (!g.attributes.normal) g.computeVertexNormals();
      const normAttr = g.attributes.normal;
      const uvAttr = g.attributes.uv;
      const indexAttr = g.index;

      const base = isZebra ? vCount : hCount;
      const tPos = isZebra ? vPos : hPos;
      const tNorm = isZebra ? vNorm : hNorm;
      const tUv = isZebra ? vUv : hUv;
      const tCol = isZebra ? vCol : hCol;
      const tIdx = isZebra ? vIdx : hIdx;

      for (let i = 0; i < posAttr.count; i++) {
        tmpV.set(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i)).applyMatrix4(obj.matrixWorld);
        tmpN.set(normAttr.getX(i), normAttr.getY(i), normAttr.getZ(i)).transformDirection(obj.matrixWorld);

        tPos.push(tmpV.x, tmpV.y, tmpV.z);
        tNorm.push(tmpN.x, tmpN.y, tmpN.z);
        tCol.push(rgb[0], rgb[1], rgb[2]);

        if (uvSentinel) {
          tUv.push(uvSentinel[0], uvSentinel[1]);
        } else if (isZebra && keepCurbUV && uvAttr) {
          tUv.push(uvAttr.getX(i), uvAttr.getY(i));
        } else if (isZebra) {
          tUv.push(0.055, 0.5);
        } else if (uvAttr) {
          tUv.push(uvAttr.getX(i), Math.max(0.0, uvAttr.getY(i)));
        } else {
          tUv.push(0.5, 0.5);
        }
      }

      if (indexAttr) {
        for (let i = 0; i < indexAttr.count; i++) {
          tIdx.push(base + indexAttr.getX(i));
        }
      } else {
        for (let i = 0; i < posAttr.count; i++) {
          tIdx.push(base + i);
        }
      }

      if (isZebra) {
        vCount += posAttr.count;
      } else {
        hCount += posAttr.count;
      }
    });

    const horizontalGeo = new THREE.BufferGeometry();
    horizontalGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(hPos), 3));
    horizontalGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(hNorm), 3));
    horizontalGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(hUv), 2));
    horizontalGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(hCol), 3));
    horizontalGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(hIdx), 1));

    const verticalGeo = new THREE.BufferGeometry();
    verticalGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vPos), 3));
    verticalGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(vNorm), 3));
    verticalGeo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(vUv), 2));
    verticalGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(vCol), 3));
    verticalGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(vIdx), 1));

    return { horizontalGeo, verticalGeo };
  }

  /**
   * Transforms local complex coordinates (x, y, z) to Pulau U world coordinates.
   */
  static localToWorld(lx, ly, lz, extraRotY = 0.0) {
    const s = TerrainDEMLoader.DPPD_SCALE;
    const totalRot = TerrainDEMLoader.DPPD_Y_ROT + extraRotY;
    const c = Math.cos(totalRot);
    const sn = Math.sin(totalRot);
    return {
      x: TerrainDEMLoader.DPPD_CX + s * (lx * c + lz * sn),
      y: TerrainDEMLoader.DPPD_PLATEAU_Y + s * ly,
      z: TerrainDEMLoader.DPPD_CZ + s * (-lx * sn + lz * c)
    };
  }

  /**
   * Returns exact world-space solid-body colliders for Kompleks DPPD (2 Lantai),
   * Gedung ILRC & DRPM (6 Lantai), Gedung STP-UI (8 Lantai), Gedung ITTC / Perpustakaan Lama (4 Lantai),
   * Halte Bikun DPPD-STP, & Parked Campus Vehicles.
   */
  static getSolidColliders() {
    const cols = [];
    const s = TerrainDEMLoader.DPPD_SCALE;
    const baseY = TerrainDEMLoader.DPPD_PLATEAU_Y;
    const rotEntrance = -0.25 * Math.PI;

    // 1. Gedung ITTC / Perpustakaan Lama UI — Central 4-Story Square Courtyard Heritage Building + NW 1-Story Annex
    cols.push({
      id: 'DPPD_Courtyard_CentralMonument',
      type: 'CYL',
      x: TerrainDEMLoader.DPPD_CX,
      z: TerrainDEMLoader.DPPD_CZ,
      r: 13.8 * s,
      minY: baseY,
      maxY: baseY + 15.85 * s
    });
    const nwAnnexW = DPPDComplexBuilder.localToWorld(-9.6, 0, -15.4, 0.0);
    cols.push({
      id: 'DPPD_ITTC_NWAnnex',
      type: 'CYL',
      x: nwAnnexW.x,
      z: nwAnnexW.z,
      r: 3.8 * s,
      minY: baseY,
      maxY: baseY + 3.8 * s
    });

    const courtyardTreeAnglesDeg = [82.0, -24.0, -138.0, 172.0];
    courtyardTreeAnglesDeg.forEach((cDeg, idx) => {
      const th = (cDeg * Math.PI) / 180.0;
      const wp = DPPDComplexBuilder.localToWorld(22.8 * Math.cos(th), 0, 22.8 * Math.sin(th), 0.0);
      cols.push({
        id: `DPPD_Courtyard_PlanterTree_${idx}`,
        type: 'CYL',
        x: wp.x,
        z: wp.z,
        r: 2.55 * s,
        minY: baseY,
        maxY: baseY + 4.4 * s
      });
      cols.push({
        id: `DPPD_Courtyard_TreeCanopy_${idx}`,
        type: 'ELLIPSOID',
        isCanopy: true,
        x: wp.x,
        cy: baseY + 4.85 * s,
        z: wp.z,
        rx: 3.45 * s,
        ry: 2.15 * s,
        skipWalk: true
      });
    });

    // Front Royal Palms (11.4m tall) & Ceremonial Merah-Putih Flagpole flanking Gedung DPPD UI portico
    [-13.6, 13.6].forEach((lxPalm, pIdx) => {
      const palmW = DPPDComplexBuilder.localToWorld(lxPalm, 0, 48.0, rotEntrance);
      cols.push({
        id: `DPPD_FrontRoyalPalmTrunk_${pIdx}`,
        type: 'CYL',
        x: palmW.x,
        z: palmW.z,
        r: 0.48 * s,
        minY: baseY,
        maxY: baseY + 10.6 * s
      });
      cols.push({
        id: `DPPD_FrontRoyalPalmCanopy_${pIdx}`,
        type: 'ELLIPSOID',
        isCanopy: true,
        x: palmW.x,
        cy: baseY + 10.75 * s,
        z: palmW.z,
        rx: 3.65 * s,
        ry: 1.65 * s,
        skipWalk: true
      });
    });
    const flagW = DPPDComplexBuilder.localToWorld(-9.85, 0, 49.6, rotEntrance);
    cols.push({
      id: 'DPPD_FrontFlagpole',
      type: 'CYL',
      x: flagW.x,
      z: flagW.z,
      r: 0.85 * s,
      minY: baseY,
      maxY: baseY + 11.9 * s
    });

    // 2. Gedung DPPD UI — 2-Story Curved Heritage Nusantara Wing + Front Gable & Portico
    const alphaMin = (DPPDComplexBuilder.DPPD_ALPHA_MIN_DEG * Math.PI) / 180.0;
    const alphaMax = (DPPDComplexBuilder.DPPD_ALPHA_MAX_DEG * Math.PI) / 180.0;
    for (let k = 0; k < 8; k++) {
      const a = alphaMin + ((k + 0.5) / 8) * (alphaMax - alphaMin);
      const wp = DPPDComplexBuilder.localToWorld(36.6 * Math.sin(a), 0, 36.6 * Math.cos(a), rotEntrance);
      cols.push({
        id: `DPPD_2Story_HeritageWing_${k}`,
        type: 'CYL',
        x: wp.x,
        z: wp.z,
        r: 8.6 * s,
        minY: baseY,
        maxY: baseY + 16.6 * s
      });
    }
    const porchW = DPPDComplexBuilder.localToWorld(0, 0, 44.2, rotEntrance);
    cols.push({
      id: 'DPPD_2Story_FrontCentralGableAndPorch',
      type: 'CYL',
      x: porchW.x,
      z: porchW.z,
      r: 6.8 * s,
      minY: baseY,
      maxY: baseY + 18.2 * s
    });

    // 3. Gedung STP-UI — 8-Story Curved Innovation Tower (-134° .. -44°) + Drop-Off Portico
    const stpTh0 = (DPPDComplexBuilder.STP_START_DEG * Math.PI) / 180.0;
    const stpTh1 = (DPPDComplexBuilder.STP_END_DEG * Math.PI) / 180.0;
    for (let k = 0; k < 9; k++) {
      const th = stpTh0 + ((k + 0.5) / 9) * (stpTh1 - stpTh0);
      const wp = DPPDComplexBuilder.localToWorld(37.5 * Math.cos(th), 0, 37.5 * Math.sin(th), 0.0);
      cols.push({
        id: `DPPD_STPUI_8Story_Sector_${k}`,
        type: 'CYL',
        x: wp.x,
        z: wp.z,
        r: 7.0 * s,
        minY: baseY,
        maxY: baseY + 47.2 * s
      });
    }
    const stpCenterTh = 0.5 * (stpTh0 + stpTh1);
    const porticoW = DPPDComplexBuilder.localToWorld(
      46.8 * Math.cos(stpCenterTh),
      0,
      46.8 * Math.sin(stpCenterTh),
      0.0
    );
    cols.push({
      id: 'DPPD_STPUI_Portico',
      type: 'CYL',
      x: porticoW.x,
      z: porticoW.z,
      r: 4.2 * s,
      minY: baseY,
      maxY: baseY + 6.8 * s
    });

    // 4. Gedung ILRC & DRPM UI — 6-Story Flat-Roofed Curved Research Laboratory (-4° .. +74°) + Twin-Pier Atrium Portal
    const ilrcTh0 = (DPPDComplexBuilder.ILRC_START_DEG * Math.PI) / 180.0;
    const ilrcTh1 = (DPPDComplexBuilder.ILRC_END_DEG * Math.PI) / 180.0;
    for (let k = 0; k < 8; k++) {
      const th = ilrcTh0 + ((k + 0.5) / 8) * (ilrcTh1 - ilrcTh0);
      const wp = DPPDComplexBuilder.localToWorld(37.4 * Math.cos(th), 0, 37.4 * Math.sin(th), 0.0);
      cols.push({
        id: `DPPD_ILRC_6Story_Sector_${k}`,
        type: 'CYL',
        x: wp.x,
        z: wp.z,
        r: 6.7 * s,
        minY: baseY,
        maxY: baseY + 36.25 * s
      });
    }
    const ilrcCenterTh = 0.5 * (ilrcTh0 + ilrcTh1);
    const ilrcPortalW = DPPDComplexBuilder.localToWorld(
      45.8 * Math.cos(ilrcCenterTh),
      0,
      45.8 * Math.sin(ilrcCenterTh),
      0.0
    );
    cols.push({
      id: 'DPPD_ILRC_AtriumCanopy',
      type: 'CYL',
      x: ilrcPortalW.x,
      z: ilrcPortalW.z,
      r: 3.6 * s,
      minY: baseY,
      maxY: baseY + 6.4 * s
    });

    // 5. Parked Campus Vehicles in East Visitor Parking Bay + Halte Bikun Shelter
    const parkTh0 = (-170.0 * Math.PI) / 180.0;
    const parkTh1 = (-140.0 * Math.PI) / 180.0;
    for (let c = 0; c < 4; c++) {
      const thCar = parkTh0 + 0.09 + ((c + 0.5) / 5) * (parkTh1 - parkTh0 - 0.10);
      const wp = DPPDComplexBuilder.localToWorld(42.2 * Math.cos(thCar), 0, 42.2 * Math.sin(thCar), 0.0);
      cols.push({
        id: `DPPD_ParkedCar_${c}`,
        type: 'CYL',
        x: wp.x,
        z: wp.z,
        r: 2.35 * s,
        minY: baseY,
        maxY: baseY + 1.55 * s
      });
    }

    const thHalte = (-112.0 * Math.PI) / 180.0;
    const halteW = DPPDComplexBuilder.localToWorld(70.60 * Math.cos(thHalte), 0, 70.60 * Math.sin(thHalte), 0.0);
    cols.push({
      id: 'DPPD_HalteBikun',
      type: 'CYL',
      x: halteW.x,
      z: halteW.z,
      r: 2.4 * s,
      minY: baseY,
      maxY: baseY + 3.7 * s
    });

    return cols;
  }
}
