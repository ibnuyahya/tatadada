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
      // Exact Pulau U Road & Landscape Linear RGBs ("warna aspal sama" & seamless grass!)
      asphaltCrown: mkLin([0.20, 0.21, 0.23]),
      asphaltGutter: mkLin([0.16, 0.17, 0.19]),
      uiYellowLine: mkLin([0.96, 0.77, 0.12]),
      whiteEdgeLine: mkLin([0.94, 0.95, 0.97]),
      concreteCurb: mkLin([0.92, 0.91, 0.87]),
      campusGrass: mkLin([0.138, 0.338, 0.074]),
      sidewalkPaver: mkLin([0.48, 0.47, 0.44]),
      courtyardGranite: mkLin([0.56, 0.55, 0.52]),
      curbStriped: mkLin([0.95, 0.96, 0.98], 'Z'),

      // Architectural & Vehicle Palette
      terracottaRoof: mk(0x8e3322),
      terracottaRidge: mk(0x692215),
      terracottaFloor: mk(0xa84835),
      creamWall: mk(0xf0ead9),
      warmStonePanel: mk(0xe4dfd3),
      facadeWhite: mk(0xf3f5f8),
      ilrcConcrete: mk(0xdedbd4),
      ilrcBrickPortal: mk(0x9a3d2c),
      concreteTrim: mk(0x8b867e),
      granitePlinth: mk(0x63666c),
      darkWoodFascia: mk(0x3b2318),
      teakBench: mk(0x8b5a2b),
      LushShrub: mk(0x2f5e1e),
      canopyLeaf: mk(0x376e24),
      treeBark: mk(0x5a4332),
      mulchSoil: mk(0x614734),
      windowGlass: mk(0x122030, 'H', [0.5, -720.0]),
      darkFriezeGlass: mk(0x0f1a28),
      warmLuminaire: mk(0xfff1c4, 'H', [0.5, -890.0]),
      whiteMullion: mk(0xf5f7fa),
      darkAnodized: mk(0x232832),
      darkIron: mk(0x1f232b),
      copperACP: mk(0xc4794e),
      copperDark: mk(0x9b5833),
      brassSign: mk(0xe2b93b),
      uiYellowFascia: mk(0xf2be1a),
      solarPanel: mk(0x142654),
      galvSteel: mk(0x9aa3ad),
      disabilityBlue: mk(0x2563eb),
      hydrantRed: mk(0xdc2626),
      carWhite: mk(0xf1f5f9),
      carSilver: mk(0xcbd5e1),
      carDarkGrey: mk(0x333842),
      carNavy: mk(0x1e3a5f)
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
    K: ['10001', '10010', '10100', '11000', '10100', '10010', '10001'],
    L: ['10000', '10000', '10000', '10000', '10000', '10000', '11111'],
    M: ['10001', '11011', '10101', '10101', '10001', '10001', '10001'],
    N: ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
    O: ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
    P: ['11110', '10001', '10001', '11110', '10000', '10000', '10000'],
    R: ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
    S: ['01111', '10000', '10000', '01110', '00001', '00001', '11110'],
    T: ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
    U: ['10001', '10001', '10001', '10001', '10001', '10001', '01110'],
    V: ['10001', '10001', '10001', '10001', '10001', '01010', '00100'],
    '-': ['00000', '00000', '00000', '01110', '00000', '00000', '00000']
  };

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

    // =========================================================================
    // 2. GEDUNG DPPD UI — STRICTLY 2 STORIES (2 LANTAI: L1 VERANDAH ARCADE + L2 RIBBON WINDOWS)
    //    Front Heritage Nusantara Pavilion facing Gedung Rektorat UI (0, 0) across Teluk U!
    // =========================================================================
    const alphaMin = (DPPDComplexBuilder.DPPD_ALPHA_MIN_DEG * Math.PI) / 180.0;
    const alphaMax = (DPPDComplexBuilder.DPPD_ALPHA_MAX_DEG * Math.PI) / 180.0;
    // Note: in entranceGroup, x = r * sin(alpha), z = r * cos(alpha) -> polar angle in entranceGroup XZ is thE = pi/2 - alpha
    const dppdTh0 = Math.PI * 0.5 - alphaMax;
    const dppdTh1 = Math.PI * 0.5 - alphaMin;

    // 2A. Raised Granite & Terracotta Verandah Plinth (y = -0.35 .. 0.65)
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.granitePlinth, dppdTh0, dppdTh1, 28.4, 44.8, -0.35, 0.55, 44);
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.terracottaFloor, dppdTh0, dppdTh1, 28.5, 44.7, 0.55, 0.68, 44);

    // Low Manicured Front Stone Planter Beds & Shrubs (y = 0.0 .. 0.92 — NEVER buries Lantai 1!)
    for (const [degA, degB] of [
      [-43.5, -13.5],
      [13.5, 43.5]
    ]) {
      const tA = Math.PI * 0.5 - (degB * Math.PI) / 180.0;
      const tB = Math.PI * 0.5 - (degA * Math.PI) / 180.0;
      DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.granitePlinth, tA, tB, 44.75, 46.55, 0.0, 0.72, 20);
      DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.mulchSoil, tA, tB, 44.95, 46.35, 0.70, 0.78, 20);
      for (let sh = 0; sh < 9; sh++) {
        const thS = tA + ((sh + 0.5) / 9) * (tB - tA);
        const bush = new THREE.Mesh(new THREE.SphereGeometry(0.72, 10, 8), mats.LushShrub);
        bush.scale.set(1.25, 0.72, 1.0);
        bush.position.set(45.65 * Math.cos(thS), 1.02, 45.65 * Math.sin(thS));
        entranceGroup.add(bush);
      }
    }

    // 2B. LANTAI 1 (y = 0.68 .. 6.25) & LANTAI 2 (y = 6.25 .. 11.75) Curved Masonry & Glazed Walls
    // Recessed L1 wall (r = 29.0 .. 43.0) creating a shaded 1.7m Nusantara front verandah colonnade at r = 44.3!
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.creamWall, dppdTh0, dppdTh1, 29.0, 43.0, 0.68, 1.85, 44);
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.windowGlass, dppdTh0, dppdTh1, 29.15, 42.85, 1.85, 5.35, 44);
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.creamWall, dppdTh0, dppdTh1, 29.0, 43.2, 5.35, 6.10, 44);

    // Projecting L1/L2 Inter-Story White Spandrel Slab & Tropical Sunshade Drip Cornice (y = 6.10 .. 6.65)
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.whiteMullion, dppdTh0, dppdTh1, 28.55, 44.85, 6.10, 6.65, 48);

    // LANTAI 2 Wall, Mullioned Ribbon Windows, & Top Eave Ring Beam (y = 6.65 .. 11.75)
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.creamWall, dppdTh0, dppdTh1, 28.85, 43.95, 6.65, 7.65, 44);
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.windowGlass, dppdTh0, dppdTh1, 29.05, 43.75, 7.65, 10.85, 44);
    DPPDComplexBuilder._addCurvedBand(entranceGroup, mats.whiteMullion, dppdTh0, dppdTh1, 28.55, 44.65, 10.85, 11.75, 48);

    // 2C. Rhythmic L1 Colonnade Pillars, L1 & L2 Window Mullions, & Architectural AC Louver Grilles (Outer + Inner!)
    const dppdBays = 22;
    for (let i = 0; i <= dppdBays; i++) {
      const alpha = alphaMin + (i / dppdBays) * (alphaMax - alphaMin);
      const deg = (alpha * 180.0) / Math.PI;
      const inCenterGable = Math.abs(deg) < 10.8;
      const sA = Math.sin(alpha);
      const cA = Math.cos(alpha);

      if (!inCenterGable) {
        // Outer L1 Nusantara Verandah Square Column + Stone Base & Capital
        const col = new THREE.Mesh(new THREE.BoxGeometry(0.62, 5.45, 0.62), mats.whiteMullion);
        col.position.set(44.25 * sA, 3.38, 44.25 * cA);
        col.rotation.y = alpha;
        entranceGroup.add(col);

        const colBase = new THREE.Mesh(new THREE.BoxGeometry(0.80, 0.55, 0.80), mats.granitePlinth);
        colBase.position.set(44.25 * sA, 0.95, 44.25 * cA);
        colBase.rotation.y = alpha;
        entranceGroup.add(colBase);

        // Outer L1 & L2 Masonry Pier + White Window Mullions
        const pierL1 = new THREE.Mesh(new THREE.BoxGeometry(0.54, 4.65, 0.42), mats.creamWall);
        pierL1.position.set(42.92 * sA, 3.55, 42.92 * cA);
        pierL1.rotation.y = alpha;
        entranceGroup.add(pierL1);

        const pierL2 = new THREE.Mesh(new THREE.BoxGeometry(0.56, 4.25, 0.46), mats.creamWall);
        pierL2.position.set(43.85 * sA, 8.78, 43.85 * cA);
        pierL2.rotation.y = alpha;
        entranceGroup.add(pierL2);
      }

      // Inner Courtyard L1 & L2 Pilasters (facing Plaza Taman Inovasi Tengah!)
      const inPier = new THREE.Mesh(new THREE.BoxGeometry(0.54, 10.45, 0.46), mats.creamWall);
      inPier.position.set(28.92 * sA, 5.90, 28.92 * cA);
      inPier.rotation.y = alpha;
      entranceGroup.add(inPier);

      // Sub-mullions & neat L2 AC louver grilles inside each bay
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
            m2.position.set(43.82 * Math.sin(aSub), 9.25, 43.82 * Math.cos(aSub));
            m2.rotation.y = aSub;
            entranceGroup.add(m2);
          }

          // Neat architectural louvered AC box tucked beneath L2 sill on alternating bays
          if (i % 2 === 0) {
            const acBox = new THREE.Mesh(new THREE.BoxGeometry(1.05, 0.72, 0.48), mats.whiteMullion);
            acBox.position.set(44.12 * sm, 7.12, 44.12 * cm);
            acBox.rotation.y = aMid;
            entranceGroup.add(acBox);

            const acGrille = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.52, 12), mats.darkAnodized);
            acGrille.rotation.x = Math.PI * 0.5;
            acGrille.position.set(44.14 * sm, 7.12, 44.14 * cm);
            acGrille.rotation.y = aMid;
            entranceGroup.add(acGrille);
          }
        }

        // Inner courtyard window mullions
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

    // Clean Single Watertight 2-Story End-Wall Fenestration at alphaMin (-44°) & alphaMax (+44°)
    for (const endAlpha of [alphaMin, alphaMax]) {
      const endGroup = new THREE.Group();
      endGroup.rotation.y = endAlpha;
      entranceGroup.add(endGroup);
      const sgn = endAlpha > 0 ? 1 : -1;
      for (const rW of [33.0, 39.2]) {
        for (const yW of [3.55, 9.20]) {
          const frame = new THREE.Mesh(new THREE.BoxGeometry(0.24, 3.10, 3.80), mats.whiteMullion);
          frame.position.set(sgn * 0.08, yW, rW);
          endGroup.add(frame);
          const pane = new THREE.Mesh(new THREE.BoxGeometry(0.28, 2.75, 3.45), mats.windowGlass);
          pane.position.set(sgn * 0.10, yW, rW);
          endGroup.add(pane);
        }
      }
    }

    // Rear Courtyard Portal Canopy on Gedung DPPD (alpha = 0°, r = 26.2 .. 29.0) facing Plaza Taman Inovasi Tengah
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

    // 2E. Central 2-Story Front Nusantara Gable Pavilion, UI Makara Pediment, 1-Story Portico, Stairs, Ramp & Flagpole!
    const gHalfW = 8.4;
    const gZBack = 35.8;
    const gZFront = 45.4;
    const gEaveY = 11.75;
    const gPeakY = 18.05;

    // 2-Story Front Central Projection Block (L1 Grand Entrance Doors + L2 Conference Ribbon Windows)
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

    // Front L1 Grand Glazed Entrance Lobby & L2 5-Bay Conference Windows
    const l1LobbyGlass = new THREE.Mesh(new THREE.BoxGeometry(11.6, 4.55, 0.30), mats.windowGlass);
    l1LobbyGlass.position.set(0, 3.15, gZFront + 0.04);
    entranceGroup.add(l1LobbyGlass);

    const l2RibbonGlass = new THREE.Mesh(new THREE.BoxGeometry(13.6, 3.25, 0.30), mats.windowGlass);
    l2RibbonGlass.position.set(0, 9.22, gZFront + 0.04);
    entranceGroup.add(l2RibbonGlass);

    for (let m = -3; m <= 3; m++) {
      const mx = m * 2.24;
      const mul2 = new THREE.Mesh(new THREE.BoxGeometry(0.22, 3.40, 0.38), mats.whiteMullion);
      mul2.position.set(mx, 9.22, gZFront + 0.08);
      entranceGroup.add(mul2);
      if (Math.abs(m) <= 2) {
        const mul1 = new THREE.Mesh(new THREE.BoxGeometry(0.24, 4.65, 0.38), mats.whiteMullion);
        mul1.position.set(m * 2.85, 3.15, gZFront + 0.08);
        entranceGroup.add(mul1);
      }
    }

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

    // 1-Story Front Drop-Off Portico Canopy, 4 Sculpted Columns, Ceremonial Terracotta Steps, Accessible Ramp, & Flagpole
    const porchPodium = new THREE.Mesh(new THREE.BoxGeometry(12.8, 0.68, 4.6), mats.terracottaFloor);
    porchPodium.position.set(0, 0.34, 47.6);
    entranceGroup.add(porchPodium);

    for (let st = 0; st < 3; st++) {
      const step = new THREE.Mesh(
        new THREE.BoxGeometry(8.8 + st * 0.5, 0.22, 0.55),
        st % 2 === 0 ? mats.terracottaFloor : mats.concreteCurb
      );
      step.position.set(0, 0.56 - st * 0.21, 50.15 + st * 0.48);
      entranceGroup.add(step);
    }

    // Inclusive Blue-Tile Disability Ramp on West (+X in entranceGroup) flank of portico
    const ramp = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.22, 3.6), mats.disabilityBlue);
    ramp.position.set(5.45, 0.34, 50.3);
    ramp.rotation.x = 0.16;
    entranceGroup.add(ramp);
    for (const rx of [4.62, 6.28]) {
      const rail = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.85, 3.5), mats.galvSteel);
      rail.position.set(rx, 0.86, 50.3);
      entranceGroup.add(rail);
    }

    for (const px of [-5.2, -1.85, 1.85, 5.2]) {
      const pCol = new THREE.Mesh(new THREE.BoxGeometry(0.58, 5.05, 0.58), mats.whiteMullion);
      pCol.position.set(px, 3.18, 49.45);
      entranceGroup.add(pCol);
      const pBase = new THREE.Mesh(new THREE.BoxGeometry(0.76, 0.58, 0.76), mats.granitePlinth);
      pBase.position.set(px, 0.96, 49.45);
      entranceGroup.add(pBase);
    }

    const porchSlab = new THREE.Mesh(new THREE.BoxGeometry(13.6, 0.46, 5.2), mats.whiteMullion);
    porchSlab.position.set(0, 5.82, 47.85);
    entranceGroup.add(porchSlab);

    // Symmetrical Nusantara Gabled Portico Roof (aligned 100% flush with main central gable axis!)
    const pHalfW = 6.55;
    const pZBack = 45.30;
    const pZFront = 50.45;
    const pEaveY = 6.05;
    const pPeakY = 8.28;
    const porchPediment = new THREE.Mesh(
      DPPDComplexBuilder._buildPolygonPrismGeo(
        [
          [-pHalfW, pEaveY],
          [pHalfW, pEaveY],
          [0.0, pPeakY - 0.18]
        ],
        pZFront - pZBack
      ),
      mats.creamWall
    );
    porchPediment.position.z = pZBack;
    entranceGroup.add(porchPediment);

    const porchInset = new THREE.Mesh(
      DPPDComplexBuilder._buildPolygonPrismGeo(
        [
          [-pHalfW + 1.15, pEaveY + 0.28],
          [pHalfW - 1.15, pEaveY + 0.28],
          [0.0, pPeakY - 0.68]
        ],
        0.16
      ),
      mats.warmStonePanel
    );
    porchInset.position.z = pZFront - 0.04;
    entranceGroup.add(porchInset);

    const pSlopeLen = Math.hypot(pHalfW + 0.65, pPeakY - pEaveY + 0.22);
    const pSlopeAng = Math.atan2(pPeakY - pEaveY, pHalfW);
    for (const sgn of [-1, 1]) {
      const pSlope = new THREE.Mesh(
        new THREE.BoxGeometry(pSlopeLen, 0.24, pZFront - pZBack + 1.10),
        mats.terracottaRoof
      );
      pSlope.position.set(
        sgn * (pHalfW + 0.65) * 0.48,
        (pEaveY + pPeakY) * 0.5 - 0.04,
        (pZBack + pZFront) * 0.5 + 0.18
      );
      pSlope.rotation.z = -sgn * pSlopeAng;
      entranceGroup.add(pSlope);

      const pBarge = new THREE.Mesh(
        new THREE.BoxGeometry(pSlopeLen + 0.08, 0.30, 0.18),
        mats.darkWoodFascia
      );
      pBarge.position.set(
        sgn * (pHalfW + 0.65) * 0.48,
        (pEaveY + pPeakY) * 0.5 - 0.08,
        pZFront + 0.72
      );
      pBarge.rotation.z = -sgn * pSlopeAng;
      entranceGroup.add(pBarge);
    }
    const porchRidge = new THREE.Mesh(
      new THREE.BoxGeometry(0.40, 0.30, pZFront - pZBack + 1.25),
      mats.terracottaRidge
    );
    porchRidge.position.set(0, pPeakY + 0.10, (pZBack + pZFront) * 0.5 + 0.18);
    entranceGroup.add(porchRidge);

    // Photocell warm portico downlights (v = -890.0)
    for (const lx of [-3.5, 0.0, 3.5]) {
      const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.65, 0.08, 0.65), mats.warmLuminaire);
      lamp.position.set(lx, 5.56, 48.2);
      entranceGroup.add(lamp);
    }

    // Ceremonial Merah-Putih Flagpole & Pair of Framing Royal Palms (clear of portico, windows, & crosswalk!)
    const flagPedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.82, 0.72, 16), mats.granitePlinth);
    flagPedestal.position.set(-9.85, 0.36, 49.6);
    entranceGroup.add(flagPedestal);
    const flagPole = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.13, 11.2, 10), mats.whiteMullion);
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

    // =========================================================================
    // 3. GEDUNG ILRC UI — STRICTLY 6 STORIES WITH 100% UNIFORM FLAT ROOFLINE (RATA TINGGINYA!)
    //    Modeled directly after real Gedung ILRC UI Depok photos:
    //    - 100.0% continuous, razor-level horizontal top white crown fascia & parapet (y = 36.25 from end to end — ZERO stepped wings or rooftop boxes!)
    //    - 6 continuous curved precast-white horizontal spandrel & sunshade bands (outer wings + inner courtyard!)
    //    - Central 6-story symmetrical terracotta-brick twin-pier glazed atrium entrance portal terminating cleanly underneath the flat top crown fascia!
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
      52.8,
      0.02,
      0.16,
      16
    );
    DPPDComplexBuilder._addCurvedBand(contextGroup, mats.windowGlass, ilrcTh0, ilrcTh1, ilrcRIn + 0.45, ilrcROut - 0.55, 0.48, 5.80, 48);

    // Build all 6 clearly countable floor slabs, spandrel bands, & projecting sunshade ledges (L1..L6)!
    // Outer sunshades terminate cleanly against the twin terracotta portal piers while inner courtyard bands run full-arc!
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
      // Outer right & left wing projecting curved sunshade ledges (flanking the twin terracotta piers cleanly, matching real photo!)
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

    // 100.0% FLAT, CONTINUOUS, UNBROKEN FULL-ARC TOP ROOF CROWN FASCIA & COPING (y = 34.15 .. 36.25 across ENTIRE ilrcTh0..ilrcTh1!)
    // Zero rooftop penthouse hump, zero raised central signboard box, zero stepped wing — 0.00m height variation from end to end!
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

    // Both ILRC End Walls (ilrcTh0 & ilrcTh1): Clean Precast-White Stair-Core Frame & 6-Story Glazed Stairwell Slot
    // Terminates flush at y = 34.15 right beneath the continuous 34.15..36.25 flat roof crown (zero corner height step!)
    for (const [thEnd, sgn] of [
      [ilrcTh0, -1],
      [ilrcTh1, 1]
    ]) {
      const endGrp = new THREE.Group();
      endGrp.rotation.y = -thEnd;
      contextGroup.add(endGrp);

      const stairTrim = new THREE.Mesh(new THREE.BoxGeometry(8.4, 34.15, 0.36), mats.facadeWhite);
      stairTrim.position.set(37.4, 17.075, sgn * 0.16);
      endGrp.add(stairTrim);

      const stairGlass = new THREE.Mesh(new THREE.BoxGeometry(3.0, 31.2, 0.44), mats.windowGlass);
      stairGlass.position.set(37.4, 17.6, sgn * 0.20);
      endGrp.add(stairGlass);

      for (let fl = 0; fl < 6; fl++) {
        const bar = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.32, 0.50), mats.facadeWhite);
        bar.position.set(37.4, ilrcFloorTops[fl] - 0.42, sgn * 0.22);
        endGrp.add(bar);
      }
    }

    // ICONIC ILRC 6-STORY SYMMETRICAL TERRACOTTA-BRICK TWIN-PIER MONUMENTAL ATRIUM PORTAL (at +35°, matching real photo!)
    // Both piers rise from y = 0.0 to y = 34.15 — terminating cleanly UNDERNEATH the continuous flat white top crown fascia!
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
    // Recessed 6-story central glass atrium curtain wall + vertical mullions between the twin terracotta piers
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.windowGlass,
      rPierTh1,
      lPierTh0,
      ilrcROut - 0.58,
      ilrcROut + 0.06,
      0.48,
      34.15,
      12
    );
    for (const frac of [-0.5, 0.0, 0.5]) {
      const thM = ilrcCenterTh + frac * (lPierTh0 - rPierTh1);
      const aMul = new THREE.Mesh(new THREE.BoxGeometry(0.18, 27.9, 0.18), mats.darkAnodized);
      aMul.position.set((ilrcROut + 0.08) * Math.cos(thM), 20.15, (ilrcROut + 0.08) * Math.sin(thM));
      aMul.rotation.y = -thM;
      contextGroup.add(aMul);
    }

    // Projecting ILRC Ground-Floor Drop-Off Canopy + Photocell Downlights!
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
    for (const sgn of [-1, 1]) {
      const thP = ilrcCenterTh + sgn * 0.11;
      const cCol = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.34, 5.55, 12), mats.ilrcBrickPortal);
      cCol.position.set((ilrcROut + 4.1) * Math.cos(thP), 2.78, (ilrcROut + 4.1) * Math.sin(thP));
      contextGroup.add(cCol);
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
    // 4. GEDUNG STP-UI — STRICTLY 8 STORIES (8 LANTAI: L1 LOBBY + L2-L3 COPPER DIAGRID + L4-L7 WHITE/BOX + L8 CROWN)
    //    Modeled directly after real Gedung Science Techno Park UI Depok photos:
    //    - Wrapping Copper-Terracotta ACP Vertical End Frames & Overhanging Top Roof Cornice
    //    - L2-L3 Angled Copper-ACP V-Diagrid Screen over 2-Story Mullioned Glazing
    //    - L4-L7 4 Crisp Curved White Spandrel Floors with Staggered Dark Box-Window Accent Frames
    //    - L8 Dark-Glazed Penthouse Frieze bearing Golden UI Makara + High-Contrast 3D "SCIENCE TECHNO PARK / UNIVERSITAS INDONESIA"
    //    - Full 8-Story Inner Courtyard & End-Wall Spandrels/Mullions (ZERO black void wall!)
    // =========================================================================
    const stpTh0 = (DPPDComplexBuilder.STP_START_DEG * Math.PI) / 180.0;
    const stpTh1 = (DPPDComplexBuilder.STP_END_DEG * Math.PI) / 180.0;
    const stpCenterTh = 0.5 * (stpTh0 + stpTh1); // -84°
    const stpRIn = 31.2;
    const stpROut = 43.8;
    const stpWingPad = 0.036; // Angular width of left & right copper-ACP framing pilasters
    const stpInnerTh0 = stpTh0 + stpWingPad;
    const stpInnerTh1 = stpTh1 - stpWingPad;
    // Exact 8 countable floor ceiling heights:
    const stpFloorTops = [6.85, 12.45, 18.05, 23.45, 28.85, 34.25, 39.65, 45.25];

    // Plinth & L1 Recessed Dark-Glazed Exhibition Lobby
    DPPDComplexBuilder._addCurvedBand(contextGroup, mats.granitePlinth, stpTh0, stpTh1, stpRIn - 0.3, stpROut + 0.3, -0.35, 0.48, 52);
    DPPDComplexBuilder._addCurvedBand(contextGroup, mats.windowGlass, stpInnerTh0, stpInnerTh1, stpRIn + 0.4, stpROut - 0.55, 0.48, 6.10, 48);
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

    // Signature L2-L3 Angled Copper-Terracotta ACP V-Diagrid Structural Ribs (matching real STP-UI facade!)
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
      // Continuous curved sunshade fin cleanly bounded inside [stpInnerTh0, stpInnerTh1] (ZERO floating ribs past end walls!)
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

    // L8 (Top 8th-Floor Dark-Glazed Crown Frieze: y = 39.65 .. 45.25)
    // Outer side uses darkFriezeGlass backing under signage + night-lit window slots so white 3D letters pop in bright daylight!
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

    // Both STP-UI End Walls (stpTh0 & stpTh1): Articulated 8-Story Glazed Stair/Elevator Core & Floor Transoms!
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

    // STP-UI Grand L1 Drop-Off Canopy Portico at stpCenterTh (-84°) + Photocell Downlights + "STP-UI" 3D Canopy Letters!
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
      mats.warmLuminaire,
      stpCenterTh - 0.11,
      stpCenterTh + 0.11,
      stpROut + 1.4,
      stpROut + 4.5,
      5.78,
      5.84,
      10
    );
    DPPDComplexBuilder._addCurvedBand(
      contextGroup,
      mats.courtyardGranite,
      stpCenterTh - 0.18,
      stpCenterTh + 0.18,
      stpROut,
      52.8,
      0.02,
      0.16,
      16
    );
    for (const sgn of [-1, 1]) {
      const thP = stpCenterTh + sgn * 0.115;
      const pCol = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.42, 5.85, 12), mats.darkAnodized);
      pCol.position.set((stpROut + 4.7) * Math.cos(thP), 2.95, (stpROut + 4.7) * Math.sin(thP));
      contextGroup.add(pCol);
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

    // =========================================================================
    // 5. PLAZA TAMAN INOVASI TENGAH (OPEN LANDSCAPED CENTRAL COURTYARD REPLACING 4-STORY BOX!)
    //    Concentric Granite/Terracotta Plaza, UI Makara Monument, Covered Link Pergolas, Trees, Benches & Bollards
    // =========================================================================
    const courtyardOuterPad = new THREE.Mesh(
      new THREE.CylinderGeometry(31.2, 31.2, 0.16, 64),
      mats.courtyardGranite
    );
    courtyardOuterPad.position.y = 0.02;
    dppdGroup.add(courtyardOuterPad);

    const courtyardBrickRing = new THREE.Mesh(
      new THREE.CylinderGeometry(19.5, 19.5, 0.18, 56),
      mats.terracottaFloor
    );
    courtyardBrickRing.position.y = 0.03;
    dppdGroup.add(courtyardBrickRing);

    const courtyardInnerGranite = new THREE.Mesh(
      new THREE.CylinderGeometry(16.8, 16.8, 0.20, 56),
      mats.courtyardGranite
    );
    courtyardInnerGranite.position.y = 0.04;
    dppdGroup.add(courtyardInnerGranite);

    // Central Tiered Botanical Planter & Sculptural UI Makara Innovation Obelisk (r = 0 .. 4.8)
    const centerPlanterRim = new THREE.Mesh(new THREE.CylinderGeometry(4.8, 5.1, 0.72, 32), mats.granitePlinth);
    centerPlanterRim.position.y = 0.38;
    dppdGroup.add(centerPlanterRim);
    const centerPlanterGrass = new THREE.Mesh(new THREE.CylinderGeometry(4.4, 4.4, 0.76, 32), mats.campusGrass);
    centerPlanterGrass.position.y = 0.40;
    dppdGroup.add(centerPlanterGrass);

    const obeliskBase = new THREE.Mesh(new THREE.CylinderGeometry(1.55, 1.85, 1.15, 8), mats.granitePlinth);
    obeliskBase.position.y = 1.15;
    dppdGroup.add(obeliskBase);
    const obeliskShaft = new THREE.Mesh(new THREE.BoxGeometry(1.35, 3.85, 1.35), mats.copperACP);
    obeliskShaft.position.y = 3.45;
    obeliskShaft.rotation.y = Math.PI * 0.25;
    dppdGroup.add(obeliskShaft);
    const obeliskCrown = DPPDComplexBuilder._buildUIMakaraEmblem(mats, 1.05);
    obeliskCrown.position.set(0, 5.95, 0);
    obeliskCrown.rotation.y = Math.PI * 0.75; // Faces Gedung DPPD & Rektorat axis
    dppdGroup.add(obeliskCrown);

    // 3 Radial Granite Promenade Spokes linking Center Plaza to DPPD (+135°), ILRC (+35°), & STP-UI (-89°)
    for (const spokeDeg of [135.0, 35.0, -89.0]) {
      const thS = (spokeDeg * Math.PI) / 180.0;
      const spoke = new THREE.Mesh(new THREE.BoxGeometry(23.0, 0.06, 4.2), mats.whiteMullion);
      spoke.position.set(16.8 * Math.cos(thS), 0.13, 16.8 * Math.sin(thS));
      spoke.rotation.y = -thS;
      dppdGroup.add(spoke);
    }

    // Open Landscaped Granite Courtyard Gateway Promenades between DPPD, ILRC, & STP-UI
    // (Zero 1-story side roofs abutting ILRC so Gedung ILRC's 6-story flat-top silhouette stays 100% pure on both flanks!)
    for (const [degA, degB, rOutProm] of [
      [74.5, 90.5, 44.2],
      [-36.0, -12.0, 52.8],
      [-151.5, -134.5, 35.2]
    ]) {
      const t0 = (Math.min(degA, degB) * Math.PI) / 180.0;
      const t1 = (Math.max(degA, degB) * Math.PI) / 180.0;
      DPPDComplexBuilder._addCurvedBand(contextGroup, mats.courtyardGranite, t0, t1, 30.6, rOutProm, 0.02, 0.15, 16);
    }

    // 4 Landscaped Courtyard Planter Islands with Shade Trees, Teak Benches, & Warm Photocell Bollards
    const courtyardTreeAnglesDeg = [82.0, -24.0, -138.0, 172.0];
    for (const cDeg of courtyardTreeAnglesDeg) {
      const th = (cDeg * Math.PI) / 180.0;
      const rIsle = 12.2;
      const px = rIsle * Math.cos(th);
      const pz = rIsle * Math.sin(th);

      const planter = new THREE.Mesh(new THREE.CylinderGeometry(2.65, 2.85, 0.52, 20), mats.granitePlinth);
      planter.position.set(px, 0.34, pz);
      contextGroup.add(planter);

      const turf = new THREE.Mesh(new THREE.CylinderGeometry(2.35, 2.35, 0.56, 20), mats.campusGrass);
      turf.position.set(px, 0.36, pz);
      contextGroup.add(turf);

      contextGroup.add(DPPDComplexBuilder._buildCourtyardTree(mats, px, 0.55, pz, 1.05));

      // Teak-and-concrete courtyard bench facing center monument
      const bench = new THREE.Group();
      bench.position.set((rIsle - 3.35) * Math.cos(th), 0.14, (rIsle - 3.35) * Math.sin(th));
      bench.rotation.y = -th + Math.PI * 0.5;
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
      const bx = 22.8 * Math.cos(th);
      const bz = 22.8 * Math.sin(th);
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
    // East Visitor Parking Bay (th = -170° .. -140°, r = 35.5 .. 52.8) — same Pulau U asphalt color!
    const parkTh0 = (-170.0 * Math.PI) / 180.0;
    const parkTh1 = (-140.0 * Math.PI) / 180.0;
    DPPDComplexBuilder._addCurvedBand(contextGroup, mats.asphaltCrown, parkTh0, parkTh1, 35.8, 52.85, 0.02, 0.085, 18);
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

    // Campus Entrance Security Post & Monolith Sign cleanly INSIDE inner lawn at th = -136° (ZERO obstruction on ring road!)
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

    // 10 Ring-Road Campus Streetlamps along Inner Lawn Shoulder (r = 51.6, photocell warm luminaires v = -890.0)
    for (let k = 0; k < 10; k++) {
      const deg = -168.0 + k * 36.0;
      if (Math.abs(deg - 135.0) < 12.0 || Math.abs(deg - -156.0) < 12.0) continue;
      const th = (deg * Math.PI) / 180.0;
      const lamp = new THREE.Group();
      lamp.position.set(51.6 * Math.cos(th), 0.0, 51.6 * Math.sin(th));
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
   * Gedung ILRC (6 Lantai), Gedung STP-UI (8 Lantai), Plaza Taman Inovasi Tengah,
   * Halte Bikun DPPD-STP, & Parked Campus Vehicles.
   */
  static getSolidColliders() {
    const cols = [];
    const s = TerrainDEMLoader.DPPD_SCALE;
    const baseY = TerrainDEMLoader.DPPD_PLATEAU_Y;
    const rotEntrance = -0.25 * Math.PI;

    // 1. Plaza Taman Inovasi Tengah — Central Makara Obelisk & Planter Ring (open walkable plaza around it!)
    cols.push({
      id: 'DPPD_Courtyard_CentralMonument',
      type: 'CYL',
      x: TerrainDEMLoader.DPPD_CX,
      z: TerrainDEMLoader.DPPD_CZ,
      r: 5.0 * s,
      minY: baseY,
      maxY: baseY + 6.5 * s
    });

    const courtyardTreeAnglesDeg = [82.0, -24.0, -138.0, 172.0];
    courtyardTreeAnglesDeg.forEach((cDeg, idx) => {
      const th = (cDeg * Math.PI) / 180.0;
      const wp = DPPDComplexBuilder.localToWorld(12.2 * Math.cos(th), 0, 12.2 * Math.sin(th), 0.0);
      cols.push({
        id: `DPPD_Courtyard_PlanterTree_${idx}`,
        type: 'CYL',
        x: wp.x,
        z: wp.z,
        r: 2.85 * s,
        minY: baseY,
        maxY: baseY + 6.2 * s
      });
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

    // 3. Gedung STP-UI — 8-Story Curved Innovation Tower (-132° .. -36°) + Drop-Off Portico
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

    // 4. Gedung ILRC UI — 6-Story Flat-Roofed Curved Research Laboratory (-4° .. +74°) + Twin-Pier Atrium Portal
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
