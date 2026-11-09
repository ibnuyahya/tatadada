# NEXUS Omni-Scale Procedural Engine (v2.0 — Google Antigravity Edition)
## Project Brief: Gedung Rektorat UI Kampus Depok (`rektorat-ui-3d`)

### 1. Visi Proyek
Membangun model 3D prosedural Kelas-A dan simulator inspeksi/lingkungan interaktif untuk **Gedung Pusat Administrasi Universitas Indonesia (Gedung Rektorat UI — Kampus Depok)** karya **Prof. Ir. Gunawan Tjahjono, M.Arch., Ph.D. (1984–1987)** di atas WebGL / Three.js murni menggunakan **NEXUS Omni-Scale Procedural Framework v2.0**.

Seluruh geometri dibangun 100% secara matematis dari **Single Source of Spatial Truth (SSoST)** tanpa *Patchwork Primitive Assembly*, memadukan:
- **Kernel 1 (Parametric Profile Lofting):** Selubung menara berundak bersimetri dihedral $D_4$, sabuk teritisan tropis, dan kurva kemuncak **Atap Tajug / Tumpang Nusantara**.
- **Kernel 2 (Exact SDF & Manifold Boolean/Smooth-Union):** Integrasi 4 Menara Sudut (*Soko Guru*), kolong *pilotis* lantai dasar, podium berundak, tangga seremonial, dan kavitas pita kaca horizontal.

---

### 2. Target Performa & Kualitas Visual Mutlak
- **Frame Rate Target:** **120 FPS Ultra** stabil di Desktop & Android modern.
- **Batas Pemanggilan Gambar:** **$\le 25\text{ Draw Calls}$** untuk aset statis pasca-optimasi `NexusBatchOptimizer.optimizeHierarchy()`.
- **Keamanan Geometri & Topologi:** `0` nilai `NaN`/`Infinity`, penyimpangan amplop dimensi `< 0.5%` terhadap Tabel Ground-Truth, sambungan antar-subdomain rapat air (`0.0000 m Gap`), dan seluruh vektor normal menghadap keluar ($\mathbf{N} \cdot \mathbf{V}_{\text{out}} > 0$).
- **Partikel & Fluida 100% GPU:** Hujan tropis, kabut tipis pagi Danau Kenanga, kunang-kunang/partikel malam, dan riak air FBM rotasi 5–6 oktaf dihitung 100% di GLSL shader tanpa loop CPU.
- **Disiplin Kolorimetri & Malam Hari:** Bebas *double-gamma washout*; menggunakan **True Emissive Masking** (`#000000` mutlak pada beton/genteng pasif, hanya pita jendela interior aktif, lentera puncak Senat, lampu kolong podium, dan aksen emblem Makara UI yang memancarkan cahaya).
- **Presisi Spasial Lintas-Skala:** Menggunakan **Relative-To-Center (RTC)** (`p_local = P_world - C_center`) melalui `FloatingOriginManager.js` agar bebas *Float32 vertex jitter* dan *Z-fighting*.

---

### 3. Aturan Mutlak Ekosistem Deployment (`danial.my.id`)
- **Target Hosting:** GitHub Pages (`github.com/tatadada`) dengan sub-path `https://danial.my.id/rektorat-ui-3d/` terindeks ke **TATADADA Portal** (`https://danial.my.id/`).
- **Strict Relative Paths (`./`):** Seluruh `import` ES6, `fetch`, dan URL modul WAJIB diawali `./` (dilarang keras menggunakan `/core/...`).
- **Native ES6 Modules & Lazy-Loading:** Menggunakan `<script type="module">` dan `await import()` agar *initial load* `< 1 detik`.

---

### 4. Ergonomi UI `studio.html` & `index.html` (Desktop & Android Portrait)
1. **Default Collapsed Control Panel & Clean Viewport:**
   - Saat *startup*, panel kontrol wajib dalam kondisi **terlipat (*collapsed*)** (hanya 1 tombol *pill* ringkas di sudut layar).
   - **Dilarang keras** menampilkan *Ghost Photo Overlay* atau garis bantu/crosshair liar di tengah kanvas.
2. **Non-Occluding Viewport Offset (`applyNonOccludingViewportOffset`):**
   - **Android / Portrait (`aspect < 1.0` atau `width < 768px`):** Panel berbentuk **Compact Bottom-Sheet Drawer** (`max-height: 34vh`, *glassmorphism*). Titik pusat optik kamera digeser otomatis ke tengah ruang bebas di atas *Bottom-Sheet* menggunakan `camera.setViewOffset` + kompensasi FOV vertikal.
   - **Desktop / Landscape (`aspect >= 1.0`):** Panel berbentuk **Slim Side-Drawer** (`max-width: 280px`). Pusat optik kamera digeser ke samping *drawer* dengan `camera.setViewOffset`.
   - Saat panel ditutup kembali, `camera.clearViewOffset()` dipanggil otomatis.
3. **Kontrol Inspeksi Kelas-A:**
   - Preset Kamera Ortogonal Tri-Sudut (**Elevasi Horisontal $0^\circ\text{--}15^\circ$**, **Oblique $45^\circ$**, **Nadir $90^\circ$ Horizon-Locked**).
   - Sakelar **Studio Clay / Zebra Reflection Lines** (`uZebraInspectMode`).
   - Sakelar pencahayaan **Siang / Sore / Malam**.
   - Mode terbang **Drone 6-DOF Sinematik Anti-Kolisi** (`resolveSoftSpatialCollision`).

---

### 5. Arsitektur Direktori 4-Layer
```text
rektorat-ui-3d/
├── .agents/
│   ├── projectbrief.md
│   ├── systemPatterns.md
│   └── groundTruthSpec.md
├── index.html
├── studio.html
├── main.js
├── core/
│   ├── MasterSurfaceBuilder.js
│   ├── PhysicsKernel.js
│   ├── AudioSynthesizer.js
│   ├── CameraDirector.js
│   ├── WeatherOpticsEngine.js
│   ├── ProceduralTextureFactory.js
│   ├── BatchRenderer.js
│   └── GateVerifier.js
├── world-gis/
│   ├── FloatingOriginManager.js
│   ├── TerrainDEMLoader.js
│   ├── SplineCorridorBuilder.js
│   ├── AutogenBiomeSpawner.js
│   └── DynamicTrafficAI.js
├── library/
│   └── landmark/Gedung_Rektorat_UI/
│       ├── AssetMetaAndEnvelope.js
│       ├── MasterSurfaceKernel.js
│       ├── PrimaryStructure.js
│       ├── SecondarySubdomains.js
│       ├── KinematicRig.js
│       ├── MaterialShaderBank.js
│       └── index.js
└── data-routes/
    └── kampus-ui-context.json
```
