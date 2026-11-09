# NEXUS Omni-Scale System Patterns & 18 Golden Laws (v2.0)

Dokumen ini merupakan konstitusi teknis permanen proyek di dalam memori `.agents/`. Setiap modul dan sub-agen wajib mematuhi pola arsitektur dan 18 Hukum Emas di bawah ini tanpa pengecualian.

---

## A. Metodologi Pemodelan Kelas-A 6-Tahap & Single Source of Spatial Truth (SSoST)

1. **Tahap 1 — Kalibrasi Amplop Ortografis & Tabel Ground-Truth (`AssetMetaAndEnvelope.js`)**
   - Menetapkan koordinat baku (`(0,0,0)` pusat alas, `+Y` Atas, `+Z` Depan/Selatan, `+X` Kanan/Timur), `scaleUnit`, `realDimensionsMeters`, `foundationSkirtDepth`, `obstacleColliders`, dan `orthoValidationPresets`.
2. **Tahap 2 — Kerangka Tulang Punggung Utama & Jaring Kurva Karakter (`MasterSurfaceKernel.js` - Bagian A)**
   - Mendefinisikan tulang punggung aksial, sumbu simetri dihedral, dan tabel stasiun profil morfometrik.
3. **Tahap 3 — Blok Permukaan Induk Rapat Air / Watertight Master Clay Buck (`MasterSurfaceKernel.js` - Bagian B)**
   - Mengekspos fungsi deterministik tunggal `evaluateMasterSurface(u, v, targetPos, targetNorm)` dan/atau `evaluateMasterSDF(p)` yang menghasilkan posisi 3D $\mathbf{P}(u,v)$ dan normal keluar kontinu $\mathbf{N}(u,v)$.
4. **Tahap 4 — Pemotongan Subdomain & Pewarisan Permukaan Panduan (`PrimaryStructure.js`)**
   - Mengekstraksi panel fungsional langsung dari domain $\Omega_k \subset [0,1]^2$ melalui `MasterSurfaceBuilder.sliceSubdomain(...)` sehingga jarak sambungan antar-subdomain bernilai nol mutlak ($\Vert \mathbf{P}_A - \mathbf{P}_B \Vert = 0$).
5. **Tahap 5 — Kavitas Sekunder, Shutlines, Flanges, & Normal-Offset Applique (`SecondarySubdomains.js`)**
   - Membentuk ceruk jendela, sabuk spandrel, *sun-shading*, lis teritisan atap, dan ornamen relief sepanjang vektor normal permukaan induk:
     $$\mathbf{P}_{\text{applique}}(u, v) = \mathbf{P}_{\text{master}}(u, v) + d(u, v) \cdot \mathbf{N}_{\text{master}}(u, v)$$
6. **Tahap 6 — Applique Tersier, Kinematic Rigging, Shader Mikro, & Single-Pass Batching (`KinematicRig.js` & `MaterialShaderBank.js`)**
   - Merakit elemen artikulasi/animasi, menyuntikkan `injectOmniScaleSurfaceShader` (Contact AO, Triplanar Grain, Studio Zebra, Filmic Dehaze), dan menjalankan `NexusBatchOptimizer.optimizeHierarchy()` ($\le 25$ Draw Calls).

---

## B. Protokol Verifikasi Bertahap Universal 4-Gate ("Verified by Design")

- **Gate 1 (Pasca-Tahap 1):** `0` nilai `NaN`/`Infinity` pada seluruh `BufferGeometry`; deviasi *Bounding Box* aktual vs `realDimensionsMeters` $< 0.5\%$; akurasi `foundationSkirtDepth`.
- **Gate 2 (Pasca-Tahap 3):** Topologi manifold rapat air, seluruh normal menghadap keluar ($\mathbf{N} \cdot \mathbf{V}_{\text{out}} > 0$), dan transisi garis pantul *Studio Zebra* mulus tanpa *faceted artifacts* liar.
- **Gate 3 (Pasca-Tahap 4 & 5):** *Zero-Gap Continuity Audit* ($\max \Vert \mathbf{P}_A - \mathbf{P}_B \Vert < 10^{-4}$), *Normal-Offset Applique Audit* (bebas mengambang/tenggelam), dan *Kinematic Sweep Audit* (bebas *self-intersection*).
- **Gate 4 (Pasca-Tahap 6):** Validasi Tri-Sudut (Elevasi, Oblique $45^\circ$, Nadir $90^\circ$ Horizon-Locked), audit optik & `emissiveMap` `#000000` pada permukaan pasif, `renderer.info.render.calls <= 25` pada target 120 FPS, serta UI *Default Collapsed* + `applyNonOccludingViewportOffset`.

---

## C. Kontrak Antarmuka `StandardNexusAsset` (v2.0)

Setiap aset di `./library/<kategori>/<Nama_Aset>/index.js` wajib mengekspor kelas yang mematuhi tanda tangan:
- `static meta = { id, name, category, kernelType, scaleUnit, realDimensionsMeters, foundationSkirtDepth, obstacleColliders, orthoValidationPresets }`
- `static evaluateMasterSurface(u, v, targetPos, targetNorm)`
- `static build(sharedMaterials, sharedGeometries)`
- `static emissiveMaskGenerator(canvasWidth, canvasHeight)`
- `static setNightIllumination(meshRoot, isNight)`
- `static updateKinematics(meshRoot, dt, telemetryState)`

---

## D. 18 Hukum Emas Enjiniring NEXUS v2.0

1. **Hukum 1 — Geometri Lofted Kontinu:** Bangun bodi lengkung/organik/transisi menggunakan satu jaring `THREE.BufferGeometry` *lofted* kontinu (minimal $28 \times 36$ sampel pada bagian lengkung), bukan tumpukan kotak patah-patah.
2. **Hukum 2 — Satu Fungsi Penampang Melintang Identik (`Shared Cross-Section Profile`):** Segmen bersebelahan wajib berbagi fungsi penampang yang 100% identik pada stasiun batasnya (`0.00 m Gap`).
3. **Hukum 3 — Disiplin Arah Normal Segitiga (`Outward Winding Order`), Sasis, & UV:** Pastikan *winding* segitiga menghasilkan $\mathbf{N}_{\text{out}} = \mathbf{T}_u \times \mathbf{T}_v$, tambahkan *solid bulkhead caps*, gunakan UV vertikal ternormalisasi `hNorm`, dan cegah pembalikan cermin pada tipografi/logo.
4. **Hukum 4 — Roda Baja & Mekanika Penggerak Terbuka Jelas:** Untuk wahana rel/jalan, ekspos mekanika roda, rem cakram, suspensi, dan peredam secara presisi.
5. **Hukum 5 — Fondasi Anti-Ngambang (`Sub-Terrain Skirt`) & `Viaduct Exclusion Zone`:** Seluruh bangunan darat wajib memiliki ekstrusi fondasi bawah (`-15 m` s.d. `-18 m`), pilar air `-68 m`, dan zona eksklusi objek di atas jembatan.
6. **Hukum 6 — Pelacakan Indeks Jaring Kabel (`Explicit Strand Indexing`):** Lacak indeks verteks dasar tiap jalur kabel/serat dalam `prevWireBase[strandIndex]` agar bebas silang *zig-zag*.
7. **Hukum 7 — Keselarasan Perspektif Indoor vs Outdoor:** Selaraskan visibilitas interior dan tampilan eksterior nyata saat pergantian mode kamera.
8. **Hukum 8 — Optik Kaca 3D, Isolasi Hujan, & Kinematika Wiper:** Isolasi partikel hujan di luar ruang interior dan lipat wiper rapat saat mati.
9. **Hukum 9 — Anatomi Manusia 3D 18-Sendi & AI NPC Biomekanik:** Dilarang menggunakan orang kotak; gunakan rig manusia 3D 18-sendi berartikulasi alami (termasuk variasi busana sivitas akademika / jaket kuning / hijab rapi `+Z`).
10. **Hukum 10 — Vegetasi Organik Multi-Spesies & Pemisahan Material Vertikal-Horisontal:** Gunakan kanopi pohon berlapis organik (trembesi/mahoni/palem tepi Danau Kenanga) dan pisahkan UV/material bidang lantai horisontal dari dinding vertikal agar tekstur tidak tertarik.
11. **Hukum 11 — Fisika & Kinematika Halus:** Gunakan persamaan diferensial gerak nyata dan redaman eksponensial mulus.
12. **Hukum 12 — Arsitektur 120 FPS Ultra (`Single-Pass Batching <= 25 Draw Calls` & `100% GPU Vertex Particles`):** Lebur `Mesh` statis per material dan `LineSegments` melalui `NexusBatchOptimizer`; hitung seluruh partikel cuaca/atmosfer 100% di `vertexShader`.
13. **Hukum 13 — Disiplin Kolorimetri Anti-Washed-Out, Larangan Double-Gamma, & True Emissive Masking:** Hindari konversi sRGB ganda dengan `ACESFilmicToneMapping`, suntikkan Filmic Dehaze, kunci bagian pasif `emissiveMap` pada `#000000` mutlak, dan kunci `envMapIntensity <= 0.14` untuk permukaan *matte*.
14. **Hukum 14 — FBM Rotasi Multi-Oktaf, Proyeksi Bola Analitis, & Analytical Contact AO:** Gunakan matriks rotasi domain `mat2(0.8, 0.6, -0.6, 0.8)` per oktaf FBM pada permukaan air Danau Kenanga & awan; suntikkan Contact AO dunia & Triplanar Micro-Grain via `onBeforeCompile`.
15. **Hukum 15 — Kunci Kutub Horison Kamera, Redaman Eksponensial 6-DOF, & Anti-Kolisi Potensial Lunak:** Pertahankan `camera.up.set(0, 1, 0)`, gunakan `setHorizonLockedNadirView` (`eps = 0.45 m`), redaman `1 - exp(-k * dt)`, dan `resolveSoftSpatialCollision`.
16. **Hukum 16 — Dekomposisi Morfologi Tri-Hierarki & Validasi Ortogonal Tri-Sudut:** Konstruksi dibagi menjadi **Primer/Basal (Kaki)**, **Sekunder/Ritmik (Badan)**, dan **Tersier/Mahkota (Kepala)**, serta divalidasi dari Elevasi, Oblique $45^\circ$, dan Nadir $90^\circ$.
17. **Hukum 17 — SSoST, Subdomain Slicing ($\Vert \mathbf{P}_A - \mathbf{P}_B \Vert = 0$), & Normal-Offset Applique:** Seluruh panel dipotong langsung dari domain $(u, v)$ permukaan induk, dan detail sekunder digeser sepanjang vektor normal $\mathbf{N}(u, v)$.
18. **Hukum 18 — De-Monolitisasi Modul Anti-Regresi & Disiplin 4-Gate:** Pecah aset ke dalam 6 sub-modul tahapan Kelas-A dan kunci setiap tahap menggunakan `GateVerifier`.
