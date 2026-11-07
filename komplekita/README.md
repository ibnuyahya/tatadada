# Kompleks Bundaran Gedung DPPD (eks-KSDP), ITTC Perpustakaan Lama, STP-UI & ILRC Universitas Indonesia — NEXUS 3D v2.0

Simulasi WebGL / Three.js prosedural murni (**100% Tanpa File `.glb`/`.fbx` Eksternal**) berstandar **NEXUS Omni-Scale Procedural Engine v2.0** untuk keempat gedung utama di dalam pulau bundaran **Jl. Prof. Dr. Nugroho Notosusanto, Kampus UI Depok** (`-6.365429, 106.826801`).

Seluruh proporsi spasial, orientasi mata angin, topologi atap, fasad, lansekap, dan penanda jalan dikalibrasi secara fotogrametris terhadap **206 foto referensi unik** (`./references/images/ref_002.jpg` s.d. `ref_207.jpg`) serta matriks kebenaran lapangan di [`./references/GROUND_TRUTH_ATLAS.md`](./references/GROUND_TRUTH_ATLAS.md).

---

## 1. Metodologi Fotogrametri 6-Pilar (Anti-Bias Street View)

1. **Jangkar Geodetik Nadir 90° Terlebih Dahulu (`ref_174.jpg`, `ref_206.jpg`, Koordinat GPS):**
   - Diameter garis tengah jalan bundaran terkunci pada **`151.0 m` (`R_center = 75.50 m`)** dengan arus lalu lintas 2 lajur **berlawanan arah jarum jam (counter-clockwise)**.
   - Variasi *zoom* / *Field-of-View* (`45y` s.d. `90y`) pada tangkapan layar Google Street View dikompensasi menggunakan sudut sektor $(\theta)$ dan koordinat kamera GPS asli.
2. **Kompensasi Area Gelap & Oklusi Pohon Rimbun (Anti-Blindspot):**
   - Area selasar/loggia dan dinding yang tampak gelap pekat akibat *auto-exposure* kamera Street View direkonstruksi dengan pantulan lingkungan sore **14:00 WIB (`PMREMGenerator`)**, `HemisphereLight`, dan `skyFillLight` sehingga modul kolom, jendela pita reflektif, dan dinding plester krem tetap terbaca jelas di balik kanopi pohon.
3. **Koreksi Artefak Lensa Matahari (`ref_042–061` vs `ref_025`, `ref_029`):**
   - Rona ungu/magenta pada foto sisi selatan ILRC yang menghadap matahari pagi dikoreksi kembali ke material fisik aslinya: **gundukan tanah serasah daun kering cokelat hangat (`#6b4c35`)** yang ditahan oleh **pagar turap cerucuk bambu vertikal**.
4. **Eksklusi Ketat Bangunan Latar Jauh, Elemen Dinamis, & GUI Street View (`ref_088–093`, `ref_114`, `ref_163`, `ref_207`):**
   - **Nol gedung luar bundaran** (Gedung FIA UI, Menara Air UI, Gerbang FT-UI, FIB UI, maupun saung luar).
   - **Nol kendaraan/pejalan kaki** dan **nol artefak GUI Street View** (tanda silang `X`/`+` putih, lingkaran kursor jalan, kompas, atau bayangan hijau bergaris).

---

## 2. Empat Gedung Utama (Dirombak Seimbang — Class-A)

| Sektor | Gedung | Rentang Sudut / Posisi | Fitur Arsitektur Utama |
| :--- | :--- | :--- | :--- |
| **Tengah** | **ITTC Perpustakaan Lama** | Bujur sangkar kardinal Utara–Selatan (`42.0 m × 42.0 m`) | Courtyad tengah `16 m × 16 m` dengan paviliun *skylight* piramida `9.6 m × 9.6 m`, 4 lantai dengan grid kolom struktural putih 7-bay, kanopi portico Barat (arah jam 9:00), atap limasan terakota terkunci SSoST, dan paviliun annex 1 lantai di sudut NW. |
| **Barat Daya (SW)** | **Gedung DPPD (eks-KSDP)** | `+92°` (Selatan) s.d. `+178°` (Barat), poros `+135.5°` SW | Talud rumput 2 tingkat + dinding batu kali, dinding pelana ujung NW bata ekspos menghadap akses Barat, atap sayap poligonal 6 faset terakota yang berujung limasan miring di SE, pelana utama lancip A-frame krem dengan jendela pita 11-bay + parabola mini, loggia L2 + unit AC split, kanopi L1 terakota (`"GEDUNG / KSDP UI"` & `"GEDUNG DPPD UI"`), totem kuningan UI–Huawei, menara spanduk kembar + plakat peta logam, dan tiang bendera Merah-Putih kinematik. |
| **Utara (N)** | **Gedung STP-UI** | `-150°` (WNW) s.d. `-34°` (ENE), puncak `26.80 m` | 8 lantai melengkung dengan bilah sudut ACP tembaga penuh & tangga darurat sirip tembaga di SW & NE, lobi piloti kaca L1 + kanopi perak berkolom `V` (`"SCIENCE TECHNO PARK UNIVERSITAS INDONESIA"`), fasad L2–L3 berisi 24 modul trapesium tembaga miring bergantian `/` & `\` + kisi vertikal, fasad putih L5–L8 dengan 5 sabuk *sunshade* lengkung & jendela boks 3D, logo Makara Emas + huruf timbul L8 NE, mahkota atap tembaga + 18 baris panel surya PV, planter miring `"STP-UI"`, totem hitam-kuning, dan pipa hidran merah. |
| **Timur–Tenggara (E–SSE)** | **Gedung ILRC & DRPM** | `-31°` (ENE) s.d. `+83°` (SSE), puncak `27.20 m` | Sabuk *cornice* atap putih melengkung kontinu dengan *soffit* cokelat-terakota hangat yang menjembatani **Atrium Terbuka 5 Lantai**, sayap Timur berlouver horizontal dengan **Kotak Kaca Observasi Kantilever L4–L5** (`+8.5°`), selasar DRPM + gerbang `"P PARKIR KHUSUS PIMPINAN"` + gardu listrik, portal putih ILRC + logo Makara & KAN + ramp *koral sikat*, sayap Selatan dengan gundukan serasah daun berpagar bambu, dan **Menara Bata 6 Lantai** di ujung selatan dengan sirip miring & 3 jendela tudung menonjol. |

---

## 3. Struktur Direktori (`./` Strict Relative Paths)

- [`./index.html`](./index.html) & [`./studio.html`](./studio.html) — Antarmuka simulasi utama dengan *drawer* adaptif non-oklusi (default tertutup).
- [`./main.js`](./main.js) — Orkestrator utama `BundaranKompleksAsset` & `NexusStudioApp`.
- [`./core/MasterSurfaceKernels.js`](./core/MasterSurfaceKernels.js) — Kernel geometri SSoST, pita polar melengkung, *ridge cap*, dan atap limasan.
- [`./core/ProceduralTextureFactory.js`](./core/ProceduralTextureFactory.js) — Generator tekstur PBR prosedural, `PMREMGenerator` 14:00 WIB, shader *Contact AO + Filmic Dehaze + Clay/Zebra*, dan **Unified 1024×1024 Signage Atlas** (1 draw call untuk seluruh papan nama & logo).
- [`./core/BatchRenderer.js`](./core/BatchRenderer.js) — `NexusBatchOptimizer` penggabung hierarki statis menjadi `<= 21 Draw Calls`.
- [`./core/CameraDirector.js`](./core/CameraDirector.js) — Pengarah kamera *Horizon-Locked Nadir 90°* (`eps = 0.45 m`), transisi mulus, *frustum offset* adaptif, dan mode *Drone 6-DOF* anti-tembus.
- [`./core/WeatherOpticsEngine.js`](./core/WeatherOpticsEngine.js) — Pencahayaan sore 14:00 WIB UI Depok & mode Malam GI (`#000000` *passive emissive masking*).
- [`./core/VerificationGateRunner.js`](./core/VerificationGateRunner.js) — Penguji otomatis 4-Gate SSoST.
- [`./world-gis/MasterPlanConfig.js`](./world-gis/MasterPlanConfig.js) & [`./world-gis/BundaranSiteBuilder.js`](./world-gis/BundaranSiteBuilder.js) — Konfigurasi geodetik & konstruktor bundaran, selokan batu kali, jembatan, dan vegetasi tropis *instanced*.
- [`./library/perpustakaan-lama/index.js`](./library/perpustakaan-lama/index.js) — Modul ITTC Perpustakaan Lama.
- [`./library/gedung-dppd/`](./library/gedung-dppd/index.js) — Modul Gedung DPPD (`PrimaryStructure.js`, `SecondarySubdomains.js`, `KinematicRig.js`).
- [`./library/stp-ui/index.js`](./library/stp-ui/index.js) — Modul Gedung STP-UI.
- [`./library/ilrc-drpm/index.js`](./library/ilrc-drpm/index.js) — Modul Gedung ILRC & DRPM.
