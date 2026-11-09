# Tabel Parameter Ground-Truth (Tahap 1 — Kalibrasi Fotogrametri DJI Mavic v2.1)
## Aset Aktif: Gedung Pusat Administrasi / Rektorat Universitas Indonesia (Kampus Depok)

- **ID Aset:** `GEDUNG_REKTORAT_UI_DEPOK`
- **Referensi Fotogrametri Nyata:** 5 Foto Udara **DJI Mavic** Kampus UI Depok (`Elevasi Depan Rotunda 0°`, `Oblique 45° Barat-Daya`, dan `3 Proyeksi Nadir 90° Orthophoto`), karya **Prof. Ir. Gunawan Tjahjono, M.Arch., Ph.D.** (1984–1987).
- **Koreksi Morfologi Kritis Hasil Ekstraksi DJI Mavic:**
  1. **Menara Utama Pagoda Krusiform 4-Sayap (`36.8 × 36.8 m` teritisan, `32.0 × 32.0 m` dinding):** Setiap fasad terbelah dua oleh **ceruk vertikal kaca gelap menerus (*central dark-glass X-mullion atrium slot*, lebar `4.60 m`, mundur `2.80 m`)**. Lantai dasar ditopang **kolonnade pilar beton putih-krem tinggi (`2 lantai`, `y = 0.00–5.65 m`)**, diikuti **6 tingkat teritisan genteng terakota melebar per lantai (*6 flared pagoda eave skirts*)** dengan pita jendela 6-modul ber-*mullion* putih, dimahkotai **4 Atap Limas Sudut Besar (`Tier 7`, `y = 25.15–30.85 m`)**.
  2. **Leher Mahkota Putih, Relief Makara UI, & Atap Puncak Limas Tunggal Curam (`~49°`):** Di antara ke-4 atap limas sudut berdiri dinding mahkota beton putih-krem yang memuat **relief batu Emblem Makara UI** diapit jendela *clerestory* gelap kiri-kanan, lalu dinaungi **1 Atap Tajug Limas Piramidal Tunggal Sangat Curam (`25.8 × 25.8 m`, `y = 32.85–47.75 m`)** bermahkota tudung puncak putih (*white apex cap*) dan penangkal petir (`y = 50.00 m`).
  3. **Tata Massa Kompleks Podium & Pelataran Asimetris Nyata (`108.00 × 124.00 m`):**
     - **`+Z` (Selatan / Fasad Rotunda):** **Selasar Agung Beratap Pelana Curam** (`x = -2.60 m`, `z = 15.8–39.0 m`) $\rightarrow$ **Pelataran Kloister Segiempat Beratap Genteng (`Pelataran Panji Fakultas UI`, `36.0 × 24.5 m`)** berisi 14 bendera panji fakultas warna-warni & tiang Merah-Putih tengah $\rightarrow$ **Gerbang Pendopo Depan** (`z = 60.0–72.0 m`) tepat di tepi kanstin belang hitam-putih **Lapangan Rotunda UI**.
     - **`+X` (Timur / Tepi Danau Kenanga) & `-X` (Barat / Parkir Rindang):** Sepasang **Gedung Sayap Pelataran Kembar (*East & West Courtyard Annex Wings*, `x ∈ [±20.5, ±54.0 m]`)** yang masing-masing memadukan **2 blok dek atap beton datar berparapet putih** dengan **1 gedung aula besar beratap limas genteng merah (`y = 12.4 m`)**; sisi luar sayap Timur berbatasan langsung dengan turap batu **Danau Kenanga**!
     - **`(+X, +Z)` Diagonal `45°` Tenggara:** **Selasar Diagonal** menuju **Pendopo Segidelapan Ber-*Skylight* Kaca Puncak** (`(+29.5, +29.5 m)`) yang bersambung dengan **Selasar Panjang Kampus** ke arah selatan.
     - **`-Z` (Utara / Belakang):** **Pendopo Limas Persegi Utara** (`x = +5.0 m`, `z = -43.5 m`, batas luar `z = -52.0 m`) dan pelataran keramik kotak abu-abu bermotif **belah ketupat putih (`◇◇◇`)** khas Rektorat UI.

---

## 1. Parameter Spasial, Amplop Ortografis, & Sistem Koordinat

| Parameter | Nilai Terukur | Satuan / Keterangan |
| :--- | :--- | :--- |
| `scaleUnit` | `METER` | Skala arsitektural 1:1 dalam meter |
| Titik Asal `(0, 0, 0)` | `(0.00, 0.00, 0.00)` | Pusat simetri horizontal Menara Utama pada elevasi pelataran dasar (`y = 0.00 m`) |
| Orientasi Sumbu Utama | `+Y` Atas, `+Z` Selatan/Rotunda, `+X` Timur/Danau | `+Z` menghadap Pelataran Bendera & Lapangan Rotunda UI; `+X` menghadap langsung **Danau Kenanga** |
| Lebar Kompleks Aset ($W$) | `108.00` m | `x ∈ [-54.00, +54.00]` (teritisan luar Sayap Barat s.d. Sayap Timur Tepi Danau Kenanga) |
| Panjang Kompleks Aset ($L$) | `124.00` m | `z ∈ [-52.00, +72.00]` (teritisan luar Pendopo Utara `-52.00 m` s.d. Gerbang Rotunda `+72.00 m`) |
| Tinggi Total Puncak ($H$) | `50.00` m | `y = 0.00 m` s.d. `47.75 m` (tudung puncak putih Tajug Utama) + penangkal petir hingga `50.00 m` |
| Denah Menara Utama | `32.00 × 32.00` m (`36.80 m` eave) | 4 sayap sudut + ceruk atrium kaca gelap tengah `4.60 m` pada ke-4 sisi kardinal |
| `foundationSkirtDepth` | `18.00` m | `y ∈ [-18.00, 0.00]` rapat air (Hukum Emas 5: anti-ngambang di tepi Danau Kenanga) |

---

## 2. Dekomposisi Stasiun Aksial Vertikal Menara SSoST ($v \in [0, 1]$)

| Stasiun | Rentang Elevasi $y$ (m) | Hierarki Morfologi | Spesifikasi Geometri Hasil Kalibrasi DJI Mavic |
| :--- | :--- | :--- | :--- |
| **S0 — Skirt** | `-18.00` s.d. `0.00` | Sub-Terrain Skirt | Fondasi bawah tanah rapat air `34.00 × 34.00 m` di bawah menara + skirt pelataran `-18.00 m`. |
| **S1 — Kolonnade Putih** | `0.00` s.d. `5.65` | **Kaki Menara (2 Lantai)** | Plinth beton 2-undak + sabuk tanaman perdu hijau + deret **pilar beton putih-krem tinggi** (5 bentang per sayap) dengan dinding kaca lobi gelap mundur di belakangnya. |
| **S2 — 6 Lantai Teritisan Pagoda** | `5.65` s.d. `25.15` | **Badan Ritmik (6 Tingkat @ `3.25 m`)** | Pada ke-8 muka sayap sudut: **6 sabuk atap teritisan miring genteng merah** (`flare 15.85 m -> 17.90 m`), spandrel beton putih-krem, dan pita jendela gelap 6-modul ber-*mullion* putih. Pada ke-4 sumbu tengah kardinal (`|s| < 2.30 m`): **ceruk vertikal kaca gelap ber-rangka berlian (*X-diamond atrium*)** menerus tanpa terputus teritisan! |
| **S3 — 4 Limas Sudut & Leher Makara** | `25.15` s.d. `32.85` | **Transisi Mahkota (`Tier 7`)** | Ke-4 sayap sudut ditutup **4 Atap Limas Piramidal Besar Genteng Merah** (`bentang luar ±18.55 m`, puncak bubungan datar putih di `y = 30.85 m`). Di tengah ke-4 sisi berdiri **Dinding Mahkota Beton Putih-Krem** dengan **Relief Emblem Makara UI** (`y = 26.20–32.85 m`) diapit jendela *clerestory* gelap kiri-kanan (`y = 30.85–32.85 m`). |
| **S4 — Atap Limas Tajug Puncak** | `32.85` s.d. `47.75` | **Mahkota Utama Curam (`49°`)** | **1 Atap Tajug Limas Piramidal Tunggal Sangat Curam** (`25.80 × 25.80 m` di `y = 32.85 m` meruncing lurus-cekung halus ke `y = 47.75 m`) dengan 4 jurai putih tipis dan **Tudung Puncak Putih (*White Apex Mustaka*)** di `y = 46.95–47.75 m`. |
| **S5 — Penangkal Petir** | `47.75` s.d. `50.00` | **Puncak Finial** | Tiang penangkal petir logam ramping + lampu suar penerbangan pada `y = 49.88 m`. |

---

## 3. Palet Material Fotometrik DJI Mavic & True Emissive Masking (Hukum Emas 13)

| Elemen Arsitektur | Albedo Siang (Kalibrasi Foto Mavic) | Roughness / Metalness | Perilaku Malam (`emissiveMap`) |
| :--- | :--- | :--- | :--- |
| Beton Putih-Krem Rektorat (*Off-White Precast*) | Crisp Tropical Off-White (`#EAE5DB`) | `roughness: 0.84`, `metalness: 0.03`, `envMapIntensity: 0.12` | **Hitam Mutlak `#000000`** (0% bleed) + Contact AO bawah 6 teritisan |
| Genteng Tanah Liat Merah Terakota | Weathered Terracotta Red (`#A44838` – `#8C3A2C`) | `roughness: 0.76`, `metalness: 0.03`, `envMapIntensity: 0.13` | **Hitam Mutlak `#000000`** |
| Paving Pelataran Motif Berlian UI (`◇◇◇`) | Grid Batu Abu-Gelap `#585A56` + Belah Ketupat Putih `#EAE8E1` | `roughness: 0.85`, `metalness: 0.03`, `envMapIntensity: 0.11` | **Hitam Mutlak `#000000`** (material horisontal khusus — Hukum Emas 10) |
| Dek Beton Datar Sayap Annex | Weathered Tropical Concrete Slab (`#8A8E91`) | `roughness: 0.90`, `metalness: 0.02` | **Hitam Mutlak `#000000`** |
| Pita Kaca 6-Modul & Atrium Berlian Tengah | Deep Slate-Teal Architectural Glass (`#111C21`) + Mullion Putih | `roughness: 0.14`, `metalness: 0.22` | **Aktif Terseleksi:** Ruang kerja hangat (`#FFD58E`), rangka `#000000` |
| Panel Relief **Makara UI** | Off-White Panel (`#EAE5DB`) + Relief Batu-Emas Hangat (`#8E887B` / `#D6B458`) | `roughness: 0.48`, `metalness: 0.35` | **Backlit Halo Emas UI (`#FFD24C`)** dengan latar pasif `#000000` |

---

## 4. Preset Kamera Validasi Tri-Sudut (Terkalibrasi Sudut Foto DJI Mavic!)

- `elevationFront` (**Rotunda $0^\circ$ — Persis Foto Mavic #1**): `pos: { x: -2.5, y: 27.0, z: 98.0 }, target: { x: -1.0, y: 18.0, z: 4.0 }`
- `oblique45` (**Oblique $45^\circ$ — Persis Foto Mavic #2**): `pos: { x: -64.0, y: 54.0, z: 66.0 }, target: { x: 0.0, y: 18.0, z: -2.0 }`
- `nadir90` (**Nadir $90^\circ$ — Persis Foto Mavic #3 & #5**): `targetX: 2.0, targetZ: 12.0, heightY: 182.0, topVecX: 0.0, topVecZ: -1.0`

---

## 5. Telemetri Geometri & Permukaan Terverifikasi `NEXUS v2.5` (`100x` Architectural & World Surface Detail + `100x` Human Realism + `>10x` Ultra-High LOD)

| Subsistem Objek | Segitiga `v2.2` | Segitiga `v2.5` | Faktor Kenaikan | Draw Calls |
| :--- | ---: | ---: | ---: | ---: |
| **Monumen Gedung Rektorat UI (`LOD0` + >1.350 Ornamen Fasad 3D)** | `25.604` | **`341.934`** | **`13,35x`** | **`13`** |
| **Terrain DEM `336×336` + Turap Batu Kali Voronoi + Danau Kenanga** | `25.400` | **`256.000`** | **`10,08x`** | **`2`** |
| **Rotunda Makara + Kanstin Bevel + Topiary + Bangku Jati + Selasar** | `3.924` | **`54.292`** | **`13,84x`** | **`2`** |
| **56 Pohon Bercabang & Banir + 48 Perdu Taman + 24 Lentera Berulir** | `94.920` | **`976.212`** | **`10,28x`** | **`3`** |
| **24 Sivitas Akademika 22-Sendi Dual-Bone LBS (`528` Sendi Hidup, SSS/Batik/Payung Hujan)** | `11.884` | **`205.856`** | **`17,32x`** | **`1`** |
| **TOTAL DUNIA KAMPUS UI (`index.html`, Semua Mode Cuaca)** | **`161.732`** | **`1.835.878`** | **`11,35x`** | **`22 – 24` ($\le 25$)** |

### Profil Permukaan Mikro-Normal Analitis & Interior Paralaks 3D (`SURFACE_PROFILE` `1..11`)
- **`11/11` Profil Material Aktif & Pipeline Shader Fisik Level Ultra (`NEXUS v2.6`)**:
  - **Cinema-Grade AgX Tone Mapping (`THREE.AgXToneMapping ?? THREE.ACESFilmicToneMapping`, `exposure = 1.08`)** + Evaluasi *True Linear-HDR Radiance* tepat sebelum `#include <tonemapping_fragment>`.
  - **Pre-Convolved Multi-Regime HDR Sky & Terracotta-Campus Bounce PMREM IBL (`DAY`, `DUSK`, `NIGHT`)** via `THREE.PMREMGenerator` (`0` runtime Draw Calls tambahan), mengaktifkan *Split-Sum GGX Specular IBL + Kulla-Conty Multiscattering Energy Compensation* serta *Soft Shadows* `4096×4096` (`normalBias = 0.018`, `radius = 2.1`).
  - **Seamless 3D Isotropic Nyquist-Band-Limited Micro-Normals (`omniFbm3BL` + `omniBand`)** bebas 100% dari *2×2 quad static* maupun tarikan UV (`<= 6.3 deg` soft-clamped tangent gradient).
  - **Fujii Qualitative Oren-Nayar Rough Diffuse Retro-Reflection**, **Beer-Lambert Botanical Leaf Subsurface Translucency (`chlorophyll`) + Cuticle Fresnel**, **3D Ray-Box Parallax Interior Rooms + Solar Glass Fresnel**, **Dual-Layer Wet Dielectric Water Clearcoat (`IOR 1.333`, `F0 = 0.0204`)**, **Henyey-Greenstein (`g = 0.76`) Mie Solar Aureole & Aerial Perspective**, **Live Sun-Coupled Human Epidermal SSS & Corneal Eye HDR Catchlights**, serta **Optika Perairan 360° Beer-Lambert + Kaustik Air Matahari + Buih Tepi Turap Batu**.

---

## 6. SSoST Pulau Rektorat UI 360° & ICAO VIP Waterfront Helipad (`NEXUS v2.7`)

- **SSoST Garis Pantai Polar Tertutup $360^\circ$ (`TerrainDEMLoader.getIslandRadiusAtAngle(theta)` & `getIslandCoastDistance(x, z)`):**
  - Pusat Pulau: `ISLAND_CX = -18.0`, `ISLAND_CZ = 34.0`
  - DEM Pulau: `336×336` (`332 m × 332 m`, `GRID_HALF_SPAN = 166.0 m`), seluruh batas luar DEM terendam penuh di dasar perairan (`LAKE_BED_Y = -3.85 m`) dengan *watertight sub-terrain skirt* `-18.0 m` (Hukum Emas 5).
  - **Turap Batu Kali Keliling Pulau $360^\circ$:** `480` segmen melingkar tertutup $\times$ `13` stasiun penampang + `64` pilar penopang radial (*masonry buttresses*) & tonggak tambat (*waterside bollards*).
  - **Bentang Perairan $360^\circ$ & Hidrodinamika Fisik `NEXUS v2.8`:** Cincin perairan `320×112` (`71.680` segitiga deformatif 3D) hingga `WATER_OUTER_RADIUS = 520.0 m` (diameter `1.040 m`, `WATER_LEVEL_Y = -1.45 m`), mengintegrasikan fase angin monotonik `uWindPhase` ($\vec{\Phi}(t) = \int_0^t \vec{v}_{\text{wind}}(\tau)\,d\tau$, **0% *backward sloshing***), `6` kereta gelombang gravitasi **Trochoidal Gerstner 3D dispersif** ($\omega_i = \sqrt{9,81 \cdot k_i}$), **refraksi gelombang pantai $360^\circ$ Hukum Snell & *shoaling surf rollers* berharmonik Stokes** yang mendaki/surut secara 3D di atas turap batu kali (periode alami `5,7 dtk` & `7,9 dtk`), riak kapiler bersilangan ganda (`evalCrossedWavelets`), kaustik seluler **Voronoi**, buih ombak berpori & *whitecaps*, serta pita batu turap basah berkilau percikan ombak (`IOR 1.333`).
- **1 Unit ICAO VIP Waterfront Helipad (`HELIPAD_CX = -104.0, HELIPAD_CZ = 106.5, HELIPAD_DECK_Y = +0.48 m`):**
  - Terhubung langsung dari Lapangan Rotunda UI melalui `RotundaWestCampusBoulevard` (`x = -35.7 .. -81.6, z = 106.5`) dan ramp akses miring berkanstin zebra (`x = -81.2 .. -90.1`).
  - Podium struktural beton segi-delapan (`R = 15.2 m`, `-2.52 m .. +0.48 m`), dek FATO aspal anti-selip segi-delapan (`R = 14.65 m`), `7` rangka baja jaring pengaman kantilever ICAO (*perimeter safety net outriggers*), marka 3D kuning FATO + cincin putih TDPM 64-segmen + huruf `"H"` ICAO (`6.4 m × 4.6 m`) + chevron arah pendaratan.
  - `16` lampu perimeter TLOF hijau zamrud tertanam (menyala HDR hanya saat `DUSK`/`NIGHT`, `0.0` saat `DAY` sesuai Hukum Emas 10), tiang *windsock* `6.8 m` belang merah-putih + kerucut *windsock* 3D oranye-putih + lampu suar merah puncak di `(-87.2, 92.8)`, serta **Pendopo Transit Helipad Nusantara** beratap Limas Terakota 2-tingkat + 2 bangku jati di `(-84.8, 118.2)` + `6` lentera taman Nusantara (`30` total lentera).
  - Preset Kamera Baru: `island360` (**Pulau $360^\circ$**) & `helipadVIP` (**Helipad VIP**); Total Segitiga Dunia: **`1.911.002`**, Total Draw Calls: tetap **`22 – 24 <= 25`** (`0` tambahan Draw Call!).

---

## 7. Telemetri Waktu & Cuaca Rill Kampus UI Depok, Transisi Fisik Wajib `20–45 Detik`, & Preloader Interaktif (`NEXUS v2.9`)

- **Default Waktu WIB (`Asia/Jakarta`, `UTC+7`) & Cuaca Rill Kampus UI Kota Depok (`core/DepokLiveWeatherService.js`):**
  - Koordinat SSoST Meteorologi: `KAMPUS_UI_LAT = -6.3658`, `KAMPUS_UI_LON = 106.8286` (Gedung Rektorat UI, Kampus Depok).
  - Jam Astronomi Depok: `getDepokWIBClock()` menghitung `solarHour` kontinu `[0, 24)` di zona waktu `Asia/Jakarta` (`UTC+7` murni, independen dari zona waktu OS pengunjung), menggerakkan busur surya/bulan melintasi kubah langit (terbit `+X` Timur di atas Danau Kenanga `~05:55 WIB`, zenit `~11:58 WIB`, terbenam `-X` Barat di atas Helipad VIP `~18:00 WIB`).
  - Integrasi Open-Meteo WMO API (`fetchLiveDepokWeather`, *refresh* otomatis tiap 5 menit, *timeout* non-pemblokir `3,6 dtk` + fallback klimatologi diurnal Kota Depok saat luring).
- **Transisi Fisik Bertahap Wajib `20–45 Detik` (`WeatherOpticsEngine.MIN_TRANSITION_SEC = 20.0`, `MAX_TRANSITION_SEC = 45.0`, Default `25.0 dtk`):**
  - Setiap pergantian cuaca & waktu (baik melalui tombol manual `Cerah`, `Gerimis`, `Hujan Deras`, `Badai Petir`, `Siang`, `Sore`, `Malam`, maupun tombol `Sinkron Rill Kampus UI Depok`) wajib melewati koreografi fisik bertahap berdurasi `20–45 detik` yang dapat diatur melalui *slider* `#slider-trans-dur`:
    1. **Saat cuaca memburuk (menuju Hujan/Badai):** Awan mendung & angin bergerak masuk terlebih dahulu (`0..58%`), rintik hujan mulai turun perlahan dari gerimis menuju deras (`16..88%`), diikuti permukaan tanah/beton yang membasah & menggenang secara bertahap (`28..100%`), sementara kilat/guntur hanya menyambar ketika kematangan badai `uStormFactor > 0.62`.
    2. **Saat cuaca membaik (menuju Cerah):** Petir mereda lebih dulu (`0..50%`), curah hujan mengecil perlahan hingga berhenti (`0..64%`), awan menipis (`18..92%`), dan genangan air/tanah basah menguap paling akhir (`30..100%`).
    3. **Saat pergantian Waktu (`DAY` / `DUSK` / `NIGHT`):** Matahari & bulan menyapu busur langit melewati fase *Golden Hour* / Senja secara kontinu sementara iluminasi jendela Rektorat, 30 lentera taman, dan 16 lampu TLOF Helipad meredup/menyala $C^0$-kontinu (`0.000` mutlak pada `DAY` sesuai Hukum Emas 10).
  - **Visualisasi Tombol Transisi:** Tombol target menampilkan bilah progres bergerak `--trans-pct` + persentase & hitung mundur detik (`64% · 9 dtk`), tombol asal menampilkan status `.is-departing` (`Mereda 36%`), dan pil ringkas kanan-bawah `#sim-pill-toggle` ikut memvisualisasikan progres transisi maupun status `LIVE Depok`.
- **Sistem Preloader Arsitektural Interaktif 9-Tahap (`#nexus-preloader`):**
  - Render instan `< 16 ms` HTML/CSS/SVG murni dengan animasi cetak biru (*CAD golden wireframe*) Menara Segi-8 & Atap Tajug 3-Tumpang Rektorat UI + kompas Pulau 360° & Helipad, jam WIB Depok langsung, kartu interaktif *"Tahukah Anda?"*, penghitung segitiga bergulir (`0 -> 2.056.828`), *frame-yielding* antar 9 tahap pembangkitan prosedural, serta *GPU Shader Pre-Warming* (`renderer.compile(scene, camera)`) sebelum tirai diangkat ke `display: none` (mematuhi Hukum Emas 9).

---

## 8. Geometri Kepulauan UI 2-Pulau: Perbukitan `Pulau U`, `Selat UI`, serta Gunung Api & Landasan Pacu `Pulau I` (`NEXUS v3.0 – v3.1`)

- **Orientasi Sumbu Tampak Udara Tegak (`U` Kiri, `I` Kanan):**
  - `+X` = Atas Layar (*Screen UP*), `-X` = Bawah Layar (*Screen DOWN*), `-Z` = Kiri Layar (*Screen LEFT*), `+Z` = Kanan Layar (*Screen RIGHT*).
  - Pusat Kepulauan: `ARCHIPELAGO_CX = -155.0 m`, `ARCHIPELAGO_CZ = +340.0 m`; Cakrawala Laut Luar: `WATER_OUTER_RADIUS = 1650.0 m` (`3,3 km` diameter cakrawala laut bulat + kubah langit `R = 3400 m`).
- **1. `Pulau U` — Kampus Rektorat Landai (`0.00 m`) + Semenanjung Berbukit-Bukit Hijau (`getIslandUCoastDistance(x, z)`):**
  - **Zona Rektorat & Rotunda Tetap Landai (`0.000 m`):** Seluruh Kompleks Gedung Rektorat UI (`(0, 0)`), Pelataran Bendera, Lapangan Rotunda UI (`(-2.5, +106.5)`), dan Helipad VIP `(H)` (`(-104.0, +106.5)`) dijaga landai murni pada elevasi `0.000 m` dengan transisi $C^1$ `smootherstep`.
  - **Perbukitan Tropis Bergelombang (`~12.8 m – 24.5 m`):** Lengkungan bawah tapal kuda (`x < -114 m`) dan Lengan Semenanjung Kanan (`z ∈ [+318 m, +466 m]`) membentuk rangkaian 6 punggungan bukit hijau membulat & lembah alami (`Puncak Lengkungan Bawah = +18.96 m`, `Puncak Utara Lengan Kanan = +24.47 m`, `Puncak Tengah Lengan Kanan = +12.82 m`, `Puncak Bawah Lengan Kiri = +16.38 m`), disertai pelana pantai landai di bibir timur `Pulau U` (`sekitar x = -215, z = +454`) untuk calon jembatan masa depan.
  - Dikelilingi 360° penuh oleh turap batu kali 13-lapis tertutup (`720` segmen panjang busur seragam Newton-Raphson, `0.000 m` celah sambungan) + `84` pilar penopang radial & *bollard* tepi laut + *skirt* kedap air `-18.0 m`.
- **2. `Selat UI` (Selat Laut Terbuka Selebar `140 m`, `z ∈ [+466 m, +606 m]`, Tanpa Jembatan Terlebih Dahulu):**
  - Memisahkan `Pulau U` dan `Pulau I` secara bersih dengan dasar selat `y = -3.85 m` dan zona reservasi landai di kedua pantai yang saling berhadapan (siap disambung jembatan pada tahap pengembangan berikutnya).
- **3. `Pulau I` — Gunung Api Aktif (`Gunung Api Pulau I`) & Landasan Pacu Pesawat Lurus (`RWY 36/18`) (`NEXUS v3.1`):**
  - Dimensi Pulau `"I"`: `I_CENTER = (-160.0, +694.0)`, `I_HALF_LEN_X = 244.0 -> tinggi 488 m` (`x ∈ [-404 m, +84 m]`), `I_HALF_WID_Z = 88.0 -> lebar 176 m` (`z ∈ [+606 m, +782 m]`, `I_CORNER_R = 56.0 m`), dikelilingi 360° oleh turap batu kali (`420` segmen tertutup) + *skirt* `-18.0 m`.
  - **Gunung Api Stratovolcano Aktif Ikonik (`VOLCANO_CX = -32.0, VOLCANO_CZ = +698.0, R_base = 98.0 m`):** Lereng cekung stratovolcano berprofil $C^1$ dengan `11` alur rekahan lava radial 3D, pemadatan jaring koordinat Gaussian `2,1x` (`~0,76 m`) melintasi bibir kawah puncak (`R_rim = 19.5 m`, elevasi bibir kawah **`+56.45 m .. +60.18 m`**) dan mangkuk kaldera cekung sedalam `~11.3–15.0 m` (`lantai kawah y = +45.19 m`) berisi **rekahan kerak obsidian & magma pijar Voronoi HDR** (`E1B` pada `ProceduralTextureFactory.js`) + **kolom kepulan asap vulkanik & percikan bara kawah 100% GPU (`NexusVolcanicSmokePlume`, `288` quad volumetrik tertiup `uWindVec`)**.
  - **Landasan Pacu Pesawat Lurus ICAO (`RWY 36/18` — `SplineCorridorBuilder._buildPulauIAirstripRunway()`):** Membentang lurus di dataran rata kaki selatan gunung api (`RUNWAY_CZ = +708.0 m`, `x ∈ [-372.0 m, -138.0 m]` = `234 m` dek utama + `2 × 10 m` *blast pad* chevron kuning `[-382.0 m, -128.0 m]` = **`254 m × 22 m`**, `RUNWAY_DECK_Y = +0.58 m` di atas teras rata `+0.44 m`), lengkap dengan *piano-key threshold* 3D, angka vektor 3D **`"36"`** & **`"18"`**, *aiming point*, *centerline*, 2 *stub taxiway* (`TWY A` & `TWY B`), **Apron Parkir Pesawat Barat (`60 m × 24 m`, `z ∈ [660, 684]`)** dengan 2 marka *T-stand*, lampu tepi/ambang/ujung landasan + **4-unit PAPI** (menyala khusus malam hari, `0.000` saat siang), serta tiang *windsock* bandara `6.8 m` di `(-214.0, +686.5)` (**`0` tambahan Draw Call**!).
- **4. Anggaran Render & Preset Kamera Kepulauan (`<= 25 Draw Calls`):**
  - Total Segitiga Dunia: **`2.125.414` segitiga** (`13,14x`); Medan 2 Pulau + Turap + Laut = **`2 Draw Calls`** (`517.300` segitiga); Koridor Rotunda + Helipad VIP `Pulau U` + Landasan Pacu `Pulau I` = **`2 Draw Calls`** (`69.688` segitiga); Total Scene = **`23 – 25 <= 25 Draw Calls`** di seluruh 12 kombinasi cuaca × waktu & sepanjang transisi `20–45 detik`.
  - Preset Kamera: `archipelagoUI` (**Kepulauan UI**), `island360` (**Pulau U**), `pulauI` (**Pulau I**), `helipadVIP` (**Helipad VIP**), `runwayPulauI` (**Landasan I**), dan `volcanoCrater` (**Kawah Api**).
