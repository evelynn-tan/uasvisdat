"""
02_geospasial_lisa.py
Lanjutan 01_analisis_pca.py: data kab/kota -> bobot spasial -> Moran's I global -> LISA
-> peta interaktif -> Moran scatterplot -> grafik share penduduk menurut kategori IPM
-> treemap bahan bakar memasak -> icicle akses internet.
Berhenti sampai visualisasi geospasial (belum Streamlit / deploy).

Kebutuhan:
  pip install geopandas shapely libpysal esda mapclassify folium plotly pandas numpy openpyxl
Jalankan:  python 02_geospasial_lisa.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import folium
import plotly.express as px
import plotly.graph_objects as go
import mapclassify as mc
from branca.element import MacroElement
from jinja2 import Template
from libpysal import weights
from libpysal.weights import lag_spatial
from esda.moran import Moran, Moran_Local

# ---------- 0. KONFIGURASI ----------
BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

PETA = Path("D:/uas/peta_FIX.geojson")
EXCEL = Path("D:/uas/data/variabel_kabkota.xlsx")   # kolom: KDPKAB, penduduk, penduduk_miskin
SATUAN_PEND = "jiwa"                           # sesuaikan dengan satuan di Excel (jiwa / ribu jiwa)
DATA = Path("D:/uas/data")
CFG = {"displaylogo": False, "scrollZoom": True}

SEED = 3
PERM = 999
ALPHA = 0.05
VAR_LISA = "PPM"                                    # variabel yang dianalisis LISA
VARS = ["PPM", "TPT", "IPM", "RLS"]
LABEL = {"PPM": "% penduduk miskin", "TPT": "TPT (%)",
         "IPM": "IPM", "RLS": "Rata-rata lama sekolah (tahun)"}
SIMPLIFY_DERAJAT = 0.01                             # penyederhanaan geometri HANYA untuk tampilan

# Palet ColorBrewer 5 kelas (Fisher-Jenks), aman untuk buta warna
PALET = {"PPM": ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],   # YlOrRd
         "TPT": ["#f2f0f7", "#cbc9e2", "#9e9ac8", "#756bb1", "#54278f"]}   # Purples

# Kategori IPM (BPS) dan RLS (jenjang pendidikan)
BIN_IPM = [60, 70, 80]
LABEL_IPM = ["Rendah (< 60)", "Sedang (60 - 69,99)", "Tinggi (70 - 79,99)", "Sangat tinggi (≥ 80)"]
WARNA_IPM = ["#eff3ff", "#bdd7e7", "#6baed6", "#2171b5"]                    # Blues

BIN_RLS = [6, 9, 12]
LABEL_RLS = ["< 6 (belum lulus SD)", "6 - 8,99 (SD s.d. kls 2 SMP)",
             "9 - 11,99 (SMP s.d. kls 3 SMA)", "≥ 12 (SMA ke atas)"]
WARNA_RLS = ["#E8F5E9", "#A5D6A7", "#4CAF50", "#2E7D32"]                    # hijau (utama #4CAF50)

# Klaster LISA (Okabe-Ito)
WARNA_LISA = {"High-High (hotspot)": "#D55E00", "Low-Low (coldspot)": "#0072B2",
              "Low-High (pencilan)": "#56B4E9", "High-Low (pencilan)": "#CC79A7",
              "Tidak signifikan": "#DDDDDD"}


# ---------- 1. BACA & BERSIHKAN ----------
def muat_peta() -> gpd.GeoDataFrame:
    with open(PETA, encoding="utf-8") as f:
        g = gpd.GeoDataFrame.from_features(json.load(f)["features"], crs="EPSG:4326")
    for v in VARS:
        g[v] = pd.to_numeric(g[v], errors="coerce")
    g["KDPKAB"] = g["KDPKAB"].astype(str).str.strip()

    # gabung data tambahan dari Excel
    tb = pd.read_excel(EXCEL)
    tb.columns = tb.columns.astype(str).str.strip()
    tb = tb.rename(columns={"Jumlah Penduduk": "penduduk",
                            "Jumlah Penduduk Miskin": "penduduk_miskin"})
    tb["KDPKAB"] = tb["KDPKAB"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    for c in ["penduduk", "penduduk_miskin"]:
        tb[c] = pd.to_numeric(tb[c], errors="coerce")

    g = g.merge(tb[["KDPKAB", "penduduk", "penduduk_miskin"]],
                on="KDPKAB", how="left", validate="one_to_one")
    print("Wilayah tanpa data tambahan:", int(g["penduduk_miskin"].isna().sum()))

    g = g[["KDPKAB", "KDPPUM", "WADMPR", "Nama_Wilayah", *VARS,
           "penduduk", "penduduk_miskin", "geometry"]].copy()
    g["geometry"] = g.geometry.apply(shapely.force_2d)
    g["geometry"] = shapely.make_valid(g.geometry.values)
    g = g.to_crs(4326).reset_index(drop=True)
    return g


def cek_kualitas(g: gpd.GeoDataFrame) -> None:
    print(f"Jumlah kab/kota: {len(g)}")
    print("Kode kembar:", int(g["KDPKAB"].duplicated().sum()))
    print("Data kosong per variabel:", g[VARS + ["penduduk", "penduduk_miskin"]].isna().sum().to_dict())
    assert g["KDPKAB"].is_unique, "KDPKAB harus unik"
    assert g[VARS].notna().all().all(), "Ada nilai kosong pada variabel utama; tangani dulu"
    assert g[["penduduk", "penduduk_miskin"]].notna().all().all(), "Data penduduk belum lengkap (cek KDPKAB di Excel)"
    # konsistensi: PPM hitung vs PPM resmi (selisih besar = tahun/satuan/kode tidak sinkron)
    selisih = (g["penduduk_miskin"] / g["penduduk"] * 100 - g["PPM"]).abs()
    print(f"Selisih PPM hitung vs resmi: median={selisih.median():.2f}, maks={selisih.max():.2f}")


def tambah_tipologi_provinsi(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sambungkan hasil klaster provinsi dari skrip 01 (opsional)."""
    f = OUT / "provinsi_terolah.csv"
    g["tipologi"] = "Tidak tersedia"
    if not f.exists():
        print("provinsi_terolah.csv tidak ditemukan -> tipologi provinsi dilewati")
        return g
    alias_prov = {"DAERAH ISTIMEWA YOGYAKARTA": "DI YOGYAKARTA",
                  "KEPULAUAN BANGKA BELITUNG": "KEP. BANGKA BELITUNG",
                  "KEPULAUAN RIAU": "KEP. RIAU"}
    p = pd.read_csv(f)[["prov", "tipologi"]]
    p["kunci"] = p["prov"].str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    kunci = g["WADMPR"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    kunci = kunci.replace(alias_prov)
    g["tipologi"] = kunci.map(dict(zip(p["kunci"], p["tipologi"]))).fillna("Tidak tersedia")
    tak = sorted(set(g.loc[g["tipologi"] == "Tidak tersedia", "WADMPR"]))
    if tak:
        print("Provinsi tanpa tipologi (nama tidak cocok / pemekaran baru):", tak)
    return g


# ---------- 2. BOBOT SPASIAL, MORAN, LISA ----------
def bangun_bobot(g: gpd.GeoDataFrame):
    ids = g["KDPKAB"].tolist()
    w = weights.Queen.from_dataframe(g, ids=ids, silence_warnings=True)
    if w.islands:                                   # kepulauan -> hubungkan ke tetangga terdekat
        print(f"Pulau tanpa tetangga Queen: {len(w.islands)} -> disambungkan ke tetangga terdekat")
        knn1 = weights.KNN.from_dataframe(g.to_crs(3395), k=1, ids=ids)
        w = weights.attach_islands(w, knn1)
    w.transform = "r"                               # row-standardized
    assert w.n == len(g) and not w.islands
    print(f"Bobot spasial: n={w.n}, rata-rata tetangga={w.mean_neighbors:.2f}")
    return w


def moran_global(g, w) -> pd.DataFrame:
    baris = []
    for v in VARS:
        np.random.seed(SEED)
        m = Moran(g[v].values, w, permutations=PERM)
        baris.append({"variabel": v, "Moran_I": round(m.I, 3), "z_sim": round(m.z_sim, 2),
                      "p_sim": round(m.p_sim, 4)})
    tab = pd.DataFrame(baris)
    print("\nMoran's I global:\n", tab.to_string(index=False))
    return tab


def jalankan_lisa(g, w, var: str) -> gpd.GeoDataFrame:
    y = g[var].values
    np.random.seed(SEED)
    lm = Moran_Local(y, w, permutations=PERM)
    nama_kuadran = {1: "High-High (hotspot)", 2: "Low-High (pencilan)",
                    3: "Low-Low (coldspot)", 4: "High-Low (pencilan)"}
    g = g.copy()
    g["lisa_I"] = lm.Is.round(3)
    g["lisa_p"] = lm.p_sim.round(4)
    g["lisa_kat"] = [nama_kuadran[q] if p < ALPHA else "Tidak signifikan"
                     for q, p in zip(lm.q, lm.p_sim)]
    z = (y - y.mean()) / y.std()
    g["z_" + var] = z
    g["lag_z_" + var] = lag_spatial(w, z)
    print(f"\nLISA {var} (p<{ALPHA}, {PERM} permutasi):")
    print(g["lisa_kat"].value_counts().to_string())
    return g


# ---------- 3. KLASIFIKASI & LEGENDA ----------
def kelas_warna(series: pd.Series, palet: list[str], k: int = 5):
    """Fisher-Jenks: cocok untuk sebaran miring karena meminimalkan variasi dalam kelas
    dan tidak memaksa jumlah unit sama seperti quantile."""
    cl = mc.FisherJenks(series.values, k=k)
    warna = [palet[i] for i in cl.yb]
    batas = [float(series.min())] + [float(b) for b in cl.bins]
    return warna, batas


def kategori(series, bins, label, warna):
    idx = np.digitize(series.values, bins)
    return [label[i] for i in idx], [warna[i] for i in idx]


def _kotak(c) -> str:
    return (f'<span style="background:{c};width:14px;height:14px;display:inline-block;'
            f'margin-right:6px;border:1px solid #999"></span>')


def html_legenda(judul, batas, palet, satuan="") -> str:
    baris = "".join(f'<div>{_kotak(palet[i])}{batas[i]:.1f} - {batas[i+1]:.1f}{satuan}</div>'
                    for i in range(len(batas) - 1))
    return f'<div><b>{judul}</b>{baris}</div>'


def html_legenda_kat(judul, label, warna) -> str:
    baris = "".join(f'<div>{_kotak(c)}{l}</div>' for l, c in zip(label, warna))
    return f'<div><b>{judul}</b>{baris}</div>'


# ---------- 4. PETA INTERAKTIF (FOLIUM) ----------
FIELDS = ["Nama_Wilayah", "WADMPR", "tipologi", "PPM", "penduduk_miskin", "penduduk",
          "IPM", "kat_IPM", "TPT", "RLS", "kat_RLS", "lisa_kat", "lisa_p"]
ALIAS = ["Kab/Kota", "Provinsi", "Tipologi provinsi", "% penduduk miskin",
         f"Penduduk miskin ({SATUAN_PEND})", f"Jumlah penduduk ({SATUAN_PEND})",
         "IPM", "Kategori IPM", "TPT (%)", "RLS (tahun)", "Kategori RLS",
         "Klaster LISA", "p-value LISA"]


class TombolReset(MacroElement):
    """Tombol rumah di kiri atas untuk kembali ke tampilan seluruh Indonesia."""
    _template = Template("""
        {% macro script(this, kwargs) %}
        var tombol = L.control({position: 'topleft'});
        tombol.onAdd = function () {
            var d = L.DomUtil.create('div', 'leaflet-bar');
            d.innerHTML = '<a href="#" title="Kembali ke Indonesia" style="width:30px;height:30px;line-height:30px;text-align:center;font-size:16px;text-decoration:none">&#8962;</a>';
            d.onclick = function (e) { e.preventDefault(); {{ this._parent.get_name() }}.setView([-2.5, 118], 5); };
            return d;
        };
        tombol.addTo({{ this._parent.get_name() }});
        {% endmacro %}
    """)


def pasang_legenda(m, legenda: dict, peta_layer: dict, tetap: str = "") -> None:
    """legenda: {kunci: html}; peta_layer: {nama layer choropleth: kunci legenda}.
    Legenda choropleth berganti otomatis; legenda `tetap` selalu tampil."""
    awal = list(peta_layer.values())[0]
    isi = "".join(
        f'<div class="lg" id="lg-{k}" style="display:{"block" if k == awal else "none"}">{v}</div>'
        for k, v in legenda.items() if k != tetap)
    if tetap:
        isi += f'<div style="margin-top:6px">{legenda[tetap]}</div>'
    kotak = ('<div id="legenda" style="position:fixed;bottom:40px;left:10px;z-index:9999;'
             'background:white;padding:8px 10px;border:1px solid #999;border-radius:4px;'
             'font-size:11px;max-width:240px">' + isi + '</div>')
    js = f"""
    <script>
    document.addEventListener('DOMContentLoaded', function () {{
      var peta = {m.get_name()};
      var kunci = {json.dumps(peta_layer)};
      peta.on('overlayadd', function (e) {{
        var k = kunci[e.name];
        if (!k) return;
        document.querySelectorAll('#legenda .lg').forEach(function (el) {{
          el.style.display = (el.id === 'lg-' + k) ? 'block' : 'none';
        }});
      }});
    }});
    </script>"""
    m.get_root().html.add_child(folium.Element(kotak + js))


def layer_choropleth(m, g, kolom_warna, nama, show):
    lyr = folium.GeoJson(
        g, name=nama, show=show,
        style_function=lambda f, c=kolom_warna: {"fillColor": f["properties"][c], "color": "#666",
                                                 "weight": 0.3, "fillOpacity": 0.85},
        highlight_function=lambda f: {"weight": 2, "color": "#000"},
        tooltip=folium.GeoJsonTooltip(fields=FIELDS, aliases=ALIAS, localize=True, sticky=False),
        zoom_on_click=True,
    )
    lyr.add_to(m)
    return lyr


def buat_peta(g: gpd.GeoDataFrame) -> folium.Map:
    t = g.copy()
    t["geometry"] = t.geometry.simplify(SIMPLIFY_DERAJAT, preserve_topology=True)
    t["_pt"] = t.geometry.representative_point()

    t["w_PPM"], b_ppm = kelas_warna(t["PPM"], PALET["PPM"])
    t["w_TPT"], b_tpt = kelas_warna(t["TPT"], PALET["TPT"])
    t["kat_IPM"], t["w_IPM"] = kategori(t["IPM"], BIN_IPM, LABEL_IPM, WARNA_IPM)
    t["kat_RLS"], t["w_RLS"] = kategori(t["RLS"], BIN_RLS, LABEL_RLS, WARNA_RLS)
    t["w_LISA"] = t["lisa_kat"].map(WARNA_LISA)

    m = folium.Map(location=[-2.5, 118], zoom_start=5, tiles=None,
                   control_scale=True, prefer_canvas=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Peta dasar (Esri)", control=False).add_to(m)  # control=False: tidak tampil di daftar layer

    kolom = ["KDPKAB", *FIELDS, "w_PPM", "w_TPT", "w_IPM", "w_RLS", "w_LISA", "geometry"]
    data = t[kolom]
    L_PPM = "Choropleth: % penduduk miskin"
    L_IPM = "Choropleth: IPM (kategori BPS)"
    L_TPT = "Choropleth: TPT (%)"
    L_RLS = "Choropleth: RLS (jenjang pendidikan)"
    L_LISA = f"Klaster LISA ({VAR_LISA})"
    L_SIM = "Simbol proporsional: penduduk miskin (warna = RLS)"
    layer_choropleth(m, data, "w_PPM", L_PPM, show=True)
    layer_choropleth(m, data, "w_IPM", L_IPM, show=False)
    layer_choropleth(m, data, "w_TPT", L_TPT, show=False)
    layer_choropleth(m, data, "w_RLS", L_RLS, show=False)
    layer_choropleth(m, data, "w_LISA", L_LISA, show=False)

    # pane khusus: lingkaran selalu di atas semua choropleth, tooltip tetap berfungsi
    folium.map.CustomPane("lingkaran", z_index=650, pointer_events=True).add_to(m)

    # simbol proporsional: luas lingkaran ~ jumlah penduduk miskin, warna = kategori RLS
    fg = folium.FeatureGroup(name=L_SIM, show=False)
    vmax = t["penduduk_miskin"].max()
    for _, r in t.sort_values("penduduk_miskin", ascending=False).iterrows():   # besar dulu agar yang kecil tidak tertutup
        folium.CircleMarker(
            location=[r["_pt"].y, r["_pt"].x],
            radius=2 + 16 * np.sqrt(r["penduduk_miskin"] / vmax),
            color="#333", weight=0.6, fill=True, fill_color=r["w_RLS"], fill_opacity=1,
            pane="lingkaran",
            tooltip=(f'{r["Nama_Wilayah"]} ({r["WADMPR"]})<br>'
                     f'Penduduk miskin: {r["penduduk_miskin"]:,.0f} {SATUAN_PEND}<br>'
                     f'RLS: {r["RLS"]:.2f} tahun ({r["kat_RLS"]})'),
        ).add_to(fg)
    fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    pasang_legenda(
        m,
        {
            "PPM": html_legenda("% penduduk miskin (Fisher-Jenks)", b_ppm, PALET["PPM"], "%"),
            "TPT": html_legenda("TPT (%) (Fisher-Jenks)", b_tpt, PALET["TPT"], "%"),
            "IPM": html_legenda_kat("IPM (kategori BPS)", LABEL_IPM, WARNA_IPM),
            "RLS": html_legenda_kat("Rata-rata lama sekolah (warna hijau)", LABEL_RLS, WARNA_RLS),
            "LISA": html_legenda_kat("Klaster LISA", list(WARNA_LISA), list(WARNA_LISA.values())),
            "SIM": '<div style="color:#555">Lingkaran: luas ~ jumlah penduduk miskin; warna = kategori RLS</div>',
        },
        peta_layer={L_PPM: "PPM", L_IPM: "IPM", L_TPT: "TPT", L_RLS: "RLS", L_LISA: "LISA"},
        tetap="SIM",
    )
    judul = ('<div style="position:fixed;top:8px;left:50%;transform:translateX(-50%);z-index:9999;'
             'background:rgba(255,255,255,.92);padding:4px 12px;border-radius:4px;font-size:13px">'
             '<b>Peta kemiskinan dan pembangunan manusia kab/kota</b>'
             '<div style="font-size:10px;color:#555">Sumber: BPS (data diolah). '
             'Batas wilayah: peta digital kab/kota.</div></div>')
    m.get_root().html.add_child(folium.Element(judul))
    m.add_child(TombolReset())
    return m


# ---------- 5. GRAFIK PLOTLY ----------
def grafik_moran(g: gpd.GeoDataFrame, var: str, I: float):
    fig = px.scatter(g, x="z_" + var, y="lag_z_" + var, color="lisa_kat",
                     color_discrete_map=WARNA_LISA, category_orders={"lisa_kat": list(WARNA_LISA)},
                     hover_name="Nama_Wilayah",
                     hover_data={"WADMPR": True, var: ":.2f", "lisa_p": True,
                                 "z_" + var: ":.2f", "lag_z_" + var: ":.2f", "lisa_kat": False})
    fig.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0.3, color="#333")))
    lim = float(np.ceil(max(g["z_" + var].abs().max(), g["lag_z_" + var].abs().max())))
    fig.add_shape(type="line", x0=-lim, x1=lim, y0=-lim * I, y1=lim * I, line=dict(color="#333", dash="dash"))
    fig.add_hline(y=0, line_width=1, line_color="#999")
    fig.add_vline(x=0, line_width=1, line_color="#999")
    for x, y, t in [(lim, lim, "High-High"), (-lim, lim, "Low-High"),
                    (-lim, -lim, "Low-Low"), (lim, -lim, "High-Low")]:
        fig.add_annotation(x=x, y=y, text=t, showarrow=False, font=dict(size=11, color="#555"),
                           xanchor="right" if x > 0 else "left", yanchor="top" if y > 0 else "bottom")
    fig.update_layout(
        title=f"Moran scatterplot {LABEL[var]} (Moran's I = {I:.3f}) - Sumber: BPS",
        xaxis_title=f"{LABEL[var]} (z-score)", yaxis_title="Rata-rata z-score tetangga (spatial lag)",
        legend_title="Klaster LISA", template="plotly_white")
    return fig


def grafik_share_ipm(g: gpd.GeoDataFrame):
    kat, _ = kategori(g["IPM"], BIN_IPM, LABEL_IPM, WARNA_IPM)
    d = g.assign(kat=kat)
    tab = (d.groupby("kat").agg(penduduk=("penduduk", "sum"), miskin=("penduduk_miskin", "sum"),
                                n=("KDPKAB", "count"))
             .reindex(LABEL_IPM).fillna(0).reset_index())
    tab["Penduduk"] = tab["penduduk"] / tab["penduduk"].sum() * 100
    tab["Penduduk miskin"] = tab["miskin"] / tab["miskin"].sum() * 100
    long = tab.melt(id_vars=["kat", "n"], value_vars=["Penduduk", "Penduduk miskin"],
                    var_name="Ukuran", value_name="Persen")
    fig = px.bar(long, x="kat", y="Persen", color="Ukuran", barmode="group",
                 category_orders={"kat": LABEL_IPM},
                 color_discrete_map={"Penduduk": "#0072B2", "Penduduk miskin": "#D55E00"},
                 hover_data={"n": True, "Persen": ":.1f"})
    fig.update_layout(
        title="Sebaran penduduk dan penduduk miskin menurut kategori IPM kab/kota - Sumber: BPS",
        xaxis_title="Kategori IPM kab/kota", yaxis_title="Persen dari total nasional (%)",
        legend_title="", template="plotly_white")
    return fig


# ---------- 5b. TREEMAP BAHAN BAKAR & ICICLE INTERNET ----------
def _cari_file(nama: str) -> Path:
    for ext in (".xlsx", ".xls", ".csv", ""):
        p = DATA / (nama + ext)
        if p.is_file():
            return p
    raise FileNotFoundError(f"{nama} tidak ditemukan di {DATA}")


def _baca_data(nama: str, kolom: list[str], sheet: str, skala: float = 1.0) -> pd.DataFrame:
    """Baca satu sheet. Cari sel header persis 'Provinsi' di kolom A, ambil data di bawahnya
    menurut urutan posisi kolom, lalu bagi dengan `skala`."""
    p = _cari_file(nama)
    if p.suffix == ".csv":
        raw = pd.read_csv(p, header=None, sep=None, engine="python", encoding="utf-8-sig")
    else:
        raw = pd.read_excel(p, header=None, sheet_name=sheet)
    cocok = raw.index[raw.iloc[:, 0].astype(str).str.strip().str.lower() == "provinsi"]
    if len(cocok) == 0:
        print(f"\n[{p.name} / {sheet}] 8 baris pertama:\n{raw.head(8).to_string()}\n")
        raise ValueError(f"Sel 'Provinsi' di kolom A tidak ditemukan ({p.name}, sheet {sheet})")
    d = raw.iloc[cocok[0] + 1:, :len(kolom)].copy()
    d.columns = kolom
    d = d.dropna(subset=["prov"])
    d["prov"] = d["prov"].astype(str).str.strip()
    d = d[~d["prov"].str.lower().isin(["", "nan", "none"]) & ~d["prov"].str.upper().str.contains("INDONESIA")]
    for k in kolom[1:]:
        d[k] = pd.to_numeric(d[k], errors="coerce")
    d = d.dropna(subset=kolom[1:], how="all").fillna(0).reset_index(drop=True)
    d = d[d[kolom[1:]].sum(axis=1) > 0].reset_index(drop=True)      # buang baris kosong/nol semua
    d[kolom[1:]] = d[kolom[1:]] / skala
    assert d["prov"].is_unique, f"Provinsi kembar di {nama}"
    print(f"{nama} (sheet {sheet}): {len(d)} provinsi")
    return d


class _Pohon:
    """Pengumpul node hierarki untuk go.Treemap / go.Icicle (branchvalues='total')."""
    def __init__(self):
        self.ids, self.labels, self.parents, self.values, self.warna, self.pct = [], [], [], [], [], []

    def tambah(self, id_, label, parent, nilai, warna, pct):
        self.ids.append(id_); self.labels.append(label); self.parents.append(parent)
        self.values.append(float(nilai)); self.warna.append(float(warna)); self.pct.append(float(warna))


def _persen(a, b):
    return a / b * 100 if b else 0.0


# ---------- pengelompokan provinsi -> pulau (untuk dropdown) ----------
URUTAN_PULAU = ["Sumatera", "Jawa", "Bali dan Nusa Tenggara", "Kalimantan",
                "Sulawesi", "Maluku", "Papua"]
SEMUA = "Seluruh Indonesia (per provinsi)"
PER_PULAU = "Ringkasan per pulau"
PILIHAN_TAMPILAN = [SEMUA, PER_PULAU] + URUTAN_PULAU


def _pulau(prov: str) -> str:
    u = " ".join(str(prov).upper().split())
    if "PAPUA" in u:
        return "Papua"
    if "MALUKU" in u:
        return "Maluku"
    if "KALIMANTAN" in u:
        return "Kalimantan"
    if "SULAWESI" in u:
        return "Sulawesi"
    if "NUSA TENGGARA" in u or u == "BALI":
        return "Bali dan Nusa Tenggara"
    if any(k in u for k in ("JAWA", "JAKARTA", "YOGYAKARTA", "BANTEN")):
        return "Jawa"
    if any(k in u for k in ("SUMATERA", "ACEH", "RIAU", "JAMBI", "BENGKULU", "LAMPUNG", "BANGKA")):
        return "Sumatera"
    return "Lainnya"


def _grup_tampilan(d: pd.DataFrame, tampilan: str):
    """Kembalikan (label akar, [(pulau atau None, subset data), ...]) sesuai pilihan dropdown."""
    d = d.assign(pulau=d["prov"].map(_pulau))
    tak_dikenal = d.loc[d["pulau"] == "Lainnya", "prov"].tolist()
    if tak_dikenal:
        print("Provinsi belum terpetakan ke pulau:", tak_dikenal)
    if tampilan == PER_PULAU:
        return "Indonesia", [(p, d[d["pulau"] == p]) for p in URUTAN_PULAU + ["Lainnya"]
                             if (d["pulau"] == p).any()]
    if tampilan in URUTAN_PULAU:
        return tampilan, [(None, d[d["pulau"] == tampilan])]
    return "Indonesia", [(None, d)]


def grafik_treemap_bbm(tampilan: str = SEMUA):
    # Sheet3 berisi persen x jumlah RT (skala 100x) -> dibagi 100 agar jadi jumlah RT
    d = _baca_data("bahan_bakar_2022",
                   ["prov", "Listrik", "Gas/Elpiji", "Minyak Tanah", "Arang/Briket", "Kayu", "Lainnya"],
                   sheet="Sheet3", skala=100)
    KEL = {"Gas/Elpiji": "Bersih", "Listrik": "Bersih",
           "Minyak Tanah": "Kotor", "Arang/Briket": "Kotor", "Kayu": "Kotor",
           "Lainnya": "Tidak memasak"}
    NAMA = {"Gas/Elpiji": "LPG", "Listrik": "Listrik (bersih)", "Minyak Tanah": "Minyak tanah",
            "Arang/Briket": "Arang/briket", "Kayu": "Kayu", "Lainnya": "Lainnya (tidak memasak)"}

    def pct_kotor(x):   # persen bahan bakar kotor dari rumah tangga yang memasak
        kotor = x[["Minyak Tanah", "Arang/Briket", "Kayu"]].sum()
        return _persen(kotor, kotor + x[["Listrik", "Gas/Elpiji"]].sum())

    akar, grup = _grup_tampilan(d, tampilan)
    semua = pd.concat([sub for _, sub in grup])
    t = _Pohon()
    tot = semua[list(KEL)].sum()
    t.tambah("ID", akar, "", tot.sum(), pct_kotor(tot), 0)
    for pulau, sub in grup:
        induk = "ID"
        if pulau:                                   # tingkat pulau (mode "Ringkasan per pulau")
            ts = sub[list(KEL)].sum()
            induk = f"P|{pulau}"
            t.tambah(induk, pulau, "ID", ts.sum(), pct_kotor(ts), 0)
        for _, r in sub.iterrows():
            p, pk = r["prov"], pct_kotor(r)
            t.tambah(p, p, induk, r[list(KEL)].sum(), pk, 0)
            for kel in ["Bersih", "Kotor", "Tidak memasak"]:
                jenis = [j for j, k in KEL.items() if k == kel]
                t.tambah(f"{p}|{kel}", kel, p, r[jenis].sum(), pk, 0)
                for j in jenis:
                    t.tambah(f"{p}|{j}", NAMA[j], f"{p}|{kel}", r[j], pk, 0)

    pos = [w for w, v in zip(t.warna[1:], t.values[1:]) if v > 0] or [0.0]
    fig = go.Figure(go.Treemap(
        ids=t.ids, labels=t.labels, parents=t.parents, values=t.values,
        branchvalues="total", maxdepth=3 + (tampilan == PER_PULAU),
        marker=dict(colors=t.warna, colorscale="Cividis",
                    cmin=min(pos), cmax=max(pos), showscale=True,
                    colorbar=dict(title="% bahan bakar kotor<br>(dari RT yang memasak)", ticksuffix="%"),
                    line=dict(width=0.6, color="white")),
        customdata=np.round(t.warna, 1),
        texttemplate="%{label}<br>%{value:,.0f} RT",
        hovertemplate=("<b>%{label}</b><br>Jumlah RT: %{value:,.0f}"
                       "<br>Porsi dari induk: %{percentParent:.1%}"
                       "<br>% bahan bakar kotor (wilayah ini): %{customdata}%<extra></extra>"),
        pathbar=dict(visible=True, thickness=24),
    ))
    fig.update_layout(
        title=f"Bahan bakar utama memasak rumah tangga, 2022: {akar} (Sumber: BPS)",
        margin=dict(t=70, l=10, r=10, b=10), template="plotly_white")
    return fig


def grafik_icicle_internet(tampilan: str = SEMUA):
    # Sheet2 berisi persen x jumlah RT (skala 100x) -> dibagi 100 agar jadi jumlah RT
    d = _baca_data("internet_2022", ["prov", "pernah_kota", "pernah_desa", "tidak_kota", "tidak_desa"],
                   sheet="Sheet2", skala=100)
    WIL = {"Perkotaan": ("pernah_kota", "tidak_kota"), "Perdesaan": ("pernah_desa", "tidak_desa")}
    KOL = ["pernah_kota", "pernah_desa", "tidak_kota", "tidak_desa"]

    def pct_pernah(x):
        pn = x["pernah_kota"] + x["pernah_desa"]
        return _persen(pn, pn + x["tidak_kota"] + x["tidak_desa"])

    akar, grup = _grup_tampilan(d, tampilan)
    semua = pd.concat([sub for _, sub in grup])
    t = _Pohon()
    tot = semua[KOL].sum()
    t.tambah("ID", akar, "", tot.sum(), pct_pernah(tot), 0)
    for pulau, sub in grup:
        induk = "ID"
        if pulau:
            ts = sub[KOL].sum()
            induk = f"P|{pulau}"
            t.tambah(induk, pulau, "ID", ts.sum(), pct_pernah(ts), 0)
        for _, r in sub.iterrows():
            p = r["prov"]
            pp = r["pernah_kota"] + r["pernah_desa"]
            tt = r["tidak_kota"] + r["tidak_desa"]
            t.tambah(p, p, induk, pp + tt, _persen(pp, pp + tt), 0)
            for w, (a, b) in WIL.items():
                pw = _persen(r[a], r[a] + r[b])
                t.tambah(f"{p}|{w}", w, p, r[a] + r[b], pw, 0)
                t.tambah(f"{p}|{w}|ya", "Pernah akses internet", f"{p}|{w}", r[a], pw, 0)
                t.tambah(f"{p}|{w}|tidak", "Tidak pernah akses", f"{p}|{w}", r[b], pw, 0)

    pos = [w for w, v in zip(t.warna[1:], t.values[1:]) if v > 0] or [0.0]
    fig = go.Figure(go.Icicle(
        ids=t.ids, labels=t.labels, parents=t.parents, values=t.values,
        branchvalues="total", maxdepth=4 + (tampilan == PER_PULAU),
        marker=dict(colors=t.warna, colorscale="Viridis",
                    cmin=min(pos), cmax=max(pos), showscale=True,
                    colorbar=dict(title="% RT pernah<br>akses internet", ticksuffix="%"),
                    line=dict(width=0.6, color="white")),
        customdata=np.round(t.warna, 1),
        texttemplate="%{label}<br>%{value:,.0f} RT",
        hovertemplate=("<b>%{label}</b><br>Jumlah RT: %{value:,.0f}"
                       "<br>Porsi dari induk: %{percentParent:.1%}"
                       "<br>% pernah akses internet (wilayah ini): %{customdata}%<extra></extra>"),
        tiling=dict(orientation="v"),
        pathbar=dict(visible=True, thickness=24),
    ))
    fig.update_layout(
        title=f"Akses internet rumah tangga, 2022: {akar} (Sumber: BPS)",
        margin=dict(t=70, l=10, r=10, b=10), template="plotly_white")
    return fig


# ---------- 6. MAIN ----------
def main():
    g = muat_peta()
    cek_kualitas(g)
    g = tambah_tipologi_provinsi(g)

    w = bangun_bobot(g)                  # bobot dari geometri ASLI (belum disederhanakan)
    tab = moran_global(g, w)
    g = jalankan_lisa(g, w, VAR_LISA)
    I_var = float(tab.loc[tab["variabel"] == VAR_LISA, "Moran_I"].iloc[0])

    tab.to_csv(OUT / "moran_global.csv", index=False)
    g.drop(columns="geometry").to_csv(OUT / "kabkota_lisa.csv", index=False)
    gs = g.copy()
    gs["geometry"] = gs.geometry.simplify(SIMPLIFY_DERAJAT, preserve_topology=True)
    (OUT / "kabkota_lisa.geojson").write_text(gs.to_json(), encoding="utf-8")

    buat_peta(g).save(OUT / "peta_geospasial.html")
    grafik_moran(g, VAR_LISA, I_var).write_html(OUT / "moran_scatter.html", config=CFG)
    grafik_share_ipm(g).write_html(OUT / "share_ipm.html", config=CFG)
    grafik_treemap_bbm().write_html(OUT / "treemap_bahan_bakar.html", config=CFG)
    grafik_icicle_internet().write_html(OUT / "icicle_internet.html", config=CFG)
    print("\nSelesai. File ada di data/processed/ (peta_geospasial.html, moran_scatter.html, "
          "share_ipm.html, treemap_bahan_bakar.html, icicle_internet.html)")


if __name__ == "__main__":
    main()