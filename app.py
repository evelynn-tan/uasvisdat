"""
app.py - Dashboard Streamlit: kemiskinan, pembangunan manusia, dan tipologi provinsi.

Menggabungkan 01_analisis_pca.py (PCA + klaster provinsi) dan 02_geospasial_lisa.py
(peta kab/kota, LISA, Moran, treemap, icicle). Jalankan lokal:  streamlit run app.py

Struktur folder (semua di repo yang sama):
    app.py, 01_analisis_pca.py, 02_geospasial_lisa.py, requirements.txt
    data/raw/rumah_tangga.xlsx        (dipakai skrip 01)
    data/raw/bahan_bakar_2022.xlsx    (dipakai treemap)
    data/raw/internet_2022.xlsx       (dipakai icicle)
    data/processed/kabkota_lisa.geojson, moran_global.csv   (hasil skrip 02)
    data/processed/provinsi_terolah.csv                     (hasil skrip 01)
"""
import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"

st.set_page_config(page_title="Dashboard Kemiskinan & Pembangunan Manusia",
                   page_icon="🗺️", layout="wide")


# ---------- MUAT MODUL (nama file berawalan angka, jadi lewat importlib) ----------
@st.cache_resource(show_spinner=False)
def muat_modul(file: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, BASE / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def cek_berkas() -> list[str]:
    wajib = [RAW / "rumah_tangga.xlsx", RAW / "bahan_bakar_2022.xlsx", RAW / "internet_2022.xlsx",
             PROC / "kabkota_lisa.geojson", PROC / "moran_global.csv"]
    return [str(p.relative_to(BASE)) for p in wajib if not p.exists()]


hilang = cek_berkas()
if hilang:
    st.error("Berkas berikut belum ada di repo:\n\n" + "\n".join(f"- `{h}`" for h in hilang)
             + "\n\nJalankan skrip 01 lalu 02 secara lokal, lalu salin hasilnya ke folder `data/`.")
    st.stop()

pca_mod = muat_modul("01_analisis_pca.py", "analisis_pca")
geo_mod = muat_modul("02_geospasial_lisa.py", "geospasial_lisa")
geo_mod.DATA = RAW            # path treemap/icicle mengikuti folder repo, bukan D:/uas
CFG = {"displaylogo": False, "scrollZoom": True}


# ---------- DATA (di-cache supaya tidak dihitung ulang tiap interaksi) ----------
@st.cache_data(show_spinner="Menghitung PCA dan klaster provinsi...")
def hitung_pca():
    d = pca_mod.variabel_turunan(pca_mod.muat_data())
    pca, skor, load = pca_mod.jalankan_pca(d, pca_mod.ACTIVE)
    klaster, _ = pca_mod.klasterkan(skor, pca_mod.K)
    out = d.copy()
    out["PC1"], out["PC2"], out["PC3"] = skor[:, 0], skor[:, 1], skor[:, 2]
    out["klaster"] = klaster
    out["tipologi"] = out["klaster"].map(pca_mod.NAMA_KLASTER)
    return d, out, load, pca.explained_variance_ratio_ * 100


@st.cache_data(show_spinner="Membaca data kab/kota...")
def muat_kabkota() -> gpd.GeoDataFrame:
    return gpd.read_file(PROC / "kabkota_lisa.geojson")


@st.cache_data
def muat_moran() -> pd.DataFrame:
    return pd.read_csv(PROC / "moran_global.csv")


@st.cache_data(show_spinner="Menyusun peta interaktif...")
def html_peta() -> str:
    return geo_mod.buat_peta(muat_kabkota()).get_root().render()


@st.cache_data(show_spinner="Menyusun treemap...")
def fig_treemap(tampilan: str):
    return geo_mod.grafik_treemap_bbm(tampilan)


@st.cache_data(show_spinner="Menyusun icicle...")
def fig_icicle(tampilan: str):
    return geo_mod.grafik_icicle_internet(tampilan)


def tampil(fig):
    st.plotly_chart(fig, use_container_width=True, config=CFG)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Tentang dashboard")
    st.write("Kemiskinan, pengangguran, dan pembangunan manusia di Indonesia: "
             "peta kab/kota, klaster spasial (LISA), serta tipologi provinsi dari PCA.")
    st.caption("Sumber: BPS (data diolah).")
    st.divider()
    st.markdown("**Isi halaman (scroll ke bawah)**")
    st.markdown("1. Ringkasan\n2. Peta dan klaster spasial\n3. Tipologi provinsi (PCA)\n"
                "4. Hierarchical Visualization")
    st.markdown("**Cara membaca**")
    st.markdown("- Arahkan kursor ke objek untuk detail.\n"
                "- Peta: pilih layer di kanan atas, klik wilayah untuk zoom, tombol rumah untuk reset.\n"
                "- Hierarchical Visualization: pilih tampilan lewat dropdown, klik kotak untuk drill-down, "
                "klik penunjuk posisi di atas untuk naik.")

st.title("Kemiskinan dan Pembangunan Manusia di Indonesia")

# Satu halaman yang di-scroll; tiap bagian dipisah header dan garis pembatas (tanpa tab).

# ---------- BAGIAN 1: RINGKASAN ----------
st.header("Ringkasan")
g = muat_kabkota()
tab_moran = muat_moran()
d, out, load, ve = hitung_pca()

var = geo_mod.VAR_LISA
moran_i = float(tab_moran.loc[tab_moran["variabel"] == var, "Moran_I"].iloc[0])
n_hot = int((g["lisa_kat"] == "High-High (hotspot)").sum())
n_cold = int((g["lisa_kat"] == "Low-Low (coldspot)").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Provinsi dianalisis", len(out))
c2.metric("Kab/kota", len(g))
c3.metric(f"Moran's I ({geo_mod.LABEL[var]})", f"{moran_i:.3f}")
c4.metric("Hotspot / coldspot LISA", f"{n_hot} / {n_cold}")

st.markdown(
    f"Kemiskinan antar kab/kota **mengelompok secara spasial** (Moran's I = {moran_i:.2f}): "
    f"{n_hot} kab/kota termasuk hotspot (miskin dikelilingi miskin) dan {n_cold} coldspot. "
    "Di tingkat provinsi, PCA mengelompokkan 38 provinsi ke dalam empat tipologi.")

kiri, kanan = st.columns(2)
with kiri:
    st.subheader("Moran's I global")
    st.dataframe(tab_moran.rename(columns={"variabel": "Variabel", "Moran_I": "Moran's I",
                                           "z_sim": "z (simulasi)", "p_sim": "p (simulasi)"}),
                 hide_index=True, use_container_width=True)
    st.caption("Semua variabel menunjukkan autokorelasi spasial positif (p < 0,05).")
with kanan:
    st.subheader("Tipologi provinsi")
    ukuran = (out.groupby(["klaster", "tipologi"]).size().reset_index(name="Jumlah provinsi")
                 .sort_values("klaster").drop(columns="klaster")
                 .rename(columns={"tipologi": "Tipologi"}))
    st.dataframe(ukuran, hide_index=True, use_container_width=True)

# ---------- BAGIAN 2: PETA & LISA ----------
st.divider()
st.header("Peta dan klaster spasial")
st.subheader("Peta interaktif kab/kota")
components.html(html_peta(), height=720, scrolling=False)

st.caption("Layer: % penduduk miskin, IPM, TPT, RLS, klaster LISA, dan simbol proporsional.")
kiri, kanan = st.columns(2)
g = muat_kabkota()
gdf = pd.DataFrame(g.drop(columns="geometry"))
with kiri:
    tampil(geo_mod.grafik_moran(gdf, geo_mod.VAR_LISA, moran_i))
with kanan:
    tampil(geo_mod.grafik_share_ipm(gdf))

# ---------- BAGIAN 3: PCA ----------
st.divider()
st.header("Tipologi provinsi (PCA)")
d, out, load, ve = hitung_pca()
st.caption(f"PC1 menjelaskan {ve[0]:.1f}% varians dan PC2 {ve[1]:.1f}%; "
           f"tiga komponen pertama {ve[:3].sum():.1f}%.")
st.subheader("Biplot")
tampil(pca_mod.grafik_biplot(out, load, ve))
st.subheader("Heatmap terklaster")
tampil(pca_mod.grafik_heatmap(out))
st.subheader("Koordinat paralel")
tampil(pca_mod.grafik_paralel(out))
st.subheader("Korelasi")
tampil(pca_mod.grafik_korelasi(d))

with st.expander("Loading komponen utama dan daftar provinsi per tipologi"):
    st.dataframe(load.iloc[:, :3].round(2), use_container_width=True)
    for k in sorted(out["klaster"].unique()):
        nama = pca_mod.NAMA_KLASTER[k]
        st.markdown(f"**{k}. {nama}**: " + ", ".join(out.loc[out["klaster"] == k, "prov"]))

# ---------- BAGIAN 4: HIERARCHICAL VISUALIZATION ----------
st.divider()
st.header("Hierarchical Visualization: rumah tangga (2022)")
tampilan = st.selectbox("Tampilan wilayah", geo_mod.PILIHAN_TAMPILAN,
                        help="Seluruh Indonesia per provinsi, ringkasan per pulau, atau satu pulau saja.")

st.markdown("#### Bahan bakar utama memasak (treemap)")
st.caption("Ukuran = jumlah rumah tangga; warna = % rumah tangga yang memakai bahan bakar kotor "
           "(dari rumah tangga yang memasak).")
with st.expander("Apa itu bahan bakar bersih dan kotor?"):
    st.markdown(
        "| Kelompok | Jenis bahan bakar |\n|---|---|\n"
        "| **Bersih** | Listrik, LPG (gas/elpiji) |\n"
        "| **Kotor** | Minyak tanah, arang/briket, kayu |\n"
        "| **Tidak memasak** | Kategori \"Lainnya\" (tidak masuk perhitungan persen kotor) |\n\n"
        "**% bahan bakar kotor** = kotor / (kotor + bersih) x 100.\n\n"
        "Bahan bakar kotor menghasilkan asap dan polusi udara dalam ruangan yang berisiko bagi "
        "kesehatan, terutama perempuan dan anak. Pengelompokan ini sama dengan variabel "
        "`bb_bersih` (listrik + elpiji) pada analisis PCA.")
tampil(fig_treemap(tampilan))

st.markdown("#### Akses internet rumah tangga (icicle)")
st.caption("Ukuran = jumlah rumah tangga; warna = % rumah tangga yang pernah mengakses internet.")
tampil(fig_icicle(tampilan))