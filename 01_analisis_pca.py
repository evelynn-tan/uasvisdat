"""
01_analisis_pca.py
Olah data provinsi -> PCA -> klaster -> uji sensitivitas -> grafik Plotly.
Jalankan dari folder proyek:  python 01_analisis_pca.py
Kebutuhan: pip install pandas numpy scikit-learn scipy plotly openpyxl
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent          # folder skrip, jalan dari terminal mana pun
RAW = BASE / "data" / "raw" / "rumah_tangga.xlsx"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

K = 4                     # jumlah klaster (silhouette k=2..5 dicetak di bawah)
SEED = 3
ACTIVE = ["pln_gap", "pdrb_log", "bansos_rasio", "ipm", "bb_bersih",
          "internet", "kel_besar", "ppm", "tpt"]
LABEL = {"pln_gap": "Kesenjangan PLN (log)", "pdrb_log": "PDRB/kapita (log)",
         "bansos_rasio": "KPM bansos per 1.000 pddk", "ipm": "IPM",
         "bb_bersih": "% masak bahan bakar bersih", "internet": "% akses internet",
         "kel_besar": "% keluarga besar", "ppm": "% penduduk miskin", "tpt": "TPT (%)"}
NAMA_KLASTER = {1: "Maju dan kaya", 2: "Maju, ekonomi menengah",
                3: "Timur tertinggal", 4: "Tertinggal ekstrem"}
WARNA = {"Maju dan kaya": "#0072B2", "Maju, ekonomi menengah": "#009E73",
         "Timur tertinggal": "#E69F00", "Tertinggal ekstrem": "#D55E00"}  # ramah buta warna


# ---------- 1. BACA & BERSIHKAN ----------
def muat_data() -> pd.DataFrame:
    raw = pd.read_excel(RAW)
    d = raw.rename(columns={
        "Provinsi": "prov", "PLN": "pln", "PDRB per Capita": "pdrb_ribu",
        "Bansos": "bansos", "IPM": "ipm", "Bahan Bakar Listrik": "bb_listrik",
        "Bahan Bakar Elpiji": "bb_elpiji", "Internet": "internet",
        "Keluarga Besar": "kel_besar", "PPM": "ppm", "TPT": "tpt",
        "Penduduk": "penduduk"})
    d["prov"] = d["prov"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    d = d[d["prov"] != "INDONESIA"].reset_index(drop=True)
    for c in d.columns.drop("prov"):
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ".", regex=False))
    return d


def cek_kualitas(d: pd.DataFrame) -> None:
    print(f"Jumlah provinsi: {len(d)} (harus 38)")
    print("Data kosong:", int(d.isna().sum().sum()))
    pct = ["pln", "bb_listrik", "bb_elpiji", "internet", "kel_besar", "ppm", "tpt"]
    for c in pct:
        assert d[c].between(0, 100).all(), f"{c} di luar 0-100"
        kembar = d[c][d[c].duplicated(keep=False)].unique()
        if len(kembar):
            print(f"  nilai kembar {c}: {sorted(float(x) for x in kembar)}  (cek ke tabel sumber)")


def variabel_turunan(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["pln_gap"] = np.log(100 - d["pln"] + 1)          # makin tinggi = makin tertinggal
    d["pdrb_juta"] = d["pdrb_ribu"] / 1000
    d["pdrb_log"] = np.log(d["pdrb_juta"])
    d["bb_bersih"] = d["bb_listrik"] + d["bb_elpiji"]
    d["bansos_rasio"] = d["bansos"] / d["penduduk"]    # penduduk dalam ribu -> per 1.000 pddk
    assert (d["bb_bersih"] <= 100).all()
    return d


# ---------- 2. PCA ----------
def jalankan_pca(df: pd.DataFrame, vars_: list[str]):
    X = StandardScaler().fit_transform(df[vars_])
    pca = PCA().fit(X)
    skor = pca.transform(X)
    load = pca.components_.T.copy()
    # tanda komponen bebas -> tetapkan agar mudah dibaca:
    # PC1: IPM positif (makin tinggi = makin maju); PC2, PC3: loading terbesar positif
    for j in range(load.shape[1]):
        acuan = vars_.index("ipm") if (j == 0 and "ipm" in vars_) else np.abs(load[:, j]).argmax()
        if load[acuan, j] < 0:
            load[:, j] *= -1
            skor[:, j] *= -1
    return pca, skor, pd.DataFrame(load, index=vars_,
                                   columns=[f"PC{i+1}" for i in range(load.shape[1])])


def klasterkan(skor: np.ndarray, k: int):
    sc = skor[:, :3]
    print("Silhouette:", {kk: round(silhouette_score(sc, KMeans(kk, n_init=100, random_state=SEED).fit_predict(sc)), 3)
                          for kk in range(2, 6)})
    km = KMeans(k, n_init=100, random_state=SEED).fit(sc)
    # nomor klaster diurutkan menurut PC1: 1 = paling maju ... k = paling tertinggal
    urut = np.argsort(-km.cluster_centers_[:, 0])
    peta = {lama: baru + 1 for baru, lama in enumerate(urut)}
    return np.array([peta[x] for x in km.labels_]), km


# ---------- 3. UJI SENSITIVITAS ----------
def uji_sensitivitas(d, skor_penuh, km_penuh_label):
    papua_baru = ["PAPUA SELATAN", "PAPUA TENGAH", "PAPUA PEGUNUNGAN", "DKI JAKARTA"]
    baris = ~d["prov"].isin(papua_baru)
    _, s1, _ = jalankan_pca(d[baris], ACTIVE)
    r1 = abs(np.corrcoef(skor_penuh[baris.values, 0], s1[:, 0])[0, 1])
    _, s2, _ = jalankan_pca(d, [v for v in ACTIVE if v != "ipm"])
    r2 = abs(np.corrcoef(skor_penuh[:, 0], s2[:, 0])[0, 1])
    print(f"PC1 tanpa 3 Papua baru & DKI vs penuh: |r| = {r1:.3f}")
    print(f"PC1 tanpa IPM vs penuh:               |r| = {r2:.3f}")
    for buang in ["bb_bersih", "bansos_rasio"]:
        _, s, _ = jalankan_pca(d, [v for v in ACTIVE if v != buang])
        lab = KMeans(K, n_init=100, random_state=SEED).fit_predict(s[:, :3])
        print(f"\nKlaster tanpa {buang}:")
        print(pd.crosstab(pd.Series(km_penuh_label, name="awal"), pd.Series(lab, name="tanpa")))


# ---------- 4. GRAFIK ----------
def grafik_biplot(out, load, var_exp):
    # label provinsi hanya untuk yang penting; sisanya muncul saat hover
    tampil = {"DKI JAKARTA", "KEP. RIAU", "KALIMANTAN TIMUR", "PAPUA TENGAH", "PAPUA PEGUNUNGAN",
              "NUSA TENGGARA TIMUR", "MALUKU", "NUSA TENGGARA BARAT", "GORONTALO", "SULAWESI BARAT"}
    out = out.assign(label=out["prov"].where(out["prov"].isin(tampil), ""))
    fig = px.scatter(out, x="PC1", y="PC2", color="tipologi", text="label",
                     color_discrete_map=WARNA,
                     category_orders={"tipologi": list(NAMA_KLASTER.values())},
                     hover_name="prov",
                     hover_data={"ipm": ":.1f", "ppm": ":.1f", "PC1": ":.2f", "PC2": ":.2f",
                                 "tipologi": False, "label": False})
    fig.update_traces(textposition="top center", textfont_size=10, marker_size=10)
    skala = 3.2
    for v, (x, y) in load[["PC1", "PC2"]].iterrows():
        fig.add_annotation(x=x * skala, y=y * skala, ax=0, ay=0, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=2,
                           arrowcolor="#777", text="")
        fig.add_trace(go.Scatter(x=[x * skala], y=[y * skala], mode="text", text=[LABEL[v]],
                                 textposition="top center", textfont=dict(size=10, color="#444"),
                                 showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        title="Biplot PCA 38 provinsi (kanan = lebih maju)",
        xaxis_title=f"PC1 ({var_exp[0]:.1f}%) - tingkat kemajuan umum",
        yaxis_title=f"PC2 ({var_exp[1]:.1f}%) - keluarga besar dan TPT",
        legend_title="Tipologi", template="plotly_white",
        annotations=list(fig.layout.annotations) + [dict(
            text="Sumber: BPS (data diolah). Panah = arah variabel; arahkan kursor ke titik untuk nama provinsi.",
            xref="paper", yref="paper", x=0, y=-0.14, showarrow=False, font=dict(size=10))])
    return fig


def grafik_paralel(out):
    urutan = ["ipm", "ppm", "bansos_rasio", "pln_gap", "internet", "bb_bersih",
              "pdrb_log", "kel_besar", "tpt"]          # dikelompokkan per tema agar garis tidak banyak menyilang
    pal = [WARNA[NAMA_KLASTER[i]] for i in range(1, K + 1)]
    skala = []
    for i in range(K):                                  # skala warna diskret
        skala += [[i / K, pal[i]], [(i + 1) / K, pal[i]]]
    fig = go.Figure(go.Parcoords(
        line=dict(color=out["klaster"], colorscale=skala, cmin=0.5, cmax=K + 0.5, showscale=True,
                  colorbar=dict(title="Tipologi", tickvals=list(range(1, K + 1)),
                                ticktext=[NAMA_KLASTER[i] for i in range(1, K + 1)], len=0.6)),
        dimensions=[dict(label=LABEL[v], values=out[v]) for v in urutan]))
    fig.update_layout(title="Profil 9 variabel menurut tipologi (Sumber: BPS) - seret pada sumbu untuk menyaring",
                      template="plotly_white", margin=dict(t=90, l=60, r=60))
    return fig


def grafik_heatmap(out):
    z = StandardScaler().fit_transform(out[ACTIVE])
    urut = []
    for k in sorted(out["klaster"].unique()):
        idx = np.where(out["klaster"].values == k)[0]
        if len(idx) > 2:
            idx = idx[leaves_list(linkage(z[idx], "ward"))]
        urut += list(idx)
    urut = np.array(urut)
    prov = out["prov"].iloc[urut].tolist()
    kode = out["klaster"].iloc[urut].tolist()
    tip = [NAMA_KLASTER[k] for k in kode]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.06, 0.94],
                        vertical_spacing=0.01)
    # strip penanda tipologi di atas tiap kolom
    pal = [WARNA[NAMA_KLASTER[i]] for i in range(1, K + 1)]
    skala = []
    for i in range(K):
        skala += [[i / K, pal[i]], [(i + 1) / K, pal[i]]]
    fig.add_trace(go.Heatmap(z=[kode], x=prov, y=["Tipologi"], colorscale=skala, zmin=0.5, zmax=K + 0.5,
                             showscale=False, text=[tip], hovertemplate="%{x}<br>%{text}<extra></extra>"),
                  row=1, col=1)
    # heatmap utama
    fig.add_trace(go.Heatmap(z=z[urut].T, x=prov, y=[LABEL[v] for v in ACTIVE],
                             colorscale="RdBu_r", zmid=0,
                             # colorbar: len=0.55, y=0.3 (ganti pengaturan lama)
                             colorbar=dict(title="z-score<br>(merah = tinggi)", len=0.55, y=0.3),
                             hovertemplate="%{x}<br>%{y}<br>z = %{z:.2f}<extra></extra>"),
                  row=2, col=1)
    # legenda tipologi (marker bantu)
    for i in range(1, K + 1):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=NAMA_KLASTER[i],
                                 marker=dict(size=12, symbol="square", color=WARNA[NAMA_KLASTER[i]])))
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(tickangle=90, row=2, col=1)
    fig.update_layout(
    title="Heatmap terklaster z-score 38 provinsi (Sumber: BPS)",
    template="plotly_white", height=700, legend_title="Tipologi",
    legend=dict(x=1.02, y=1, yanchor="top"),
    margin=dict(b=230),
    annotations=[dict(
        text="Merah = nilai tinggi pada variabel itu, bukan 'lebih baik'. "
             "Contoh: pada % penduduk miskin merah berarti miskin; pada IPM merah berarti IPM tinggi.",
        xref="paper", yref="paper", x=0, y=0, yanchor="top", yshift=-200,
        showarrow=False, font=dict(size=11))])
    return fig


def grafik_korelasi(d):
    c = d[ACTIVE].corr().round(2)
    fig = px.imshow(c.values, x=[LABEL[v] for v in ACTIVE], y=[LABEL[v] for v in ACTIVE],
                    text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r")
    fig.update_layout(title="Matriks korelasi variabel aktif (Sumber: BPS)", template="plotly_white")
    return fig


# ---------- 5. MAIN ----------
def main():
    d = muat_data()
    cek_kualitas(d)
    d = variabel_turunan(d)

    pca, skor, load = jalankan_pca(d, ACTIVE)
    ve = pca.explained_variance_ratio_ * 100
    print("\nVarians (%):", np.round(ve[:4], 1), "| kumulatif 3 PC:", round(ve[:3].sum(), 1))
    print("Eigenvalue:", np.round(ve[:4] / 100 * len(ACTIVE), 2))   # dari matriks korelasi, sama dengan R
    print(load.iloc[:, :3].round(2))

    klaster, km = klasterkan(skor, K)
    out = d.copy()
    out["PC1"], out["PC2"], out["PC3"] = skor[:, 0], skor[:, 1], skor[:, 2]   # PC1: besar = maju
    out["klaster"] = klaster
    out["tipologi"] = out["klaster"].map(NAMA_KLASTER)

    print("\nUkuran klaster:", out["klaster"].value_counts().sort_index().to_dict(), "(R: 5, 24, 7, 2)")
    print(out.groupby("tipologi")[["ipm", "ppm", "internet", "bb_bersih", "pln", "pdrb_juta", "bansos_rasio"]]
          .mean().round(1))
    for k, g in out.groupby("klaster"):
        print(k, NAMA_KLASTER[k], "->", ", ".join(g["prov"]))

    print("\n--- Uji sensitivitas ---")
    uji_sensitivitas(d, skor, klaster)

    out.to_csv(OUT / "provinsi_terolah.csv", index=False)
    load.to_csv(OUT / "pca_loadings.csv")
    d[ACTIVE].corr().round(3).to_csv(OUT / "korelasi.csv")

    grafik_biplot(out, load, ve).write_html(OUT / "biplot.html")
    grafik_paralel(out).write_html(OUT / "paralel.html")
    grafik_heatmap(out).write_html(OUT / "heatmap.html")
    grafik_korelasi(d).write_html(OUT / "korelasi.html")
    print("\nSelesai. File ada di data/processed/")


if __name__ == "__main__":
    main()