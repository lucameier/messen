# app.py
# Streamlit Dashboard: Logger-Messung Batterie – One-Page Übersicht
# Start: streamlit run streamlit_app.py

import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ===================================
# PAGE CONFIG
# ===================================
st.set_page_config(page_title="Batterie Logger Dashboard", layout="wide")

# ===================================
# DATENMODELLE
# ===================================
@dataclass
class TestConfig:
    # Objekt
    fahrzeugtyp: str = ""
    fahrzeugnummer: str = ""
    ort: str = ""
    datum: str = ""
    durchfuehrende_person: str = ""
    auftrag_referenz: str = ""

    # Batteriesystem
    batteriechemie: str = ""
    hersteller_typ: str = ""
    serialnummern: List[str] = field(default_factory=list)

    u_nenn_v: Optional[float] = None
    n_zellen: Optional[int] = None
    n_strang: Optional[int] = None
    c_nenn_ah: Optional[float] = None
    e_nenn_wh: Optional[float] = None

    u_end_v_per_cell: Optional[float] = None
    u_end_v_total: Optional[float] = None
    u_end_quelle: str = ""

    # Messziel / Profil
    messziel: str = "Autonomie"
    testbedingung: str = "Speisung AUS"
    bedienprofil: str = ""
    akzeptanz_kurz: str = ""

    # Ruhestrom
    ruhestrom_typ: str = "Reisezugwagen"
    i_ruhm_max_ma: float = 40.0
    t_ruhm_min: float = 5.0

    # Abbruchkriterien
    t_uv_s: float = 120.0
    t_max_h: float = 24.0

    # Logger
    entladen_negativ: bool = True


# ===================================
# HILFSFUNKTIONEN
# ===================================
def safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return None
        return float(x)
    except:
        return None


def safe_int(x) -> Optional[int]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return None
        return int(float(x))
    except:
        return None


def compute_e_nenn(u: Optional[float], c: Optional[float]) -> Optional[float]:
    if u and c:
        return float(u) * float(c)
    return None


def compute_u_end_total(u_per_cell: Optional[float], n_cells: Optional[int]) -> Optional[float]:
    if u_per_cell and n_cells:
        return float(u_per_cell) * float(n_cells)
    return None


def detect_stability_threshold(t_s, y, threshold, below, window_s) -> Optional[float]:
    """Find first time where condition holds continuously for >= window_s."""
    if len(t_s) < 2:
        return None
    cond = (y <= threshold) if below else (y >= threshold)
    start_idx = None
    acc = 0.0
    for i in range(1, len(t_s)):
        dt = float(t_s[i] - t_s[i - 1])
        if dt < 0:
            dt = 0.0
        if cond[i]:
            if start_idx is None:
                start_idx = i
                acc = 0.0
            acc += dt
            if acc >= window_s:
                return float(t_s[start_idx])
        else:
            start_idx = None
            acc = 0.0
    return None


def integrate_discharge(df, time_col, u_col, i_col, entladen_negativ) -> Tuple[pd.DataFrame, Dict]:
    """Berechne Ladung und Energie."""
    out = df.copy().sort_values(time_col).reset_index(drop=True)
    t = pd.to_datetime(out[time_col], errors="coerce")
    out[time_col] = t
    out = out.dropna(subset=[time_col]).reset_index(drop=True)

    dt_s = out[time_col].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
    out["dt_s"] = dt_s

    I = pd.to_numeric(out[i_col], errors="coerce").fillna(np.nan)
    U = pd.to_numeric(out[u_col], errors="coerce").fillna(np.nan)

    I_entl = np.maximum(-I, 0.0) if entladen_negativ else np.maximum(I, 0.0)
    out["I_entl_A"] = I_entl

    Q_As = (I_entl * out["dt_s"]).fillna(0.0)
    out["Q_Ah_cum"] = Q_As.cumsum() / 3600.0

    P_W = (U * I_entl).fillna(0.0)
    E_Ws = (P_W * out["dt_s"]).fillna(0.0)
    out["E_Wh_cum"] = E_Ws.cumsum() / 3600.0

    metrics = {
        "Q_Ah_total": float(out["Q_Ah_cum"].iloc[-1]) if len(out) else 0.0,
        "E_Wh_total": float(out["E_Wh_cum"].iloc[-1]) if len(out) else 0.0,
    }
    return out, metrics


def to_download_bytes(text: str) -> bytes:
    return text.encode("utf-8")


# ===================================
# STREAMLIT UI - SINGLE PAGE
# ===================================
st.title("📊 Batterie Logger Dashboard")
st.caption("Einfache Konfiguration, CSV-Upload und automatische Auswertung")

# Initialisierung
if "cfg" not in st.session_state:
    st.session_state.cfg = TestConfig(datum=datetime.now().date().isoformat())

cfg: TestConfig = st.session_state.cfg

# ===================================
# SECTION 1: TESTPARAMETER
# ===================================
st.header("1️⃣ Testparameter")

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    cfg.fahrzeugtyp = st.text_input("Fahrzeugtyp", value=cfg.fahrzeugtyp, key="fahrzeugtyp")
with col2:
    cfg.fahrzeugnummer = st.text_input("Fahrzeugnummer", value=cfg.fahrzeugnummer, key="fahrzeugnummer")
with col3:
    cfg.ort = st.text_input("Ort", value=cfg.ort, key="ort")
with col4:
    cfg.datum = st.text_input("Datum", value=cfg.datum, key="datum")
with col5:
    cfg.durchfuehrende_person = st.text_input("Person", value=cfg.durchfuehrende_person, key="person")
with col6:
    cfg.auftrag_referenz = st.text_input("Auftrag/Ref", value=cfg.auftrag_referenz, key="auftrag")

st.markdown("---")

st.subheader("Batteriesystem")
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    cfg.batteriechemie = st.text_input("Chemie (VRLA/NiCd/Li)", value=cfg.batteriechemie, key="chemie")
with col2:
    cfg.hersteller_typ = st.text_input("Hersteller/Typ", value=cfg.hersteller_typ, key="hersteller")
with col3:
    cfg.u_nenn_v = safe_float(st.text_input("U_Nenn (V)", value="" if cfg.u_nenn_v is None else str(cfg.u_nenn_v), key="u_nenn"))
with col4:
    cfg.c_nenn_ah = safe_float(st.text_input("C_Nenn (Ah)", value="" if cfg.c_nenn_ah is None else str(cfg.c_nenn_ah), key="c_nenn"))
with col5:
    cfg.n_zellen = safe_int(st.text_input("n_Zellen", value="" if cfg.n_zellen is None else str(cfg.n_zellen), key="n_zellen"))
with col6:
    cfg.n_strang = safe_int(st.text_input("n_Strang", value="" if cfg.n_strang is None else str(cfg.n_strang), key="n_strang"))

col1, col2, col3 = st.columns(3)
with col1:
    cfg.e_nenn_wh = safe_float(st.text_input("E_Nenn (Wh)", value="" if cfg.e_nenn_wh is None else str(cfg.e_nenn_wh), key="e_nenn"))
with col2:
    cfg.u_end_v_per_cell = safe_float(st.text_input("U_End/Zelle (V)", value="" if cfg.u_end_v_per_cell is None else str(cfg.u_end_v_per_cell), key="u_end_cell"))
with col3:
    cfg.u_end_v_total = safe_float(st.text_input("U_End gesamt (V)", value="" if cfg.u_end_v_total is None else str(cfg.u_end_v_total), key="u_end_total"))

# Auto-Berechnungen
col1, col2 = st.columns(2)
with col1:
    e_est = compute_e_nenn(cfg.u_nenn_v, cfg.c_nenn_ah)
    if e_est and not cfg.e_nenn_wh:
        st.info(f"📊 E_Nenn ≈ {e_est:.1f} Wh (U × C)")
with col2:
    u_est = compute_u_end_total(cfg.u_end_v_per_cell, cfg.n_zellen)
    if u_est and not cfg.u_end_v_total:
        st.info(f"📊 U_End ≈ {u_est:.2f} V (pro Zelle × n)")

st.markdown("---")

st.subheader("Messziele & Abbruchkriterien")
col1, col2 = st.columns(2)
with col1:
    cfg.messziel = st.text_input("Messziel", value=cfg.messziel, key="messziel")
    cfg.testbedingung = st.text_input("Testbedingung", value=cfg.testbedingung, key="testbed")
    cfg.bedienprofil = st.text_area("Bedienprofil", value=cfg.bedienprofil, height=50, key="profil")
    cfg.akzeptanz_kurz = st.text_input("Akzeptanz", value=cfg.akzeptanz_kurz, key="akzept")

with col2:
    cfg.ruhestrom_typ = st.radio("Ruhestrom-Kategorie", 
        ["Reisezugwagen", "Triebzug/Gliederzug", "Projektspezifisch"],
        index=0 if cfg.ruhestrom_typ.startswith("Reisezugwagen") else (1 if cfg.ruhestrom_typ.startswith("Triebzug") else 2),
        key="ruhestrom_cat")
    
    if cfg.ruhestrom_typ.startswith("Reisezugwagen"):
        cfg.i_ruhm_max_ma = 40.0
    elif cfg.ruhestrom_typ.startswith("Triebzug"):
        cfg.i_ruhm_max_ma = 150.0
    else:
        cfg.i_ruhm_max_ma = st.number_input("I_Ruh,max (mA)", value=cfg.i_ruhm_max_ma, key="i_ruh")
    
    cfg.t_ruhm_min = st.number_input("t_Ruh (min)", min_value=0.0, value=cfg.t_ruhm_min, step=0.5, key="t_ruh")
    cfg.t_uv_s = st.number_input("t_UV (s)", min_value=0.0, value=cfg.t_uv_s, step=10.0, key="t_uv")
    cfg.t_max_h = st.number_input("t_max (h)", min_value=0.0, value=cfg.t_max_h, step=1.0, key="t_max")

st.session_state.cfg = cfg

# ===================================
# SECTION 2: CSV UPLOAD & MAPPING
# ===================================
st.header("2️⃣ CSV Upload & Mapping")

col1, col2, col3, col4 = st.columns(4)
with col1:
    sep = st.text_input("Trennzeichen", value=";", key="sep")
with col2:
    decimal = st.text_input("Dezimal", value=",", key="decimal")
with col3:
    encoding = st.text_input("Encoding", value="utf-8", key="encoding")
with col4:
    skiprows = st.number_input("Skip rows", min_value=0, value=0, step=1, key="skiprows")

up = st.file_uploader("📥 CSV hochladen", type=["csv", "txt"], key="csv_upload")

df_raw = None
if up is not None:
    try:
        df_raw = pd.read_csv(up, sep=sep, decimal=decimal, encoding=encoding, skiprows=int(skiprows), engine="python")
        st.success(f"✅ {df_raw.shape[0]} Zeilen, {df_raw.shape[1]} Spalten")
        
        with st.expander("📋 Preview (erste 20 Zeilen)"):
            st.dataframe(df_raw.head(20), use_container_width=True)
        
        # Mapping
        cols = list(df_raw.columns)
        col1, col2, col3 = st.columns(3)
        with col1:
            time_col = st.selectbox("📅 Zeit", cols, index=0, key="time_col")
        with col2:
            u_col = st.selectbox("🔴 U_Bat", cols, index=cols.index("U_Bat") if "U_Bat" in cols else 0, key="u_col")
        with col3:
            i_col = st.selectbox("🟠 I_Bat", cols, index=cols.index("I_Bat") if "I_Bat" in cols else 0, key="i_col")
        
        col1, col2 = st.columns(2)
        with col1:
            i_vi_cols = st.multiselect("Verbraucher I_Vi", 
                [c for c in cols if c not in [time_col, u_col, i_col]], 
                default=[c for c in cols if str(c).startswith("I_") and c not in [i_col]][:3],
                key="i_vi")
        with col2:
            t_cols = st.multiselect("Temperaturen", 
                [c for c in cols if c not in [time_col, u_col, i_col]], 
                default=[c for c in cols if "temp" in str(c).lower()][:1],
                key="t_cols")
        
        # Skalierung
        scale_map = {}
        col1, col2, col3 = st.columns(3)
        with col1:
            scale_map[u_col] = st.number_input(f"Faktor {u_col}", value=1.0, step=0.1, key=f"scale_{u_col}")
        with col2:
            scale_map[i_col] = st.number_input(f"Faktor {i_col}", value=1.0, step=0.1, key=f"scale_{i_col}")
        with col3:
            for c in i_vi_cols[:1]:
                scale_map[c] = st.number_input(f"Faktor {c}", value=1.0, step=0.1, key=f"scale_{c}")
        
        st.session_state.df_raw = df_raw
        st.session_state.mapping = {
            "time_col": time_col, "u_col": u_col, "i_col": i_col,
            "i_vi_cols": i_vi_cols, "t_cols": t_cols, "scale_map": scale_map
        }
        st.info("✅ Mapping bereit")
        
    except Exception as e:
        st.error(f"❌ Fehler: {e}")

# ===================================
# SECTION 3: ANALYSE
# ===================================
st.header("3️⃣ Analyse & Ergebnisse")

if "df_raw" in st.session_state and "mapping" in st.session_state:
    cfg = st.session_state.cfg
    df_raw = st.session_state.df_raw
    mp = st.session_state.mapping

    df = df_raw.copy()
    for col, fac in mp["scale_map"].items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * fac

    try:
        df_int, metrics = integrate_discharge(df, mp["time_col"], mp["u_col"], mp["i_col"], cfg.entladen_negativ)
        
        t0 = df_int[mp["time_col"]].iloc[0]
        df_int["t_s"] = (df_int[mp["time_col"]] - t0).dt.total_seconds()

        u_end = cfg.u_end_v_total if cfg.u_end_v_total else compute_u_end_total(cfg.u_end_v_per_cell, cfg.n_zellen)

        i_ruhm_a = float(cfg.i_ruhm_max_ma) / 1000.0
        t_ruhm_s = float(cfg.t_ruhm_min) * 60.0
        t_uv_s = float(cfg.t_uv_s)
        t_max_s = float(cfg.t_max_h) * 3600.0

        Ibat = pd.to_numeric(df_int[mp["i_col"]], errors="coerce").to_numpy()
        Ubat = pd.to_numeric(df_int[mp["u_col"]], errors="coerce").to_numpy()
        ts = df_int["t_s"].to_numpy()

        t_ruhm_hit = detect_stability_threshold(ts, np.abs(Ibat), i_ruhm_a, below=True, window_s=t_ruhm_s)
        t_uv_hit = detect_stability_threshold(ts, Ubat, float(u_end), below=True, window_s=t_uv_s) if u_end else None
        t_max_hit = t_max_s

        candidates = []
        if t_ruhm_hit is not None:
            candidates.append(("Ruhestrom", t_ruhm_hit))
        if t_uv_hit is not None:
            candidates.append(("Unterspannung", t_uv_hit))
        candidates.append(("Zeit", t_max_hit))

        criterion, t_end_s = min(candidates, key=lambda x: x[1])
        t_end = t0 + timedelta(seconds=float(t_end_s))

        df_end = df_int[df_int["t_s"] <= t_end_s].copy()
        if len(df_end) == 0:
            df_end = df_int.copy()

        u_min = float(pd.to_numeric(df_end[mp["u_col"]], errors="coerce").min())
        q_total = float(df_end["Q_Ah_cum"].iloc[-1])
        e_total = float(df_end["E_Wh_cum"].iloc[-1])

        q_pct = 100.0 * q_total / float(cfg.c_nenn_ah) if cfg.c_nenn_ah else None
        e_pct = 100.0 * e_total / float(cfg.e_nenn_wh) if cfg.e_nenn_wh else None

        # Resultate
        st.subheader("Resultate")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Kriterium", criterion)
        with col2:
            st.metric("Testende", str(t_end)[-8:])
        with col3:
            st.metric("Q (Ah)", f"{q_total:.3f}")
        with col4:
            st.metric("E (Wh)", f"{e_total:.1f}")
        with col5:
            st.metric("U_min (V)", f"{u_min:.2f}")
        with col6:
            st.metric("Q_%", "-" if q_pct is None else f"{q_pct:.1f}%")

        st.markdown("---")

        # Plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df_int[mp["time_col"]], y=df_int[mp["u_col"]], name="U_Bat (V)", yaxis="y1"))
            fig1.add_trace(go.Scatter(x=df_int[mp["time_col"]], y=df_int[mp["i_col"]], name="I_Bat (A)", yaxis="y2"))
            fig1.update_layout(
                title="U_Bat & I_Bat",
                yaxis=dict(title="U (V)"),
                yaxis2=dict(title="I (A)", overlaying="y", side="right"),
                hovermode="x unified",
                height=400,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            if u_end:
                fig1.add_hline(y=float(u_end), line_dash="dash", annotation_text="U_End")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_int[mp["time_col"]], y=df_int["Q_Ah_cum"], name="Q (Ah)", yaxis="y1"))
            fig2.add_trace(go.Scatter(x=df_int[mp["time_col"]], y=df_int["E_Wh_cum"], name="E (Wh)", yaxis="y2"))
            fig2.update_layout(
                title="Ladung & Energie",
                yaxis=dict(title="Ah"),
                yaxis2=dict(title="Wh", overlaying="y", side="right"),
                hovermode="x unified",
                height=400,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # Abbruchkriterien
        st.subheader("Abbruchkriterien-Detektion")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Ruhestrom:** {cfg.i_ruhm_max_ma:.1f} mA / {cfg.t_ruhm_min:.1f} min")
            st.write(f"→ {('✅ ' + str(t0 + timedelta(seconds=float(t_ruhm_hit)))[-8:]) if t_ruhm_hit else '❌ Nicht erreicht'}")
        with col2:
            st.write(f"**Unterspannung:** {u_end if u_end else '–'} V / {cfg.t_uv_s:.0f} s")
            st.write(f"→ {('✅ ' + str(t0 + timedelta(seconds=float(t_uv_hit)))[-8:]) if t_uv_hit else '❌ Nicht erreicht'}")
        with col3:
            st.write(f"**Zeit:** {cfg.t_max_h:.1f} h")
            st.write(f"→ {str(t0 + timedelta(seconds=float(t_max_hit)))[-8:]}")

        st.session_state.df_int = df_int
        st.session_state.df_end = df_end
        st.session_state.analysis_summary = {
            "t0": str(t0), "t_end": str(t_end), "criterion": criterion,
            "Q_Ah": q_total, "E_Wh": e_total, "U_min": u_min,
            "Q_pct": q_pct, "E_pct": e_pct
        }

    except Exception as e:
        st.error(f"❌ Fehler: {e}")

else:
    st.info("ℹ️ CSV hochladen um Analyse zu starten")

# ===================================
# SECTION 4: EXPORT
# ===================================
st.header("4️⃣ Export & Download")

col1, col2, col3 = st.columns(3)

with col1:
    cfg_json = json.dumps(asdict(cfg), ensure_ascii=False, indent=2)
    st.download_button("📥 Konfiguration JSON", data=to_download_bytes(cfg_json), file_name="config.json", mime="application/json")

if "analysis_summary" in st.session_state:
    with col2:
        summary_json = json.dumps(st.session_state.analysis_summary, ensure_ascii=False, indent=2)
        st.download_button("📥 Ergebnisse JSON", data=to_download_bytes(summary_json), file_name="resultate.json", mime="application/json")

    if "df_int" in st.session_state:
        with col3:
            buf = io.StringIO()
            st.session_state.df_int.to_csv(buf, index=False)
            st.download_button("📥 Auswertungs-CSV", data=to_download_bytes(buf.getvalue()), file_name="auswertung.csv", mime="text/csv")

# ===================================
# SECTION 5: HINWEISE
# ===================================
st.header("5️⃣ Messaufbau-Hinweise & Sicherheit")

st.markdown("""
### 🔧 Messpunkte
- **U_Bat:** Direkt an Batterieklemmen, separate abgesicherte Messleitung  
- **I_Bat:** Am Sammelleiter oder je Strang  
- **I_Vi:** Am Versorgungsleiter einzelner Verbraucher  

### ⚠️ Sicherheit (VERBINDLICH)
- Messleitungen mechanisch sichern und **IN BATTERIENÄHE ABSICHERN**
- Nullabgleich durchführen und dokumentieren
- Offset-/Driftprüfung vor und nach Messung
- Keine Veränderungen an Schutzfunktionen ohne Freigabe
- Messung sofort abbrechen bei: Erwärmung, Rauch, Geräusche, beschädigte Leitungen

### ✅ Vorbereitung vor Messung
1. Konfigurationsblatt ausfüllen
2. Fahrzeug sichern (Stillstand)
3. Messaufbau installieren + mechanisch sichern
4. Nullabgleich durchführen + dokumentieren
5. Messwerte plausibilisieren
6. Umgebungstemperatur + Startzustand dokumentieren
7. Batterie voll geladen (5h), 10-15 min Stabilisierung

### 📝 Während Messung
- Abschaltung Verbraucher mit Uhrzeit protokollieren
- Fahrzeugzustand-Wechsel dokumentieren
- Bedienhandlungen markieren
- Ungeplante Einflüsse notieren

### 📊 Logger-Konfiguration
- **Zeitbasis:** Synchronisieren, Zeitzone festlegen
- **Abtastrate:** 1 Hz (Default)
- **Datenformat:** CSV mit Zeitstempel + Kanalwerte
- **Marker:** Ereignis-Protokollierung dokumentieren

### 📋 Abbruchkriterien (automatisch erkannt)
| Kriterium | Bedingung | Default |
|-----------|-----------|---------|
| Ruhestrom | \|I_Bat\| < I_Ruh,max über t_Ruh | 40–150 mA, 5 min |
| Unterspannung | U_Bat ≤ U_End über t_UV | 120 s |
| Zeit | t - t0 ≥ t_max | 24 h |
""")
