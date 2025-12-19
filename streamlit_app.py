# app.py
# Streamlit Dashboard: Logger-Messung Batterie (Konfiguration + CSV-Upload + Auswertung)
# Start: streamlit run streamlit_app_new.py

import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Seitenkonfiguration
st.set_page_config(
    page_title="Batterie Logger Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d1e7dd;
        padding: 12px;
        border-radius: 6px;
        border-left: 4px solid #198754;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 12px;
        border-radius: 6px;
        border-left: 4px solid #ffc107;
    }
    .info-box {
        background-color: #cfe2ff;
        padding: 12px;
        border-radius: 6px;
        border-left: 4px solid #0d6efd;
    }
</style>
""", unsafe_allow_html=True)


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
    messziel: str = "Autonomie / Abschaltverhalten / Nachlauf / Ruhestrom / Energiebedarf"
    testbedingung: str = "Speisung AUS ab t0"
    bedienprofil: str = ""
    soll_nachlauf_min: Optional[float] = None
    mindest_autonomie_erstes_event_min: Optional[float] = None
    mindest_autonomie_bis_testende_min: Optional[float] = None
    max_entnommene_energie_wh: Optional[float] = None
    akzeptanz_kurz: str = ""

    # Ruhestrom
    ruhestrom_typ: str = "Reisezugwagen (Default gemäss BCA)"
    i_ruhm_max_ma: float = 40.0
    t_ruhm_min: float = 5.0

    # Abbruchkriterien / Fenster
    t_uv_s: float = 120.0
    t_max_h: float = 24.0

    # Logger / Konvention
    zeitzone: str = "Lokalzeit"
    zeitformat: str = "ISO 8601"
    abtastrate_hz: Optional[float] = 1.0

    # Vorzeichenkonvention
    entladen_negativ: bool = True


# ===================================
# HILFSFUNKTIONEN
# ===================================
def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(float(x))
    except Exception:
        return None


def compute_e_nenn(u_nenn_v: Optional[float], c_nenn_ah: Optional[float]) -> Optional[float]:
    if u_nenn_v is None or c_nenn_ah is None:
        return None
    return float(u_nenn_v) * float(c_nenn_ah)


def compute_u_end_total(u_end_v_per_cell: Optional[float], n_zellen: Optional[int]) -> Optional[float]:
    if u_end_v_per_cell is None or n_zellen is None:
        return None
    return float(u_end_v_per_cell) * float(n_zellen)


def detect_stability_threshold(
    t_s: np.ndarray,
    y: np.ndarray,
    threshold: float,
    below: bool,
    window_s: float,
) -> Optional[float]:
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


def integrate_discharge(
    df: pd.DataFrame,
    time_col: str,
    u_col: str,
    i_col: str,
    entladen_negativ: bool,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Berechnet Ladung, Energie und kumulative Werte."""
    out = df.copy()
    out = out.sort_values(time_col).reset_index(drop=True)

    t = pd.to_datetime(out[time_col], errors="coerce")
    out[time_col] = t
    out = out.dropna(subset=[time_col]).reset_index(drop=True)

    dt_s = out[time_col].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
    out["dt_s"] = dt_s

    I = pd.to_numeric(out[i_col], errors="coerce").fillna(np.nan)
    U = pd.to_numeric(out[u_col], errors="coerce").fillna(np.nan)

    if entladen_negativ:
        I_entl = np.maximum(-I, 0.0)
    else:
        I_entl = np.maximum(I, 0.0)

    out["I_entl_A"] = I_entl

    Q_As = (I_entl * out["dt_s"]).fillna(0.0)
    out["Q_Ah_cum"] = Q_As.cumsum() / 3600.0

    P_W = (U * I_entl).fillna(0.0)
    E_Ws = (P_W * out["dt_s"]).fillna(0.0)
    out["E_Wh_cum"] = E_Ws.cumsum() / 3600.0

    Qnet_As = (I.fillna(0.0) * out["dt_s"]).fillna(0.0)
    out["Qnet_Ah_cum"] = Qnet_As.cumsum() / 3600.0

    Pnet_W = (U.fillna(0.0) * I.fillna(0.0)).fillna(0.0)
    Enet_Ws = (Pnet_W * out["dt_s"]).fillna(0.0)
    out["Enet_Wh_cum"] = Enet_Ws.cumsum() / 3600.0

    metrics = {
        "Q_Ah_total": float(out["Q_Ah_cum"].iloc[-1]) if len(out) else 0.0,
        "E_Wh_total": float(out["E_Wh_cum"].iloc[-1]) if len(out) else 0.0,
        "Qnet_Ah_total": float(out["Qnet_Ah_cum"].iloc[-1]) if len(out) else 0.0,
        "Enet_Wh_total": float(out["Enet_Wh_cum"].iloc[-1]) if len(out) else 0.0,
    }
    return out, metrics


def plot_timeseries(
    df: pd.DataFrame,
    time_col: str,
    series: List[Tuple[str, str]],
    title: str,
) -> go.Figure:
    fig = go.Figure()
    for col, name in series:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[time_col], y=df[col], mode="lines", name=name))
    fig.update_layout(
        title=title,
        xaxis_title="Zeit",
        yaxis_title="Wert",
        hovermode="x unified",
        legend_title="Signal",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def to_download_bytes(text: str) -> bytes:
    return text.encode("utf-8")


# ===================================
# MESSAUFBAU-INFORMATIONEN
# ===================================
MESSAUFBAU_SECTIONS = {
    "messpunkte": {
        "titel": "🔴 Messpunkte",
        "inhalt": """
#### U_Bat – Batteriespannung
- **Messung:** Direkt an Batterieklemmen (Plus/Minus)
- **Leitungsausführung:** Separate, abgesicherte Messleitung
- **Sicherung:** In Batterienähe absichern, berührungssicher abgedeckt

#### I_Bat – Batteriestrom
- **Messung:** Am Sammelleiter (empfohlen Minusleiter) ODER je Strang
- **Messkonzept:** Gemäss Konfigurationsblatt
- **Strommessmittel:** DC-Stromzange oder Shunt
- **Dokumentation:** Nullabgleich + Pfeil-/Polaritätsrichtung

#### I_Vi – Verbraucherströme  
- **Messung:** Am Versorgungsleiter des Abgangs (einzeln)
- **Ausführung:** Nicht gemeinsam mit Rückleiter messen
- **Auflösung:** Ausreichend für mA-Bereich (falls erforderlich)
        """
    },
    "kanalzuordnung": {
        "titel": "📊 Kanalzuordnung (Logger)",
        "inhalt": """
| Kanal | Signal | Bemerkung |
|-------|--------|-----------|
| **CH1** | **U_Bat** | Batterieklemmen (separate Messleitung) |
| **CH2** | **I_Bat** | Gesamtstrom oder Summe Stränge |
| **CH3–n** | **I_Vi** | Verbraucher gemäss Konfigurationsblatt |
| **CHm** | **Marker** | Ereignismarker (falls verfügbar) |
| **CHx** | **T_Bat** | Batterietemperatur (optional) |
| **CHy** | **T_TS** | Temperatur Ladegerät-TS (optional) |
        """
    },
    "vorbereitung": {
        "titel": "✅ Vorbereitung & Installation",
        "inhalt": """
#### Vor Messbeginn:
1. ☐ Konfigurationsblatt vollständig ausfüllen
2. ☐ Fahrzeug sichern (Stillstand)
3. ☐ Messaufbau installieren und mechanisch sichern
4. ☐ Absicherung Messleitungen prüfen
5. ☐ Nullabgleich/Offset durchführen und dokumentieren
6. ☐ Messwerte plausibilisieren (Spannung, Strom, Kanäle)
7. ☐ Umgebungstemperatur und Startzustand dokumentieren

#### Batterie-Startzustand:
- Typisch: **Voll geladen** (fahrzeugeigenes Ladegerät)
- Ladezeit: Default **5 h**
- Stabilisierung: Default **10–15 min**
- Ladesystemstatus dokumentieren
        """
    },
    "sicherheit": {
        "titel": "⚠️ Sicherheit",
        "inhalt": """
#### 🔴 Grundsatz:
Arbeiten an elektrischen Anlagen **nur durch qualifiziertes Personal** gemäss geltenden Vorschriften!

#### Spannungsmessung:
- Messleitungen **mechanisch sichern** (Zugentlastung, Scheuerschutz)
- **In Batterienähe absichern** ← verbindlich!
- Absicherung muss für Systemspannung geeignet sein

#### Strommessung:
- DC-Stromzangen/Shunts mechanisch sichern
- Nullabgleich durchführen und dokumentieren
- Pfeil-/Polaritätsrichtung dokumentieren

#### Ruhestrommessung:
- Messverfahren mit ausreichender Auflösung/Genauigkeit
- Offset-/Driftprüfung vor und nach Messung

#### 🔴 Verbote:
- ❌ Keine Veränderungen an Schutzfunktionen ohne Freigabe
- ❌ Fahrzeugzugang regeln; Manipulationen protokollieren
- ❌ Messung sofort abbrechen bei: Erwärmung, Rauch, Geräusche, beschädigte Leitungen
        """
    },
    "logger_config": {
        "titel": "⚙️ Logger-Konfiguration",
        "inhalt": """
#### Zeitbasis:
- Logger-Uhr synchronisieren
- Zeitzone: Default Lokalzeit
- Zeitformat: ISO 8601

#### Datenformat:
- Abtastrate: **1 Hz** (Default)
- CSV-Export mit Zeitstempel + Kanalwerte + Einheiten
- Zeitstempel-Auflösung: **1 s**

#### Konventionen:
- Vorzeichen: Laden I > 0, **Entladen I < 0** (Default)
- Marker: Digitaler Marker und/oder manuelles Protokoll

#### Mindestanforderungen Messkette:
- Spannungsgenauigkeit: **≤ 1%** oder besser
- Stromgenauigkeit I_Bat: **≤ 1%** oder besser
- Auflösung Ruhestrom: **≤ 1 mA** oder besser
        """
    },
    "messphase": {
        "titel": "📝 Während der Messung",
        "inhalt": """
#### Ereignis-Protokollierung (verbindlich):
Mit Uhrzeit (hh:mm:ss) und Beschreibung dokumentieren:

- ☐ Abschaltung/Shutdown Verbraucher
- ☐ Fahrzeugzustand-Wechsel (Parkstellung, Besetzung)
- ☐ Bedienhandlungen (Türbetätigungen, etc.)
- ☐ Ungeplante Einflüsse (Personalzugang, Störungen)

#### Soll-Nachlaufzeit:
Falls definiert: Nachweis über Loggerdaten + Ereignismarker

#### Messdauer:
Kann bis t_max verlängert werden, um Nachlauf- und Abschaltkaskaden zu beobachten
        """
    }
}


# Abbruchkriterien-Defaults
ABBRUCHKRITERIEN_DF = pd.DataFrame({
    "Kriterium": ["Ruhestrom", "Unterspannung", "Maximale Zeit"],
    "Bedingung": [
        "|I_Bat| < I_Ruh,max über t_Ruh",
        "U_Bat ≤ U_End über t_UV",
        "t - t0 ≥ t_max"
    ],
    "Default": [
        "40–150 mA, 5 min",
        "Projektabhängig, 120 s",
        "24 h"
    ]
})

RUHESTROM_DEFAULTS = {
    "Reisezugwagen": {"I_max_mA": 40.0, "quelle": "BCA 20002483"},
    "Triebzug/Gliederzug": {"I_max_mA": 150.0, "quelle": "BCA 20002483"},
}

MINDESTANF_DF = pd.DataFrame({
    "Parameter": [
        "Spannungsgenauigkeit U_Bat",
        "Stromgenauigkeit I_Bat",
        "Auflösung Ruhestrom",
        "Zeitstempel-Auflösung"
    ],
    "Vorgabe": ["≤ 1%", "≤ 1%", "≤ 1 mA", "1 s"]
})


# ===================================
# STREAMLIT UI - HAUPTSEITE
# ===================================
st.title("📊 Batterie Logger Dashboard")
st.caption("""
Professionelle Konfiguration, Analyse und Auswertung von Logger-Messungen an Batteriesystemen
""")

# Initialisierung
if "cfg" not in st.session_state:
    st.session_state.cfg = TestConfig(datum=datetime.now().date().isoformat())

# Navigation
tabs = st.tabs([
    "1️⃣ Testparameter",
    "2️⃣ CSV Upload",
    "3️⃣ Messaufbau-Hinweise",
    "4️⃣ Analyse",
    "5️⃣ Export"
])


# ===================================
# TAB 1: TESTPARAMETER
# ===================================
with tabs[0]:
    st.header("📋 Testparameter konfigurieren")
    
    cfg: TestConfig = st.session_state.cfg
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Objekt")
        cfg.fahrzeugtyp = st.text_input("Fahrzeugtyp", value=cfg.fahrzeugtyp)
        cfg.fahrzeugnummer = st.text_input("Fahrzeugnummer", value=cfg.fahrzeugnummer)
        cfg.ort = st.text_input("Ort", value=cfg.ort)
        cfg.datum = st.text_input("Datum (YYYY-MM-DD)", value=cfg.datum)
        cfg.durchfuehrende_person = st.text_input("Durchführende Person", value=cfg.durchfuehrende_person)
        cfg.auftrag_referenz = st.text_input("Auftrag/Referenz", value=cfg.auftrag_referenz)

        st.subheader("Messziel & Profil")
        with st.expander("Messziel-Details", expanded=False):
            cfg.messziel = st.text_input("Messziel", value=cfg.messziel)
            cfg.testbedingung = st.text_input("Testbedingung", value=cfg.testbedingung)
            cfg.bedienprofil = st.text_area("Bedienprofil während Messung", value=cfg.bedienprofil, height=70)

        c1, c2, c3 = st.columns(3)
        with c1:
            cfg.soll_nachlauf_min = safe_float(st.text_input("Soll-Nachlauf (min)", value="" if cfg.soll_nachlauf_min is None else str(cfg.soll_nachlauf_min)))
        with c2:
            cfg.mindest_autonomie_erstes_event_min = safe_float(st.text_input("Mindest-Autonomie bis Event (min)", value="" if cfg.mindest_autonomie_erstes_event_min is None else str(cfg.mindest_autonomie_erstes_event_min)))
        with c3:
            cfg.mindest_autonomie_bis_testende_min = safe_float(st.text_input("Mindest-Autonomie Testende (min)", value="" if cfg.mindest_autonomie_bis_testende_min is None else str(cfg.mindest_autonomie_bis_testende_min)))

        c4, c5 = st.columns(2)
        with c4:
            cfg.max_entnommene_energie_wh = safe_float(st.text_input("Max. Energie (Wh)", value="" if cfg.max_entnommene_energie_wh is None else str(cfg.max_entnommene_energie_wh)))
        with c5:
            cfg.akzeptanz_kurz = st.text_input("Akzeptanz/Pass-Fail", value=cfg.akzeptanz_kurz)

    with colB:
        st.subheader("Batteriesystem")
        cfg.batteriechemie = st.text_input("Batteriechemie (z.B. VRLA, NiCd, Li-Ion)", value=cfg.batteriechemie)
        cfg.hersteller_typ = st.text_input("Hersteller / Typbezeichnung", value=cfg.hersteller_typ)

        sn_text = st.text_area(
            "Serialnummern (eine pro Zeile)",
            value="\n".join(cfg.serialnummern) if cfg.serialnummern else "",
            height=70,
        )
        cfg.serialnummern = [s.strip() for s in sn_text.splitlines() if s.strip()]

        c1, c2, c3 = st.columns(3)
        with c1:
            cfg.u_nenn_v = safe_float(st.text_input("U_Nenn (V)", value="" if cfg.u_nenn_v is None else str(cfg.u_nenn_v)))
        with c2:
            cfg.n_zellen = safe_int(st.text_input("Zellen", value="" if cfg.n_zellen is None else str(cfg.n_zellen)))
        with c3:
            cfg.n_strang = safe_int(st.text_input("Stränge", value="" if cfg.n_strang is None else str(cfg.n_strang)))

        c4, c5 = st.columns(2)
        with c4:
            cfg.c_nenn_ah = safe_float(st.text_input("C_Nenn (Ah)", value="" if cfg.c_nenn_ah is None else str(cfg.c_nenn_ah)))
        with c5:
            cfg.e_nenn_wh = safe_float(st.text_input("E_Nenn (Wh)", value="" if cfg.e_nenn_wh is None else str(cfg.e_nenn_wh)))

        if cfg.e_nenn_wh is None:
            est = compute_e_nenn(cfg.u_nenn_v, cfg.c_nenn_ah)
            if est is not None:
                st.info(f"**Orientierung:** E_Nenn ≈ {est:.1f} Wh (U_Nenn × C_Nenn)")

        st.markdown("---")
        st.subheader("Abbruchkriterien")
        
        with st.expander("U_End und Ruhestrom-Defaults"):
            c6, c7 = st.columns(2)
            with c6:
                cfg.u_end_v_per_cell = safe_float(st.text_input("U_End je Zelle (V)", value="" if cfg.u_end_v_per_cell is None else str(cfg.u_end_v_per_cell)))
            with c7:
                cfg.u_end_v_total = safe_float(st.text_input("U_End gesamt (V)", value="" if cfg.u_end_v_total is None else str(cfg.u_end_v_total)))

            if cfg.u_end_v_total is None:
                est_u_end = compute_u_end_total(cfg.u_end_v_per_cell, cfg.n_zellen)
                if est_u_end is not None:
                    st.info(f"**Orientierung:** U_End ≈ {est_u_end:.2f} V")

            cfg.u_end_quelle = st.text_input("Quelle U_End (Hersteller/Norm)", value=cfg.u_end_quelle)

            st.markdown("---")
            ruh_options = ["Reisezugwagen (Default gemäss BCA)", "Triebzug / Gliederzug (Default gemäss BCA)", "Projektspezifisch"]
            cfg.ruhestrom_typ = st.selectbox("Ruhestrom-Kategorie", ruh_options, index=ruh_options.index(cfg.ruhestrom_typ) if cfg.ruhestrom_typ in ruh_options else 0)

            if cfg.ruhestrom_typ.startswith("Reisezugwagen"):
                cfg.i_ruhm_max_ma = 40.0
            elif cfg.ruhestrom_typ.startswith("Triebzug"):
                cfg.i_ruhm_max_ma = 150.0
            else:
                cfg.i_ruhm_max_ma = float(st.number_input("I_Ruh,max (mA)", min_value=0.0, value=float(cfg.i_ruhm_max_ma), step=1.0))

            c8, c9, c10 = st.columns(3)
            with c8:
                cfg.t_ruhm_min = float(st.number_input("t_Ruh (min)", min_value=0.0, value=float(cfg.t_ruhm_min), step=0.5))
            with c9:
                cfg.t_uv_s = float(st.number_input("t_UV (s)", min_value=0.0, value=float(cfg.t_uv_s), step=10.0))
            with c10:
                cfg.t_max_h = float(st.number_input("t_max (h)", min_value=0.0, value=float(cfg.t_max_h), step=1.0))

        st.markdown("---")
        st.subheader("Logger-Konvention")
        with st.expander("Logger-Details"):
            cfg.entladen_negativ = st.toggle("Entladen ist negativ (Default)", value=cfg.entladen_negativ)
            cfg.abtastrate_hz = safe_float(st.text_input("Abtastrate (Hz)", value="" if cfg.abtastrate_hz is None else str(cfg.abtastrate_hz)))
            cfg.zeitzone = st.text_input("Zeitzone", value=cfg.zeitzone)
            cfg.zeitformat = st.text_input("Zeitformat", value=cfg.zeitformat)

    st.session_state.cfg = cfg

    # Zusammenfassung
    with st.expander("📌 Konfiguration Zusammenfassung"):
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric("Fahrzeugtyp", cfg.fahrzeugtyp if cfg.fahrzeugtyp else "–")
        with summary_cols[1]:
            st.metric("Batterie", cfg.hersteller_typ if cfg.hersteller_typ else "–")
        with summary_cols[2]:
            st.metric("Nennspannung", f"{cfg.u_nenn_v} V" if cfg.u_nenn_v else "–")


# ===================================
# TAB 2: CSV UPLOAD & MAPPING
# ===================================
with tabs[1]:
    st.header("📥 CSV Upload & Spalten-Mapping")

    st.subheader("CSV Datei hochladen")
    
    col_upload, col_settings = st.columns([2, 1])
    
    with col_upload:
        up = st.file_uploader("Logger CSV", type=["csv", "txt"])
    
    with col_settings:
        st.caption("CSV-Einstellungen")
        sep = st.text_input("Trennzeichen", value=";")
        decimal = st.text_input("Dezimaltrennzeichen", value=",")
        encoding = st.text_input("Encoding", value="utf-8")
        skiprows = st.number_input("Skip rows", min_value=0, value=0, step=1)

    df_raw = None
    if up is not None:
        try:
            df_raw = pd.read_csv(
                up,
                sep=sep,
                decimal=decimal,
                encoding=encoding,
                skiprows=int(skiprows),
                engine="python",
            )
            st.success(f"✅ CSV geladen: **{df_raw.shape[0]}** Zeilen, **{df_raw.shape[1]}** Spalten")
            with st.expander("Preview (erste 30 Zeilen)", expanded=False):
                st.dataframe(df_raw.head(30), use_container_width=True)
        except Exception as e:
            st.error(f"❌ CSV-Fehler: {e}")

    if df_raw is not None and len(df_raw.columns) > 0:
        st.markdown("---")
        st.subheader("Spalten-Mapping")
        
        cols = list(df_raw.columns)

        c1, c2, c3 = st.columns(3)
        with c1:
            time_col = st.selectbox("📅 Zeitspalte", cols, index=0)
        with c2:
            u_col = st.selectbox("🔴 U_Bat (V)", cols, index=cols.index("U_Bat") if "U_Bat" in cols else 0)
        with c3:
            i_col = st.selectbox("🟠 I_Bat (A)", cols, index=cols.index("I_Bat") if "I_Bat" in cols else 0)

        st.caption("Optional: Verbraucherströme und Temperaturen")
        i_vi_cols = st.multiselect(
            "Verbraucher-/Abgangsströme (I_Vi)",
            options=[c for c in cols if c not in [time_col, u_col, i_col]],
            default=[c for c in cols if str(c).startswith("I_") and c not in [i_col]][:6],
        )

        t_cols = st.multiselect(
            "Temperaturen (T_Bat, T_TS)",
            options=[c for c in cols if c not in [time_col, u_col, i_col]],
            default=[c for c in cols if "temp" in str(c).lower() or str(c).lower().startswith("t_")][:2],
        )

        st.markdown("---")
        st.subheader("Skalierungsfaktoren (optional)")
        st.caption("Falls Logger z.B. mA loggt: Faktor 0.001 setzen")

        scale_map = {}
        with st.expander("Skalierungsfaktoren pro Spalte"):
            scale_map[u_col] = float(st.number_input(f"Faktor {u_col} (V)", value=1.0, step=0.1))
            scale_map[i_col] = float(st.number_input(f"Faktor {i_col} (A)", value=1.0, step=0.1))
            for c in i_vi_cols:
                scale_map[c] = float(st.number_input(f"Faktor {c} (A)", value=1.0, step=0.1))
            for c in t_cols:
                scale_map[c] = float(st.number_input(f"Faktor {c} (°C)", value=1.0, step=0.1))

        # Persist
        st.session_state.df_raw = df_raw
        st.session_state.mapping = {
            "time_col": time_col,
            "u_col": u_col,
            "i_col": i_col,
            "i_vi_cols": i_vi_cols,
            "t_cols": t_cols,
            "scale_map": scale_map,
        }
        st.info("✅ Mapping konfiguriert – bereit für Analyse")
    else:
        st.info("ℹ️ CSV hochladen um Spalten zu mappen")


# ===================================
# TAB 3: MESSAUFBAU-HINWEISE
# ===================================
with tabs[2]:
    st.header("🔧 Messaufbau & Anleitung")
    
    st.write("""
Diese Seite enthält detaillierte Hinweise zum professionellen Aufbau der Messkette,
entsprechend den Vorgaben für universelle Logger-Messungen an Batteriesystemen.
    """)

    # Tabs für verschiedene Bereiche
    sub_tabs = st.tabs([
        "Messpunkte",
        "Kanalzuordnung",
        "Vorbereitung",
        "Sicherheit",
        "Logger-Konfiguration",
        "Messphase"
    ])

    section_keys = ["messpunkte", "kanalzuordnung", "vorbereitung", "sicherheit", "logger_config", "messphase"]
    
    for idx, (sub_tab, key) in enumerate(zip(sub_tabs, section_keys)):
        with sub_tab:
            section = MESSAUFBAU_SECTIONS[key]
            st.markdown(f"### {section['titel']}")
            st.markdown(section['inhalt'])

    st.markdown("---")
    st.subheader("📋 Abbruchkriterien-Übersicht")
    st.dataframe(ABBRUCHKRITERIEN_DF, use_container_width=True)

    st.subheader("📊 Mindestanforderungen Messkette")
    st.dataframe(MINDESTANF_DF, use_container_width=True)

    st.subheader("🏷️ Ruhestrom-Defaults (BCA 20002483)")
    for typ, vals in RUHESTROM_DEFAULTS.items():
        st.write(f"**{typ}:** {vals['I_max_mA']} mA, Quelle: {vals['quelle']}")


# ===================================
# TAB 4: ANALYSE & VISUALISIERUNG
# ===================================
with tabs[3]:
    st.header("📈 Analyse & Visualisierung")

    if "df_raw" not in st.session_state or "mapping" not in st.session_state:
        st.warning("⚠️ Bitte zuerst CSV uploaden und Spalten mappen (Tab 2)")
    else:
        cfg: TestConfig = st.session_state.cfg
        df_raw: pd.DataFrame = st.session_state.df_raw
        mp: Dict = st.session_state.mapping

        # Prepare dataframe
        df = df_raw.copy()
        for col, fac in mp["scale_map"].items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * fac

        time_col = mp["time_col"]
        u_col = mp["u_col"]
        i_col = mp["i_col"]
        i_vi_cols = mp["i_vi_cols"]
        t_cols = mp["t_cols"]

        # Integration
        try:
            df_int, metrics = integrate_discharge(df, time_col, u_col, i_col, cfg.entladen_negativ)
        except Exception as e:
            st.error(f"❌ Auswertung fehlgeschlagen: {e}")
            st.stop()

        # Time since t0
        t0 = df_int[time_col].iloc[0]
        df_int["t_s"] = (df_int[time_col] - t0).dt.total_seconds()

        # U_End
        u_end = cfg.u_end_v_total
        if u_end is None:
            u_end = compute_u_end_total(cfg.u_end_v_per_cell, cfg.n_zellen)

        # I_Ruh & Fenster
        i_ruhm_a = float(cfg.i_ruhm_max_ma) / 1000.0
        t_ruhm_s = float(cfg.t_ruhm_min) * 60.0
        t_uv_s = float(cfg.t_uv_s)
        t_max_s = float(cfg.t_max_h) * 3600.0

        # Detect
        Ibat = pd.to_numeric(df_int[i_col], errors="coerce").fillna(np.nan).to_numpy()
        Ubat = pd.to_numeric(df_int[u_col], errors="coerce").fillna(np.nan).to_numpy()
        ts = df_int["t_s"].to_numpy()

        t_ruhm_hit = detect_stability_threshold(ts, np.abs(Ibat), i_ruhm_a, below=True, window_s=t_ruhm_s)
        t_uv_hit = None
        if u_end is not None:
            t_uv_hit = detect_stability_threshold(ts, Ubat, float(u_end), below=True, window_s=t_uv_s)

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

        # Summary
        u_min = float(pd.to_numeric(df_end[u_col], errors="coerce").min())
        idx_u_min = int(pd.to_numeric(df_end[u_col], errors="coerce").idxmin())
        u_min_time = df_int.loc[idx_u_min, time_col] if idx_u_min in df_int.index else None

        q_total = float(df_end["Q_Ah_cum"].iloc[-1])
        e_total = float(df_end["E_Wh_cum"].iloc[-1])

        q_pct = None
        e_pct = None
        if cfg.c_nenn_ah is not None and cfg.c_nenn_ah != 0:
            q_pct = 100.0 * q_total / float(cfg.c_nenn_ah)
        if cfg.e_nenn_wh is not None and cfg.e_nenn_wh != 0:
            e_pct = 100.0 * e_total / float(cfg.e_nenn_wh)

        st.markdown("### 📌 Resultate (bis Testende)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Abbruchkriterium", criterion)
        with c2:
            st.metric("Testende", str(t_end)[-8:])  # nur HH:MM:SS
        with c3:
            st.metric("Ladung (Ah)", f"{q_total:.3f}")
        with c4:
            st.metric("Energie (Wh)", f"{e_total:.1f}")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("U_Bat,min (V)", f"{u_min:.2f}")
        with c6:
            st.write("⏱️ Zeit U_min")
            st.write(str(u_min_time)[-8:] if u_min_time is not None else "–")
        with c7:
            st.metric("Q_% von Nenn", "-" if q_pct is None else f"{q_pct:.1f}%")
        with c8:
            st.metric("E_% von Nenn", "-" if e_pct is None else f"{e_pct:.1f}%")

        st.markdown("---")
        st.markdown("### 📊 Plots")

        # Plot 1: U & I dual-axis
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_int[time_col], y=df_int[u_col], mode="lines", name="U_Bat (V)", yaxis="y1", line=dict(color="#1f77b4")))
        fig1.add_trace(go.Scatter(x=df_int[time_col], y=df_int[i_col], mode="lines", name="I_Bat (A)", yaxis="y2", line=dict(color="#ff7f0e")))
        fig1.update_layout(
            title="Batteriespannung & Batteriestrom",
            hovermode="x unified",
            xaxis=dict(rangeslider=dict(visible=True)),
            yaxis=dict(title="U_Bat (V)", titlefont=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4")),
            yaxis2=dict(title="I_Bat (A)", overlaying="y", side="right", titlefont=dict(color="#ff7f0e"), tickfont=dict(color="#ff7f0e")),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        if u_end is not None:
            fig1.add_hline(y=float(u_end), line_dash="dash", annotation_text="U_End", annotation_position="top left")
        st.plotly_chart(fig1, use_container_width=True)

        # Plot 2: Verbraucherströme
        if i_vi_cols:
            series = [(c, str(c)) for c in i_vi_cols if c in df_int.columns]
            fig2 = plot_timeseries(df_int, time_col, series, "Verbraucher-/Abgangsströme (I_Vi)")
            st.plotly_chart(fig2, use_container_width=True)

        # Plot 3: Cumulative Ah/Wh
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_int[time_col], y=df_int["Q_Ah_cum"], mode="lines", name="Q (Ah)", yaxis="y1", line=dict(color="#2ca02c")))
        fig3.add_trace(go.Scatter(x=df_int[time_col], y=df_int["E_Wh_cum"], mode="lines", name="E (Wh)", yaxis="y2", line=dict(color="#d62728")))
        fig3.update_layout(
            title="Kumulierte Ladung und Energie",
            hovermode="x unified",
            xaxis=dict(rangeslider=dict(visible=True)),
            yaxis=dict(title="Ah", titlefont=dict(color="#2ca02c")),
            yaxis2=dict(title="Wh", overlaying="y", side="right", titlefont=dict(color="#d62728")),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Plot 4: Temperaturen
        if t_cols:
            series = [(c, str(c)) for c in t_cols if c in df_int.columns]
            fig4 = plot_timeseries(df_int, time_col, series, "Temperaturen")
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🎯 Abbruchkriterien-Detektion")
        cA, cB, cC = st.columns(3)
        with cA:
            st.write(f"**I_Ruh,max:** {cfg.i_ruhm_max_ma:.1f} mA, t = {cfg.t_ruhm_min:.1f} min")
            if t_ruhm_hit is not None:
                st.write(f"✅ Erkannt: {str(t0 + timedelta(seconds=float(t_ruhm_hit)))[-8:]}")
            else:
                st.write("❌ Nicht erreicht")
        with cB:
            st.write(f"**U_End:** {'-' if u_end is None else f'{float(u_end):.2f} V'}, t = {cfg.t_uv_s:.0f} s")
            if t_uv_hit is not None:
                st.write(f"✅ Erkannt: {str(t0 + timedelta(seconds=float(t_uv_hit)))[-8:]}")
            else:
                st.write("❌ Nicht erreicht")
        with cC:
            st.write(f"**t_max:** {cfg.t_max_h:.1f} h")
            st.write(f"⏹️ Zeit-Abbruch: {str(t0 + timedelta(seconds=float(t_max_hit)))[-8:]}")

        # Persist für Export
        st.session_state.df_int = df_int
        st.session_state.df_end = df_end
        st.session_state.analysis_summary = {
            "t0": str(t0),
            "t_end": str(t_end),
            "criterion": criterion,
            "Q_Ah_total": q_total,
            "E_Wh_total": e_total,
            "U_Bat_min": u_min,
            "U_Bat_min_time": str(u_min_time) if u_min_time is not None else "",
            "Q_pct": q_pct,
            "E_pct": e_pct,
            "I_Ruh_max_mA": cfg.i_ruhm_max_ma,
            "t_Ruh_min": cfg.t_ruhm_min,
            "U_End_V": None if u_end is None else float(u_end),
            "t_UV_s": cfg.t_uv_s,
            "t_max_h": cfg.t_max_h,
        }


# ===================================
# TAB 5: EXPORT
# ===================================
with tabs[4]:
    st.header("💾 Export & Download")

    cfg: TestConfig = st.session_state.cfg

    st.subheader("📋 Konfiguration")
    cfg_json = json.dumps(asdict(cfg), ensure_ascii=False, indent=2)
    st.download_button(
        "📥 Konfiguration als JSON",
        data=to_download_bytes(cfg_json),
        file_name="testparameter.json",
        mime="application/json",
    )

    if "analysis_summary" in st.session_state:
        summary = st.session_state.analysis_summary
        
        st.subheader("📊 Analyse-Summary")
        summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
        st.download_button(
            "📥 Analyse-Summary als JSON",
            data=to_download_bytes(summary_json),
            file_name="analyse_summary.json",
            mime="application/json",
        )

        st.subheader("📑 Markdown Report")
        md = []
        md.append("# Logger-Auswertung Batterie\n\n")
        md.append("## Objekt\n")
        md.append(f"- **Fahrzeugtyp:** {cfg.fahrzeugtyp}\n")
        md.append(f"- **Fahrzeugnummer:** {cfg.fahrzeugnummer}\n")
        md.append(f"- **Ort:** {cfg.ort}\n")
        md.append(f"- **Datum:** {cfg.datum}\n")
        md.append(f"- **Durchführende Person:** {cfg.durchfuehrende_person}\n")
        if cfg.auftrag_referenz:
            md.append(f"- **Auftrag/Referenz:** {cfg.auftrag_referenz}\n")

        md.append("\n## Batteriesystem\n")
        md.append(f"- **Chemie/Typ:** {cfg.batteriechemie}\n")
        md.append(f"- **Hersteller/Typ:** {cfg.hersteller_typ}\n")
        if cfg.serialnummern:
            md.append(f"- **Serialnummern:** {', '.join(cfg.serialnummern)}\n")
        md.append(f"- **U_Nenn:** {cfg.u_nenn_v if cfg.u_nenn_v is not None else '–'} V\n")
        md.append(f"- **n_Zellen:** {cfg.n_zellen if cfg.n_zellen is not None else '–'}\n")
        md.append(f"- **n_Strang:** {cfg.n_strang if cfg.n_strang is not None else '–'}\n")
        md.append(f"- **C_Nenn:** {cfg.c_nenn_ah if cfg.c_nenn_ah is not None else '–'} Ah\n")
        md.append(f"- **E_Nenn:** {cfg.e_nenn_wh if cfg.e_nenn_wh is not None else '–'} Wh\n")
        md.append(f"- **U_End (gesamt):** {summary.get('U_End_V', '–')} V\n")
        if cfg.u_end_quelle:
            md.append(f"- **Quelle U_End:** {cfg.u_end_quelle}\n")

        md.append("\n## Abbruchkriterien\n")
        md.append(f"- **I_Ruh,max:** {summary['I_Ruh_max_mA']:.1f} mA\n")
        md.append(f"- **t_Ruh:** {summary['t_Ruh_min']:.1f} min\n")
        md.append(f"- **t_UV:** {summary['t_UV_s']:.0f} s\n")
        md.append(f"- **t_max:** {summary['t_max_h']:.1f} h\n")

        md.append("\n## Resultate\n")
        md.append(f"- **Startzeit t0:** {summary['t0']}\n")
        md.append(f"- **Testende:** {summary['t_end']}\n")
        md.append(f"- **Abbruchkriterium:** {summary['criterion']}\n")
        md.append(f"- **U_Bat,min:** {summary['U_Bat_min']:.2f} V (Zeit: {summary['U_Bat_min_time']})\n")
        md.append(f"- **Entnommene Ladung:** {summary['Q_Ah_total']:.3f} Ah\n")
        md.append(f"- **Entnommene Energie:** {summary['E_Wh_total']:.1f} Wh\n")

        q_pct_str = "" if summary.get("Q_pct") is None else f"{summary['Q_pct']:.1f} %"
        e_pct_str = "" if summary.get("E_pct") is None else f"{summary['E_pct']:.1f} %"

        md.append(f"- **Anteil Q_%:** {q_pct_str}\n")
        md.append(f"- **Anteil E_%:** {e_pct_str}\n")

        report_md = "".join(md)
        st.download_button(
            "📥 Report als Markdown",
            data=to_download_bytes(report_md),
            file_name="report_logger_auswertung.md",
            mime="text/markdown",
        )

        st.subheader("📊 CSV-Tabellen")
        
        if "df_int" in st.session_state:
            df_int = st.session_state.df_int
            buf = io.StringIO()
            df_int.to_csv(buf, index=False)
            st.download_button(
                "📥 Auswertung (komplett) als CSV",
                data=to_download_bytes(buf.getvalue()),
                file_name="auswertung_komplett.csv",
                mime="text/csv",
            )

        if "df_end" in st.session_state:
            df_end = st.session_state.df_end
            buf2 = io.StringIO()
            df_end.to_csv(buf2, index=False)
            st.download_button(
                "📥 Auswertung (bis Testende) als CSV",
                data=to_download_bytes(buf2.getvalue()),
                file_name="auswertung_bis_testende.csv",
                mime="text/csv",
            )
    else:
        st.info("ℹ️ Bitte zuerst eine Analyse durchführen (Tab 4)")
