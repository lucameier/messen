# 📊 Batterie Logger Dashboard

Professionelle Konfiguration, Analyse und Auswertung von Logger-Messungen an Batteriesystemen gemäss technischen Vorgaben.

## 🎯 Features

- **📋 Testparameter-Konfiguration** – Strukturierte Eingabe aller relevanten Messvorgaben
- **📥 CSV-Upload & Spalten-Mapping** – Flexible Datenimporte mit Skalierungsfaktoren
- **🔧 Messaufbau-Hinweise** – Detaillierte Anleitung mit Sicherheitshinweisen und Checklisten
  - Messpunkte & Kanalzuordnung
  - Vorbereitung & Installation
  - Sicherheitshinweise (verbindlich)
  - Logger-Konfiguration
  - Ereignis-Protokollierung
- **📈 Automatische Analyse & Visualisierung**
  - Abbruchkriterien-Detektion (Ruhestrom, Unterspannung, Zeit)
  - Berechnung entnommener Ladung (Ah) und Energie (Wh)
  - Duale Plots (Spannung/Strom, Verbraucherströme, Temperatur)
  - Kumulative Metriken
- **💾 Export** – JSON, Markdown, CSV (mit allen Berechnungen)

## 📋 Abbruchkriterien

Das Dashboard detektiert automatisch:

| Kriterium | Bedingung | Default |
|-----------|-----------|---------|
| **Ruhestrom** | \|I_Bat\| < I_Ruh,max über t_Ruh | 40–150 mA, 5 min |
| **Unterspannung** | U_Bat ≤ U_End über t_UV | Projektabhängig, 120 s |
| **Maximale Zeit** | t - t0 ≥ t_max | 24 h |

## 🚀 Installation

1. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

2. App starten:
   ```bash
   streamlit run streamlit_app.py
   ```

3. Browser öffnet sich automatisch unter `http://localhost:8501`

## 📖 Verwendung

### 1️⃣ Testparameter (Tab 1)
Konfiguriere Fahrzeug, Batterie, Messziele und Abbruchkriterien.
- Automatische Berechnung von U_End und E_Nenn aus Herstellerdaten
- Vorlagen für Standard-Fahrzeugkategorien (Reisezugwagen, Triebzug)

### 2️⃣ CSV Upload & Mapping (Tab 2)
Lade Logger-Daten hoch und ordne Spalten zu.
- Unterstützt verschiedene Trennzeichen und Dezimalformate
- Optional: Skalierungsfaktoren pro Spalte (z.B. mA → A)

### 3️⃣ Messaufbau-Hinweise (Tab 3)
Detaillierte Anleitung für die Messung:
- **Messpunkte:** U_Bat, I_Bat, I_Vi
- **Kanalzuordnung:** Logger-Konfiguration
- **Vorbereitung:** Checkliste vor Messbeginn
- **Sicherheit:** Verbindliche Sicherheitshinweise
- **Logger-Konfiguration:** Zeitbasis, Datenformat, Konventionen
- **Messphase:** Ereignis-Protokollierung

Enthält auch Tabellen mit:
- Abbruchkriterien und Stabilitätsfenstern
- Mindestanforderungen Messkette
- Ruhestrom-Defaults (BCA 20002483)

### 4️⃣ Analyse & Visualisierung (Tab 4)
Automatische Auswertung der geladenen Daten:
- **Zusammenfassung:** Abbruchkriterium, Testende, Ladung, Energie
- **Plots:**
  - U_Bat & I_Bat (Dual-Axis mit Bereichsschieber)
  - Verbraucherströme (I_Vi)
  - Kumulative Ladung & Energie
  - Temperaturen (falls vorhanden)
- **Abbruchkriterien-Detektion:** Automatische Erkennung mit Zeitpunkten

### 5️⃣ Export (Tab 5)
Lade Ergebnisse herunter:
- Testparameter als JSON
- Analyse-Summary als JSON
- Report als Markdown
- Auswertungstabellen als CSV (komplett & bis Testende)

## 📐 Formeln

### Entnommene Ladung
$$I_{\mathrm{Entl},k} = \max(-I_{\mathrm{Bat},k}, 0)$$
$$Q_\mathrm{Ah} = \frac{1}{3600} \sum_{k=0}^{n-1} I_{\mathrm{Entl},k} \cdot \Delta t_k$$

### Entnommene Energie
$$P_k = U_{\mathrm{Bat},k} \cdot I_{\mathrm{Entl},k}$$
$$E_\mathrm{Wh} = \frac{1}{3600} \sum_{k=0}^{n-1} P_k \cdot \Delta t_k$$

### Bewertung gegen Nennwerte
$$Q_\% = 100 \cdot \frac{Q_\mathrm{Ah}}{C_\mathrm{Nenn}}, \quad E_\% = 100 \cdot \frac{E_\mathrm{Wh}}{E_\mathrm{Nenn}}$$

## 📚 Normen & Referenzen

- **BCA 20002483** – Technische Regeln Batteriesystem
- Fahrzeugspezifische Unterlagen (Stromlaufplan, Sicherungs-/Abgangsliste)
- Batterieherstellerdatenblätter
- Lokale Sicherheits- und Arbeitsvorschriften (LOTO, Freischaltregeln)

## 🔐 Sicherheit

⚠️ **Wichtig:** Arbeiten an elektrischen Anlagen sind ausschliesslich durch qualifiziertes Personal gemäss geltenden Vorschriften durchzuführen!

Das Dashboard enthält detaillierte Sicherheitshinweise im Tab "Messaufbau & Hinweise".

## 📞 Support

Bei Fragen oder Verbesserungen: Dokumentation konsultieren
