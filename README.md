# 📊 Batterie Logger Dashboard

**Einfache, intuitive Konfiguration, CSV-Upload und automatische Auswertung** von Logger-Messungen an Batteriesystemen - alles auf einer Übersichtlichen Seite.

## 🎯 Features

- **📋 Testparameter** – Fahrzeug, Batterie, Messziele, Abbruchkriterien (6-Spalten Layout)
- **📥 CSV Upload & Mapping** – Flexible Datenimporte mit Skalierungsfaktoren
- **📈 Analyse & Visualisierung** – Automatische Auswertung mit Plots
  - Abbruchkriterien-Detektion (Ruhestrom, Unterspannung, Zeit)
  - Berechnung Ladung (Ah) und Energie (Wh)
  - Dual-Axis Plots (U_Bat & I_Bat, Q & E)
- **💾 Export** – JSON & CSV Download
- **5️⃣ Messaufbau-Hinweise** – Sicherheit, Vorbereitung, Logger-Konfiguration

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

## 📖 Workflow

**Eine Seite – Fünf Abschnitte:**

### 1️⃣ Testparameter
- Fahrzeug (Typ, Nummer, Ort, Datum, Person)
- Batterie (Chemie, Hersteller, U_Nenn, C_Nenn, Zellen, Stränge)
- Messziele (Messziel, Testbedingung, Bedienprofil)
- Abbruchkriterien (Ruhestrom, Unterspannung, Zeit)
- **Automatisch:** E_Nenn aus U × C; U_End aus U/Zelle × n

### 2️⃣ CSV Upload & Mapping
- CSV hochladen (Trennzeichen, Dezimal, Encoding einstellbar)
- Spalten mappen (Zeit, U_Bat, I_Bat, optional Verbraucher & Temperaturen)
- Skalierungsfaktoren setzen (z.B. mA → A)

### 3️⃣ Analyse & Ergebnisse
- **6 Metriken:** Kriterium, Testende, Ladung, Energie, U_min, Q_%
- **2 Plots:** U_Bat & I_Bat (Dual-Axis); Ladung & Energie
- **Abbruchkriterien:** Automatische Detektion mit Zeitpunkten

### 4️⃣ Export
- Konfiguration als JSON
- Ergebnisse als JSON
- Auswertungstabelle als CSV

### 5️⃣ Messaufbau-Hinweise
- Messpunkte (U_Bat, I_Bat, I_Vi)
- Sicherheit (Verbindlich!)
- Vorbereitung vor Messung
- Logger-Konfiguration
- Abbruchkriterien-Tabelle

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
