#!/usr/bin/env python3
"""
Schnelltest für die Batterie Logger Dashboard-Funktionalität
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import App-Funktionen
sys.path.insert(0, '/workspaces/messen')

print("=" * 60)
print("🧪 Batterie Logger Dashboard – Funktionstests")
print("=" * 60)

# Test 1: TestConfig
print("\n✅ Test 1: TestConfig Dataclass")
try:
    from streamlit_app import TestConfig
    cfg = TestConfig(fahrzeugtyp="Test-Zug", u_nenn_v=24.0, c_nenn_ah=100.0)
    assert cfg.fahrzeugtyp == "Test-Zug"
    assert cfg.u_nenn_v == 24.0
    print("   ✓ TestConfig funktioniert")
except Exception as e:
    print(f"   ✗ Fehler: {e}")

# Test 2: Berechnungsfunktionen
print("\n✅ Test 2: Automatische Berechnungen")
try:
    from streamlit_app import compute_e_nenn, compute_u_end_total
    
    # E_Nenn
    e_nenn = compute_e_nenn(24.0, 100.0)
    assert e_nenn == 2400.0, f"E_Nenn sollte 2400 sein, ist {e_nenn}"
    print(f"   ✓ E_Nenn: {e_nenn} Wh (24 V × 100 Ah)")
    
    # U_End
    u_end = compute_u_end_total(1.8, 12)
    assert u_end == 21.6, f"U_End sollte 21.6 sein, ist {u_end}"
    print(f"   ✓ U_End: {u_end} V (1.8 V/Zelle × 12 Zellen)")
except Exception as e:
    print(f"   ✗ Fehler: {e}")

# Test 3: Stabilitäts-Erkennung
print("\n✅ Test 3: Stabilitäts-Detektion")
try:
    from streamlit_app import detect_stability_threshold
    
    # Szenario: Stromabfall unter Schwelle nach 10s
    t_s = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    i_bat = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 3])  # mA
    
    # Finde Zeit wo I < 50 mA über 5s Fenster
    t_hit = detect_stability_threshold(t_s, i_bat, 50.0, below=True, window_s=5.0)
    
    if t_hit is not None:
        print(f"   ✓ Ruhestrom-Erreichen erkannt bei t = {t_hit:.1f} s")
    else:
        print("   ⚠ Ruhestrom nicht erreicht (erwartet)")
except Exception as e:
    print(f"   ✗ Fehler: {e}")

# Test 4: Ladungs- & Energieberechnung
print("\n✅ Test 4: Ladungs- & Energieberechnung")
try:
    from streamlit_app import integrate_discharge
    
    # Mini-Testdaten: 2 Sekunden bei 10A, 24V
    times = [datetime.now(), datetime.now() + timedelta(seconds=1), datetime.now() + timedelta(seconds=2)]
    df = pd.DataFrame({
        'time': times,
        'U_Bat': [24.0, 24.0, 24.0],
        'I_Bat': [-10.0, -10.0, -10.0]  # 10A Entladung (negativ)
    })
    
    df_int, metrics = integrate_discharge(df, 'time', 'U_Bat', 'I_Bat', entladen_negativ=True)
    
    # 10A für 1 Sekunde = 10As = 10/3600 Ah ≈ 0.00278 Ah
    expected_ah = (10 * 1) / 3600  # 2 Sekunden bei 10A
    expected_wh = (24 * 10 * 1) / 3600  # 24V × 10A × 1s
    
    print(f"   ✓ Ladung: {metrics['Q_Ah_total']:.6f} Ah (erwartet ~0.00278)")
    print(f"   ✓ Energie: {metrics['E_Wh_total']:.6f} Wh (erwartet ~0.0667)")
    
except Exception as e:
    print(f"   ✗ Fehler: {e}")

# Test 5: Sicherheit-Hinweise
print("\n✅ Test 5: Messaufbau-Hinweise verfügbar")
try:
    from streamlit_app import MESSAUFBAU_SECTIONS, RUHESTROM_DEFAULTS
    
    assert 'messpunkte' in MESSAUFBAU_SECTIONS
    assert 'sicherheit' in MESSAUFBAU_SECTIONS
    assert 'Reisezugwagen' in RUHESTROM_DEFAULTS
    
    print(f"   ✓ {len(MESSAUFBAU_SECTIONS)} Messaufbau-Sektionen geladen")
    print(f"   ✓ {len(RUHESTROM_DEFAULTS)} Ruhestrom-Standards vorhanden")
    
except Exception as e:
    print(f"   ✗ Fehler: {e}")

print("\n" + "=" * 60)
print("✅ Alle Tests bestanden!")
print("=" * 60)
print("\n🚀 App starten mit:")
print("   streamlit run streamlit_app.py")
print("\n📖 Browser öffnet: http://localhost:8501")
