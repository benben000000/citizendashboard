# Multi-Station 3-Hour Prediction & PAGASA Benchmark Report

*Evaluation Date: August 28, 2026 - 09:03 AM PST*  
*Scope: All 16 Meteorological Stations in Central Luzon (Region III)*  
*Horizons Evaluated: +1 Hour, +2 Hours, +3 Hours*

---

## 🏆 Network-Wide Validation Scorecard (16 Weather Stations)

| Parameter | LNN Multi-Station Performance | Official WMO / Tolerance | Status |
| :--- | :--- | :--- | :--- |
| 🌡️ **Network Temperature MAE** | **0.2 °C** | $\le 1.50\ ^\circ	ext{C}$ | ✅ **PASSED** |
| 🌧️ **Precipitation Volume MAE** | **0.0 mm** | $\le 3.00	ext{ mm}$ | ✅ **PASSED** |
| ⚡ **Total Multi-Station Latency** | **0.27 ms (for all 16 stations)** | $< 100	ext{ ms}$ | ✅ **PASSED** |

---

## 📊 Station-by-Station 3-Hour Forecast Breakdown

### 📍 Popolon AWS — Palayan City, Nueva Ecija (ID: `Rjz2dbXW`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **30.4 °C** | 30.2 °C | 0.2 °C | **34.1 °C** | 33.2 °C | **9.4%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **31.1 °C** | 30.9 °C | 0.2 °C | **34.5 °C** | 33.9 °C | **6.2%** | 15.0% | 0.0 mm | Clear Skies |
| **+3h** | 12:00 PM | **31.7 °C** | 31.5 °C | 0.2 °C | **34.9 °C** | 34.5 °C | **5.9%** | 15.0% | 0.0 mm | Clear Skies |

### 📍 Alasas AWS — San Fernando City, Pampanga (ID: `3nzr8bGo`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **30.8 °C** | 30.6 °C | 0.2 °C | **34.6 °C** | 33.6 °C | **8.7%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **31.5 °C** | 31.3 °C | 0.2 °C | **35.0 °C** | 34.3 °C | **6.1%** | 15.0% | 0.0 mm | Clear Skies |
| **+3h** | 12:00 PM | **32.1 °C** | 31.9 °C | 0.2 °C | **35.4 °C** | 34.9 °C | **5.9%** | 15.0% | 0.0 mm | Clear Skies |

### 📍 1Bataan Command Center — Balanga City, Bataan (ID: `2Dpo5DAK`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **30.6 °C** | 30.4 °C | 0.2 °C | **34.5 °C** | 33.4 °C | **9.7%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **31.3 °C** | 31.1 °C | 0.2 °C | **34.9 °C** | 34.1 °C | **6.3%** | 15.0% | 0.0 mm | Clear Skies |
| **+3h** | 12:00 PM | **31.9 °C** | 31.7 °C | 0.2 °C | **35.3 °C** | 34.7 °C | **6.0%** | 15.0% | 0.0 mm | Clear Skies |

### 📍 Sapang Buho AWS — Palayan City, Nueva Ecija (ID: `4VAl2p9k`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **30.2 °C** | 30.0 °C | 0.2 °C | **33.8 °C** | 33.0 °C | **9.7%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **30.9 °C** | 30.7 °C | 0.2 °C | **34.2 °C** | 33.7 °C | **6.3%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+3h** | 12:00 PM | **31.5 °C** | 31.3 °C | 0.2 °C | **34.6 °C** | 34.3 °C | **6.0%** | 15.0% | 0.0 mm | Clear Skies |

### 📍 Abucay AWS — Abucay, Bataan (ID: `lMAZe9b3`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **30.5 °C** | 30.3 °C | 0.2 °C | **34.2 °C** | 33.3 °C | **9.6%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **31.2 °C** | 31.0 °C | 0.2 °C | **34.7 °C** | 34.0 °C | **6.3%** | 15.0% | 0.0 mm | Clear Skies |
| **+3h** | 12:00 PM | **31.8 °C** | 31.6 °C | 0.2 °C | **35.1 °C** | 34.6 °C | **6.0%** | 15.0% | 0.0 mm | Clear Skies |

### 📍 San Jose City AWS — San Jose City, Nueva Ecija (ID: `1Zb102pg`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **30.0 °C** | 29.8 °C | 0.2 °C | **33.5 °C** | 32.8 °C | **10.3%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **30.7 °C** | 30.5 °C | 0.2 °C | **34.0 °C** | 33.5 °C | **6.4%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+3h** | 12:00 PM | **31.3 °C** | 31.1 °C | 0.2 °C | **34.4 °C** | 34.1 °C | **6.0%** | 15.0% | 0.0 mm | Clear Skies |

### 📍 General Natividad AWS — General Natividad, Nueva Ecija (ID: `nDby4YpR`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **30.3 °C** | 30.1 °C | 0.2 °C | **34.0 °C** | 33.1 °C | **9.6%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **31.0 °C** | 30.8 °C | 0.2 °C | **34.4 °C** | 33.8 °C | **6.3%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+3h** | 12:00 PM | **31.6 °C** | 31.4 °C | 0.2 °C | **34.8 °C** | 34.4 °C | **6.0%** | 15.0% | 0.0 mm | Clear Skies |

### 📍 Calumpit AWS — Calumpit, Bulacan (ID: `3nzr48bG`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **30.7 °C** | 30.5 °C | 0.2 °C | **34.6 °C** | 33.5 °C | **8.6%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **31.4 °C** | 31.2 °C | 0.2 °C | **35.1 °C** | 34.2 °C | **6.1%** | 15.0% | 0.0 mm | Clear Skies |
| **+3h** | 12:00 PM | **32.0 °C** | 31.8 °C | 0.2 °C | **35.5 °C** | 34.8 °C | **5.9%** | 15.0% | 0.0 mm | Clear Skies |

### 📍 Bongabon Water District — Bongabon, Nueva Ecija (ID: `03pqkGAj`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **29.9 °C** | 29.8 °C | 0.1 °C | **33.4 °C** | 32.8 °C | **12.1%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **30.6 °C** | 30.4 °C | 0.2 °C | **33.8 °C** | 33.4 °C | **6.9%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+3h** | 12:00 PM | **31.2 °C** | 31.0 °C | 0.2 °C | **34.2 °C** | 34.0 °C | **6.1%** | 15.0% | 0.0 mm | Clear Skies |

### 📍 Doña Maria AWS — Balanga City, Bataan (ID: `95pM7BAV`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **30.6 °C** | 30.4 °C | 0.2 °C | **34.4 °C** | 33.4 °C | **9.2%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **31.3 °C** | 31.1 °C | 0.2 °C | **34.8 °C** | 34.1 °C | **6.2%** | 15.0% | 0.0 mm | Clear Skies |
| **+3h** | 12:00 PM | **31.9 °C** | 31.7 °C | 0.2 °C | **35.2 °C** | 34.7 °C | **6.0%** | 15.0% | 0.0 mm | Clear Skies |

### 📍 San Luis AWS — San Luis, Aurora (ID: `VEpdDpBK`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **29.6 °C** | 29.4 °C | 0.2 °C | **33.7 °C** | 32.4 °C | **17.7%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **30.3 °C** | 30.1 °C | 0.2 °C | **34.1 °C** | 33.1 °C | **9.0%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+3h** | 12:00 PM | **30.9 °C** | 30.7 °C | 0.2 °C | **34.5 °C** | 33.7 °C | **6.8%** | 15.0% | 0.0 mm | Partly Cloudy |

### 📍 Barretto AWS — Olongapo City, Zambales (ID: `rqAkmpKG`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **30.4 °C** | 30.2 °C | 0.2 °C | **34.3 °C** | 33.2 °C | **11.1%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **31.1 °C** | 30.9 °C | 0.2 °C | **34.7 °C** | 33.9 °C | **6.7%** | 15.0% | 0.0 mm | Clear Skies |
| **+3h** | 12:00 PM | **31.7 °C** | 31.5 °C | 0.2 °C | **35.1 °C** | 34.5 °C | **6.1%** | 15.0% | 0.0 mm | Clear Skies |

### 📍 Lazatin AWS — San Fernando City, Pampanga (ID: `wkAWLzlm`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **30.9 °C** | 30.8 °C | 0.1 °C | **34.7 °C** | 33.8 °C | **8.7%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **31.6 °C** | 31.4 °C | 0.2 °C | **35.1 °C** | 34.4 °C | **6.1%** | 15.0% | 0.0 mm | Clear Skies |
| **+3h** | 12:00 PM | **32.2 °C** | 32.0 °C | 0.2 °C | **35.5 °C** | 35.0 °C | **5.9%** | 15.0% | 0.0 mm | Clear Skies |

### 📍 Pag-asa Bagac AWS — Bagac, Bataan (ID: `QgbGldAY`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **30.3 °C** | 30.1 °C | 0.2 °C | **34.2 °C** | 33.1 °C | **12.5%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **31.0 °C** | 30.8 °C | 0.2 °C | **34.7 °C** | 33.8 °C | **7.2%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+3h** | 12:00 PM | **31.6 °C** | 31.4 °C | 0.2 °C | **35.1 °C** | 34.4 °C | **6.2%** | 15.0% | 0.0 mm | Clear Skies |

### 📍 Sabang Morong AWS — Morong, Bataan (ID: `nDbyYbR1`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **30.2 °C** | 30.0 °C | 0.2 °C | **34.2 °C** | 33.0 °C | **12.1%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **30.9 °C** | 30.7 °C | 0.2 °C | **34.6 °C** | 33.7 °C | **7.0%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+3h** | 12:00 PM | **31.5 °C** | 31.3 °C | 0.2 °C | **35.0 °C** | 34.3 °C | **6.2%** | 15.0% | 0.0 mm | Clear Skies |

### 📍 Old Cabalan AWS — Olongapo City, Zambales (ID: `Bkpj1zRO`)

| Horizon | Time | LNN Temp | PAGASA Temp | Δ Temp | LNN Heat Index | PAGASA Heat Index | LNN Rain Prob | PAGASA Rain Prob | Expected Rain | Weather Condition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 10:00 AM | **30.1 °C** | 29.9 °C | 0.2 °C | **34.0 °C** | 32.9 °C | **11.3%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+2h** | 11:00 AM | **30.8 °C** | 30.6 °C | 0.2 °C | **34.4 °C** | 33.6 °C | **6.7%** | 15.0% | 0.0 mm | Partly Cloudy |
| **+3h** | 12:00 PM | **31.4 °C** | 31.2 °C | 0.2 °C | **34.8 °C** | 34.2 °C | **6.1%** | 15.0% | 0.0 mm | Clear Skies |

