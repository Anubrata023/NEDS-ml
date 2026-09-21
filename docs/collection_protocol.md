# Stage-B Hardware Data Collection Protocol — Venjex Smart Lathi

This protocol outlines the standard operating procedure (SOP) for collecting $\ge 2,000$ labeled sensor scans using the physical **Smart Lathi** prototype.

---

## 1. Objective

To build a high-quality, drift-resilient, and session-invariant dataset (`data/ours/*.parquet`) for training model pipelines (Person B) and deploying to edge microcontrollers (Person C).

---

## 2. Proxy Analytes & Canonical Label Mapping

All analytes are legal, standard laboratory/commercial samples selected to safely model target chemical signatures:

| Canonical Label | Target Threat / Category | Legal Proxy Analytes | Recommended Concentration / Sample |
|-----------------|--------------------------|----------------------|-----------------------------------|
| `clean` | Baseline Ambient | Ambient room, AC indoor, outdoor corridor, fresh air | Clean ambient environment |
| `benign_odour` | False Alarm Background | Perfume/Deodorant, Agarbatti, Tea/Coffee, Cooked food, Phenyl cleaner, Sanitizer | 1–2 sprays or standard mug distance |
| `narcotic_voc` | Narcotic VOCs | Ethanol (Rectified Spirit), Acetone (Nail Polish Remover), Toluene/Thinner, Camphor, IPA | 5–10 mL in beaker / open vial |
| `nitrate_trace` | Explosive Precursors | Ammonium Nitrate fertiliser, Urea, Dilute Ammonia solution (10%), Diesel | 10 g solid / 10 mL solution in vial |

> [!IMPORTANT]
> **Proxy Mapping Note:** Labelling `ethanol → narcotic_voc` is a methodology proxy mapping for chemical sensor response evaluation, not a claim that ethanol itself is a narcotic.

---

## 3. Hardware Setup & Scan SOP

### Pre-Scan Setup
1. **Sensor Warm-Up:** Power the Smart Lathi device for at least **15 minutes** before starting collection to allow MOX heater elements to stabilize.
2. **Environmental Log:** Note ambient room temperature ($^\circ\text{C}$) and relative humidity ($\%$) using on-board sensors.

### Scan Execution Steps (Per Scan)
1. **Purge Cycle (60s):** Expose probe tip to clean ambient air for 60 seconds.
2. **Pre-Scan Baseline (5s):** Hold device at target sample position. Start logging 5 seconds before active sampling.
3. **Active Sampling (10s):** Trigger active sampling pump. Collect 10 seconds of high-frequency MOX transient readings + BME688 10-step heater profile sweep.
4. **Post-Purge (60s):** Flush chamber with clean ambient air before positioning next sample.

### Operational Parameters
- **Distances:** Log distance parameter explicitly:
  - `10 cm` (Close proximity scan)
  - `30 cm` (Standard standoff distance)
- **Metadata Logging:** Every scan file MUST store:
  - `scan_id`: `lathi_YYYYMMDD_HHMMSS`
  - `label`: Canonical label string
  - `collection_day`: Integer day index (1, 2, 3, ...)
  - `room_id`: Location tag (e.g., `room_201`, `lab_a`, `corridor_b`)
  - `distance_cm`: 10 or 30
  - `temp_c`, `rh_pct`, `press_hpa`: Environmental sensor values

---

## 4. Multi-Day & Multi-Room Grouping Mandate

> [!WARNING]
> **Preventing Overfitting:** Scans MUST be collected across at least **3 different calendar days** and **2 distinct physical rooms**.

Models will be evaluated using `GroupKFold(groups=collection_day)` and `GroupKFold(groups=room_id)` to guarantee that the model learns true chemical features rather than single-session ambient noise or room temperature biases.

---

## 5. Target Yield & Deliverables

- **Total Target:** $\ge 2,000$ complete scans
- **Distribution Target:**
  - `clean`: 500 scans
  - `benign_odour`: 700 scans (heavy emphasis on false-positive prevention)
  - `narcotic_voc`: 400 scans
  - `nitrate_trace`: 400 scans
- **Output Path:** `data/ours/stage_b_dataset.parquet`
