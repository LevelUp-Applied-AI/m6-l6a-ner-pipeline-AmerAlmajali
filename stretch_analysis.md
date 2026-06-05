# Stretch 6A-S1: Custom NER Rules — Analysis

## Before/After Entity Count Comparison

| Entity Label  | Base spaCy | Ruler Before | Ruler After | Δ Before | Δ After |
|---------------|-----------|--------------|-------------|----------|---------|
| ORG           | 184       | 179          | 185         | -5       | +1      |
| GPE           | 165       | 180          | 165         | +15      | 0       |
| DATE          | 256       | 256          | 256         | 0        | 0       |
| LAW           | 5         | 9            | 5           | +4       | 0       |
| EVENT         | 8         | 4            | 8           | -4       | 0       |
| WORK_OF_ART   | 6         | 6            | 6           | 0        | 0       |
| PERSON        | 36        | 31           | 36          | -5       | 0       |
| QUANTITY      | 92        | 72           | 92          | -20      | 0       |
| MONEY         | 63        | 65           | 63          | +2       | 0       |
| LOC           | 93        | 86           | 93          | -7       | 0       |
| POLICY        | 0         | 4            | 0           | +4       | 0       |
| CLIMATE_EVENT | 0         | 6            | 4           | +6       | +4      |
| REPORT        | 0         | 3            | 0           | +3       | 0       |
| THRESHOLD     | 0         | 26           | 0           | +26      | 0       |
| CARDINAL      | 138       | 136          | 138         | -2       | 0       |
| PERCENT       | 103       | 102          | 103         | -1       | 0       |
| **TOTAL**     | **1202**  | **1217**     | **1207**    | **+15**  | **+5**  |

> Note: evaluation is restricted to the 10 gold-annotated texts only.  
> Predictions on unannotated texts are excluded as they are not verifiable.

---

## Evaluation Delta (Standard Labels Only, Gold Texts Only)

| Metric    | Base spaCy | Ruler Before | Ruler After | Δ Before | Δ After |
|-----------|-----------|--------------|-------------|----------|---------|
| Precision | 0.657     | 0.894        | 0.662       | +0.237   | +0.005  |
| Recall    | 0.647     | 0.868        | 0.662       | +0.221   | +0.015  |
| F1        | 0.652     | 0.881        | 0.662       | +0.229   | +0.010  |

| System       | TP | FP | FN |
|--------------|----|----|-----|
| Base spaCy   | 44 | 23 | 24  |
| Ruler Before | 59 | 7  | 9   |
| Ruler After  | 45 | 23 | 23  |

> ⚠️ Gold standard covers only ~10 texts, so results indicate trends rather than statistically robust conclusions.

---

## Pipeline Position: Before vs After

When the ruler runs **before** the statistical NER, its matches take full priority and produce all meaningful gains.  
F1 improves from 0.652 to 0.881 (+0.229), true positives increase from 44 to 59, and false positives drop sharply from 23 to 7.  
This shows that the rules not only recover missed entities but also correct incorrect model predictions by overriding them before they are finalized.

When the ruler runs **after**, the statistical NER takes priority and suppresses almost all rule-based matches.  
POLICY drops from 4 to 0, REPORT from 3 to 0, and THRESHOLD from 26 to 0.  
Only entities completely missed by the model survive (e.g., some COP mentions), which explains the minimal improvement (F1 +0.010).

This confirms that **pipeline position is critical**, and rule-based components must run before the model to have any real impact.

---

## Custom Label Qualitative Evaluation

### POLICY — 4 matches (ruler before), all correct
All observed POLICY matches were semantically correct.  
"Paris Agreement" appeared in contexts such as *"fall short of the Paris Agreement targets"*, which is a valid policy reference.  
"Nationally Determined Contribution" was correctly identified in national climate commitment contexts.  
No false positives were observed.

---

### CLIMATE_EVENT — 6 matches (ruler before), all correct
All matches correspond to real climate conferences.  
Examples include COP28 (*"ahead of COP28"*), COP26 (*"announced the targets at COP26"*), and COP27 in policy discussions.  
These were previously either missed or inconsistently labeled as ORG by the base model.

---

### REPORT — 3 matches (ruler before), all correct
All REPORT matches were accurate.  
"Sixth Assessment Report" was correctly identified in *"The IPCC released its Sixth Assessment Report..."*, recovering a gold entity that the base model missed.  
"Adaptation Gap Report" and "Emissions Gap Report" were also correctly identified.  
No false positives were observed.

---

### THRESHOLD — 26 matches (ruler before), correct intent with noise
The THRESHOLD pattern successfully captured policy-relevant targets such as "1.5 degrees Celsius".  
However, it also matched measurement contexts such as "1.48 degrees Celsius" and "2 degrees Celsius above the seasonal average".

These spans are syntactically correct but semantically incorrect for the THRESHOLD label, as they represent observed values rather than policy targets.  
This reflects a recall-oriented design: broader patterns improve coverage but introduce controlled semantic noise.

---

## Analysis

The custom EntityRuler running before the statistical NER produces substantial improvements by targeting systematic model failures.  
F1 increases from 0.652 to 0.881 (+0.229), precision improves significantly (0.657 → 0.894), and recall also increases (0.647 → 0.868).  
False positives drop sharply (23 → 7), while false negatives decrease (24 → 9).

The main gains come from:
- Recovering missed entities such as "Paris Agreement" (LAW/POLICY)
- Correctly identifying climate events (COP28, COP26, COP27)
- Recovering report entities such as "Sixth Assessment Report"
- Fixing systematic label errors (e.g., LOC → GPE for "Dubai", "New York")

The THRESHOLD rule introduces some noise due to its broad design, highlighting a trade-off between recall and precision.  
However, this noise is controlled and explainable, and could be reduced with additional context constraints.

In contrast, the ruler-after configuration is largely ineffective, as the statistical NER blocks rule-based matches.  
This demonstrates that **rule-based systems are most effective when applied before the model and when focused on known weaknesses rather than general patterns**.