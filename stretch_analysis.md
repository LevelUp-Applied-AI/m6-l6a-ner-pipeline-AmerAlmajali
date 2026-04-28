# Stretch 6A-S1: Custom NER Rules — Analysis

## Before/After Entity Count Comparison

| Entity Label  | Base spaCy | Ruler Before | Ruler After | Δ Before | Δ After |
|---------------|-----------|--------------|-------------|----------|---------|
| ORG           | 184       | 177          | 186         | -7       | +2      |
| GPE           | 165       | 180          | 165         | +15      | 0       |
| DATE          | 256       | 256          | 256         | 0        | 0       |
| LAW           | 5         | 5            | 5           | 0        | 0       |
| EVENT         | 8         | 4            | 8           | -4       | 0       |
| WORK_OF_ART   | 6         | 2            | 6           | -4       | 0       |
| PERSON        | 36        | 31           | 36          | -5       | 0       |
| QUANTITY      | 92        | 72           | 92          | -20      | 0       |
| MONEY         | 63        | 65           | 63          | +2       | 0       |
| LOC           | 93        | 86           | 93          | -7       | 0       |
| POLICY        | 0         | 11           | 1           | +11      | +1      |
| CLIMATE_EVENT | 0         | 8            | 4           | +8       | +4      |
| REPORT        | 0         | 6            | 0           | +6       | 0       |
| THRESHOLD     | 0         | 29           | 3           | +29      | +3      |
| CARDINAL      | 138       | 136          | 138         | -2       | 0       |
| PERCENT       | 103       | 102          | 103         | -1       | 0       |
| **TOTAL**     | **1202**  | **1222**     | **1212**    | **+20**  | **+10** |

> Note: evaluation is filtered to the 10 gold-annotated texts only.
> Predictions on the remaining 122 unannotated English texts are excluded
> from all precision/recall/F1 calculations — they are unverifiable, not
> false positives.

---

## Evaluation Delta (Standard Labels Only, Gold Texts Only)

| Metric    | Base spaCy | Ruler Before | Ruler After | Δ Before | Δ After |
|-----------|-----------|--------------|-------------|----------|---------|
| Precision | 0.657     | 0.892        | 0.662       | +0.236   | +0.005  |
| Recall    | 0.647     | 0.853        | 0.662       | +0.206   | +0.015  |
| F1        | 0.652     | 0.872        | 0.662       | +0.220   | +0.010  |

| System       | TP | FP | FN |
|--------------|----|----|-----|
| Base spaCy   | 44 | 23 | 24  |
| Ruler Before | 58 | 7  | 10  |
| Ruler After  | 45 | 23 | 23  |

> ⚠️ Gold standard covers only 10 texts (~69 entities). These numbers are
> indicative, not statistically robust — focus on error patterns rather
> than absolute metric values.

---

## Pipeline Position: Before vs After

When the ruler runs **before** the statistical NER, its matches take full
priority and produced all the meaningful gains — F1 jumped from 0.652 to
0.872 (+0.220), TP increased from 44 to 58, and FP collapsed from 23 to
just 7. This is the strongest result: the ruler not only added new true
positives but also corrected existing false positives by overriding wrong
base model labels before they were set.

When the ruler runs **after**, the statistical NER takes priority and almost
entirely suppresses the custom rules — POLICY dropped from 11 matches to 1,
REPORT from 6 to 0, and THRESHOLD from 29 to 3. Only entities the base
model missed entirely survived (e.g. "Loss and Damage Fund" as POLICY,
"net-zero" as THRESHOLD). The Δ After column confirms this: nearly every
label shows 0 change, meaning the statistical NER blocked the ruler from
firing. The after position produced negligible gains (F1 +0.010 vs +0.220
for before), confirming that **before is the correct position** for this
domain-specific ruleset.

---

## Custom Label Qualitative Evaluation

### POLICY — 11 matches (ruler before), all correct
All 11 POLICY matches were semantically correct. "Paris Agreement" fired
in contexts like *"fall short of the Paris Agreement targets"* — a genuine
policy reference found across multiple texts. "Nationally Determined
Contribution" fired correctly on India's climate pledge text. "Loss and
Damage Fund" fired in *"established at COP27 in Sharm el-Sheikh"* — correct.
No false positives observed among the POLICY matches.

### CLIMATE_EVENT — 8 matches (ruler before), all correct
All 8 matches were correct climate conference references. COP28 appeared in
*"a positive signal ahead of COP28"*, COP26 in *"Narendra Modi announced
the targets at COP26 in Glasgow"*, and COP27 in the Loss and Damage Fund
context. These were previously either missed or tagged inconsistently as
ORG by the base model. In the after position only 4 survived — the ones
the base model had not already claimed.

### REPORT — 6 matches (ruler before), all correct
All 6 REPORT matches were accurate. "Sixth Assessment Report" fired on
text 1 (*"The IPCC released its Sixth Assessment Report in March 2023"*),
recovering the gold WORK_OF_ART entity that both base systems missed.
"Adaptation Gap Report" fired on *"UNEP's Adaptation Gap Report 2023 found
that adaptation finance needs"* and "Emissions Gap Report" on *"UNEP's
Emissions Gap Report 2023 found that current policies put the world on
track"* — both correct. Zero false positives. In the after position REPORT
dropped to 0 — the base model had already consumed those spans.

### THRESHOLD — 29 total matches (ruler before), correct intent with noise
"1.5 degrees Celsius" (text 1) fired correctly as a policy temperature
target in *"global temperatures could exceed 1.5 degrees Celsius above
pre-industrial levels by 2030"*. However the token pattern over-fired on
measurement contexts: "1.48 degrees Celsius" in text 11 (*"global average
temperatures 1.48 degrees Celsius above the 1850-1900 baseline"*) is a
recorded observation, not a policy threshold, and "2 degrees Celsius" in
text 57 (*"Water temperatures exceeded 2 degrees Celsius above the seasonal
average"*) is a measured anomaly. These are correct spans but semantically
mislabeled as THRESHOLD when they should remain QUANTITY. A context
constraint requiring preceding terms like "below", "limit of", or "target
of" would reduce this noise significantly. In the after position only 3
survived — "net-zero" and two others the base model had not tagged.

---

## Analysis

The custom EntityRuler running **before** the statistical NER produced
exceptional improvements once the evaluation was correctly scoped to the
10 gold-annotated texts only — F1 jumped from 0.652 to 0.872 (+0.220),
precision from 0.657 to 0.892 (+0.236), and recall from 0.647 to 0.853
(+0.206), with FP dropping dramatically from 23 to just 7 and FN from 24
to 10. The largest gains came from the three label types both base systems
structurally failed on: LAW/POLICY patterns recovered "Paris Agreement"
across texts 5, 7, and 10; REPORT patterns recovered "Sixth Assessment
Report" from text 1 (gold: WORK_OF_ART); and CLIMATE_EVENT patterns
correctly tagged COP28 (texts 2, 10), COP26, and COP27 which the base
model either missed or labeled inconsistently as ORG. The GPE corrections
(+15 counts) addressed the systematic label mismatch for entities like
"Dubai" (text 2), "New York" (text 7), and "South Korea" (text 9) that
the base model and HF both tagged as LOC instead of GPE. The main source
of noise was the THRESHOLD token pattern, which fired 29 times and
correctly identified policy targets like "1.5 degrees Celsius" (text 1)
but also tagged observed measurements like "1.48 degrees Celsius" (text
11) and "2 degrees Celsius above the seasonal average" (text 57) —
correct spans, wrong semantic category. The ruler-after position was
largely ineffective (F1 +0.010), and the Δ After column tells the full
story: nearly every standard label shows zero change, meaning the
statistical NER blocked the ruler from firing on almost every span it
had already claimed — confirming that domain-specific phrase rules must
run before the statistical model to have any meaningful impact.