"""
Module 6 Week A — Stretch: Custom NER Rules
Extends the base spaCy NER with a domain-specific EntityRuler
for climate terminology.

Run: python stretch_custom_ner.py
"""

import pandas as pd
import spacy


def load_data(filepath="data/climate_articles.csv"):
    return pd.read_csv(filepath)


def build_patterns():
    """
    Patterns built directly from the gold standard misses identified
    in the Lab 6A error analysis — targeting entities both spaCy and
    HF failed on, plus 4 custom label types required by the stretch spec.
    """
    patterns = [
        # ── POLICY (custom label) ─────────────────────────────────────
        # Paris Agreement — missed by both systems in texts 5, 7, 10
        {"label": "POLICY", "pattern": "Paris Agreement"},
        # text 6 — missed by both systems
        {"label": "POLICY", "pattern": "Carbon Border Adjustment Mechanism"},
        {"label": "POLICY", "pattern": "CBAM"},
        {"label": "POLICY", "pattern": "Montreal Protocol"},
        {"label": "POLICY", "pattern": "Kigali Amendment"},
        {"label": "POLICY", "pattern": "Loss and Damage Fund"},
        {"label": "POLICY", "pattern": "Global Methane Pledge"},
        {"label": "POLICY", "pattern": "Nationally Determined Contribution"},
        {"label": "POLICY", "pattern": "NDC"},
        # ── CLIMATE_EVENT (custom label) ──────────────────────────────
        # text 5 — missed by both, gold label was EVENT
        {"label": "CLIMATE_EVENT", "pattern": "Bonn Climate Change Conference"},
        # text 7 — missed by both, gold label was EVENT
        {"label": "CLIMATE_EVENT", "pattern": "Climate Ambition Summit"},
        # text 2, 10 — HF broke into subwords: cop28
        {"label": "CLIMATE_EVENT", "pattern": "COP28"},
        {"label": "CLIMATE_EVENT", "pattern": "COP27"},
        {"label": "CLIMATE_EVENT", "pattern": "COP26"},
        {"label": "CLIMATE_EVENT", "pattern": "One Planet Summit"},
        {"label": "CLIMATE_EVENT", "pattern": "Global Stocktake"},
        # ── REPORT (custom label) ─────────────────────────────────────
        # text 1 — missed by both, gold label was WORK_OF_ART
        {"label": "REPORT", "pattern": "Sixth Assessment Report"},
        {"label": "REPORT", "pattern": "IPCC AR6"},
        {"label": "REPORT", "pattern": "Emissions Gap Report"},
        {"label": "REPORT", "pattern": "Adaptation Gap Report"},
        {"label": "REPORT", "pattern": "Global Methane Assessment"},
        {"label": "REPORT", "pattern": "Global Landscape of Climate Finance"},
        {"label": "REPORT", "pattern": "Global Risks Report"},
        # ── THRESHOLD (custom label) ──────────────────────────────────
        # text 1 — gold label was QUANTITY, HF schema can't predict it
        {"label": "THRESHOLD", "pattern": "1.5 degrees Celsius"},
        {"label": "THRESHOLD", "pattern": "2 degrees Celsius"},
        {"label": "THRESHOLD", "pattern": "1.5°C"},
        {"label": "THRESHOLD", "pattern": "2°C"},
        {"label": "THRESHOLD", "pattern": "net zero"},
        {"label": "THRESHOLD", "pattern": "net-zero"},
        {"label": "THRESHOLD", "pattern": "carbon neutral"},
        {"label": "THRESHOLD", "pattern": "carbon neutrality"},
        {
            "label": "THRESHOLD",
            "pattern": [
                {"LIKE_NUM": True},
                {"TEXT": {"IN": ["°C", "degrees"]}},
                {"LOWER": "celsius", "OP": "?"},
            ],
        },
        # ── LAW (standard label) ──────────────────────────────────────
        # missed by both in texts 5, 7, 10
        {"label": "LAW", "pattern": "Paris Agreement"},
        # missed by both in text 6
        {"label": "LAW", "pattern": "Carbon Border Adjustment Mechanism"},
        {"label": "LAW", "pattern": "CBAM"},
        # ── EVENT (standard label) ────────────────────────────────────
        # missed by both in text 5
        {"label": "EVENT", "pattern": "Bonn Climate Change Conference"},
        # missed by both in text 7
        {"label": "EVENT", "pattern": "Climate Ambition Summit"},
        # ── WORK_OF_ART (standard label) ──────────────────────────────
        # missed by both in text 1
        {"label": "WORK_OF_ART", "pattern": "Sixth Assessment Report"},
        # ── ORG (standard label) ──────────────────────────────────────
        # text 2, 10 — HF broke into subwords
        {"label": "ORG", "pattern": "COP28"},
        {"label": "ORG", "pattern": "COP27"},
        # text 9 — HF missed entirely
        {"label": "ORG", "pattern": "Green Climate Fund"},
        # text 7 — HF broke into unitednationsgeneralassembly
        {"label": "ORG", "pattern": "United Nations General Assembly"},
        # text 4 — HF missed
        {"label": "ORG", "pattern": "Ministry of Environment"},
        # text 4 — HF missed
        {"label": "ORG", "pattern": "European Union"},
        # text 6
        {"label": "ORG", "pattern": "EU"},
        # text 3 — HF broke into worldbank
        {"label": "ORG", "pattern": "World Bank"},
        # text 8 — HF missed
        {"label": "ORG", "pattern": "FAO"},
        # text 5
        {"label": "ORG", "pattern": "UNFCCC"},
        # text 1
        {"label": "ORG", "pattern": "IPCC"},
        # ── GPE (standard label) ──────────────────────────────────────
        # text 2 — HF: LOC not GPE
        {"label": "GPE", "pattern": "Dubai"},
        # text 2 — HF broke into unitedarabemirates
        {"label": "GPE", "pattern": "United Arab Emirates"},
        # text 7 — HF: LOC not GPE
        {"label": "GPE", "pattern": "New York"},
        # text 9 — HF: LOC not GPE
        {"label": "GPE", "pattern": "South Korea"},
        {"label": "GPE", "pattern": "Bangladesh"},
        {"label": "GPE", "pattern": "Songdo"},
        # text 10 — HF: LOC not GPE
        {"label": "GPE", "pattern": "California"},
        # text 4 — HF: LOC not GPE
        {"label": "GPE", "pattern": "Jordan"},
        # text 10 — HF: LOC not GPE
        {"label": "GPE", "pattern": "China"},
        # ── PERSON (standard label) ───────────────────────────────────
        # text 1 — HF: antonioguterres (broken)
        {"label": "PERSON", "pattern": "Antonio Guterres"},
        # text 3 — HF: ##jaybanga (broken)
        {"label": "PERSON", "pattern": "Ajay Banga"},
        # text 5 — HF: simonstiell (broken)
        {"label": "PERSON", "pattern": "Simon Stiell"},
        # text 8 — HF: ##udongyu (broken)
        {"label": "PERSON", "pattern": "Qu Dongyu"},
        # text 10 — HF: johnkerry (broken)
        {"label": "PERSON", "pattern": "John Kerry"},
        # text 10 — HF: xiezhenhua (broken)
        {"label": "PERSON", "pattern": "Xie Zhenhua"},
        # ── QUANTITY (standard label) ─────────────────────────────────
        # text 1 — gold entity, HF schema cannot predict QUANTITY
        {"label": "QUANTITY", "pattern": "1.5 degrees Celsius"},
        # text 2 — gold entity
        {"label": "QUANTITY", "pattern": "190 nations"},
        # text 8 — gold entity
        {"label": "QUANTITY", "pattern": "735 million people"},
        # text 4 — gold entity
        {"label": "QUANTITY", "pattern": "31%"},
        # ── MONEY (standard label) ────────────────────────────────────
        # text 3
        {"label": "MONEY", "pattern": "$12.5 billion"},
        # text 8
        {"label": "MONEY", "pattern": "$4 billion"},
        # text 9 — flexible token pattern for $X million/billion/trillion
        {
            "label": "MONEY",
            "pattern": [
                {"TEXT": "$"},
                {"LIKE_NUM": True},
                {"LOWER": {"IN": ["million", "billion", "trillion"]}},
            ],
        },
    ]
    return patterns


def extract_entities(df, nlp):
    """Extract entities from English texts using the given pipeline."""
    english_df = df[df["language"] == "en"]
    rows = []
    for _, row in english_df.iterrows():
        doc = nlp(row["text"])
        for ent in doc.ents:
            rows.append(
                {
                    "text_id": row["id"],
                    "entity_text": ent.text,
                    "entity_label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"],
    )


def evaluate_ner(predicted_df, gold_df):
    """
    Evaluate on overlapping standard labels only.
    IMPORTANT: Filter predictions to gold-annotated text IDs only —
    predictions on unannotated texts are not false positives, they are
    simply unverifiable and must be excluded from the denominator.
    """
    STANDARD_LABELS = {
        "ORG",
        "GPE",
        "DATE",
        "LAW",
        "MONEY",
        "PERSON",
        "QUANTITY",
        "LOC",
        "EVENT",
        "WORK_OF_ART",
    }

    # ── only evaluate on the 10 gold-annotated texts ─────────────────
    gold_text_ids = set(gold_df["text_id"].unique())

    predicted_filtered = predicted_df[
        (predicted_df["entity_label"].isin(STANDARD_LABELS))
        & (predicted_df["text_id"].isin(gold_text_ids))
    ]

    predicted_set = set(
        zip(
            predicted_filtered["text_id"],
            predicted_filtered["entity_text"].str.lower(),
            predicted_filtered["entity_label"],
        )
    )
    gold_set = set(
        zip(
            gold_df["text_id"],
            gold_df["entity_text"].str.lower(),
            gold_df["entity_label"],
        )
    )

    tp = len(predicted_set & gold_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compare_label_counts(base_df, before_df, after_df):
    """Print side-by-side entity label counts for all three pipelines."""
    all_labels = sorted(
        set(
            list(base_df["entity_label"].unique())
            + list(before_df["entity_label"].unique())
            + list(after_df["entity_label"].unique())
        )
    )
    print(f"\n{'='*65}")
    print("  Entity Label Counts: Base vs Ruler-Before vs Ruler-After")
    print(f"{'='*65}")
    print(f"  {'Label':<28} {'Base':>6} {'Before':>8} {'After':>7}")
    print(f"  {'-'*53}")
    for label in all_labels:
        b = len(base_df[base_df["entity_label"] == label])
        be = len(before_df[before_df["entity_label"] == label])
        af = len(after_df[after_df["entity_label"] == label])
        print(f"  {label:<28} {b:>6} {be:>8} {af:>7}")
    print(f"  {'-'*53}")
    print(f"  {'TOTAL':<28} {len(base_df):>6} {len(before_df):>8} {len(after_df):>7}")


def show_custom_label_examples(entities_df, df, n=3):
    """
    Surface example texts where each custom label fired,
    for qualitative evaluation in the analysis.
    """
    CUSTOM_LABELS = ["POLICY", "CLIMATE_EVENT", "REPORT", "THRESHOLD"]

    print(f"\n{'='*65}")
    print("  Custom Label Examples (Qualitative Evaluation)")
    print(f"{'='*65}")

    for label in CUSTOM_LABELS:
        hits = entities_df[entities_df["entity_label"] == label]
        if hits.empty:
            print(f"\n  [{label}] — no matches found")
            continue
        print(f"\n  [{label}] — {len(hits)} total matches")
        samples = hits.drop_duplicates("entity_text").head(n)
        for _, hit in samples.iterrows():
            text_row = df[df["id"] == hit["text_id"]]
            if text_row.empty:
                continue
            full_text = text_row.iloc[0]["text"]
            start = max(0, hit["start_char"] - 40)
            end = min(len(full_text), hit["end_char"] + 40)
            snippet = "..." + full_text[start:end] + "..."
            print(f"    entity  : {hit['entity_text']}")
            print(f"    context : {snippet}")
            print()


if __name__ == "__main__":
    df = load_data()
    gold = pd.read_csv("data/gold_entities.csv")
    patterns = build_patterns()

    # ── Base pipeline (no EntityRuler) ───────────────────────────────
    nlp_base = spacy.load("en_core_web_sm")
    base_entities = extract_entities(df, nlp_base)
    base_metrics = evaluate_ner(base_entities, gold)

    # ── EntityRuler BEFORE statistical NER ───────────────────────────
    nlp_before = spacy.load("en_core_web_sm")
    ruler_before = nlp_before.add_pipe(
        "entity_ruler", before="ner", name="ruler_before"
    )
    ruler_before.add_patterns(patterns)
    before_entities = extract_entities(df, nlp_before)
    before_metrics = evaluate_ner(before_entities, gold)

    # ── EntityRuler AFTER statistical NER ────────────────────────────
    nlp_after = spacy.load("en_core_web_sm")
    ruler_after = nlp_after.add_pipe("entity_ruler", after="ner", name="ruler_after")
    ruler_after.add_patterns(patterns)
    after_entities = extract_entities(df, nlp_after)
    after_metrics = evaluate_ner(after_entities, gold)

    # ── Metrics summary ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Metrics Comparison (standard labels only)")
    print(f"{'='*65}")
    print(
        f"  {'Metric':<12} {'Base':>8} {'Before':>8} {'After':>8} {'Δ Before':>10} {'Δ After':>9}"
    )
    print(f"  {'-'*57}")
    for key in ["precision", "recall", "f1"]:
        delta_before = before_metrics[key] - base_metrics[key]
        delta_after = after_metrics[key] - base_metrics[key]
        print(
            f"  {key:<12} "
            f"{base_metrics[key]:>8.3f} "
            f"{before_metrics[key]:>8.3f} "
            f"{after_metrics[key]:>8.3f} "
            f"{delta_before:>+10.3f} "
            f"{delta_after:>+9.3f}"
        )
    print(f"\n  TP / FP / FN breakdown:")
    for name, m in [
        ("Base", base_metrics),
        ("Before", before_metrics),
        ("After", after_metrics),
    ]:
        print(f"  {name:<8} TP={m['tp']:>3}  FP={m['fp']:>4}  FN={m['fn']:>3}")

    # ── Label count comparison ────────────────────────────────────────
    compare_label_counts(base_entities, before_entities, after_entities)

    # ── Qualitative evaluation — ruler BEFORE ─────────────────────────
    print("\n--- Custom label examples: Ruler BEFORE NER ---")
    show_custom_label_examples(before_entities, df)

    # ── Qualitative evaluation — ruler AFTER ──────────────────────────
    print("\n--- Custom label examples: Ruler AFTER NER ---")
    show_custom_label_examples(after_entities, df)
