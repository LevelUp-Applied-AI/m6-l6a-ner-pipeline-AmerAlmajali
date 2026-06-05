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
    patterns = [
        # ── POLICY (custom label) ─────────────────────────────────────
        {"label": "POLICY", "pattern": "Paris Agreement"},
        {"label": "POLICY", "pattern": "Carbon Border Adjustment Mechanism"},
        {"label": "POLICY", "pattern": "Nationally Determined Contribution"},
        # ── LAW (standard label) — gold texts 5, 7, 10 use LAW not POLICY
        {"label": "LAW", "pattern": "Paris Agreement"},
        {"label": "LAW", "pattern": "Carbon Border Adjustment Mechanism"},
        # ── CLIMATE_EVENT (custom label) ──────────────────────────────
        {
            "label": "CLIMATE_EVENT",
            "pattern": [{"TEXT": {"REGEX": r"^COP\d+$"}}],
        },
        {"label": "CLIMATE_EVENT", "pattern": "Bonn Climate Change Conference"},
        # ── EVENT (standard label) — gold text 5 uses EVENT not CLIMATE_EVENT
        {"label": "EVENT", "pattern": "Bonn Climate Change Conference"},
        {"label": "EVENT", "pattern": "Climate Ambition Summit"},
        # ── REPORT (custom label) ─────────────────────────────────────
        {
            "label": "REPORT",
            "pattern": [
                {
                    "LOWER": {
                        "IN": ["first", "second", "third", "fourth", "fifth", "sixth"]
                    }
                },
                {"LOWER": "assessment"},
                {"LOWER": "report"},
            ],
        },
        {
            "label": "REPORT",
            "pattern": [
                {"IS_TITLE": True},
                {"LOWER": "gap"},
                {"LOWER": "report"},
            ],
        },
        # ── WORK_OF_ART (standard) — gold text 1 uses WORK_OF_ART not REPORT
        {
            "label": "WORK_OF_ART",
            "pattern": [
                {
                    "LOWER": {
                        "IN": ["first", "second", "third", "fourth", "fifth", "sixth"]
                    }
                },
                {"LOWER": "assessment"},
                {"LOWER": "report"},
            ],
        },
        # ── THRESHOLD (custom label) ──────────────────────────────────
        {
            "label": "THRESHOLD",
            "pattern": [
                {"LIKE_NUM": True},
                {"LOWER": {"IN": ["degrees", "°c"]}},
                {"LOWER": "celsius", "OP": "?"},
            ],
        },
        {"label": "THRESHOLD", "pattern": "net zero"},
        {"label": "THRESHOLD", "pattern": "carbon neutral"},
        # ── QUANTITY (standard) — gold text 1 uses QUANTITY not THRESHOLD
        {"label": "QUANTITY", "pattern": "1.5 degrees Celsius"},
        {"label": "QUANTITY", "pattern": "190 nations"},
        {"label": "QUANTITY", "pattern": "735 million people"},
        {"label": "QUANTITY", "pattern": "31%"},
        # ── ORG (standard label) ──────────────────────────────────────
        # COP28, COP27 etc — gold texts 2, 10 use ORG
        {
            "label": "ORG",
            "pattern": [{"TEXT": {"REGEX": "^COP\d+$"}}],
        },
        {"label": "ORG", "pattern": "Green Climate Fund"},
        {"label": "ORG", "pattern": "United Nations General Assembly"},
        {"label": "ORG", "pattern": "Ministry of Environment"},
        {"label": "ORG", "pattern": "European Union"},
        {"label": "ORG", "pattern": "World Bank"},
        # ── GPE (standard label) — HF labeled all as LOC ──────────────
        {"label": "GPE", "pattern": "Dubai"},
        {"label": "GPE", "pattern": "United Arab Emirates"},
        {"label": "GPE", "pattern": "New York"},
        {"label": "GPE", "pattern": "South Korea"},
        {"label": "GPE", "pattern": "Bangladesh"},
        {"label": "GPE", "pattern": "Songdo"},
        {"label": "GPE", "pattern": "California"},
        {"label": "GPE", "pattern": "Jordan"},
        {"label": "GPE", "pattern": "China"},
        # ── PERSON (standard label) — HF broke with subwords ──────────
        {"label": "PERSON", "pattern": "Antonio Guterres"},
        {"label": "PERSON", "pattern": "Ajay Banga"},
        {"label": "PERSON", "pattern": "Simon Stiell"},
        {"label": "PERSON", "pattern": "Qu Dongyu"},
        {"label": "PERSON", "pattern": "John Kerry"},
        {"label": "PERSON", "pattern": "Xie Zhenhua"},
        # ── MONEY (standard label) ────────────────────────────────────
        {"label": "MONEY", "pattern": "$12.5 billion"},
        {"label": "MONEY", "pattern": "$4 billion"},
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
