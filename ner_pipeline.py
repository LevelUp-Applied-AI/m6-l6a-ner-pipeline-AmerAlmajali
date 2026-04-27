"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import pandas as pd
import numpy as np
import spacy
import unicodedata
from transformers import pipeline as hf_pipeline


def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    df = pd.read_csv(filepath)

    return df
    # TODO: Load the CSV and return the DataFrame
    pass


def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    word_counts = df["text"].apply(lambda t: len(t.split()))
    return {
        "shape": df.shape,
        "lang_counts": df["language"].value_counts().to_dict(),
        "category_counts": df["category"].value_counts().to_dict(),
        "text_length_stats": {
            "mean": float(word_counts.mean()),
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
        },
    }
    # TODO: Compute shape, language/category value_counts, and word-count
    #       statistics on df['text']
    pass


def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
    normalized = unicodedata.normalize("NFC", text)
    doc = nlp(normalized)
    results = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        lemma = token.lemma_.lower()
        results.append(lemma)
    return results

    # TODO: NFC-normalize the text, run it through nlp(), drop
    #       punctuation/whitespace tokens, return lowercased lemmas
    pass


def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER.

    Args:
        df: DataFrame with columns id, text, language, ...
        nlp: A loaded spaCy Language object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
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

    # TODO: Filter df to English rows, process each text with nlp,
    #       collect entities into rows, return as a DataFrame
    pass


def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face `pipeline('ner', ...)` object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    english_df = df[df["language"] == "en"]

    rows = []

    for _, row in english_df.iterrows():
        text = row["text"]
        ner_results = ner_pipeline(text)

        merged = []
        current = None

        for token in ner_results:
            if token["entity"].startswith("B-"):
                if current:
                    merged.append(current)
                current = {
                    "entity": token["entity"][2:],  # remove B-
                    "word": token["word"],
                    "start": token["start"],
                    "end": token["end"],
                    "score": token["score"],
                }

            elif token["entity"].startswith("I-") and current:

                word_piece = token["word"].replace("##", "")
                current["word"] += word_piece
                current["end"] = token["end"]
                current["score"] = min(current["score"], token["score"])

            else:
                if current:
                    merged.append(current)
                    current = None

        if current:
            merged.append(current)

        for ent in merged:
            rows.append(
                {
                    "text_id": row["id"],
                    "entity_text": ent["word"],
                    "entity_label": ent["entity"],
                    "start_char": ent["start"],
                    "end_char": ent["end"],
                }
            )

    return pd.DataFrame(
        rows,
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"],
    )
    # TODO: Filter df to English rows, run each text through
    #       ner_pipeline, merge ## subword tokens, strip B-/I- prefix
    #       from labels (IOB format), return as a DataFrame
    pass


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    spacy_counts = spacy_df["entity_label"].value_counts().to_dict()
    hf_counts = hf_df["entity_label"].value_counts().to_dict()

    spacy_set = set(zip(spacy_df["text_id"], spacy_df["entity_text"]))
    hf_set = set(zip(hf_df["text_id"], hf_df["entity_text"]))

    both = spacy_set & hf_set
    spacy_only = spacy_set - hf_set
    hf_only = hf_set - spacy_set

    return {
        "spacy_counts": spacy_counts,
        "hf_counts": hf_counts,
        "total_spacy": len(spacy_df),
        "total_hf": len(hf_df),
        "both": both,
        "spacy_only": spacy_only,
        "hf_only": hf_only,
    }
    # TODO: Count entities per label for each system, compute totals,
    #       and derive the three overlap sets by matching on
    #       (text_id, entity_text)
    pass


def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    Computes entity-level precision, recall, and F1. An entity is a
    true positive if both the entity text and label match a gold entry
    for the same text_id.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0-1).
    """
    predicted_set = set(
        zip(
            predicted_df["text_id"],
            predicted_df["entity_text"].str.lower(),
            predicted_df["entity_label"],
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

    return {"precision": precision, "recall": recall, "f1": f1}
    # TODO: Match predicted entities to gold entities by text_id +
    #       entity_text + entity_label, compute precision/recall/F1
    pass


def detailed_error_report(predicted_df, gold_df, system_name):
    """
    Print a detailed breakdown of TP, FP (spurious), and FN (missed)
    entities compared to the gold standard.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text, entity_label.
        gold_df:      DataFrame with columns text_id, entity_text, entity_label.
        system_name:  String label for display (e.g. 'spaCy' or 'HF').
    """
    predicted_set = set(
        zip(
            predicted_df["text_id"],
            predicted_df["entity_text"].str.lower(),
            predicted_df["entity_label"],
        )
    )
    gold_set = set(
        zip(
            gold_df["text_id"],
            gold_df["entity_text"].str.lower(),
            gold_df["entity_label"],
        )
    )

    tp_set = predicted_set & gold_set
    fp_set = predicted_set - gold_set  # predicted but not in gold  (spurious)
    fn_set = gold_set - predicted_set  # in gold but not predicted  (missed)

    print(f"\n{'='*60}")
    print(f"  {system_name} — Detailed Error Report")
    print(f"{'='*60}")
    print(f"  True Positives  (correct):  {len(tp_set)}")
    print(f"  False Positives (spurious): {len(fp_set)}")
    print(f"  False Negatives (missed):   {len(fn_set)}")

    # ── Correctly found ──────────────────────────────────────────
    print(f"\n--- Correctly Found (TP) ---")
    tp_df = pd.DataFrame(
        sorted(tp_set), columns=["text_id", "entity_text", "entity_label"]
    )
    print(tp_df.to_string(index=False))

    # ── Missed by the system ─────────────────────────────────────
    print(f"\n--- Missed by {system_name} (FN) — in gold but not predicted ---")
    fn_df = pd.DataFrame(
        sorted(fn_set), columns=["text_id", "entity_text", "entity_label"]
    )
    if fn_df.empty:
        print("  (none)")
    else:
        # group by label so the pattern is obvious
        for label, group in fn_df.groupby("entity_label"):
            print(f"\n  [{label}]")
            print(group[["text_id", "entity_text"]].to_string(index=False))

    # ── Spurious predictions ─────────────────────────────────────
    print(f"\n--- Spurious by {system_name} (FP) — predicted but not in gold ---")
    fp_df = pd.DataFrame(
        sorted(fp_set), columns=["text_id", "entity_text", "entity_label"]
    )
    if fp_df.empty:
        print("  (none)")
    else:
        for label, group in fp_df.groupby("entity_label"):
            print(f"\n  [{label}]")
            print(group[["text_id", "entity_text"]].to_string(index=False))


if __name__ == "__main__":
    # Load spaCy and HF models once, reuse across functions
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    # Load and explore
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")

        # Preprocess a sample to verify your function
        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        # spaCy NER across the English corpus
        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")

        # HF NER across the English corpus
        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")

        # Compare the two systems
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")

        # Evaluate against gold standard
        gold = pd.read_csv("data/gold_entities.csv")

        print(
            f"\nGold standard: {len(gold)} entities across "
            f"{gold['text_id'].nunique()} texts"
        )
        print(f"Gold label breakdown:\n{gold['entity_label'].value_counts()}")

        if spacy_entities is not None:
            spacy_metrics = evaluate_ner(spacy_entities, gold)
            print(f"\nspaCy evaluation: {spacy_metrics}")
            detailed_error_report(spacy_entities, gold, "spaCy")

        if hf_entities is not None:
            hf_metrics = evaluate_ner(hf_entities, gold)
            print(f"\nHF evaluation: {hf_metrics}")
            detailed_error_report(hf_entities, gold, "HF")

        # ── Side-by-side summary ──────────────────────────────────
        if spacy_entities is not None and hf_entities is not None:
            print(f"\n{'='*60}")
            print("  Final Metrics Comparison")
            print(f"{'='*60}")
            print(f"  {'Metric':<12} {'spaCy':>10} {'HF':>10}")
            print(f"  {'-'*34}")
            for key in ["precision", "recall", "f1"]:
                print(
                    f"  {key:<12} {spacy_metrics[key]:>10.3f} {hf_metrics[key]:>10.3f}"
                )
