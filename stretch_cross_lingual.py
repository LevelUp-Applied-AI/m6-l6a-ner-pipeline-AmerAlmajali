# ============================================================
# Stretch 6B-S2: Cross-Lingual Embedding Comparison
# + DistilBERT vs mBERT English quality comparison
# ============================================================

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


# ── Step 1: Load and filter data ─────────────────────────────────────────────
def load_and_filter_data(csv_path: str):
    df = pd.read_csv(csv_path)
    print("=== Dataset Overview ===")
    print(f"Total rows : {len(df)}")
    print(f"Language counts:\n{df['language'].value_counts()}\n")
    en_df = df[df["language"] == "en"].reset_index(drop=True)
    ar_df = df[df["language"] == "ar"].reset_index(drop=True)
    # print("=== English articles (id | category | first 60 chars) ===")
    # for i, row in en_df.iterrows():
    #     print(f"  [{i}] id={row['id']} | {row['category']} | {row['text'][:60]}...")

    # print("\n=== Arabic articles (id | category | first 60 chars) ===")
    # for i, row in ar_df.iterrows():
    #     print(f"  [{i}] id={row['id']} | {row['category']} | {row['text'][:60]}...")
    return en_df, ar_df


# ── Step 2: Select matched pairs ─────────────────────────────────────────────
def select_pairs(en_df, ar_df):
    en_ids = [1, 2, 3, 9, 11, 12, 16, 18, 21, 55]
    ar_ids = [79, 81, 80, 119, 86, 87, 88, 91, 92, 93]
    en_selected = (
        en_df[en_df["id"].isin(en_ids)].set_index("id").loc[en_ids].reset_index()
    )
    ar_selected = (
        ar_df[ar_df["id"].isin(ar_ids)].set_index("id").loc[ar_ids].reset_index()
    )
    en_texts = en_selected["text"].tolist()
    ar_texts = ar_selected["text"].tolist()
    return en_texts, ar_texts, en_selected, ar_selected


# ── Step 3: Get single embedding (mean pooling) ───────────────────────────────
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    embedding = (sum_embeddings / sum_mask).squeeze(0)
    return embedding.numpy()


# ── Step 4: Extract all embeddings ───────────────────────────────────────────
def extract_all_embeddings(texts, tokenizer, model, label):
    print(f"=== Extracting {label} embeddings ===")
    embeddings = []
    for i, text in enumerate(texts):
        emb = get_embedding(text, tokenizer, model)
        embeddings.append(emb)
        print(f"  [{i+1}/{len(texts)}] done — shape: {emb.shape}")
    matrix = np.vstack(embeddings)
    print(f"✅ {label} matrix shape: {matrix.shape}\n")
    return matrix


# ── Step 5: Compute cross-lingual similarity matrix ──────────────────────────
def compute_similarity_matrix(en_embeddings, ar_embeddings, en_selected, ar_selected):
    print("=== Compute Cross-Lingual Similarity Matrix ===")
    sim_matrix = cosine_similarity(en_embeddings, ar_embeddings)
    print(f"Matrix shape : {sim_matrix.shape}")
    print(
        f"Max  : {sim_matrix.max():.4f} | Min  : {sim_matrix.min():.4f} | Mean : {sim_matrix.mean():.4f}"
    )
    print("\n--- Diagonal (same-topic pairs) ---")
    for i in range(10):
        print(
            f"  Pair[{i}] EN({en_selected.loc[i,'id']}) <-> AR({ar_selected.loc[i,'id']}) | score = {sim_matrix[i][i]:.4f}"
        )
    return sim_matrix


# ── Step 6: Plot cross-lingual heatmap ───────────────────────────────────────
def plot_heatmap(
    sim_matrix, en_selected, ar_selected, output_path="cross_lingual_heatmap.png"
):
    print("=== Plot Heatmap ===")
    en_labels = [
        f"EN({row['id']}): {row['text'][:35]}" for i, row in en_selected.iterrows()
    ]
    ar_labels = [
        f"AR({row['id']}): {row['text'][:35]}" for i, row in ar_selected.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(18, 12))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0.4,
        vmax=0.8,
        xticklabels=ar_labels,
        yticklabels=en_labels,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 8},
        ax=ax,
    )
    for i in range(10):
        ax.add_patch(
            plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="blue", linewidth=2.5)
        )
    ax.set_title(
        "Cross-Lingual Embedding Similarity\nbert-base-multilingual-cased | EN (rows) vs AR (cols)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Arabic Texts", fontsize=11)
    ax.set_ylabel("English Texts", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Heatmap saved: {output_path}")


# ── Step 7: Analyze cross-lingual results ────────────────────────────────────
def analyze_results(sim_matrix, en_selected, ar_selected):
    print("=== Cross-Lingual Analysis ===")
    diagonal = [sim_matrix[i][i] for i in range(10)]
    off_diagonal = [sim_matrix[i][j] for i in range(10) for j in range(10) if i != j]
    for i in range(10):
        print(
            f"  Pair[{i}] EN({en_selected.loc[i,'id']}) <-> AR({ar_selected.loc[i,'id']}) | score = {sim_matrix[i][i]:.4f}"
        )
    print(
        f"\n  Same-topic  mean : {np.mean(diagonal):.4f} | std : {np.std(diagonal):.4f}"
    )
    print(
        f"  Off-topic   mean : {np.mean(off_diagonal):.4f} | std : {np.std(off_diagonal):.4f}"
    )
    print(f"  Gap (same - off) : {np.mean(diagonal) - np.mean(off_diagonal):.4f}")
    return diagonal, off_diagonal


# ── Step 8: Compare mBERT vs DistilBERT on English texts ─────────────────────
def compare_mbert_vs_distilbert(
    en_texts, mbert_tokenizer, mbert_model, distil_tokenizer, distil_model, en_selected
):
    print("=== Step 8: mBERT vs DistilBERT English Quality Comparison ===")

    print("\n--- Extracting with mBERT ---")
    mbert_en_emb = extract_all_embeddings(
        en_texts, mbert_tokenizer, mbert_model, "mBERT-EN"
    )

    print("\n--- Extracting with DistilBERT ---")
    distil_en_emb = extract_all_embeddings(
        en_texts, distil_tokenizer, distil_model, "DistilBERT-EN"
    )

    mbert_sim = cosine_similarity(mbert_en_emb, mbert_en_emb)
    distil_sim = cosine_similarity(distil_en_emb, distil_en_emb)

    mbert_off = [mbert_sim[i][j] for i in range(10) for j in range(10) if i != j]
    distil_off = [distil_sim[i][j] for i in range(10) for j in range(10) if i != j]

    print("\n--- Within-English Similarity (off-diagonal) ---")
    print(f"  mBERT      mean: {np.mean(mbert_off):.4f} | std: {np.std(mbert_off):.4f}")
    print(
        f"  DistilBERT mean: {np.mean(distil_off):.4f} | std: {np.std(distil_off):.4f}"
    )
    print(
        f"  Difference (Distil - mBERT): {np.mean(distil_off) - np.mean(mbert_off):.4f}"
    )

    en_labels = [
        f"EN({row['id']}): {row['text'][:25]}" for i, row in en_selected.iterrows()
    ]

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    sns.heatmap(
        mbert_sim,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.4,
        vmax=1.0,
        xticklabels=en_labels,
        yticklabels=en_labels,
        linewidths=0.5,
        annot_kws={"size": 7},
        ax=axes[0],
    )
    axes[0].set_title(
        "mBERT\nWithin-English Similarity", fontsize=12, fontweight="bold"
    )
    axes[0].tick_params(axis="x", rotation=45, labelsize=7)
    axes[0].tick_params(axis="y", rotation=0, labelsize=7)

    sns.heatmap(
        distil_sim,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.4,
        vmax=1.0,
        xticklabels=en_labels,
        yticklabels=en_labels,
        linewidths=0.5,
        annot_kws={"size": 7},
        ax=axes[1],
    )
    axes[1].set_title(
        "DistilBERT\nWithin-English Similarity", fontsize=12, fontweight="bold"
    )
    axes[1].tick_params(axis="x", rotation=45, labelsize=7)
    axes[1].tick_params(axis="y", rotation=0, labelsize=7)

    plt.suptitle(
        "mBERT vs DistilBERT — English Embedding Quality",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig("mbert_vs_distilbert.png", dpi=150, bbox_inches="tight")
    print("✅ Comparison plot saved: mbert_vs_distilbert.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load models once
    MBERT_NAME = "bert-base-multilingual-cased"
    DISTIL_NAME = "distilbert-base-uncased"

    mbert_tokenizer = AutoTokenizer.from_pretrained(MBERT_NAME)
    mbert_model = AutoModel.from_pretrained(MBERT_NAME)
    mbert_model.eval()
    print(
        f"✅ mBERT loaded | Parameters: {sum(p.numel() for p in mbert_model.parameters()):,}"
    )

    distil_tokenizer = AutoTokenizer.from_pretrained(DISTIL_NAME)
    distil_model = AutoModel.from_pretrained(DISTIL_NAME)
    distil_model.eval()
    print(
        f"✅ DistilBERT loaded | Parameters: {sum(p.numel() for p in distil_model.parameters()):,}"
    )

    # Step 1
    en_df, ar_df = load_and_filter_data("data/climate_articles.csv")

    # Step 2
    en_texts, ar_texts, en_selected, ar_selected = select_pairs(en_df, ar_df)

    # Steps 3 & 4
    en_embeddings = extract_all_embeddings(
        en_texts, mbert_tokenizer, mbert_model, "English-mBERT"
    )
    ar_embeddings = extract_all_embeddings(
        ar_texts, mbert_tokenizer, mbert_model, "Arabic-mBERT"
    )

    # Step 5
    sim_matrix = compute_similarity_matrix(
        en_embeddings, ar_embeddings, en_selected, ar_selected
    )

    # Step 6
    plot_heatmap(sim_matrix, en_selected, ar_selected)

    # Step 7
    diagonal, off_diagonal = analyze_results(sim_matrix, en_selected, ar_selected)

    # Step 8
    compare_mbert_vs_distilbert(
        en_texts,
        mbert_tokenizer,
        mbert_model,
        distil_tokenizer,
        distil_model,
        en_selected,
    )

    print("\n All steps complete!")
