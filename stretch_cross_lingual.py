# ============================================================
# Stretch 6B-S2: Cross-Lingual Embedding Comparison
# ============================================================

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


import pandas as pd


def load_and_filter_data(csv_path: str):
    df = pd.read_csv(csv_path)

    print("=== Dataset Overview ===")
    print(f"Total rows : {len(df)}")
    print(f"Columns    : {list(df.columns)}")
    print(f"\nLanguage counts:\n{df['language'].value_counts()}\n")

    en_df = df[df["language"] == "en"].reset_index(drop=True)
    ar_df = df[df["language"] == "ar"].reset_index(drop=True)

    # print("=== English articles (id | category | first 60 chars) ===")
    # for i, row in en_df.iterrows():
    # print(f"  [{i}] id={row['id']} | {row['category']} | {row['text'][:60]}...")

    # print("\n=== Arabic articles (id | category | first 60 chars) ===")
    # for i, row in ar_df.iterrows():
    # print(f"  [{i}] id={row['id']} | {row['category']} | {row['text'][:60]}...")

    return en_df, ar_df


en_df, ar_df = load_and_filter_data("data/climate_articles.csv")


def select_pairs(en_df, ar_df):
    en_ids = [1, 2, 3, 9, 11, 12, 16, 18, 21, 55]
    ar_ids = [79, 81, 80, 119, 86, 87, 88, 91, 92, 93]

    en_selected = (
        en_df[en_df["id"].isin(en_ids)].set_index("id").loc[en_ids].reset_index()
    )
    ar_selected = (
        ar_df[ar_df["id"].isin(ar_ids)].set_index("id").loc[ar_ids].reset_index()
    )

    print("=== Selected Pairs ===")
    for i in range(10):
        print(f"\nPair [{i}]")
        print(
            f"  EN id={en_selected.loc[i,'id']} | {en_selected.loc[i,'text'][:70]}..."
        )
        print(
            f"  AR id={ar_selected.loc[i,'id']} | {ar_selected.loc[i,'text'][:70]}..."
        )

    en_texts = en_selected["text"].tolist()
    ar_texts = ar_selected["text"].tolist()

    return en_texts, ar_texts, en_selected, ar_selected


en_texts, ar_texts, en_selected, ar_selected = select_pairs(en_df, ar_df)
print("\n✅ Step 2 done")


def load_model_and_tokenizer(model_name="bert-base-multilingual-cased"):
    print(f"=== Step 3: Loading model: {model_name} ===")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model (~680MB, please wait)...")
    model = AutoModel.from_pretrained(model_name)

    model.eval()

    print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return tokenizer, model


tokenizer, model = load_model_and_tokenizer()

print("\n✅ Step 3 done")


def get_embedding(text, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling over last hidden states
    token_embeddings = outputs.last_hidden_state  # (1, seq_len, 768)
    attention_mask = inputs["attention_mask"]  # (1, seq_len)
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    embedding = (sum_embeddings / sum_mask).squeeze(0)

    return embedding.numpy()


# Test on first pair
emb = get_embedding(en_texts[0], tokenizer, model)
print(f"✅ Embedding shape : {emb.shape}")
print(f"✅ First 5 values  : {emb[:5]}")


def extract_all_embeddings(texts, tokenizer, model, label):
    print(f"=== Extracting {label} embeddings ===")
    embeddings = []
    for i, text in enumerate(texts):
        emb = get_embedding(text, tokenizer, model)
        embeddings.append(emb)
        print(f"  [{i+1}/10] done — shape: {emb.shape}")

    matrix = np.vstack(embeddings)
    print(f"✅ {label} matrix shape: {matrix.shape}\n")
    return matrix


en_embeddings = extract_all_embeddings(en_texts, tokenizer, model, "English")
ar_embeddings = extract_all_embeddings(ar_texts, tokenizer, model, "Arabic")


def compute_similarity_matrix(en_embeddings, ar_embeddings):
    print("=== Step 6: Compute Similarity Matrix ===")

    sim_matrix = cosine_similarity(en_embeddings, ar_embeddings)

    print(f"Matrix shape : {sim_matrix.shape}")
    print(f"Max  : {sim_matrix.max():.4f}")
    print(f"Min  : {sim_matrix.min():.4f}")
    print(f"Mean : {sim_matrix.mean():.4f}")

    print("\n--- Diagonal (same-topic pairs) ---")
    for i in range(10):
        print(
            f"  Pair [{i}] EN_id={en_selected.loc[i,'id']} <-> AR_id={ar_selected.loc[i,'id']} | score = {sim_matrix[i][i]:.4f}"
        )

    return sim_matrix


sim_matrix = compute_similarity_matrix(en_embeddings, ar_embeddings)
print("\n✅ Step 6 done")


def plot_heatmap(
    sim_matrix, en_selected, ar_selected, output_path="cross_lingual_heatmap.png"
):
    print("=== Step 7: Plot Heatmap ===")

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

    # Highlight diagonal with blue box
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


plot_heatmap(sim_matrix, en_selected, ar_selected)
print("\n✅ Step 7 done")


def analyze_results(sim_matrix, en_selected, ar_selected):
    print("=== Step 8: Analysis ===")

    diagonal = [sim_matrix[i][i] for i in range(10)]
    off_diagonal = [sim_matrix[i][j] for i in range(10) for j in range(10) if i != j]

    print(f"--- Same-topic pairs (diagonal) ---")
    for i in range(10):
        en_id = en_selected.loc[i, "id"]
        ar_id = ar_selected.loc[i, "id"]
        print(
            f"  Pair[{i}] EN({en_id}) <-> AR({ar_id}) | score = {sim_matrix[i][i]:.4f}"
        )

    print(f"\n--- Summary ---")
    print(f"  Same-topic  mean : {np.mean(diagonal):.4f}")
    print(f"  Same-topic  std  : {np.std(diagonal):.4f}")
    print(f"  Off-topic   mean : {np.mean(off_diagonal):.4f}")
    print(f"  Off-topic   std  : {np.std(off_diagonal):.4f}")
    print(f"  Gap (same - off) : {np.mean(diagonal) - np.mean(off_diagonal):.4f}")

    print(f"\n--- Top 3 highest same-topic pairs ---")
    top3 = sorted(enumerate(diagonal), key=lambda x: x[1], reverse=True)[:3]
    for i, score in top3:
        print(f"  Pair[{i}] score={score:.4f}")
        print(f"    EN: {en_selected.loc[i,'text'][:80]}...")
        print(f"    AR: {ar_selected.loc[i,'text'][:80]}...")

    print(f"\n--- Lowest same-topic pair ---")
    i, score = sorted(enumerate(diagonal), key=lambda x: x[1])[0]
    print(f"  Pair[{i}] score={score:.4f}")
    print(f"    EN: {en_selected.loc[i,'text'][:80]}...")
    print(f"    AR: {ar_selected.loc[i,'text'][:80]}...")

    return diagonal, off_diagonal


diagonal, off_diagonal = analyze_results(sim_matrix, en_selected, ar_selected)
print("\n✅ Step 8 done")
