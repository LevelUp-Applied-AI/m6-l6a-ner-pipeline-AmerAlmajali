# Stretch 6B-S2: Cross-Lingual Embedding Analysis
## bert-base-multilingual-cased | English–Arabic Climate Texts

---

## Paragraph 1: How well does the multilingual model capture cross-lingual similarity?

The multilingual BERT model (bert-base-multilingual-cased) demonstrates a clear and 
consistent ability to capture semantic similarity across English and Arabic climate texts. 
Across all 10 same-topic pairs, the diagonal similarity scores ranged from 0.6504 to 0.7375, 
with a mean of 0.7003. The strongest cross-lingual pairs were EN(1)↔AR(79) on the IPCC Sixth 
Assessment Report (0.7375), EN(55)↔AR(93) on Global Carbon Project emissions (0.7300), and 
EN(21)↔AR(92) on the Copernicus September 2023 temperature record (0.7283) — all texts that 
share precise numerical facts and named entities (e.g., "1.5°C", "36.8 billion tonnes", 
"1.75°C") that the model's shared WordPiece vocabulary encodes consistently across languages. 
The weakest pair was EN(2)↔AR(81) on COP28 (0.6504), likely because the Arabic text 
emphasizes loss and damage compensation while the English text focuses on the fossil fuel 
transition agreement — same event, different angles. Critically, the mean same-topic score 
(0.7003) was notably higher than the mean off-topic score (0.6053), producing a gap of 0.0950. 
This gap confirms that the ranking is preserved: same-topic cross-lingual pairs consistently 
score higher than random cross-lingual pairs, which is the key requirement for bilingual 
retrieval and search applications.

---

## Paragraph 2: What does this mean for building bilingual NLP tools in the MENA region?

The results carry practical implications for deploying NLP systems across Arabic and English 
in the MENA region. The 0.0950 gap between same-topic and off-topic pairs shows that a single 
mBERT model can power bilingual semantic search — for example, an Arabic query about CO2 
concentrations would correctly retrieve its English counterpart above unrelated English 
documents. However, the comparison with DistilBERT reveals a meaningful trade-off: 
DistilBERT's within-English similarity mean was 0.8570 versus mBERT's 0.7405, a difference 
of 0.1166. This confirms that mBERT sacrifices some English-specific embedding quality in 
exchange for cross-lingual capability, as its 177M parameters are distributed across 104 
languages rather than optimized for English alone. For MENA deployment, this means that 
organizations running Arabic-only or English-only pipelines would benefit from monolingual 
models, while those building bilingual search engines, cross-lingual document classification, 
or Arabic-English retrieval systems should use mBERT or its stronger successor 
(XLM-RoBERTa). The Jordanian context is particularly relevant: with Arabic-language climate 
policy documents (e.g., Jordan's Renewable Energy Law, the Azraq refugee camp solar reports) 
needing to be retrieved alongside English IPCC and World Bank sources, a single mBERT-based 
embedding pipeline offers a practical path to unified bilingual infrastructure without 
maintaining two separate models.

---

## Key Numbers Summary

| Metric | Value |
|--------|-------|
| Same-topic cross-lingual mean | 0.7003 |
| Off-topic cross-lingual mean | 0.6053 |
| Gap (same - off) | **0.0950** |
| mBERT within-English mean | 0.7405 |
| DistilBERT within-English mean | 0.8570 |
| English quality cost of multilingualism | **0.1166** |
| Highest cross-lingual pair | EN(1)↔AR(79) IPCC = 0.7375 |
| Lowest cross-lingual pair | EN(2)↔AR(81) COP28 = 0.6504 |