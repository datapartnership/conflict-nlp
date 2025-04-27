# 🦙 BERTopic + LLaMA 2 Protest Topic Analysis

This notebook applies **BERTopic** and **LLaMA 2** to identify, cluster, and label topics from Iran protest narratives.

## Objective

To extract protest topics using BERTopic's clustering mechanism and improve topic interpretability by using Few-Shot Prompting with LLaMA 2 for label generation.

## Model Summary

**Model Used:** `bertopic.BERTopic` + `meta-llama/Llama-2-7b-chat-hf`  
**Embedding Model:** `all-mpnet-base-v2`  
**Clustering Models:** `UMAP`, `HDBSCAN`  
**Evaluation Metrics:**
- Coherence Score (`c_v`)
- Topic Diversity

## Methodology

1. **Data Loading:** Load ACLED protest dataset.
2. **Text Preprocessing:** Cleaning, stopword removal, tokenization.
3. **Sentence Embeddings:** Generate high-dimensional document embeddings.
4. **Dimensionality Reduction:** Use UMAP to project embeddings.
5. **Clustering:** Detect topic clusters using HDBSCAN.
6. **Topic Representation:** Generate topic keywords and representations.
7. **Few-Shot Prompting with LLaMA 2:** Use few-shot examples to label topics.
8. **Model Evaluation:** Coherence and diversity evaluation.

## Evaluation Metrics

### 1. C_V Coherence Score
Semantic closeness among top topic keywords.

$$
\text{C}_V = \frac{1}{|W|^2} \sum_{i,j} \text{NPMI}(w_i, w_j) \times \text{Sim}(w_i, w_j)
$$

### 2. Topic Diversity
Distinctiveness among different topic word sets.

$$
\text{Topic Diversity} = \frac{\text{Unique Top Words}}{k \times T}
$$

## Files

- `notebooks/llama2_bertopic_analysis.ipynb` — Main notebook for BERTopic + LLaMA 2 modeling.
- `docs/bertopic-readme.md` — This documentation file.

## 🔁 How to Reproduce

```bash
pip install bertopic umap-learn hdbscan pandas sentence-transformers transformers torch
jupyter notebook notebooks/llama2_bertopic_analysis.ipynb
```