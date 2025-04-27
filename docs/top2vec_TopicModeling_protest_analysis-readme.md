# 🧠 Top2Vec Protest Topic Analysis

This notebook applies **Top2Vec** to discover latent topics from the Iran protest notes dataset.

## Objective

To automatically learn semantically coherent protest topics without needing to pre-specify the number of topics, by using document and word embeddings.

## Model Summary

**Model Used:** `top2vec.Top2Vec`  
**Embedding Models:** `doc2vec`, `universal-sentence-encoder`  
**Evaluation Metrics:**
- Coherence Score (`c_v`)
- Topic Diversity

## Methodology

1. **Data Loading:** Load protest notes from the ACLED dataset.
2. **Text Preprocessing:** Lowercasing, tokenization, stopword removal, and cleaning.
3. **Embedding Generation:** Use pretrained embedding models to jointly learn document-word embeddings.
4. **Dimensionality Reduction:** Use UMAP to lower embedding dimensions.
5. **Clustering:** Use HDBSCAN to detect dense clusters as topics.
6. **Model Evaluation:** Evaluate using coherence scores and topic diversity.

## Evaluation Metrics

### 1. C_V Coherence Score
Measures semantic consistency among top words in each topic.

$$
\text{C}_V = \frac{1}{|W|^2} \sum_{i,j} \text{NPMI}(w_i, w_j) \times \text{Sim}(w_i, w_j)
$$

### 2. Topic Diversity
Measures uniqueness across topic keywords.

$$
\text{Topic Diversity} = \frac{\text{Unique Top Words}}{k \times T}
$$

## Files

- `notebooks/top2vec_protest_analysis.ipynb` — Main notebook for Top2Vec modeling.
- `docs/top2vec-readme.md` — This documentation file.

## 🔁 How to Reproduce

```bash
pip install top2vec pandas nltk matplotlib wordcloud
jupyter notebook notebooks/top2vec_protest_analysis.ipynb
```