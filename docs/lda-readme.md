# Latent Dirichlet Allocation (LDA) Topic Modeling on Iran Protest Dataset

This notebook performs Latent Dirichlet Allocation (LDA) topic modeling on protest notes from the ACLED dataset for Iran. The goal is to identify coherent and distinct topics underlying protest narratives and evaluate the model's performance through coherence and topic diversity metrics.

## Objective

To discover and interpret themes in protest narratives using LDA by tuning model hyperparameters (alpha and eta) and visualizing the results through word clouds, word frequency plots, document-topic distributions, and interactive PyLDAvis maps.

## Model Summary 

**Model Used:** `gensim.models.LdaModel`  
**Preprocessing Tools:** `NLTK`, `gensim.utils.simple_preprocess`  
**Evaluation Metrics:**
- Coherence Score (`c_v`)
- Topic Diversity

**Optimization:** Grid Search on:
- Number of topics: 5
- Alpha: `'symmetric'`, `'asymmetric'`, `0.01`, `0.05`, `0.1`, `0.5`, `'auto'`
- Eta: `'symmetric'`, `0.01`, `0.05`, `0.1`, `0.5`, `'auto'`

## Methodology

1. **Data Loading:** Load the ACLED protest dataset for Iran using `pandas`, selecting the `notes` column.

2. **Text Preprocessing:**
-  Convert notes to lowercase to normalize casing.
- Tokenize using `simple_preprocess`,  which removes punctuation and splits sentences into word tokens.
- Remove standard and domain-specific stopwords (e.g., month names, "plan", "building") which appear frequently but didn’t add topic-specific value.
- Remove short words (length ≤ 2) to reduce noise.
- The resulting output is a list of tokenized and cleaned words for each protest note.

3. **Exploratory Visualization:** Generate a word cloud using token frequencies to verify that preprocessing removed noisy/common terms and retained relevant keywords.

4. **Dictionary and Corpus Creation:**
- Build a dictionary using `corpora.Dictionary`.
- Filter extremes: remove tokens in <5 docs or >30% of docs to remove rare and overly common words to reduce noise.
- Transform each document into a bag-of-words representation using doc2bow.

5. **Model Optimization and Grid Search:**
   1. Initially, the model is run with topic numbers ranging from 5 to 10 to evaluate performance across different granularities. Based on coherence scores, the best performance is observed with 5 topics, which is then used for the final grid search over alpha and eta values.
   2. Loop through combinations of `alpha` and `eta` values.
   3. Train each model using:
    - passes=100: ensures sufficient training
    - update_every=0: batch learning
    - chunksize=5000: controls how many documents are processed at a time
   4. Evaluate each using coherence (`c_v coherence metric`) and select the best.

6. **Topic Diversity Evaluation:** For the best model, compute topic diversity using the top 10 words for each topic.

7.  **Visualization:**
   - Top 10 words per topic (bar charts).
   - Document-topic match table (10 samples).
   - Topic probability distribution (2 documents).
   - PyLDAvis interactive topic visualization (HTML export).

## Evaluation Metrics:
Two keys metrics are used to evaluate the performance of the LDA model:

### 1. C_V Coherence Score: 
The **C_V coherence metric** evaluates how semantically consistent the top words within a topic are. It combines several steps:
- Uses a sliding window to compute co-occurrence between topic words.
- Builds a context vector for each word using co-occurrence counts.
- Measures cosine similarity between word pairs based on these vectors.
-  Averages these similarities across all top word pairs in each topic.

*Intuition:* If the top words in a topic tend to co-occur in similar contexts, the topic is considered coherent.

**Formula:**
```
CV_Coherence = (1 / (N choose 2)) * Σ cosine_similarity(v_wi, v_wj)
```
Where:
- `v_wi` and  `v_wj` are the context vectors of word `wi` and `wj`
- `N` is the number of top words per topic

**Tool:** Implemented in `gensim.models.CoherenceModel` with `coherence='c_v'`  
**Scale:** Ranges from 0 to 1 — higher is better

### 2. Topic Diversity

**Topic Diversity** measures how unique the top words are across all topics. It checks whether different topics are using the same words repeatedly, which can indicate redundancy.

**Formula:**
```
Topic Diversity = |Unique Top Words| / (k × T)
```

Where:
- `k` = number of top words per topic (e.g., 10)
- `T` = total number of topics
- Numerator is the number of unique words found in the top k words across all topics.

**Scale:**
- 1.0 means all topics have completely distinct top words
- Lower values indicate overlap and less informative separation between topics

**Implementation:**  
Calculated by extracting the top `k` words from each topic and measuring the uniqueness ratio.

Using both metrics gives a balanced evaluation:
- **C_V Coherence score** checks how meaningful each topic is internally.
- **Topic Diversity score** checks whether topics are distinct from each other.

## Files
- `protests-classification-lda.ipynb` — Main notebook for LDA modeling.
- `lda_topics_visualization.html` — PyLDAvis interactive visualization.
- `lda-readme.md` — This documentation file.


## 🔁 How to Reproduce

```bash
1. Ensure Python 3.8+ environment
# 2. Install dependencies:
pip install pandas gensim nltk matplotlib pyLDAvis wordcloud
# 3. Run notebook:
jupyter notebook protests-classification-lda.ipynb
