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

