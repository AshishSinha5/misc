# Author Indentification via NLP Techniques

This project aims to explore various text classification techniques in NLP literature. Here I apply suit of NLP algorithms to solve the problem of Author Identification using exerpts from their literary works. 

## Dataset

Data is taken from Kaggle's [Spooky Author Idenfication Challenge](https://www.kaggle.com/c/spooky-author-identification/). It containes the excerpts from the literary works od three authors - 
  - [Edgar Allen Poe](https://en.wikipedia.org/wiki/Edgar_Allan_Poe)
  - [H. P. Lovecraft](https://en.wikipedia.org/wiki/H._P._Lovecraft)
  - [Mary Shelley](https://en.wikipedia.org/wiki/Mary_Shelley)
 
The training data contains a total of **19579** instances with a test containing **8392**

## EDA

## Workflow/Experiments

- ML models
  - Logistic (with tfidf, cntvec, glove)
  - NB (with tfidf and cntvec)
  - XGboost (with glove, tfidf, cntvec)
  - Tuening
- DL models
  - Embedding Bag model
  - Entity Embedding mode;
  - Glove embedding model
    - normalized 
    - un-normalized 
  - Tuning with optuna
- Pretrained Model
  - Tranformers
