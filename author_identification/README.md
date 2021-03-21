# Author Indentification via NLP Techniques

This project aims to explore various text classification techniques in NLP literature. Here I apply suit of NLP algorithms to solve the problem of Author Identification using excerpts from their literary works. 

## Dataset

Data is taken from Kaggle's [Spooky Author Idenfication Challenge](https://www.kaggle.com/c/spooky-author-identification/). It containes the excerpts from the literary works od three authors - 
  - [Edgar Allen Poe](https://en.wikipedia.org/wiki/Edgar_Allan_Poe)
  - [H. P. Lovecraft](https://en.wikipedia.org/wiki/H._P._Lovecraft)
  - [Mary Shelley](https://en.wikipedia.org/wiki/Mary_Shelley)
 
The training data contains a total of **19579** instances with a test containing **8392**

## EDA & Feature Engineering

Sample excerpts look as follows - 
- EAP - 'This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit, and return to the point whence I set out, without being aware of the fact; so perfectly uniform seemed the wall.'
- HPL - 'It wearied Carter to see how solemnly people tried to make earthly reality out of old myths which every step of their boasted science confuted, and this misplaced seriousness killed the attachment he might have kept for the ancient creeds had they been content to offer the sonorous rites and emotional outlets in their true guise of ethereal fantasy.'
- MWS - 'The revenue of its possessor, which had always found a mode of expenditure congenial to his generous nature, was now attended to more parsimoniously, that it might embrace a wider portion of utility.'

Frequency distribution for the three classes (authors) look as follows - 

![freq_dist](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/dist.png)

There does not appear to be severe class imbalance. We do not need to perform any upsampling/downsampling methods for our models. We can just stratify on *authors* while deviding our data into trtain and validation sets.

Let us look at the various meta features (e.g. length of excerpts, number of punctuations, stopwords, etc.)

Feature            |  EAP | HPL | MWS|
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
Length of excerpt  |  ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/EAP_len.png) | ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/HPL_len.png) | ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/MWS_len.png)
\# Stop Words/excerpt | ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/EAP_num_stop_words.png) | ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/HPL_num_stop_words.png) | ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/MWS_num_stop_words.png)
\# Punctuations/excerpt | ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/EAP_puncts.png) | ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/HPL_puncts.png) | ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/MWS_puncts.png)
Average length of tokens | ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/EAP_ave_len_word.png) | ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/HPL_ave_len_word.png) | ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/MWS_ave_len_word.png)
\# Tokens/excerpt | ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/EAP_num_words.png) | ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/HPL_num_words.png) | ![](https://github.com/AshishSinha5/misc/blob/master/author_identification/plots/MWS_num_words.png)

Looking at the graphs we notice that there appear to be some outliers and there seems to be specific patterns of how a particular author constructs their sentences, so these features might help build our classification model.






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
