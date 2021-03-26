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

## Evaluation Metrics

1 - **Cross-Entropy Loss** <br>
2 - **Accuracy**

## Preprocessing

1 - Remove accented charecters <br>
2 - Convert to lower cse <br>
3 - Remove all the punctuations <br>
4 - Remove all stop-words

## ML Models

We now turn towards actual task of classification, for which we first explore the basic ML Models such as Naive Bayes, Logistic Regression followed by XGBoost Model. We consider TF-IDF and count vectors as our features.

Model | Validation Loss | Validation Accuracy 
:-----------------------:|:--------------------:|:-------------------:
Logistic Regression with TFIDF vectors | 0.5374 |  0.829
Logistic_regression with Count Vectorizer | 0.482 | 0.805
Naive Bayes with TFIDF vectors | 0.555 | 0.829
Naive Bayes with Count Vectorizer | 0.436 | 0.847

We see that Naive Bayes Model with count vectorizer performs the best, we can further improve this model by hypreparameter tuning by grid search, hyperparameter under consieration is the smoothing parameter for the n-gram model.


Model | Validation Loss | Validation Accuracy 
:-----------------------:|:--------------------:|:-------------------:
Naive Bayes Tuned with Count Vectorizer | 0.4450 |  0.847
Naive Bayes Tuned with TFIDF | 0.4421 | 0.840

We improved aur results when using TF-IDF features but Coun vectorizer still performs better

## DL Models
We now move towards Deep Learning Modelsk where we'll use PyTorch to build these models. The broad pipeline of how we'll go about building and improving these models is shown below.

### EmbeddingBagModel
Embedding Bag model takes in the entire sentences as input and creates an embedding for the sentence as a whole, using torch.nn.EmbeddingBag
### EntityEmbeddingModel
Entity Embedding Model builds o the previous model, as it creats an embedding for each of the tokens in the sentence which is then fed into LSTM layer follwed by fully connectted layers. 
### GLoveEmbeddingModel
Similar to previous model but this uses pretrained glove embeddings for the tokens rather than training the embeddings on the go.
