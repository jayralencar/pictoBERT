# PictoBERT: Transformers for Next Pictogram Prediction

Original code implementation of the paper "Transformers for Next Pictogram Prediction".

Pictogram is the term used by the Augmentative and Alternative Communication (AAC) community for an image with a label that represents a place, person, action, object and animal. AAC systems like the shown below allow message construction and communication by arranging pictograms in sequence.

![image](https://user-images.githubusercontent.com/7529265/117816187-a02cbb00-b23c-11eb-9ffd-b54c1f4816b1.png)


Pictogram prediction is an important task for AAC systems for it can facilitate communication. Previous works used n-gram statistical models or knowledge bases to accomplishing this task. Our proposal is an adaptation of the BERT (Bidirectional Encoder Representations from Transformers) model to perform pictogram prediction. We changed the BERT vocabulary and input embeddings to allow the usage of word-senses, considering that a word-sense represents better a pictogram. We call our version PictoBERT.

![image](https://user-images.githubusercontent.com/7529265/117816802-48db1a80-b23d-11eb-9362-37670baa048a.png)

We trained the model using the CHILDES (Child Language Data Exchange System) corpora as a dataset. We annotated the North American English version of CHILDES with word-senses using [supWSD](https://github.com/rodriguesfas/PySupWSDPocket). PictoBERT performance was compared to n-gram models and achieved good results, as show in the table bellow. 

![image](https://user-images.githubusercontent.com/7529265/117849118-5bb01800-b25a-11eb-8e73-b3c1f77f6cc9.png)


The PictoBERT is capable of predicting pictograms in different contexts. And its main characteristic is the ability to transfer learning for it allows other models focused on users' specific needs to be trained.

![image](https://user-images.githubusercontent.com/7529265/117823613-0b2dc000-b244-11eb-8cf7-a23934b8a45e.png)


## Software requirements

* [Pytorch](https://pytorch.org/)
* [Pytorch Lightning](https://www.pytorchlightning.ai/)
* [Tokenizers](https://github.com/huggingface/tokenizers)
* [Transformers](https://huggingface.co/transformers/)
* [Gensim](https://radimrehurek.com/gensim/auto_examples/index.html)
* [Pandas](https://pandas.pydata.org/)
* [Keras](https://keras.io/)
* [PySupWSD](https://github.com/rodriguesfas/PySupWSDPocket)
* [NLTK](https://keras.io/)

## Execution

Clone the repository in your machine and open the notebooks.
```
git clone https://github.com/jayralencar/pictoBERT.git
```
Or open the notebooks in Google Colab by clicking  in the anchors bellow.

The process for training PictoBERT is based on five main steps:
1. [Create SemCHILDES](https://colab.research.google.com/github/jayralencar/pictoBERT/blob/main/SemCHILDES.ipynb)
2. [Train Tokenizer, and Encode and split dataset](https://colab.research.google.com/github/jayralencar/pictoBERT/blob/main/Train_Tokenizer_and_Prepare_Dataset.ipynb)
3. [Create model](https://colab.research.google.com/github/jayralencar/pictoBERT/blob/main/Create_Models.ipynb)
4. [Train and test PictoBERT](https://colab.research.google.com/github/jayralencar/pictoBERT/blob/main/Training_PictoBERT.ipynb)
5. [Train and test n-gram models](https://colab.research.google.com/github/jayralencar/pictoBERT/blob/main/N_gram_models.ipynb)

The notebooks execution must follow this sequence, but downloadable versions of the resources are available in each step..
