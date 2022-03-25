

# PictoBERT: Transformers for Next Pictogram Prediction

Original code implementation of the paper "PictoBERT: Transformers for Next Pictogram Prediction".

Pictogram is the term used by the Augmentative and Alternative Communication (AAC) community for an image with a label that represents a place, person, action, object and animal. AAC systems like the shown below allow message construction and communication by arranging pictograms in sequence.

![image](https://user-images.githubusercontent.com/7529265/117816187-a02cbb00-b23c-11eb-9ffd-b54c1f4816b1.png)


Pictogram prediction is an important task for AAC systems for it can facilitate communication. Previous works used n-gram statistical models or knowledge bases to accomplishing this task. Our proposal is an adaptation of the BERT (Bidirectional Encoder Representations from Transformers) model to perform pictogram prediction. We changed the BERT vocabulary and input embeddings to allow the usage of word-senses, considering that a word-sense represents better a pictogram. We call our version PictoBERT.

![using_flow_2_hl](https://user-images.githubusercontent.com/7529265/160137035-f487523a-a924-443f-9497-0a950d2823de.png)

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

You can run the PictoBERT scripts using Google Colab or clone the repository in your machine and open the notebooks.
```
git clone https://github.com/jayralencar/pictoBERT.git
```
We present each of the notebooks below and their relationship with the paper's content. You may execute the notebooks following the sequence we give below. However, downloadable versions of the resources are available in each step.

### PictoBERT
In the paper, we present PictoBERT construction (Section 4.1) in three steps: corpus construction, BERT adaptation and pretraining.

#### Dataset Creation
The dataset creation is described in Section 4.1.1 of the paper and consists of downloading and annotating the North American English part of the CHILDES dataset.


<table class="tfo-notebook-buttons" align="left">
	<td>
		<strong>SemCHILDES.ipynb</strong>
	</td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github//jayralencar/pictoBERT/blob/main/SemCHILDES.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/jayralencar/pictoBERT/blob/main/SemCHILDES.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
<a target="_blank" href="http://jayr.clubedosgeeks.com.br/pictobert/all_mt_2.txt"><img src="https://icons.iconarchive.com/icons/custom-icon-design/mono-general-2/16/download-icon.png" />NA-EN SemCHILDES</a>
</td>
</table>
<br><br>

In addition, we also annotated the British English part of CHILDES with semantic roles to use for fine-tuning PictoBERT to perform pictogram prediction based on a grammatical structure.

<table  class="tfo-notebook-buttons"  align="left">
<td> <strong>Create_SRL_semCHILDES.ipynb</strong></td>
<td>
<a  target="_blank"  href="https://colab.research.google.com/github//jayralencar/pictoBERT/blob/main/Create_SRL_semCHILDES.ipynb"><img  src="https://www.tensorflow.org/images/colab_logo_32px.png"  />Run in Google Colab</a>
</td>
<td>
<a  target="_blank"  href="https://github.com/jayralencar/pictoBERT/blob/main/Create_SRL_semCHILDES.ipynb"><img  src="https://www.tensorflow.org/images/GitHub-Mark-32px.png"  />View source on GitHub</a>
</td>
  <td>
<a target="_blank" href="http://jayr.clubedosgeeks.com.br/pictobert/sem_childes_uk_clean_2.txt"><img src="https://icons.iconarchive.com/icons/custom-icon-design/mono-general-2/16/download-icon.png" />UK-EN SemCHILDES</a>
</td>
</table>

#### Updating BERT Vocabulary and Embeddings Layer
For updating BERT vocabulary and Embeddings Layer, as described in Section 4.1.2 of the paper, we first trained a Word Level tokenizer and prepared the dataset for future training.

<table  class="tfo-notebook-buttons"  align="left">
<td>
<b>Train_Tokenizer_and_Prepare_Dataset.ipynb</b>
</td>
<td>
<a  target="_blank"  href="https://colab.research.google.com/github//jayralencar/pictoBERT/blob/main/Train_Tokenizer_and_Prepare_Dataset.ipynb "><img  src="https://www.tensorflow.org/images/colab_logo_32px.png"  />Run in Google Colab</a>
</td>
<td>
<a  target="_blank"  href="https://github.com/jayralencar/pictoBERT/blob/main/Train_Tokenizer_and_Prepare_Dataset.ipynb"><img  src="https://www.tensorflow.org/images/GitHub-Mark-32px.png"  />View source on GitHub</a>
</td>
<td>
<a target="_blank" href="http://jayr.clubedosgeeks.com.br/pictobert/childes_all_new.json"><img src="https://icons.iconarchive.com/icons/custom-icon-design/mono-general-2/16/download-icon.png" />PictoBERT Tokenizer</a>
<br>
<a target="_blank" href="http://jayr.clubedosgeeks.com.br/pictobert/train_childes_all.pt"><img src="https://icons.iconarchive.com/icons/custom-icon-design/mono-general-2/16/download-icon.png" />Train dataset</a>
<br>
<a target="_blank" href="http://jayr.clubedosgeeks.com.br/pictobert/test_childes_all.pt"><img src="https://icons.iconarchive.com/icons/custom-icon-design/mono-general-2/16/download-icon.png" />Test dataset</a>
<br>
<a target="_blank" href="http://jayr.clubedosgeeks.com.br/pictobert/val_childes_all.pt"><img src="https://icons.iconarchive.com/icons/custom-icon-design/mono-general-2/16/download-icon.png" />Val dataset</a>
</td>
</table>

Then, we created the models by changing the BERT embeddings and vocabulary:
<table  class="tfo-notebook-buttons"  align="left">
<td><b>Create_Models.ipynb</b></td>
<td>
<a  target="_blank"  href="https://colab.research.google.com/github//jayralencar/pictoBERT/blob/main/Create_Models.ipynb"><img  src="https://www.tensorflow.org/images/colab_logo_32px.png"  />Run in Google Colab</a>
</td>
<td>
<a  target="_blank"  href="https://github.com/jayralencar/pictoBERT/blob/main/Create_Models.ipynb"><img  src="https://www.tensorflow.org/images/GitHub-Mark-32px.png"  />View source on GitHub</a>
</td>
<td>
<a target="_blank" href="http://jayr.clubedosgeeks.com.br/pictobert/pictobert-large-contextual.zip"><img src="https://icons.iconarchive.com/icons/custom-icon-design/mono-general-2/16/download-icon.png" />PictoBERT contextualized</a>
<br>
<a target="_blank" href="http://jayr.clubedosgeeks.com.br/pictobert/pictobert-large-gloss.zip"><img src="https://icons.iconarchive.com/icons/custom-icon-design/mono-general-2/16/download-icon.png" />PictoBERT gloss-based</a>
</td>
</table>

#### Pre-Training PictoBERT
As described in section 4.1.3 of the paper, we splited semCHILDES in a 98/1/1 split for training, validation, and test. We used a batch size of 128 sequences with 32 tokens. Each data batch was collated to choose 15% of the tokens for prediction. We used a learning rate of $1 \times 10 ^{-4}$,  with $\beta_1 = 0.9$, $\beta_2 = 0.999$, L2 weight decay of 0.01 and linear decay of learning rate. Training PictoBERT was performed in a single 16GB NVIDIA Tesla V100 GPU for 500 epochs for each version.

<table  class="tfo-notebook-buttons"  align="left">
<td><b>Training_PictoBERT.ipynb</b></td>
<td>
<a  target="_blank"  href="https://colab.research.google.com/github//jayralencar/pictoBERT/blob/main/Training_PictoBERT.ipynb "><img  src="https://www.tensorflow.org/images/colab_logo_32px.png"  />Run in Google Colab</a>
</td>
<td>
<a  target="_blank"  href="https://github.com/jayralencar/pictoBERT/blob/main/Training_PictoBERT.ipynb"><img  src="https://www.tensorflow.org/images/GitHub-Mark-32px.png"  />View source on GitHub</a>
</td>
<td>
<a target="_blank" href="http://jayr.clubedosgeeks.com.br/pictobert/pictobert-large-contextual.zip"><img src="https://icons.iconarchive.com/icons/custom-icon-design/mono-general-2/16/download-icon.png" />PictoBERT contextualized</a>
<br>
<a target="_blank" href="http://jayr.clubedosgeeks.com.br/pictobert/pictobert-large-gloss.zip"><img src="https://icons.iconarchive.com/icons/custom-icon-design/mono-general-2/16/download-icon.png" />PictoBERT gloss-based</a>
</td>
</table>
