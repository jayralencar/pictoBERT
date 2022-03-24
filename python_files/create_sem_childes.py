# %%
from pysupwsdpocket import PySupWSDPocket
nlp = PySupWSDPocket(lang='en', model='semcor_omsti', model_path="./pysupwsdpocket_models/")    

# %% [markdown]
# ## CHILDES
# 
# The Child Language Data Exchange System (CHILDES) is a corpus established in 1984 by Brian MacWhinney and Catherine Snow to serve as a central repository for data of first language acquisition[¹](https://en.wikipedia.org/wiki/CHILDES). It counts with a list of different corpora from many languages that can be downloaded in XML or CHA format.
# 
# In this notebook we download only one corpus, but SemCHILDES is composed by the entire American English CHILDES.

# %%
!mkdir corpora
corpora_files = ["Bates.zip", "Bernstein.zip", "Bliss.zip", "Bloom.zip", "Bohannon.zip", "Braunwald.zip", "Brent.zip", "Brown.zip", "Clark.zip", "Demetras1.zip", "Demetras2.zip", "Evans.zip", "Feldman.zip", "Garvey.zip", "Gathercole.zip", "Gelman.zip", "Gleason.zip", "Gopnik.zip", "HSLLD.zip", "Haggerty.zip", "Hall.zip", "Hicks.zip", "Higginson.zip", "Kuczaj.zip", "MacWhinney.zip", "McCune.zip", "McMillan.zip", "Morisset.zip", "Nelson.zip", "NewEngland.zip", "NewmanRatner.zip", "Peters.zip", "PetersonMcCabe.zip", "Post.zip", "Rollins.zip", "Sachs.zip", "Sawyer.zip", "Snow.zip", "Soderstrom.zip", "Sprott.zip", "Suppes.zip", "Tardif.zip", "Valian.zip", "VanHouten.zip", "VanKleeck.zip", "Warren.zip", "Weist.zip"]
for corpus_file in corpora_files:
    !wget https://childes.talkbank.org/data-xml/Eng-NA/$corpus_file -O corpora/$corpus_file

# %%
for corpus_file in corpora_files:
    !unzip corpora/$corpus_file -d corpora

# %% [markdown]
# ### Extract data from CHILDES
# 
# The data extraction is made by parsing the CHILDES' XML files.

# %%
import os
from glob import glob
PATH = "./corpora"
all_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.xml'))]
print(len(all_files))
print(all_files[0])

# %% [markdown]
# #### Find participants
# 
# This information is important for making queries in the future. For example, get sentences by children age.

# %%
def find_participants(root):
    participants = []
    for participant in root.find(ns+"Participants"):
      participants.append(participant.attrib)
    return participants

# %% [markdown]
# #### Parse utterances

# %%
def parse_utterance(u):
    wsd_doc = []
    if 'text' in u: # some utterances in CHILDES have just researchers comments or actions like (he screamed)
        doc = nlp.wsd(u['text'])
        for token in doc.tokens():
          wsd_doc.append(token.__dict__)
    return wsd_doc

# %% [markdown]
# #### Process utterances

# %%
from tqdm import tqdm
def process_utterances(root, process_faster=False):
    utterances = []
    for u in root.findall(ns+'u'):
      utterance_dict = u.attrib
      utterance_dict['original_tokens'] = []
      tokens = []
      for token in u.getchildren():
        if token.tag == ns+"w":
          tags = [a.tag for a in token.getchildren()]
          if ns+"shortening" in tags:
            try:
                tokens.append(token.find(ns+'mor').find(ns+"mw").find(ns+"stem").text)
            except:
                pass
          elif token.text is not None:
            tokens.append(token.text)
        elif token.tag == ns+"g": # group of words
          token = token.find(ns+'w')
          if token is not None:
              tags = [a.tag for a in token.getchildren()]
              if ns+"shortening" in tags:
                try:
                    tokens.append(token.find(ns+'mor').find(ns+"mw").find(ns+"stem").text)
                except:
                    pass
              elif token.text is not None:
                tokens.append(token.text)

        elif token.tag == ns+"t": # punctuation
          if token.attrib['type'] == 'p':
            tokens.append(".")
          elif token.attrib['type'] == 'q':
            tokens.append("?")
        elif token.tag == ns+"tagMarker": #comma
          tokens.append(',')
      if len(tokens) > 1:
        utterance_dict['text'] = " ".join(tokens)
      if not process_faster:
          utterance_dict['wsd_doc'] = parse_utterance(utterance_dict)
      utterances.append(utterance_dict)

    return utterances

# %%
!mkdir dicts

# %%
import xml.etree.ElementTree as ET
from tqdm import tqdm
import warnings
import json
from os.path import exists

warnings.filterwarnings('ignore')
all_dicts = []

process_n_files = len(all_files) # change to len(all_files) to use all

faster_processing = True # To process faster, you can use pysupwsd process_corpus method. However, by using this method we cannot use the sentences metadata (e.g., children age).
if faster_processing:
    !mkdir only_texts

for xml_file in tqdm(all_files):
    if faster_processing:
        txt_file = "only_texts/{0}.txt".format("_".join(xml_file.split('/')[-2:]))
        if exists(txt_file):
            continue
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = "{http://www.talkbank.org/ns/talkbank}"
    sem_dict = root.attrib
    sem_dict['file'] = "/content/corpora/MacWhinney/030018a.xml"
    sem_dict['participants'] = find_participants(root)
    sem_dict['utterances'] = process_utterances(root,faster_processing)
    
    if faster_processing:
        ft = open("only_texts/{0}.txt".format("_".join(xml_file.split('/')[-2:])),'w')
        ft.writelines([l['text']+"\n" for l in sem_dict['utterances'] if 'text' in l])
        ft.close() 
    
    all_dicts.append(sem_dict)
    json.dump(sem_dict, open("dicts/{0}.json".format("_".join(xml_file.split('/')[-2:])),'w'))

# %% [markdown]
# #### Create corpus for BERT input

# %%
!pip install nltk

# %%
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet as wn

# %%
import os
from glob import glob
f = open("data/semCHILDES.txt",'w')
if faster_processing:
    PATH = "only_texts/"
    only_texts_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.txt'))]
    
    for text_file in tqdm(only_texts_files):
        corpus = nlp.parse_corpus(text_file)
        for doc in corpus:
            new_sentence = []
            for t in doc.tokens():
              token = t.__dict__
              if token['word'] in ['me','and','or',',']:
                  new_sentence.append(token['word'])
              elif token['lemma'] in ["can","a","to","how","what",'this',"that"]:
                  new_sentence.append(token['lemma'])
              elif token['senses'][0]['id'] != 'U':
                  new_sentence.append(token['senses'][0]['id'])
              elif token['pos'] in ['IN','PRP','.','WRB','CC',"PRP$","DT"]:
                  new_sentence.append(token['lemma'])
              elif token['pos'] in ['NNP']:
                  new_sentence.append('proper_noun')
              elif token['pos'] in ['NN',"NNS"]:
                  n_token = None
                  synsets = wn.synsets(token['lemma'],'n')
                  if len(synsets) > 0:
                      synset = synsets[0]
                      for l in synset.lemmas():
                          if l.name() == token['lemma']:
                              n_token = l.key()
                  if n_token is not None:
                    new_sentence.append(n_token)
                  else:
                    new_sentence.append(token['lemma']) # it may be words that are common on children vocabulary.
            if len(new_sentence) > 1:
                f.write(" ".join(new_sentence)+"\n")
f.close()


# %%
f = open("data/semCHILDES.txt",'w')
if not faster_processing:
    for sem_dict in all_dicts:
        for u in tqdm(sem_dict['utterances']):
          if 'wsd_doc' in u:
            new_sentence = []
            for token in u['wsd_doc']:
              if token['word'] in ['me','and','or',',']:
                  new_sentence.append(token['word'])
              elif token['lemma'] in ["can","a","to","how","what",'this',"that"]:
                  new_sentence.append(token['lemma'])
              elif token['senses'][0]['id'] != 'U':
                  new_sentence.append(token['senses'][0]['id'])
              elif token['pos'] in ['IN','PRP','.','WRB','CC',"PRP$","DT"]:
                  new_sentence.append(token['lemma'])
              elif token['pos'] in ['NNP']:
                  new_sentence.append('proper_noun')
              elif token['pos'] in ['NN',"NNS"]:
                  n_token = None
                  synsets = wn.synsets(token['lemma'],'n')
                  if len(synsets) > 0:
                      synset = synsets[0]
                      for l in synset.lemmas():
                          if l.name() == token['lemma']:
                              n_token = l.key()
                  if n_token is not None:
                    new_sentence.append(n_token)
                  else:
                    new_sentence.append(token['lemma']) # it may be words that are common on children vocabulary.
            if len(new_sentence) > 1:
                f.write(" ".join(new_sentence)+"\n")
    f.close()



