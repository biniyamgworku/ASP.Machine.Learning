###################################################
""" Code written for the Exercise 2 Natural Language processsing of the Python and Machine Learning.
     By Worku Biniyam
    Submitted to Michael E. Rose (PhD)
    Date of submission date August 30, 2022.
"""
# ------------------------- QUESTION ONE---------------
# ------------------------------------------------------------
#  Natural Language Processing
# ------------------------------------------------------------
# 2.a  Loading the dataset
import nltk
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pathlib import Path
import pickle
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from string import digits, punctuation
import re

Data=list(Path("data/speeches").glob("R0*"))
corpus = []
for i in Data:
        try:
            e=open(i,mode='r',
                   encoding="utf-8").read()
            corpus.append(e)
        except UnicodeDecodeError:
            print(f'{i}')
#--------------------------------------------
#2.b

_stopwords = nltk.corpus.stopwords.words ("english")
_stemmer = nltk.snowball.SnowballStemmer ("english")
len (_stopwords)

def tokenize_and_stem (text):
    """ Return tokens of text deprived of number and punctuation."""
    d = {p: "" for p in digits + punctuation}  # creating dictionary
    text = text.translate(str.maketrans(d))
    return[_stemmer.stem(t) for t in nltk.word_tokenize(text.lower())]
  count = CountVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem)
count.fit(corpus)
print(_stopwords)  # to see the list
count_matrix = count.transform(corpus)


tfidf = TfidfVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem, ngram_range=(1, 3))
tfidf_matrix = tfidf.fit_transform(corpus)
df_tfidf = pd.DataFrame(tfidf_matrix.todense().T, index=tfidf.get_feature_names_out())

# ----------------------------------------------
# 2.c

pickle_out = open("./output/speech_matrix.pk", "wb")
pickle.dump(tfidf_matrix, pickle_out)
pickle_out.close()

df = pd.DataFrame(df_tfidf)
df.to_csv("output/speech_matrix.csv")
# ----------------------------------------------
# 3.a
pickle_in = open("./output/speech_matrix.pk", "rb")
speech_matrix = pickle.load(pickle_in)
print(f'Showing the pickled data:{speech_matrix}')
pickle_in.close()
# ----------------------------------------------
# 3.b

array_dendro = count_matrix.toarray()
Z = linkage(array_dendro, method="complete", metric="cosine")
# ----------------------------------------------
# 3.c
plt.figure()
dendrogram(Z, color_threshold=0.85, no_labels=True)
plt.savefig("./output/speeches_dendrogram.pdf")
# --------------------------------------------
# 4.a Loading the dataset

FNAME = open("./data/Stellenanzeigen.txt", mode="r", encoding="utf-8").read()

newspaper = re.findall(r"(.*),\s\d{1,2}\.\s\w+\s\d{4}", FNAME)
date = re.findall(r".*,\s(\d{1,2}\.\s\w+\s\d{4})", FNAME)
ads = re.findall(r".*,\s\d{1,2}\.\s\w+\s\d{4}\s+(.*\n?.*)", FNAME)

job_ads_df = pd.DataFrame({"Newspaper": newspaper, "Date": date,
                           "Job Ad": ads})
job_ads_df["Date"] = job_ads_df["Date"].str.replace("März", "3.")
job_ads_df["Date"] = job_ads_df["Date"].astype("datetime64[ns]")
# --------------------------------- END ------------------------------------------------------------
# --------------------------------- END ------------------------------------------------------------
