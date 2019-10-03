


import pandas as pd
import numpy as np
from wandblog import log
import wandb
run = wandb.init(job_type='eval')
config = run.config
config.lowercase=True
config.ngram_min=1
config.ngram_max=2

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import TfidfVectorizer
count_vect = TfidfVectorizer(lowercase=config.lowercase,
                             ngram_range=(config.ngram_min,
                                          config.ngram_max),
                             #token_pattern=config.token_pattern
                                        )
count_vect.fit(fixed_text)

counts = count_vect.transform(fixed_text)

"""
Using MultinomialNB
Accuracy: 65.3321601407831
"""
#from sklearn.naive_bayes import MultinomialNB
#nb = MultinomialNB()#alpha=config.alpha)

"""
Using Perceptron
Accuracy: 63.28640563132424
"""
"""
from sklearn.linear_model import Perceptron
nb = Perceptron()#alpha=config.alpha)
"""

"""
Using SVM/SVC
Accuracy: 59.2608886933568
"""
"""
from sklearn.svm import SVC
nb = SVC()#alpha=config.alpha)
"""

"""
Using Decision Tree
Accuracy: 60.66871975362957
"""
"""
from sklearn.tree import DecisionTreeClassifier
nb = DecisionTreeClassifier()#alpha=config.alpha)
"""

"""
Using Random Forest
Accuracy: 64.29828420589529 (with default n_estimators)
Accuracy: 66.46505000219586 (with n_estimators=1000)
Accuracy: 63.40739111306643 (with default n_estimators, ngram_range=(1,2))
Accuracy: 65.02419709634844 (with n_estimators=100, ngram_range=(1,2))

Accuracy: 64.94720633523977 (with TfidfVectorizer, n_estimators=100, ngram_range=(1,2))
"""
from sklearn.ensemble import RandomForestClassifier
nb = RandomForestClassifier(n_estimators=100)#alpha=config.alpha)

from sklearn.model_selection import cross_val_score, cross_val_predict

scores = cross_val_score(nb, counts, fixed_target)
print(scores)
print(scores.mean())

predictions = cross_val_predict(nb, counts, fixed_target)
log(run, fixed_text, fixed_target, predictions)
