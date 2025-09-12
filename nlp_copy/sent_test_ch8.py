#%%
import os, codecs

root = os.getcwd()
print(root)

def read_in(folder):
    """ Reads in all the files in the folder.
    Returns a dict with file name as key and
    all the test as a str"""
    files = os.listdir(folder)
    a_dict = {}
    for a_file in sorted(files):
        if not a_file.startswith('.'):
            with codecs.open(folder + a_file,
                             encoding = 'ISO-8859-1',
                             errors = 'ignore') as f:
                file_id = a_file.split('.')[0].strip()
                a_dict[file_id] = f.read()
            f.close()
    return a_dict

#%% test

folder = 'data_ch7/text/'
test_dict = read_in(folder)
print(test_dict)
print(len(test_dict))
print(test_dict.items())



#%% returns reviews in dicts with file name as key

folder = 'data_ch7/txt_sentoken/'

pos_dict = read_in(folder + 'pos/')
print(len(pos_dict))
print(pos_dict.get(next(iter(pos_dict))))

neg_dict = read_in(folder + 'neg/')
print(len(neg_dict))
print(neg_dict.get(next(iter(neg_dict))))

#%% Use SpaCy pipeline to set up data
import spacy
nlp = spacy.load('en_core_web_md')

def spacy_preprocess_reviews(source):
    source_docs = {}
    
    index = 0
    for review_id in source.keys():
        source_docs[review_id] = nlp(source.get(review_id).
                                     replace('\n', ' '),
                                    disable=['ner'])
        if index>0 and (index%200) == 0:
            print(str(index) + ' review processed')
        index += 1
        
    print('Dataset processed')
    return source_docs

pos_docs = spacy_preprocess_reviews(pos_dict)
neg_docs = spacy_preprocess_reviews(neg_dict)

#%% Check some spacy container data

first_key = list(pos_docs.keys())[0]
rev_doc = pos_docs.get(first_key)
for token in rev_doc:
    print(token.lemma_)

# %%

def tokenize(text):
    """ Returns list with individual words"""
    text.replace('\n', ' ')
    return text.split()

# rev_str = list(test_dict.values())[0]
# rev_tok = tokenize(rev_str)
# print(rev_tok)

def split_cr(text):
    """ Split on row because cr after every full stop"""
    return text.split('\n')

def statistics(a_dict):
    length = 0
    sent_length = 0
    num_sents = 0
    vocab = []
    for review in a_dict.values():
        length += len(tokenize(review))
        sents = split_cr(review)
        num_sents += len(sents)
        for sent in sents:
            sent_length += len(tokenize(sent))
        vocab += tokenize(review)
    avg_length = float(length)/len(a_dict)
    avg_sent_length = float(sent_length)/num_sents
    vocab_size = len(set(vocab))
    diversity = float(length)/float(vocab_size)
    return avg_length, avg_sent_length, vocab_size, diversity

avg_length, avg_sent_length, vocab_size, diversity = statistics(test_dict)

print(avg_length, avg_sent_length, vocab_size, diversity)


# %% Code filter content of the reviews and prep for feature extraction
import random
import string
from spacy.lang.en.stop_words import STOP_WORDS as stopwords_list

random.seed(42)
punctuation_list = [punct for punct in string.punctuation]

def text_filter(a_dict, label, exclude_lists):
    data = []
    for rev_id in a_dict.keys():
        tokens = []
        for token in a_dict.get(rev_id):
            if not token.text in exclude_lists:
                tokens.append(token.text)
        data.append((' '.join(tokens), label))
    return data

def prepare_data(pos_docs, neg_docs, exclude_list):
    data = text_filter(pos_docs, 1, exclude_list)
    data += text_filter(neg_docs, -1, exclude_list)
    random.shuffle(data)
    texts = []
    labels = []
    for item in data:
        texts.append(item[0])
        labels.append(item[1])
    return texts, labels

text_, labels = prepare_data(pos_docs, neg_docs, punctuation_list)

# %% Train test split

def split(texts, labels, proportion):
    train_data = []
    train_targets = []
    test_data = []
    test_targets = []
    for i in range(0, len(texts)):
        if i < proportion*len(texts):
            train_data.append(texts[i])
            train_targets.append(labels[i])
        else:
            test_data.append(texts[i])
            test_targets.append(labels[i])
    return train_data, train_targets, test_data, test_targets

train_data, train_targets, test_data, test_targets = split(text_, labels,0.8)




# %%
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
# Returns matrix earch row is a review, each column a word
train_counts = count_vect.fit_transform(train_data)

#%% 
print(train_counts.shape)
#print(len(count_vect.vocabulary_))
#print(train_counts.toarray()[:8,:8])

# %% Model and test on count vectorizer matrix
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(train_counts, train_targets)

test_counts = count_vect.transform(test_data)
predicted = clf.predict(test_counts)


# %% Set up pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer

text_clf = Pipeline([
                    ('vect', CountVectorizer(min_df=10, max_df=0.5)),
                    ('binarizer', Binarizer()),
                    ('clf', MultinomialNB())
])

text_clf.fit(train_data, train_targets)
print(text_clf)
predicted = text_clf.predict(test_data)


# %%
from sklearn import metrics

print('\nConfusion Matrix')
print(metrics.confusion_matrix(test_targets, predicted))
print(metrics.classification_report(test_targets, predicted))


# %% Do train/test with k-fold cross-validation
# the original text file is run through the pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict

scores = cross_val_score(text_clf, text_, labels, cv=10)
print(scores)
print('Accuracy: ' + str(sum(scores)/10))

predicted = cross_val_predict(text_clf, text_, labels, cv=10)

print('\nConfusion matrix:')
print(metrics.confusion_matrix(labels, predicted))
print(metrics.classification_report(labels, predicted))

# %%

text_clf = Pipeline([
                    ('vect', CountVectorizer(ngram_range=(1,2))),
                    ('binarizer', Binarizer()),
                    ('clf', MultinomialNB())
                    ])

scores = cross_val_score(text_clf, text_, labels, cv=10)
print(scores)
print('Accuracy: ' + str(sum(scores)/10))

predicted = cross_val_predict(text_clf, text_, labels, cv=10)
print('\nConfusion matrix')
print(metrics.confusion_matrix(labels, predicted))
print(metrics.classification_report(labels, predicted))



# %% ====================== REDO WITH Million Song Data
import pandas as pd

df_feat = pd.read_csv('msd_data/features.csv')
df_lab = pd.read_csv('msd_data/labels.csv')
df = pd.merge(left=df_feat, right=df_lab, on='trackID')
df['genre'] = df['genre'].replace({"soul and reggae":0, "pop":1, "punk":2,
        "jazz and blues":3, "dance and electronica":4,
        "folk":5, "classic pop and rock":6, "metal":7})

#%% Make a df with raw data and cols to be used

df_text = df[['trackID','tags','genre']]

labels = df[['genre']]
#display(df_text.head())
#print(df_text.info())

# %% 
df_text2 = df_text.copy(deep=True)

df_text2['proc_text'] = df_text2['tags'].astype(str).apply(lambda x: x.replace(',',''))
#display(df_text.head())

texts = df_text2.set_index('trackID')['proc_text'].to_dict()
print(len(texts))
tests_list = df_text2['proc_text'].to_list()


# %%

text_clf = Pipeline([
                    ('vect', CountVectorizer()),
                    ('binarizer', Binarizer()),
                    ('clf', MultinomialNB())
])

scores = cross_val_score(text_clf, tests_list, labels, cv=10)
print(scores)
print("Accuracy: " + str(sum(scores)/10))
predicted = cross_val_predict(text_clf, tests_list, labels, cv=10)
print('\nConfusion Matrix')
print(metrics.confusion_matrix(labels, predicted))
print(metrics.classification_report(labels,predicted))


# %%
