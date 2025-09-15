#%% 7.1 Code to read in the positive and negative movie reviews
import os, codecs

def read_in(folder):
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


#%% Code to initialize two Python dictionaries for the reviews of different polarity

folder = 'data_ch7/txt_sentoken/'

pos_dict = read_in(folder + 'pos/')
print(len(pos_dict))
#print(pos_dict.get(next(iter(pos_dict))))

neg_dict = read_in(folder + 'neg/')
print(len(neg_dict))
print(neg_dict.get(next(iter(neg_dict))))

# %% 7.3 Code to calculate statistics on the review dataset

def tokenize(text):
    text.replace('\n', ' ')
    return text.split()

def statistics(a_dict):
    length = 0
    sent_length = 0
    num_sents = 0
    vocab = []
    for review in a_dict.values():
        length += len(tokenize(review))
        sents = review.split('\n')
        num_sents += len(sents)
        for sent in sents:
            sent_length += len(tokenize(sent))
        vocab += tokenize(review)
    avg_length = float(length)/len(a_dict)
    avg_sent_length = float(sent_length)/num_sents
    vocab_size = len(set(vocab))
    diversity = float(length)/float(vocab_size)
    return avg_length, avg_sent_length, vocab_size, diversity

categories = ['Positive', 'Negative']
rows = []
rows.append(['Category', 'Avg_Len(Review)','Avg_Len(Sent)', 'Vocabulary Size','Diversity'])
stats = {}
stats['Positive'] = statistics(pos_dict)
stats['Negative'] = statistics(neg_dict)
for cat in categories:
    rows.append([cat, f'{stats.get(cat)[0]:.6f}',
                f'{stats.get(cat)[1]:.6f}',
                f'{stats.get(cat)[2]:.6f}',
                f'{stats.get(cat)[3]:.6f}'])

columns = zip(*rows)
columns_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=columns_widths[i]) for i in range(0, len(row))))


# %% 7.4 Code to measure the difference between two lists of words

def vocab_difference(list1, list2):
    vocab1 = []
    vocab2 = []
    for rev in list1:
        vocab1 += tokenize(rev)
    for rev in list2:
        vocab2 += tokenize(rev)
    return sorted(list(set(vocab1) - set(vocab2)))

pos_wordlist = pos_dict.values()
neg_wordlist = neg_dict.values()

print(vocab_difference(pos_wordlist, neg_wordlist)[1500:1600])
print(vocab_difference(neg_wordlist, pos_wordlist)[1500:1600])
print()
print(str(len(vocab_difference(pos_wordlist, neg_wordlist))) +
      ' unique words in positive reviews only')
print(str(len(vocab_difference(neg_wordlist, pos_wordlist))) +
      ' unique words in negative reviews only')


# %% 7.5 Code to run spacy's linguistic pipeline and store the results
import spacy
nlp = spacy.load('en_core_web_md')

def spacy_preprocess_reviews(source):
    source_docs = {}
    
    index = 0
    for review_id in source.keys():
        source_docs[review_id] = nlp(
            source.get(review_id).replace('\n', ''),
            disable=['ner'])
        if index>0 and (index%200)==0:
            print(str(index) + ' review processed')
        index += 1
    
    print('Dataset processed')        
    return source_docs
        
pos_docs = spacy_preprocess_reviews(pos_dict)        
neg_docs = spacy_preprocess_reviews(neg_dict)

# %% 7.6 Code to calculate statistics on word lemmas

def statistics_lem(source_docs):
    length = 0
    vocab = []
    for review_id in source_docs.keys():
        review_doc = source_docs.get(review_id)
        lemmas = []
        for token in review_doc:
            lemmas.append(token.lemma_)
        length += len(lemmas)
        vocab += lemmas
    avg_length = float(length) / len(source_docs)
    vocab_size = len(set(vocab))
    diversity = float(length) / float(vocab_size)
    return avg_length, vocab_size, diversity

categories = ['Positive', 'Negative']
rows = []
rows.append(['Categories', 'Avg_Len(Review)','Vocabulary Size','Diversity'])
stats = {}
stats['Positive'] = statistics_lem(pos_docs)
stats['Negative'] = statistics_lem(neg_docs)
for cat in categories:
    rows.append([cat, f'{stats.get(cat)[0]:.6f}',
                 f'{stats.get(cat)[1]:.6f}',
                 f'{stats.get(cat)[2]:.6f}'])
columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=columns_widths[i]) for i in range(0, len(row))))

# %% 7.7 Code to detect the non-overlapping lemmas between two types of reviews

def vocab_lem_difference(source_docs1, source_docs2):
    vocab1 = []
    vocab2 = []
    for rev_id in source_docs1.keys():
        rev = source_docs1.get(rev_id)
        for token in rev:
            vocab1.append(token.lemma_)

    for rev_id in source_docs2.keys():
        rev = source_docs2.get(rev_id)
        for token in rev:
            vocab2.append(token.lemma_)
    return sorted(list(set(vocab1) - set(vocab2)))

print(str(len(vocab_lem_difference(pos_docs, neg_docs))) + 
      ' unique lemmas in prositive reviews only')
print(str(len(vocab_lem_difference(neg_docs, pos_docs))) + 
      ' unique lemmas in negative reviews only')


#%% 7.8 Code to populate sentiment word dictionaries with sentiment values

def collect_wordlist(input_file):
    word_dict = {}
    with codecs.open(input_file, encoding='ISO-8859-1', errors='ignore') as f:
        for a_line in f.readlines():
            cols = a_line.split("\t")
            if len(cols)>2:
                word = cols[0].strip()
                score = float(cols[1].strip())
                word_dict[word] = score
    f.close()
    return word_dict

adj_90 = collect_wordlist('data_ch7/sentiment_words/adjectives/1990.tsv')
print(adj_90.get('cool'))
print(len(adj_90))
adj_00 = collect_wordlist('data_ch7/sentiment_words/adjectives/2000.tsv')
print(adj_00.get('cool'))
print(len(adj_00))
all_90 = collect_wordlist('data_ch7/sentiment_words/frequent_words/1990.tsv')
print(len(all_90))
all_00 = collect_wordlist('data_ch7/sentiment_words/frequent_words/2000.tsv')
print(len(all_00))
movie_words = collect_wordlist('data_ch7/sentiment_words/subreddits/movies.tsv')
print(len(movie_words))

# %% 7.9 Code to apply and evaluate the sentiment lexicon-based approach

def bin_decision(a_dict, label, sent_dict):
    decisions = []
    for rev_id in a_dict.keys():
        score = 0
        for token in a_dict.get(rev_id):
            if token.text in sent_dict.keys():
                if sent_dict.get(token.text)<0:
                    score -= 1
                else:
                    score += 1
        if score < 0:
            decisions.append((-1, label))
        else:
            decisions.append((1, label))
    return decisions

def weighted_decisions(a_dict, label, sent_dict):
    decisions = []
    for rev_id in a_dict.keys():
        score = 0
        for token in a_dict.get(rev_id):
            if token.text in sent_dict.keys():
                score += sent_dict.get(token.text)
        if score < 0:
            decisions.append((-1, label))
        else:
            decisions.append((1, label))
    return decisions

def get_accuracy(pos_docs, neg_docs, sent_dict):
    decisions_pos = weighted_decisions(pos_docs, 1, sent_dict)
    decisions_neg = weighted_decisions(neg_docs, -1, sent_dict)
    decisions_all = decisions_pos + decisions_neg
    lists = [decisions_pos, decisions_neg, decisions_all]
    accuracies = []
    for i in range(0, len(lists)):
        match = 0
        for item in lists[i]:
            if item[0] == item[1]:
                match += 1
        accuracies.append(float(match) / float(len(lists[i])))
    return accuracies

categories = ['adj_90', 'adj_00','all_90','all_00','movies']
rows = []
rows.append(['List', 'Acc(positive)','Acc(negative)','Acc(all)'])
accs = {}
accs['adj_90'] = get_accuracy(pos_docs, neg_docs, adj_90)
accs['adj_00'] = get_accuracy(pos_docs, neg_docs, adj_00)
accs['all_90'] = get_accuracy(pos_docs, neg_docs, all_90)
accs['all_00'] = get_accuracy(pos_docs, neg_docs, all_00)
accs['movies'] = get_accuracy(pos_docs, neg_docs, movie_words)

for cat in categories:
    rows.append([cat, f'{accs.get(cat)[0]:.6f}',
                 f'{accs.get(cat)[1]:.6f}',
                 f'{accs.get(cat)[2]:.6f}',])

columns = zip(*rows)
columns_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i]) for i in range(0, len(row))))


#%% ####################  CH 8  ####################################

# %% 8.1 Code to access SentiWordNet and check induvidual words
import nltk
nltk.download('wordnet')
nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet as swn

print(list(swn.senti_synsets('joy')))
print(list(swn.senti_synsets('trouble')))

#%%8.2 Code to explore the difference in the sentiment scores for word senses

joy1 = swn.senti_synset('joy.n.01')
joy2 = swn.senti_synset('joy.n.02')

trouble1 = swn.senti_synset('trouble.n.03')
trouble2 = swn.senti_synset('trouble.n.04')

categories = ['Joy1', 'Joy2', 'Trouble1','Trouble2']
rows = []
rows.append(['List', 'Positive Score', 'Negative Score'])
accs = {}
accs['Joy1'] = [joy1.pos_score(), joy2.neg_score()]
accs['Joy2'] = [joy2.pos_score(), joy2.neg_score()]
accs['Trouble1'] = [trouble1.pos_score(), trouble2.neg_score()]
accs['Trouble2'] = [trouble2.pos_score(), trouble2.neg_score()]

for cat in categories:
    rows.append([cat, f'{accs.get(cat)[0]:.3f}',
                 f'{accs.get(cat)[1]:.3f}'])

columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i])
                  for i in range(0, len(row))))




#%% 8.4 Code to aggregate sentiment scores based on SentiWordNet
from nltk.corpus import wordnet as wn

def convert_tags(pos_tag):
    if pos_tag.startswith("JJ"):
        return wn.ADJ
    elif pos_tag.startswith('NN'):
        return wn.NOUN
    elif pos_tag.startswith('RB'):
        return wn.ADV
    elif pos_tag.startswith('VB') or pos_tag.startswith('MD'):
        return wn.VERB
    return None

def swn_decisions(a_dict, label):
    decisions = []
    for rev_id in a_dict.keys():
        score = 0
        neg_count = 0
        pos_count = 0
        for token in a_dict.get(rev_id):
            wn_tag = convert_tags(token.tag_)
            for wn_tag in (wn.ADJ, wn.ADV, wn.NOUN, wn.VERB):
                synsets = list(swn.senti_synsets(token.lemma_, pos=wn_tag))
                if len(synsets)>0:
                    temp_score = 0.0
                    for synset in synsets:
                        temp_score += synset.pos_score() - synset.neg_score()
                    score += temp_score/len(synsets)    
        if score < 0:
            decisions.append((-1, label))
        else: decisions.append((1, label))
    return decisions

#%% 8.5 Code to evaluate the results for this approach

def get_swn_accuracy(pos_docs, neg_docs):
    decisions_pos = swn_decisions(pos_docs, 1)
    decisions_neg = swn_decisions(neg_docs, -1)
    decisions_all = decisions_pos + decisions_neg
    lists = [decisions_pos, decisions_neg, decisions_all]
    accuracies = []
    for i in range(0, len(lists)):
        match = 0
        for item in lists[i]:
            if item[0] == item[1]:
                match += 1
        accuracies.append(float(match) / float(len(lists[i])))
    return accuracies

accuracies = get_swn_accuracy(pos_docs, neg_docs)

rows = []
rows.append(['List', 'Acc(positive)', 'Acc(negative)', 'Acc(all)'])
rows. append(['SentiWordNet', f'{accuracies[0]:.6f}',
              f'{accuracies[1]:.6f}',
              f'{accuracies[2]:.6f}'])

columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=columns_widths[i]) for i in range(0, len(row))))
    
#%% 8.6 Code to filter the content of the reviews and prepare it for feature exctraction

import random
import string
from spacy.lang.en.stop_words import STOP_WORDS as stopwords_list
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

def prepare_data(pos_docs, neg_docs, exclude_lists):
    data = text_filter(pos_docs, 1, exclude_lists)
    data += text_filter(neg_docs, -1, exclude_lists)
    random.seed(42)
    random.shuffle(data)
    
    texts = []
    labels = []
    for item in data:
        texts.append(item[0])
        labels.append(item[1])
    return texts, labels

texts, labels = prepare_data(pos_docs, neg_docs, punctuation_list)

print(len(texts), len(labels))
print(texts[0])
    
    
# %% 8.7  Code to split the data into the training and test sets

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

train_data, train_targets, test_data, test_targets = split(texts, labels,0.8)

print(len(train_data))
print(len(test_data))

# %%
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(train_data)

print(train_counts.shape)

# %% 8.9 Code to apply CountVectorizer to test set and run classification
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(train_counts, train_targets)

test_counts = count_vect.transform(test_data)
predicted = clf.predict(test_counts)

for text, label in list(zip(test_data, predicted))[:10]:
    if label == 1:
        print('%r => %s' % (text[:100], 'pos'))
    else:
        print('%r => %s' % (text[:100], 'neg'))

#%% 8.10  Code to define pipline 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer

text_clf = Pipeline([('vect', CountVectorizer(min_df=10, max_df=0.5)),
                     ('binarizer', Binarizer()),
                     ('clf', MultinomialNB())
                     ])

text_clf.fit(train_data, train_targets)
print(text_clf)
predicted = text_clf.predict(test_data)

#%%

from sklearn import metrics

print('\nConfusion matrix:')
print(metrics.confusion_matrix(test_targets, predicted))
print(metrics.classification_report(test_targets, predicted))


# %% 8.12 Code to run k-fold cross-validation

from sklearn.model_selection import cross_val_score, cross_val_predict

scores = cross_val_score(text_clf, texts, labels, cv=10)
print(scores)
print('Accuracy: ' + str(sum(scores)/10))
predicted = cross_val_predict(text_clf, texts, labels, cv=10)

print('\nConfusion matrix:')
print(metrics.confusion_matrix(labels, predicted))
print(metrics.classification_report(labels, predicted))

#%% 8.13 Code to update the Pipeline with n-gram features

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
                      ('binarizer', Binarizer()),
                      ('clf', MultinomialNB())
                      ])

scores = cross_val_score(text_clf, texts, labels, cv=10)
print(scores)
print('Accuracy: ' + str(sum(scores)/10))
predicted = cross_val_predict(text_clf, texts, labels, cv=10)
print('\nConfusion matrix:')
print(metrics.confusion_matrix(labels, predicted))
print(metrics.classification_report(labels, predicted))

#%% 8.14 Code to add negationshandling to your text preprocessing
from nltk.sentiment.util import mark_negation

def text_filter_neg(a_dict, label, exclude_lists):
    data = []
    for rev_id in a_dict.keys():
        tokens = []
        for sent in a_dict.get(rev_id).sents:
            neg_tokens = mark_negation(sent.text.split())
            for token in neg_tokens:
                if not token in exclude_lists:
                   tokens.append(token)
        data.append((' '.join(tokens), label))
    return data

def prepare_data_neg(pos_docs, neg_docs, exclude_lists):
    data = text_filter_neg(pos_docs, 1, exclude_lists)
    data += text_filter_neg(neg_docs, -1, exclude_lists)
    random.seed = 42
    random.shuffle(data)
    texts = []
    labels = []
    for item in data:
        texts.append(item[0])      
        labels.append(item[1])
    return texts, labels
            
texts_neg, labels_neg = prepare_data_neg(pos_docs, neg_docs, punctuation_list)
print(len(texts_neg))         
print(texts_neg[0])

# %% 8.15 Code to update the Pipeline and run the classifier
text_clf = Pipeline([
                    ('vect', CountVectorizer(ngram_range=(1,2))),
                    ('binarizer', Binarizer()),
                    ('clf', MultinomialNB())])

scores = cross_val_score(text_clf, texts, labels, cv=10)
print(scores)
print('Accuracy: ' + str(sum(scores)/10))
predicted = cross_val_predict(text_clf, texts, labels, cv=10)
print('\nConfusion matrix')
print(metrics.confusion_matrix(labels, predicted))
print(metrics.classification_report(labels, predicted))


# %%
