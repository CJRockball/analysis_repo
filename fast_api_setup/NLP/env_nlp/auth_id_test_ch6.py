#%%
import os
import codecs
import nltk
from nltk import word_tokenize, sent_tokenize
nltk.download('punkt')
import random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

random.seed(42)
root = os.getcwd()
#print(root)

# CH 5.

def read_txt(fname, author):
    """ Make list of tuples with each sentence tokenized in a list
    and author name as a string"""
    f = open(fname, 'r', encoding='utf-8')
    t = f.read()
    one_list = sent_tokenize(t)
    list_of_lists = [(word_tokenize(sent), author) for sent in one_list]

    return list_of_lists

book_train_list = [('David Copperfield.txt', 'dickens'), ('Oliver Twist.txt', 'dickens'), 
                   ('hound of baskerville.txt', 'doyle'), ('The Sign of The Four.txt', 'doyle')]
                   
book_test_list = [('Tale of Two Cities.txt', 'dickens'), ('A Study in Scarlett.txt', 'doyle')]

all_train_list = []
for name,auth in book_train_list:
    all_train_list += read_txt(name, auth)
    
train_auth_list = [author for (sent, author) in all_train_list]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_train_list = []
strat_pretest_list = []
for train_index, pretest_index in sss.split(all_train_list, train_auth_list):
    strat_train_list = [all_train_list[index] for index in train_index]
    strat_pretest_list = [all_train_list[index] for index in pretest_index]

test_list = []
for name, auth in book_test_list:
    test_list += read_txt(name, auth)

#%%
print(len(strat_train_list))
print(len(strat_pretest_list))
print([auth for (sent, auth) in strat_train_list[5000:5010]])

#%% Claclulate basic stats on text by author

def statistics(text_data):
    for sent,label in text_data:
        words_per_sentence = len(sent)
        word_len = []
        for word in sent:
            word_len.append(len(word))
        avg_word_len = sum(word_len) / words_per_sentence
    return avg_word_len, words_per_sentence

def word_occ_count(text_data):
    occ_count = {}
    for sent,label in text_data:
        for word in sent:
            if word in occ_count:
                occ_count[word] += 1
            else: occ_count[word] = 1
    return occ_count

dickens_test = [(sent, auth) for sent, auth in test_list if auth == 'dickens']

a, b = statistics(dickens_test)
print(f'Words per sentence: {b}, Average word length: {a}')

occ_count = word_occ_count(dickens_test)
df_occ_count = pd.DataFrame(data=occ_count.items(), columns=['word', 'count_']).sort_values(by='count_', ascending=False)
display(df_occ_count.head())
print(f'Average number of occurences per word: {df_occ_count.count_.mean()}')

#%% Scatterplot of word count
import matplotlib.pyplot as plt
import numpy as np

df_occ_count['log_count'] = np.log(df_occ_count.count_)

plt.figure()
plt.scatter(df_occ_count.word, df_occ_count.log_count)
ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=False)
ax.set_xticks([])
plt.show()



# %% Get sentence features for nltk {word1: true, word2: true}

def get_features(text):
    features = {}
    word_list = [word for word in text]
    for word in word_list:
        features[word] = True
    return features

train_features = [(get_features(sents), label) for (sents,label) in strat_train_list]
pretest_features = [(get_features(sents), label) for (sents,label) in strat_pretest_list]


#%% Train classifier on all the words in the books
from nltk import NaiveBayesClassifier, classify

clf = NaiveBayesClassifier.train(train_features)

acc_train = classify.accuracy(clf, train_features)
acc_pretest = classify.accuracy(clf, pretest_features)

print(f'Accuracy on training set = {str(acc_train)}')
print(f'Accuracy on pretest set = {str(acc_pretest)}')
clf.show_most_informative_features(10)

test_features = [(get_features(sents), label) for (sents, label) in test_list]
print(f'Test set size = {str(len(test_features))} sentences')
print(f'Accuracy on the test set = {str(classify.accuracy(clf, test_features))}')


# %% Try to decrease the number of features by not using the 200 most used
# and the 20% least used words
# Start by counting words, same as above butt different functions
from collections import Counter

# Get all the words from combined book list
words_list = []
def extract_words(text, words_list):
    words_list += set([word for word in text])
    return words_list

for (sent, label) in strat_train_list:
    words_list = extract_words(sent, words_list)

counts = Counter(words_list)
print(counts)

# %% Remove most common and rate features
from nltk import DecisionTreeClassifier
highest_count = 19320

selected_words = []
for item in counts.items():
    count  = float(item[1])
    if count > 200 and count/highest_count < 0.2:
        selected_words.append(item[0])
print(len(selected_words))

def get_selected_features(text, selected_words):
    features = {}
    word_list = [word for word in text]
    for word in word_list:
        if word in selected_words:
            features[word] = True
    return features   


train_features = [(get_selected_features(sents, selected_words), label) for (sents, label) in strat_train_list]
pretest_features = [(get_selected_features(sents, selected_words), label) for (sents, label) in strat_pretest_list]
test_features = [(get_selected_features(sents, selected_words), label) for (sents, label) in test_list]

classifier = DecisionTreeClassifier.train(train_features)

print(f'Accuracy on the training set = {str(classify.accuracy(classifier, train_features))}')
print(f'Accuracy on the pretest set = {str(classify.accuracy(classifier, pretest_features))}')
print(f'Accuracy on the test set = {str(classify.accuracy(classifier, test_features))}')


# %% CH 6 do classification with text features rather than words

def avg_number_chars(text):
    total_chars = 0.0
    for word in text:
        total_chars += len(word)
    avg_chars = float(total_chars) / float(len(text))
    return avg_chars

def number_words(text):
    return float(len(text))

def initialize_dataset(source):
    all_features = []
    targets = []
    for (sents, label) in source:
        feature_list = []
        feature_list.append(avg_number_chars(sents))
        feature_list.append(number_words(sents))
        all_features.append(feature_list)
        if label == 'dickens':
            targets.append(0)
        else: targets.append(1)
    return all_features, targets

train_data, train_targets = initialize_dataset(strat_train_list)
pretest_data, pretest_targets = initialize_dataset(strat_pretest_list)
test_data, test_targets = initialize_dataset(test_list)


# %%
print('pretest')
print(pretest_data[:3])
print(pretest_targets[:3])
print(len(pretest_data), len(pretest_targets))
print(f'Number of Doyle data in set: {sum(pretest_targets)}')
print(f'train data\n train data: {len(train_data)}, train target: {len(train_targets)}')
print(f'test data\n test data: {len(test_data)}, test_data: {len(test_targets)}')

# %% Can no longer use the nltk classifiers. Use sklearn
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import metrics

text_clf = DecisionTreeClassifier(random_state=42)
text_clf.fit(train_data, train_targets)
predicted = text_clf.predict(pretest_data)

def evaluate(predicted, targets):
    print(np.mean(predicted == targets))
    print(metrics.confusion_matrix(targets, predicted))
    print(metrics.classification_report(targets, predicted))

evaluate(predicted, pretest_targets)
predicted = text_clf.predict(test_data)
evaluate(predicted, test_targets)

#%% Calculate number and proportion of time words appear

def word_counts(text):
    counts = {}
    for word in text:
        counts[word.lower()] = counts.get(word.lower(), 0) + 1
    return counts

def proportion_words(text, wordlist):
    count = 0
    for word in text:
        if word.lower() in wordlist:
            count += 1
        return float(count / float(len(text)))



#%% Add new features, stop_word count, and model
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_md')

def initialize_dataset(source):
    all_features = []
    targets = []
    for (sent, label) in source:
        feature_list = []
        feature_list.append(avg_number_chars(sent))
        feature_list.append(number_words(sent))
        counts = word_counts(sent)
        for word in STOP_WORDS:
            if word in counts.keys():
                feature_list.append(counts.get(word))
            else:
                feature_list.append(0)
        stop_w_prop = proportion_words(sent, STOP_WORDS)
        feature_list.append(stop_w_prop)
        all_features.append(feature_list)
        
        if label == 'dickens':
            targets.append(0)
        else: targets.append(1)
    return all_features, targets
    
train_data, train_targets = initialize_dataset(strat_train_list)
pretest_data, pretest_targets = initialize_dataset(strat_pretest_list)
test_data, test_targets = initialize_dataset(test_list)

print(len(train_data), len(train_targets))
print(len(pretest_data), len(pretest_targets))
print(len(test_data), len(test_targets))

# %%

text_clf = DecisionTreeClassifier(random_state=42)
text_clf.fit(train_data, train_targets)
predicted = text_clf.predict(pretest_data)
evaluate(predicted, pretest_targets)

predicted = text_clf.predict(test_data)
evaluate(predicted, test_targets)

# %%

def preprocess(source):
    source_docs = {}
    index = 0
    for (sent, label) in source:
        text = ' '.join(sent)
        source_docs[text] = nlp(text)
        if index > 0 and (index%2000) == 0:
            print(str(index) + ' texts processed')
        index += 1
    print('All done')
    return source_docs

train_docs = preprocess(strat_train_list)
pretest_docs = preprocess(strat_pretest_list)
test_docs = preprocess(test_list)

# %%

from collections import Counter
pos_list = ['C','D','E','F','I','J','M',
            'N','P','R','T','U','V','W']

def pos_count(text, source_docs, pos_list):
    pos_counts = {}
    doc = source_docs.get(' '.join(text))
    tags = []
    for word in doc:
        tags.append(str(word.tag_)[0])
    counts = Counter(tags)
    for pos in pos_list:
        if pos in counts.keys():
            pos_counts[pos] = counts.get(pos)
        else:
            pos_counts[pos] = 0
    return pos_counts

def initialize_dataset(source, source_docs):
    all_features = []
    targets = []
    for (sent, label) in source:
        feature_list = []
        feature_list.append(avg_number_chars(sent))
        feature_list.append(number_words(sent))
        counts = word_counts(sent)
        for word in STOP_WORDS:
            if word in counts.keys():
                feature_list.append(counts.get(word))
            else:
                feature_list.append(0)    
        feature_list.append(proportion_words(sent, STOP_WORDS))
        p_counts = pos_count(sent, source_docs, pos_list)
        for pos in p_counts.keys():
            feature_list.append(float(p_counts.get(pos))/float(len(sent)))
        all_features.append(feature_list)
        if label == 'dickens':
            targets.append(0)
        else:
            targets.append(1)
    return all_features, targets

#%%

def run():
    train_data, train_targets = initialize_dataset(strat_train_list, train_docs)
    pretest_data, pretest_targets = initialize_dataset(strat_pretest_list, pretest_docs)
    test_data, test_targets = initialize_dataset(test_list, test_docs)

    print(len(train_data), len(train_targets))
    print(len(pretest_data), len(pretest_targets))
    print(len(test_data), len(test_targets))
    print()
    
    text_clf = DecisionTreeClassifier(random_state=42)
    text_clf.fit(train_data, train_targets)
    predicted = text_clf.predict(pretest_data)
    evaluate(predicted, pretest_targets)

    predicted = text_clf.predict(test_data)
    evaluate(predicted, test_targets)

run()

# %%
import operator

def select_suffixes(cutoff):
    all_suffixes = []
    for doc in train_docs.values():
        for word in doc:
            all_suffixes.append(str(word.suffix_).lower())
    counts = Counter(all_suffixes)
    sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    selected_suffixes = []
    for i in range(0, round(len(counts)*cutoff)):
        selected_suffixes.append(sorted_counts[i][0])
    return selected_suffixes

selected_suffixes = select_suffixes(0.3)
print(len(selected_suffixes))
print(selected_suffixes)


# %%

def suffix_count(text, source_docs, suffix_list):
    suffix_counts = {}
    doc = source_docs.get(" ".join(text))
    suffixes = []
    for word in doc:
        suffixes.append(str(word.suffix_))
    counts = Counter(suffixes)
    for suffix in suffix_list:
        if suffix in counts.keys():
            suffix_counts[suffix] = counts.get(suffix)
        else:
            suffix_counts[suffix] = 0
    return suffix_counts

def initialize_dataset(source, source_docs):
    all_features = []
    targets = []
    for (sent, label) in source:
        feature_list = []
        feature_list.append(avg_number_chars(sent))
        feature_list.append(number_words(sent))
        counts = word_counts(sent)
        for word in STOP_WORDS:
            if word in counts.keys():
                feature_list.append(counts.get(word))
            else:
                feature_list.append(0)
        feature_list.append(proportion_words(sent, STOP_WORDS))
        p_counts = pos_count(sent, source_docs, pos_list)
        for pos in p_counts.keys():
            feature_list.append(float(p_counts.get(pos))/float(len(sent)))
        s_counts = suffix_count(sent, source_docs, selected_suffixes)
        for suffix in s_counts.keys():
            feature_list.append(float(s_counts.get(suffix))/ float(len(sent)))
        
        all_features.append(feature_list)
        if label == 'dickens':
            targets.append(0)
        else:
            targets.append(1)
    return all_features, targets

run()



# %%

def unique_vocabulary(label1, label2, cutoff):
    voc1 = []
    voc2 = []
    for (sent, label) in strat_train_list:
        if label == label1:
            for word in sent:
                voc1.append(word.lower())
        elif label1 == label2:
            for word in sent:
                voc2.append(word.lower())
    counts1 = Counter(voc1)            
    sorted_counts1 = sorted(counts1.items(), key=operator.itemgetter(1), reverse=True)
    counts2 = Counter(voc2)
    sorted_counts2 = sorted(counts2.items(), key=operator.itemgetter(1), reverse=True)

    unique_voc = []
    for i in range(0, round(len(sorted_counts1)*cutoff)):
        if not sorted_counts1[i][0] in counts2.keys():
            unique_voc.append(sorted_counts1[i][0])
    for i in range(0, round(len(sorted_counts2)*cutoff)):
        if not sorted_counts2[i][0] in counts1.keys():
            unique_voc.append(sorted_counts2[i][0])
    return unique_voc

unique_voc = unique_vocabulary('dickens', 'doyle', 0.5)
print(len(unique_voc))
print(unique_voc[:10])

# %%

def unique_counts(text, unique_voc):
    unique_counts = {}
    words = []
    for word in text:
        words.append(word.lower())
    counts = Counter(words)
    for word in unique_voc:
        if word in counts.keys():
            unique_counts[word] = counts.get(word)
        else:
            unique_counts[word] = 0
    return unique_counts

def initialize_dataset(source, source_docs):
    all_features = []
    targets = []
    for (sent, label) in source:
        feature_list = []
        feature_list.append(avg_number_chars(sent))
        feature_list.append(number_words(sent))
        counts = word_counts(sent)
        for word in STOP_WORDS:
            if word in counts.keys():
                feature_list.append(counts.get(word))
            else: 
                feature_list.append(0)
        feature_list.append(proportion_words(sent, STOP_WORDS))
        p_counts = pos_count(sent, source_docs, pos_list)
        for pos in p_counts.keys():
            feature_list.append(float(p_counts.get(pos)) / float(len(sent)))
        s_counts = suffix_count(sent, source_docs, selected_suffixes)
        for suffix in s_counts.keys():
            feature_list.append(float(s_counts.get(suffix)) / 
                                float(len(sent)))        
        u_counts = unique_counts(sent, unique_voc)
        for word in u_counts.keys():
            feature_list.append(u_counts.get(word))
        all_features.append(feature_list)
        if label == 'dickens':
            targets.append(0)
        else:
            targets.append(1)
    return all_features, targets

run()


# %%





