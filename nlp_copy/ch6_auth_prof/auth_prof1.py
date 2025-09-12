#%%
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
nltk.download('punkt')
from sklearn.model_selection import StratifiedShuffleSplit
from nltk import NaiveBayesClassifier, classify
from collections import Counter
from nltk import DecisionTreeClassifier

print(gutenberg.fileids())

# Get text for train and test
author1_train = gutenberg.sents('austen-emma.txt') + gutenberg.sents('austen-persuasion.txt')
author1_test = gutenberg.sents('austen-sense.txt')

author2_train = gutenberg.sents('shakespeare-caesar.txt') + gutenberg.sents('shakespeare-hamlet.txt')
author2_test = gutenberg.sents('shakespeare-macbeth.txt')

#%% Create train test data
# Get train/test split 
# Combine all text in one list
all_sents = [(sent, 'austen') for sent in author1_train]
all_sents += [(sent, 'shakespeare') for sent in author2_train]

# Get list of author order in training list
values = [author for (sent, author) in all_sents]
# Get array of rows one for train, one for test
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# Iterate through, splitting to train/test list X=all_sents, y=values
strat_train_set = []
strat_pretest_set = []
for train_index, pretest_index in split.split(all_sents, values):
    strat_train_set = [all_sents[index] for index in train_index]
    strat_pretest_set = [all_sents[index] for index in pretest_index]

# Make a auth1, auth2 combined test set
test_set = [(sent, 'austen') for sent in author1_test]
test_set += [(sent, 'shakespeare') for sent in author2_test]

#%% Get word features from sentences
def get_features(text):
    features = {}
    word_list = [word for word in text]
    for word in word_list:
        features[word] = True
    return features

train_features = [(get_features(sents), label) for (sents, label) in strat_train_set] 
pretest_features = [(get_features(sents), label) for (sents, label) in strat_pretest_set]
  
#Train model check accuracy and most informative features
classifier = NaiveBayesClassifier.train(train_features)

acc_train = classify.accuracy(classifier, train_features)
acc_pretest = classify.accuracy(classifier, pretest_features)

print(f'Accuracy on the training set = {str(acc_train)}')
print(f'Accuracy on the perTest set = {str(acc_pretest)}')
classifier.show_most_informative_features(10)

# Get word features from test set, predict labels, calculate accuracy
test_features = [(get_features(sents), label) for (sents,label) in test_set]
print(f'Test set size = {str(len(test_features))} senteces')
print(f'Accuracy on the test set = {str(classify.accuracy(classifier, test_features))}')


# %%

words = []
def extract_words(text,words):
    words += set([word for word in text])
    return words

for (sents, label) in strat_train_set:
    words = extract_words(sents,words)

counts = Counter(words)
print(counts)

maximum = float(13414)

selected_words = []
for item in counts.items():
    count = float(item[1])
    if count > 200 and count/maximum < 0.2:
        selected_words.append(item[0])
print(len(selected_words))

def get_features(text, selected_words):
    features = {}
    word_list = [word for word in text]
    for word in word_list:
        if word in selected_words:
            features[word] = True
    return features


train_features = [(get_features(sents, selected_words), label) for (sents, label) in strat_train_set]
pretest_features = [(get_features(sents, selected_words), label) for (sents, label) in strat_pretest_set]
test_features = [(get_features(sents, selected_words), label) for (sents, label) in test_set]

classifier = DecisionTreeClassifier.train(train_features)

print(f'Accuracy on the training set = {str(classify.accuracy(classifier, train_features))}')
print(f'Accuracy on the pretest set = {str(classify.accuracy(classifier, pretest_features))}')
print(f'Accuracy on the test set = {str(classify.accuracy(classifier, test_features))}')



# %% =================  CH 6  ===========================================

def avg_number_chars(text):
    total_chars = 0.0
    for word in text:
        total_chars += len(word)
    return float(total_chars) / float(len(text))

def number_words(text):
    return float(len(text))

#%% 6.2 Code to extract features and map them to the labels

def initialize_dataset(source):
    all_features = []
    targets = []
    for (sents, label) in source:
        feature_list = []
        feature_list.append(avg_number_chars(sents))
        feature_list.append(number_words(sents))
        all_features.append(feature_list)
        if label == 'austen':
            targets.append(0)
        else:
            targets.append(1)
    return all_features, targets

train_data, train_targets = initialize_dataset(strat_train_set)
pretest_data, pretest_targets = initialize_dataset(strat_pretest_set)
test_data, test_targets = initialize_dataset(test_set)

print(len(train_data), len(train_targets))
print(len(pretest_data), len(pretest_targets))
print(len(test_data), len(test_targets))

#%% 6.3 Code to train and test a classifier with sklearn

from sklearn.tree import DecisionTreeClassifier

text_clf = DecisionTreeClassifier(random_state=42)
text_clf.fit(train_data, train_targets)
predicted = text_clf.predict(pretest_data)

#%%
import numpy as np
from sklearn import metrics

def evaluate(predicted, targets):
    print(np.mean(predicted == targets))
    print(metrics.confusion_matrix(targets, predicted))
    print(metrics.classification_report(targets, predicted))

evaluate(predicted, pretest_targets)
predicted = text_clf.predict(test_data)
evaluate(predicted, test_targets)

#%% 6.5 Code to calculate the number and 
# proportion of times certain words occur

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
    return float(count) / float(len(text))

#%% COde to add stopword counts and proportion as features
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
        feature_list.append(proportion_words(sent, STOP_WORDS))
        all_features.append(feature_list)
        if label == 'austen':
            targets.append(0)
        else: 
            targets.append(1)
    return all_features, targets

train_data, train_targets = initialize_dataset(strat_train_set)
pretest_data, pretest_targets = initialize_dataset(strat_pretest_set)
test_data, test_targets = initialize_dataset(test_set)

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

#%%



#%%



#%%

import spacy

print(spacy.__version__)



# %%