#%%
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg

gutenberg.fileids()

# %% 5.2 Code to define training and test set

nltk.download('punkt')

author1_train = gutenberg.sents('austen-emma.txt') + gutenberg.sents('austen-persuasion.txt')
print(author1_train)
print(len(author1_train))

author1_test = gutenberg.sents('austen-sense.txt')
print(len(author1_test))

author2_train = gutenberg.sents('shakespeare-caesar.txt') + gutenberg.sents('shakespeare-hamlet.txt')
print(author2_train)
print(len(author2_train))

author2_test = gutenberg.sents('shakespeare-macbeth.txt')
print(author2_test)
print(len(author2_test))

#%% 5.3 Code to calculate simple statistics on texts

def statistics(gutenberg_data):
    for work in gutenberg_data:
        num_chars = len(gutenberg.raw(work))
        num_words = len(gutenberg.words(work))
        num_sents = len(gutenberg.sents(work))
        num_vocab = len(set(w.lower() for w in gutenberg.words(work)))

        print(round(num_chars/num_words),
            round(num_words/num_sents),
            round(num_words/num_vocab), work)

gutenberg_data = ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt',
                  'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt']

statistics(gutenberg_data)

#%% 5.4 Run StratifiedShuffleSplit on the data

from sklearn.model_selection import StratifiedShuffleSplit

all_sents = [(sent, 'austen') for sent in author1_train]
all_sents += [(sent, 'shakespeare') for sent in author2_train]
print(f'Dataset size = {str(len(all_sents))} sentences')

values = [author for (sent, author) in all_sents]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_train_set = []
strat_pretest_set = []
for train_index, pretest_index in split.split(all_sents, values):
    strat_train_set = [all_sents[index] for index in train_index]
    strat_pretest_set = [all_sents[index] for index in pretest_index]

#%% 5.5 Check the proportions of the data in the two classes
def cat_proportions(data, cat):
    count = 0
    for item in data:
        if item[1] == cat:
            count += 1
    return float(count) / float(len(data))

categories = ['austen', 'shakespeare']
rows = []
rows.append(['Category', 'Overall', 'Stratified train', 'Stratified pretest'])
for cat in categories:
    rows.append([cat, f'{cat_proportions(all_sents, cat):.6f}',
                f'{cat_proportions(strat_train_set, cat):.6f}',
                f'{cat_proportions(strat_pretest_set, cat):.6f}'])

columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i])
                  for i in range(0, len(row))))

#%% 5.6 Code to create the test_set data structure

test_set = [(sent, 'austen') for sent in author1_test]
test_set += [(sent, 'shakespeare') for sent in author2_test]


# %% 5.7 Code to extract word as features

def get_features(text):
    features = {}
    word_list = [word for word in text]
    for word in word_list:
        features[word] = True
    return features

train_features = [(get_features(sents), label) for (sents, label) in strat_train_set] 
pretest_features = [(get_features(sents), label) for (sents, label) in strat_pretest_set]

print(len(train_features))
print(train_features[0][0])
print(train_features[100][0])

#%% 5.8 Code to train the Naive Bayes classifier on train and test on pretest set

from nltk import NaiveBayesClassifier, classify

print(f'Training set size = {str(len(train_features))} sentences')
print(f'Pretest set size = {str(len(pretest_features))} sentences')
classifier = NaiveBayesClassifier.train(train_features)

print(f'Accuracy on the training set = {str(classify.accuracy(classifier, train_features))}')
print(f'Accuracy on the perTest set = {str(classify.accuracy(classifier, pretest_features))}')
classifier.show_most_informative_features(10)


# %% 5.9 Code to test the classifier on the test set

test_features = [(get_features(sents), label) for (sents,label) in test_set]
print(f'Test set size = {str(len(test_features))} senteces')
print(f'Accuracy on the test set = {str(classify.accuracy(classifier, test_features))}')

# %% 5.10 Code to estimate document frequencies for all words in the training set

from collections import Counter

words = []

def extract_words(text,words):
    words += set([word for word in text])
    return words

for (sents, label) in strat_train_set:
    words = extract_words(sents,words)

counts = Counter(words)
print(counts)

# %% 5.11 Code to run DecisionTreeClassifier with the selected features
from nltk import DecisionTreeClassifier

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







# %%
