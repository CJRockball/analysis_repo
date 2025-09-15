#%%
import os
import codecs
import nltk
from nltk import word_tokenize
nltk.download('punkt')
import random

random.seed(42)

def read_in(folder):
    files = os.listdir(folder)
    a_list = []
    for a_file in files:
        if not a_file.startswith("."):
            f = codecs.open(folder + a_file, 'r', encoding="ISO-8859-1", errors='ignore')
            a_list.append(f.read())
            f.close()
    return a_list
            
# Load data in list

spam_list = read_in('data/enron1/spam/')
ham_list = read_in('data/enron1/ham/')

print(len(spam_list))
print(len(ham_list))
print(spam_list[0])
print('***')
print(ham_list[1])

# %% Put all mails in the same list, with label


all_emails = [(email_content, 'spam') for email_content in spam_list]
all_emails += [(email_content, 'ham') for email_content in ham_list]


print(f'Dataset size = {str(len(all_emails))} emails')

# %% tokenize text

def tokenize(input):
    word_list = []
    for word in word_tokenize(input):
        word_list.append(word)
    return word_list

sentence = "What's the best way to split a sentence into words?"
print(tokenize(sentence))

#%%

def get_features(text):
    features = {}
    word_list = [word for word in word_tokenize(text.lower())]
    for word in word_list:
        features[word] = True
    return features
    
all_features = [(get_features(email), label) for (email, label) in all_emails]

print(get_features("Patricipate In Out Lottery Now!"))
print(len(all_features))
print(len(all_features[0][0]))
print(all_features[99])
    
# %%

from nltk import NaiveBayesClassifier, classify

def train(features, proportion):
    train_size = int(len(features) * proportion)
    train_set = features[:train_size]
    test_set = features[train_size:]
    print(f"Training set size = {str(len(train_set))} emails")
    print(f"Test set size = {str(len(test_set))} emails")
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier

train_set, test_set, classifier = train(all_features, 0.8)

# %%

def evaluate(train_set, test_set, classifier):
    print(f'Accuracy on the training set = {str(classify.accuracy(classifier, train_set))}')
    print(f'Accuracy of the test set = {str(classify.accuracy(classifier,test_set))}')
    classifier.show_most_informative_features(50)

evaluate(train_set, test_set, classifier)

# %% Check word in context

from nltk.text import Text

def concordance(data_list, search_word):
    for email in data_list:
        word_list = [word for word in word_tokenize(email.lower())]
        text_list = Text(word_list)
        if search_word in word_list:
            text_list.concordance(search_word)

print('STOCKS in HAM')
concordance(ham_list, 'stocks')

print("\nSTOCKS in SPAM:")
concordance(spam_list, 'stocks')

# %%

test_spam_list = ['Participate in our new lottery!', 'Try out this new medicine']
test_ham_list = ['See the minutes from the last meeting attached', 'Investors are coming to our office on Monday']

test_emails = [(email_content, 'spam') for email_content in test_spam_list]
test_emails += [(email_content, 'ham') for email_content in test_ham_list]

new_test_set = [(get_features(email), label) for (email, label) in test_emails]

evaluate(train_set, new_test_set, classifier)

# %%

for email in test_spam_list:
    print(email)
    print(classifier.classify(get_features(email)))

for email in test_ham_list:
    print(email)
    print(classifier.classify(get_features(email)))





# %%
