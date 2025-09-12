# %% 8.1 Code to access SentiWordNet and check induvidual words
import nltk
nltk.download('wordnet')
nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet as swn

print(list(swn.senti_synset('joy')))
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






#%%
    
    