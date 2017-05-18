import sys
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

filename = sys.argv[1]

with open(filename) as f:
    next(f)
    word_pairs = []
    for line in f:
        lemm_words = [lemmatizer.lemmatize(word).lower() for word in line.strip().split(',')[1:]]
        word_pairs.append(lemm_words)

with open('lemm_' + filename, 'w') as f:
    f.write('id,word1,word2\n')
    for i, word_pair in enumerate(word_pairs):
        f.write('{},{},{}\n'.format(i, word_pair[0], word_pair[1]))


