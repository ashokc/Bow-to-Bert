# fasttext_sentence_similarity.py
import subprocess, os, sys
import numpy as np

with open('sentences.txt', 'r') as f:
    sentences = f.readlines()
    sentences = [x.strip() for x in sentences]
    if len(sentences) != 3:
        print ("Need 3 sentences for testing similarity")
        sys.exit(0)

words_by_sentence, words = {}, set()
wordVectors = {}
for i,sentence in enumerate(sentences):
    words_by_sentence[i] = sentence.rstrip().split(' ')
    words.update(words_by_sentence[i])

print("Loading word vectors...this will take a while")
filename = os.environ["PRE_TRAINED_HOME"] + '/fasttext/crawl-300d-2M-subword.vec'
f = open(filename, "r", encoding='utf-8')
lines = f.readlines()
for i,line in enumerate(lines):
    if i % 10000 == 0:
        print(i,"/ 2000000")
    line = line
    token = line.split(' ')
    wordVectors[token[0]] = token[1:]

f.close()

wordVectorLength, zeroVectorCount = 300, 0
docVectors = np.zeros( (3, wordVectorLength), dtype='float32')

for i,word in enumerate(words):
    tokens = wordVectors.get(word)
    wv = np.asarray(tokens, dtype='float32')
    if word in wordVectors:
        for i in range(3):
            if word in words_by_sentence[i]:
                docVectors[i] = docVectors[i] + wv/np.linalg.norm(wv)
    else:
        zeroVectorCount = zeroVectorCount + [word]

print ('# words not found in fasttext..', zeroVectorCount)

for i in range(3):
    docVectors[i] = docVectors[i]/np.linalg.norm(docVectors[i])

print ('Cosine Similarity:',sentences[0], '&', sentences[1], ':', np.dot(docVectors[0], docVectors[1]))
print ('Cosine Similarity:',sentences[0], '&', sentences[2], ':', np.dot(docVectors[0], docVectors[2]))
print ('Cosine Similarity:',sentences[1], '&', sentences[2], ':', np.dot(docVectors[1], docVectors[2]))

