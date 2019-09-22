# fasttext_sentence_similarity.py
import subprocess, os, sys
import numpy as np

with open('sentences.txt', 'r') as f:
    sentences = f.readlines()
    sentences = [x.strip() for x in sentences]
    if len(sentences) != 3:
        print ("Need 3 sentences for testing similarity")
        sys.exit(0)

words_by_sentence, words = {}, []
for i,sentence in enumerate(sentences):
    words_by_sentence[i] = sentence.rstrip().split(' ')
    words = words + words_by_sentence[i]

wordVectorLength, zeroVectorCount = 300, 0
filename = os.environ["PRE_TRAINED_HOME"] + '/fasttext/crawl-300d-2M-subword.vec'
docVectors = np.zeros( (3, wordVectorLength), dtype='float32')
for word in words:
    w = "^'" + word + " ' "
    s = subprocess.check_output('/bin/grep ' + w + filename, shell=True).decode("utf-8")
    tokens = s.rstrip().split(' ')
    wv = np.asarray(tokens[1:], dtype='float32')
    if (len(wv) == wordVectorLength):
        for i in range(3):
            if word in words_by_sentence[i]:
                docVectors[i] = docVectors[i] + wv/np.linalg.norm(wv)
    else:
        zeroVectorCount = zeroVectorCount + 1
print ('# words not found in fasttext..', zeroVectorCount)

for i in range(3):
    docVectors[i] = docVectors[i]/np.linalg.norm(docVectors[i])

print ('Cosine Similarity:',sentences[0], '&', sentences[1], ':', np.dot(docVectors[0], docVectors[1]))
print ('Cosine Similarity:',sentences[0], '&', sentences[2], ':', np.dot(docVectors[0], docVectors[2]))
print ('Cosine Similarity:',sentences[1], '&', sentences[2], ':', np.dot(docVectors[1], docVectors[2]))

