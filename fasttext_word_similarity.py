
import sys, subprocess, os
import numpy as np

args = sys.argv
if (len(args) < 4):
    print ("Need 3 words for testing similarity")
    sys.exit(0)
else:
    word1 = args[1]
    word2 = args[2]
    word3 = args[3]

wordVectorLength = 300
filename = os.environ["PRE_TRAINED_HOME"] + '/fasttext/crawl-300d-2M-subword.vec'
words = [word1, word2, word3]
wordvectors = []
for word in words:
    w = "^'" + word + " ' "
    s = subprocess.check_output('/bin/grep ' + w + filename, shell=True).decode("utf-8")
    tokens = s.rstrip().split(' ')
    wv = np.asarray(tokens[1:], dtype='float32')
    if (len(wv) == wordVectorLength):
        wordvectors.append(wv/np.linalg.norm(wv))
    else:
        print (word + ' not found in fasttext...')
        sys.exit(0)

print ('Cosine Similarity:',words[0], '&', words[1], ':', np.dot(wordvectors[0], wordvectors[1]))
print ('Cosine Similarity:',words[0], '&', words[2], ':', np.dot(wordvectors[0], wordvectors[2]))
print ('Cosine Similarity:',words[1], '&', words[2], ':', np.dot(wordvectors[1], wordvectors[2]))

#grep ^'john ' glove.6B.300d.txt
