import random as rn
import numpy as np
import json

bertVectors, sentences = {}, []
with open("./bertWordVectors.jsonl") as f:
    jsonlines = f.readlines()
    jsonlines = [x.strip() for x in jsonlines]
    for j, jsonline in enumerate(jsonlines):
        json_content = json.loads(jsonline) 
        allTokens = [feature['token'] for feature in json_content['features']]
        sentences.append(' '.join(allTokens[1:-1]))  # Exclude CLS & SEP
        bertVectors[j] = {}
        for i, token in enumerate(allTokens[1:-1]):
            wv = np.array(json_content['features'][i+1]['layers'][1]['values'])
            bertVectors[j][token] = wv/np.linalg.norm(wv)

def checkPairs (iStart, word):
    print (sentences[iStart], ' <=> ', sentences[iStart+1] , '\t\t\t<=> ', round(np.dot(bertVectors[iStart][word], bertVectors[iStart+1][word]),3))
    print (sentences[iStart], ' <=> ', sentences[iStart+2] , '\t\t\t<=> ', round(np.dot(bertVectors[iStart][word], bertVectors[iStart+2][word]),3))
#    print (sentences[iStart+1], ' <=> ', sentences[iStart+2] , '\t\t\t<=> ', round(np.dot(bertVectors[iStart+1][word], bertVectors[iStart+2][word]),3))

#0 Arms bend at the elbow
#1 Germany sells arms to Saudi Arabia 
#2 Wave your arms around

checkPairs (0, 'arms')

#3 Boil the solution with salt
#4 The problem has no solution
#5 Heat the solution to 75 degrees

print('\n')
checkPairs (3, 'solution')

#6 economics an arts subject
#7 All income is subject to tax 
#8 I have one subject for credit this quarter

print('\n')
checkPairs (6, 'subject')

#9 The key broke in the lock
#10 The key problem was not one of quality but of quantity
#11 Where is the key

print('\n')
checkPairs (9, 'key')

print('\n')
checkPairs (12, 'play')

