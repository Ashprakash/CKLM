# -*- coding: utf-8 -*-

"""
Created on Mon Sep 24 02:14:23 2018
"""

import json
import os
import pandas as pd

from os import listdir
from os.path import isfile, join
from collections import defaultdict
from stanfordcorenlp import StanfordCoreNLP

props={'annotators': 'lemma', 'pipelineLanguage':'en', 'outputFormat':'json', 'parse.maxlen':"1000", "timeout":"5000"}
nlp = StanfordCoreNLP(r'/mnt/c/Masters Studies/interests/question answering/stanford-corenlp-full-2018-02-27/stanford-corenlp-full-2018-02-27')

def getLemmatizedVerb(verb, nlp):
    global props
#     wordnet_lemmatizer = WordNetLemmatizer()
    zz=nlp.annotate(verb, properties=props)
    zz=json.loads(zz)
    return zz['sentences'][0]['tokens'][0]['lemma']

def getLemmatizedVReplacedSentence(sentence, verb, nlp):
    global props
    zz=nlp.annotate(sentence, properties=props)
    zz=json.loads(zz)
    toks = zz['sentences'][0]['tokens']
    reqTokens = []
    for tok in toks:
        if tok['lemma']==verb:
            reqTokens.append("V")
        else:
            reqTokens.append(tok['word'].lower())
    newSentence = " ".join(reqTokens)       
    return newSentence

def process_qasrl_output(qasrl_output, pronoun):
    qa_pairs_array = []
    json_obj = json.loads(qasrl_output)
    sent_tokens = json_obj["words"]
    verbs_objs = json_obj["verbs"]
    for verbs_obj in verbs_objs:
        verb = verbs_obj["verb"]
        qa_pairs = verbs_obj["qa_pairs"]
        for qa_pair in qa_pairs:
            ques = qa_pair["question"]
            answers = qa_pair["spans"]
            ans_is_pronoun = False
            for ans in answers:
                ans_text = ans["text"]
                if ans_text==pronoun:
                    ans_is_pronoun = True
                    break
            if ans_is_pronoun:
                qa_pair = {'ques':ques, 'ans':pronoun, 'verb':verb}
                qa_pairs_array.append(qa_pair)
            else:
                for ans in answers:
                    qa_pair = {'ques':ques, 'ans':ans["text"], 'verb':verb}
                    qa_pairs_array.append(qa_pair)
    
    return qa_pairs_array
            
        


if __name__=="__main__":
    relationsPath = "data/qasrl_outputs_processed"
    mypath = "data/qasrl_outputs"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    paragraphToOutputFiles = defaultdict(list)
    _ = {paragraphToOutputFiles[f.split('_')[0]].append(f) for f in onlyfiles}
    xl = pd.ExcelFile("/mnt/c/Masters Studies/interests/question answering/qasrl-forked/ProPara.xlsx")
    df = xl.parse('TrainDevTest')
    tempDict = df.to_dict("records")
    pidToPartition = {x['Paragraph ID']:x['Partition'] for x in tempDict}
    nlp = StanfordCoreNLP(r'/mnt/c/Masters Studies/interests/question answering/stanford-corenlp-full-2018-02-27/stanford-corenlp-full-2018-02-27')
    
    count = 1
    for paragraphString, sentences in paragraphToOutputFiles.items():
        paragraphId = int(paragraphString)
        if (count%50==0):
            try:
                nlp.close()
            except Exception as e:
                pass
            nlp = StanfordCoreNLP(r'/mnt/c/Masters Studies/interests/question answering/stanford-corenlp-full-2018-02-27/stanford-corenlp-full-2018-02-27') 
        partition = pidToPartition[paragraphId]
        print(str(paragraphId))
        paragraphFile = open(join(relationsPath, partition, str(paragraphId) + ".sample"), "w")
        for sentence in sentences:
            sentenceTime = int(sentence[len(sentence)-5])
            jsonObject = {}
            if (os.stat(join(mypath, sentence)).st_size!=0):
                jsonObject = json.load(open(join(mypath, sentence)))
            else:
                continue
            for verbInst in jsonObject['verbs']:
                for qaPair in verbInst['qa_pairs']:
                    question = qaPair['question']
                    lemVerb = getLemmatizedVerb(verbInst['verb'], nlp)
                    lemQuestion = getLemmatizedVReplacedSentence(question, lemVerb, nlp)
                    for answer in qaPair['spans']:
                        paragraphFile.write("observedAt(")
                        paragraphFile.write("\"" + lemVerb.lower() + "\"")
                        paragraphFile.write(",")
                        paragraphFile.write("\"" + lemQuestion + "\"")
                        paragraphFile.write(",")
                        paragraphFile.write("\"" + answer['text'].lower() + "\"")
                        paragraphFile.write(",")
                        paragraphFile.write(str(sentenceTime))
                        paragraphFile.write(")\n")
        try:
            nlp.close()               
        except Exception as e:
            pass
#        print("no process to close")
        paragraphFile.close()  
        count+=1          




