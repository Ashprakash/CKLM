import re
import json
import ast
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive

use_predictor = True
verbose_output = True

class PretrainedModel:
    """
    A pretrained model is determined by both an archive file
    (representing the trained model)
    and a choice of predictor.
    """
    def __init__(self, archive_file: str, predictor_name: str) -> None:
        self.archive_file = archive_file
        self.predictor_name = predictor_name

    def predictor(self) -> Predictor:
        archive = load_archive(self.archive_file)
        return Predictor.from_archive(archive, self.predictor_name)


if use_predictor:
    model = PretrainedModel('./esim-elmo-2018.05.17.tar.gz','textual-entailment')
    predictor = model.predictor()
    #predictor = Predictor.from_path("./decomposable-attention-elmo-2018.02.19.tar.gz")
wn_lemmatizer = WordNetLemmatizer()

#ques_type_lists = [["What V?"],["What was V?"],["Who V someone something ?","Who V to do something ?"]]

all_ques = set()
aux_verbs_list = ["does","was","am","is","did","are","were","have","has","had","be"]
sim_verbs_lists = [["rush","zoom"],["upset","frustrate"],["make","get"],["protect","guard"],["beat","defeat"],["appear","emerge","surface"],["stop","prevent"],["convey","show"],["respond","answer"],["offer","give","pass"],["lift","carry"],["ache","hurt"],["beat","defeat"],["release","throw"],["say","ask"],["follow","obey"],["love","like"]]
#sim_verbs_lists = [["rushed","zoomed"],["upset","frustrated"],["made","gotten"],["fell","fall"],["protect","guard"],["beat","defeated"],["appeared","emerged","surfaced"],["stop","prevent"],["conveys","showing"],["respond","answer"],["offered","gave","pass","passed","give"],["lifting","carried"],["ached","hurting"],["beat","defeated"],["releases","threw"],["said","asked"],["follows","obey"]]

i_pronouns = ["i","me","mine","myself","my"]
he_pronouns = ["he","his","him","himself"]
she_pronouns = ["she","her","herself"]
you_pronouns = ["you","yourself","your"]

list_of_sim_words = [("we","us")]
list_of_dissim_words = [("there","i"),("them","you"),("i","it")]

similar_file = open("similar_pair.json","r")
similar_pair = json.load(similar_file)

def are_same(phrase1,phrase2):
    if phrase1==phrase2:
        return True
    if (phrase1,phrase2) in list_of_sim_words:
        return True
    if (phrase2,phrase1) in list_of_sim_words:
        return True
    if phrase1 in i_pronouns and phrase2 in i_pronouns:
          return True
    if phrase1 in he_pronouns and phrase2 in he_pronouns:
          return True
    if phrase1 in she_pronouns and phrase2 in she_pronouns:
          return True  
    if phrase1 in you_pronouns and phrase2 in you_pronouns:
          return True

    return False

def are_not_same(phrase1,phrase2):
    if phrase1 in i_pronouns and (phrase2 in he_pronouns or phrase2 in she_pronouns or phrase2 in you_pronouns):
        return True
    if phrase1 in he_pronouns and (phrase2 in i_pronouns or phrase2 in she_pronouns or phrase2 in you_pronouns):
        return True
    if phrase1 in she_pronouns and (phrase2 in he_pronouns or phrase2 in i_pronouns or phrase2 in you_pronouns):
        return True
    if phrase1 in you_pronouns and (phrase2 in he_pronouns or phrase2 in she_pronouns or phrase2 in i_pronouns):
        return True

    if (phrase1,phrase2) in list_of_dissim_words:
        return True
    if (phrase2,phrase1) in list_of_dissim_words:
        return True

    return False

def populate_ques_type_list():
    global ques_type_lists
    file_path = "similar_questions_lists.txt"
    with open(file_path,"r") as f:
        ques_type_lists = json.loads(f.read())


def get_ques_with_ans(word, qa_pairs):
    list_of_qas = []
    for qa_pair in qa_pairs:
        ques = qa_pair["ques"]
        ans = qa_pair["ans"]
        ans_tokens = ans.split(" ")
        for ans_token in ans_tokens:
            if word==ans_token:
                list_of_qas.append((ques,ans,qa_pair["verb"]))
                break
    return list_of_qas


def generate_qa_dict(qa_pairs):
    ques_ans_dict = {}
    for qa_pair in qa_pairs:
        ques_ans_dict[qa_pair["ques"]] = qa_pair["ans"]
    return ques_ans_dict

def same_type_from_list(ques1,ques2):
    #print("IN TOP TOP: ",ques1,"   ",ques2)
    for ques_type_list in ques_type_lists:
        if (ques1 in ques_type_list) and (ques2 in ques_type_list):
            #print("IN TOP: ",ques1,"   ",ques2)
            return True
        #else:
        #    print("NO")

def same_type_score(ques1, ques2):
    #print("QUES1: ",ques1)
    #print("QUES2: ",ques2)
    if ques1==ques2:
        return 1.0
    elif same_type_from_list(ques1,ques2):
        return 1.0
    else:
        return 0.0

def get_words_similarity(word1,word2):
    if word1==word2:
        return 1.0
    if word1 in aux_verbs_list and word2 in aux_verbs_list:
        return 1.0
    for sim_verbs_list in sim_verbs_lists:
        if word1 in sim_verbs_list and word2 in sim_verbs_list:
            return 1.0

    syn1s = wn.synsets(word1)
    if syn1s==[]:
        return 0.0
    syn1 = syn1s[0]
    syn2s = wn.synsets(word2)
    if syn2s==[]:
        return 0.0
    syn2 = syn2s[0]
    sim = syn1.path_similarity(syn2)
    return sim

def get_lemma(word,pos_tag):
    if word.lower()=="felt":
        return "feel"
    lemma = wn_lemmatizer.lemmatize(word.lower(),pos=pos_tag)
    return lemma.lower()

def is_similar(ws_qa_pair,know_qa_pair):
    ws_ques = ws_qa_pair["ques"]
    ws_verb = ws_qa_pair["verb"]
    know_ques = know_qa_pair["ques"]
    know_verb = know_qa_pair["verb"]

    all_ques.add(ws_ques)
    all_ques.add(know_ques)

    total_sim_score = 0.0
    total_sim_score += same_type_score(ws_ques,know_ques)
    ws_verb_lemma = get_lemma(ws_verb,'v')
    know_verb_lemma = get_lemma(know_verb,'v')

    if ws_verb_lemma==know_verb_lemma:
        total_sim_score += 1.0
    else:
        verb_sim = get_words_similarity(ws_verb_lemma,know_verb_lemma)
        if verb_sim is not None:
            total_sim_score += verb_sim
        else:
            total_sim_score += 0.0

    if total_sim_score > 1.4:
        return True
    else:
        return False

def get_similar_ques(ws_qa_pairs, know_qa_pairs):
    dict_of_similar_ques = {}
    dict_of_similar_ans = {}
    for ws_qa_pair in ws_qa_pairs:
        sim_ques = ""
        sim_ans = ""
        for know_qa_pair in know_qa_pairs:
            sim_score = is_similar(ws_qa_pair,know_qa_pair)
            if sim_score is True:
                sim_ques = know_qa_pair["ques"]
                sim_ans = know_qa_pair["ans"]
                dict_of_similar_ques[(ws_qa_pair["ques"],ws_qa_pair["verb"])] = (sim_ques, ws_qa_pair["verb"])
                if (ws_qa_pair["ans"],ws_qa_pair["verb"]) in dict_of_similar_ans:
                    set_of_sim_ans = dict_of_similar_ans[(ws_qa_pair["ans"],ws_qa_pair["verb"])]
                else:
                    set_of_sim_ans = set()
                set_of_sim_ans.add((sim_ans,ws_qa_pair["verb"]))
                dict_of_similar_ans[(ws_qa_pair["ans"],ws_qa_pair["verb"])] = set_of_sim_ans
        if (ws_qa_pair["ques"],ws_qa_pair["verb"]) not in dict_of_similar_ques.keys():
            dict_of_similar_ques[(ws_qa_pair["ques"],ws_qa_pair["verb"])] = ("",ws_qa_pair["verb"])
            ans_set = set()
            dict_of_similar_ans[(ws_qa_pair["ans"],ws_qa_pair["verb"])] = ans_set

    return dict_of_similar_ques,dict_of_similar_ans

def get_max(ent_contr_scores):
    dist_bw_ent_and_contr = 0.0
    (sec_max_ent,sec_max_contr) = ent_contr_scores[0]
    (max_ent,max_contr) = ent_contr_scores[0]
    for (ent,contr) in ent_contr_scores:
        if ent!=1.0 and sec_max_ent < ent:
            sec_max_ent = ent
            sec_max_contr = contr
        if ent > max_ent:
            max_ent = ent
            max_contr = contr
    return (max_ent,max_contr,sec_max_ent,sec_max_contr)

def words_are_similar(word1,word2):
    if word1==word2:
        return True
    else:
        if word1 in i_pronouns and word2 in i_pronouns:
            return True
        elif word1 in he_pronouns and word2 in he_pronouns:
            return True
        elif word1 in she_pronouns and word2 in she_pronouns:
            return True
        elif word1 in you_pronouns and word2 in you_pronouns:
            return True
    return False

def get_conf(bucket_ch1,bucket_ch2):
    choice1_conf = 0.0
    choice2_conf = 0.0
    if len(bucket_ch1) > 0:
        if len(bucket_ch2) > 0:
            (ch1_ent,ch1_contr,ch1_sec_ent,ch1_sec_ent) = get_max(bucket_ch1)
            (ch2_ent,ch2_contr,ch2_sec_ent,ch2_sec_ent) = get_max(bucket_ch2)
            if verbose_output:
                print("MAX SCORES FOR CHOICE1: ",(ch1_ent,ch1_contr,ch1_sec_ent,ch1_sec_ent))
                print("MAX SCORES FOR CHOICE2: ",(ch2_ent,ch2_contr,ch2_sec_ent,ch2_sec_ent))
            if ch1_ent > ch2_ent:
                choice1_conf += 1.0
            elif ch1_ent < ch2_ent:
                choice2_conf += 1.0
            else:
                if ch1_sec_ent > ch2_sec_ent:
                    choice1_conf += 1
                else:
                    choice2_conf += 1

        else:
            choice1_conf += 1.0
    else:
        if len(bucket_ch2) > 0:
            (ch2_ent,ch2_contr,ch2_sec_ent,ch2_sec_ent) = get_max(bucket_ch2)
            if verbose_output:
                print("MAX SCORES FOR CHOICE2: ",(ch2_ent,ch2_contr,ch2_sec_ent,ch2_sec_ent))
            choice2_conf += 1.0
    return choice1_conf,choice2_conf

def get_similarity_score(choice_set, k_ans_set, psl, know_no):
    for (e_ans, verb) in choice_set:
        for (e1_ans, e1_verb) in k_ans_set:
            if e_ans.lower() != e1_ans.lower():
                key = e_ans+'$$'+e1_ans
                pair_score = 0.0
                if key not in similar_pair:
                    score = predictor.predict(hypothesis = e_ans, premise = e1_ans)
                    ent_score = score["label_probs"][0]
                    pair_score = ent_score
                    similar_pair[key] = pair_score
                    
                psl["similarity"].append(e_ans+'$$'+e1_ans+'$$'+'K'+know_no+'$$'+str(similar_pair[key]))
                

def main(problem, ws_qa_pairs, know_qa_pairs, psl, know_no):

    ws_sent = problem["ws_sent"]
    ws_pronoun = problem["pronoun"]
    ws_ans = problem["ans"]
    ws_choice1 = problem["choice1"]
    ws_choice2 = problem["choice2"]
    know_sent = problem["know_sent"]

    if verbose_output:
        print("SENT: ",ws_sent)
        print("PRONOUN: ",ws_pronoun)
        print("KNOW SENT: ",know_sent)
        print("ANSWER: ",ws_ans)
        print("CHOICE1: ",ws_choice1)
        print("CHOICE2: ",ws_choice2)

    ans_is_choice1 = False
    if ws_ans.lower()==ws_choice1.lower():
        ans_is_choice1 = True
    print("WS_PAIR "+str(ws_qa_pairs))
    print("KNOW_PAIR "+str(know_qa_pairs))
    # Converting WSC and Knowledge question/answers into dictionaries for easy access
    ws_qa_dict = generate_qa_dict(ws_qa_pairs)
    know_qa_dict = generate_qa_dict(know_qa_pairs)
    print("WS_DICT "+str(ws_qa_dict))
    print("KNOW_DICT "+str(ws_qa_dict))
    # Getting similar (corresponding) questions and answers from WSC sentence and Knowledge sentence
    dict_of_similar_ques, dict_of_sim_ans = get_similar_ques(ws_qa_pairs, know_qa_pairs)
    print("Sim questions "+str(dict_of_similar_ques))
    print("Sim ans "+str(dict_of_sim_ans))
    if verbose_output:
        print(dict_of_similar_ques)
        print(dict_of_sim_ans)

    # Getting WSC question/answers which contain the concerned pronoun in the answers
    ws_qas_with_pronoun_in_ans = get_ques_with_ans(ws_pronoun, ws_qa_pairs)

    # Finding if an answer exists in the WSC question/answers with just the concerned pronoun as the answer
    one_word_pronoun_ans_exists = False
    for (ques,ans,verb) in ws_qas_with_pronoun_in_ans:
        if ans==ws_pronoun or ans.lower()==ws_pronoun:
            one_word_pronoun_ans_exists = True

    # Finding the questions/answers such that the answer(s) correspond to the choice 1 of the given problem
    choice1_set_of_sim_ans = set()
    sim_ans_keys = dict_of_sim_ans.keys()
    for (ans1, verb1) in sim_ans_keys:
        if ans1.lower()== ws_choice1.lower():
            choice1_set_of_sim_ans = choice1_set_of_sim_ans | dict_of_sim_ans[(ans1,verb1)]
        else:
            ans1_tokens = ans1.split(" ")
            if ws_choice1.lower() in ans1_tokens or ws_choice1 in ans1_tokens:
                choice1_set_of_sim_ans = choice1_set_of_sim_ans | dict_of_sim_ans[(ans1,verb1)]

    # Finding the questions/answers such that the answer(s) correspond to the choice 2 of the given problem
    choice2_set_of_sim_ans = set()
    sim_ans_keys = dict_of_sim_ans.keys()
    for (ans1,verb1) in sim_ans_keys:
        if ans1.lower()==ws_choice2.lower():
            choice2_set_of_sim_ans = choice2_set_of_sim_ans | dict_of_sim_ans[(ans1,verb1)]
        else:
            ans1_tokens = ans1.split(" ")
            if ws_choice2.lower() in ans1_tokens or ws_choice2 in ans1_tokens:
                choice2_set_of_sim_ans = choice2_set_of_sim_ans | dict_of_sim_ans[(ans1,verb1)]

    # There are 8 possibilities based on the questions/answers found in the knowledge sentence corresponding to the questions/answers wrt answer choice 1 (q1), answer choice2 (q2) and the concerned pronoun (q3) in the WSC sentence.
    # If q1=F, q2=F, q3=F
    # Something will be done in case any such problem found
    #print("CHOICE2 SIM aS: ",choice2_set_of_sim_ans)
    #print("CHOICE1 SIM aS: ",choice1_set_of_sim_ans)
    ent_comparisons_for_choice1 = set()
    ent_comparisons_for_choice2 = set()
    choice1_ent_contr_scores = [(0.0,0.0)]
    choice2_ent_contr_scores = [(0.0,0.0)]
    for (ques,ans,verb) in ws_qas_with_pronoun_in_ans:
        k_ques_list = dict_of_similar_ques[(ques,verb)]
        if (ans,verb) in dict_of_sim_ans.keys():
            k_ans_list = dict_of_sim_ans[(ans,verb)]
        elif (ans.lower(),verb) in dict_of_sim_ans.keys():
            k_ans_list = dict_of_sim_ans[(ans.lower(),verb)]
        else:
            k_ans_list = []

        if len(k_ans_list)==0:
            # If q1=F, q2=F, q3=F
            if len(choice1_set_of_sim_ans) == 0 and len(choice2_set_of_sim_ans) == 0:

                #replace each occurrence of the concerned pronoun with answer choice 1 and answer choice 2 in WS sentence and
                sent_tokens = ws_sent.split(" ")
                sent_tokens = [ws_choice1 if token.lower()== ws_pronoun.lower() else token for token in sent_tokens]
                new_sent = " ".join(sent_tokens)

                know_sent_tokens = know_sent.split(" ")
                know_sent_tokens = [i_pronouns[0] if token.lower() in i_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [he_pronouns[0] if token.lower() in he_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [she_pronouns[0] if token.lower() in she_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [you_pronouns[0] if token.lower() in you_pronouns else token for token in know_sent_tokens]

                know_sent = " ".join(know_sent_tokens)
                ent_comparisons_for_choice1.add(("FFF", new_sent, know_sent))

                sent_tokens = ws_sent.split(" ")
                sent_tokens = [ws_choice2 if token.lower()==ws_pronoun.lower() else token for token in sent_tokens]
                new_sent = " ".join(sent_tokens)

                ent_comparisons_for_choice2.add(("FFF",new_sent, know_sent))

            # If q1=T, q2=T, q3=F
            if len(choice1_set_of_sim_ans) > 0 and len(choice2_set_of_sim_ans) > 0:
                sent_tokens = ws_sent.split(" ")
                sent_tokens = [ws_choice1 if token.lower()==ws_pronoun.lower() else token for token in sent_tokens]
                new_sent = " ".join(sent_tokens)

                know_sent_tokens = know_sent.split(" ")
                know_sent_tokens = [i_pronouns[0] if token.lower() in i_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [he_pronouns[0] if token.lower() in he_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [she_pronouns[0] if token.lower() in she_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [you_pronouns[0] if token.lower() in you_pronouns else token for token in know_sent_tokens]

                for (choice1_sim_ans,choice1_sim_verb) in choice1_set_of_sim_ans:
                    know_sent_tokens = [ws_choice1 if token.lower()==choice1_sim_ans.lower() else token for token in know_sent_tokens]
                    know_sent = " ".join(know_sent_tokens)
                    ent_comparisons_for_choice1.add(("TTF",new_sent,know_sent))
                    psl["context"].append(prob['choice1']+'$$'+choice1_sim_ans+'$$'+'K'+know_no)
                    psl["context"].append(prob['pronoun']+'$$'+choice1_sim_ans+'$$'+'K'+know_no)

                sent_tokens = ws_sent.split(" ")
                sent_tokens = [ws_choice2 if token.lower()==ws_pronoun.lower() else token for token in sent_tokens]
                new_sent = " ".join(sent_tokens)

                know_sent_tokens = know_sent.split(" ")
                know_sent_tokens = [i_pronouns[0] if token.lower() in i_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [he_pronouns[0] if token.lower() in he_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [she_pronouns[0] if token.lower() in she_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [you_pronouns[0] if token.lower() in you_pronouns else token for token in know_sent_tokens]

                for (choice2_sim_ans,choice2_sim_verb) in choice2_set_of_sim_ans:
                    know_sent_tokens = [ws_choice2 if token.lower()==choice2_sim_ans.lower() else token for token in know_sent_tokens]
                    know_sent = " ".join(know_sent_tokens)
                    ent_comparisons_for_choice2.add(("TTF",new_sent,know_sent))
                    psl["context"].append(prob['choice2']+'$$'+choice2_sim_ans+'$$'+'K'+know_no)
                    psl["context"].append(prob['pronoun']+'$$'+choice2_sim_ans+'$$'+'K'+know_no)
                psl["case"].append("TTF"+'$$'+'K'+know_no)
            # If q1=F, q2=T, q3=F
            elif len(choice1_set_of_sim_ans) == 0 and len(choice2_set_of_sim_ans) > 0:
                sent_tokens = ws_sent.split(" ")
                sent_tokens = [ws_choice2 if token.lower()==ws_pronoun.lower() else token for token in sent_tokens]
                new_sent = " ".join(sent_tokens)

                know_sent_tokens = know_sent.split(" ") 
                know_sent_tokens = [i_pronouns[0] if token.lower() in i_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [he_pronouns[0] if token.lower() in he_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [she_pronouns[0] if token.lower() in she_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [you_pronouns[0] if token.lower() in you_pronouns else token for token in know_sent_tokens]

                for (choice2_sim_ans,choice2_sim_verb) in choice2_set_of_sim_ans:
                    know_sent_tokens = [ws_choice2 if token.lower()==choice2_sim_ans.lower() else token for token in know_sent_tokens]
                    know_sent = " ".join(know_sent_tokens)
                    ent_comparisons_for_choice2.add(("FTF",new_sent,know_sent))
                    psl["context"].append(prob['choice2']+'$$'+choice2_sim_ans+'$$'+'K'+know_no)
                    psl["context"].append(prob['pronoun']+'$$'+choice2_sim_ans+'$$'+'K'+know_no)
                psl["case"].append("FTF"+'$$'+'K'+know_no)
            # If q1=T, q2=F, q3=F
            elif len(choice1_set_of_sim_ans) > 0 and len(choice2_set_of_sim_ans) == 0:
                sent_tokens = ws_sent.split(" ")
                sent_tokens = [ws_choice1 if token.lower()==ws_pronoun.lower() else token for token in sent_tokens]
                new_sent = " ".join(sent_tokens)

                know_sent_tokens = know_sent.split(" ")
                know_sent_tokens = [i_pronouns[0] if token.lower() in i_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [he_pronouns[0] if token.lower() in he_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [she_pronouns[0] if token.lower() in she_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [you_pronouns[0] if token.lower() in you_pronouns else token for token in know_sent_tokens]

                for (choice1_sim_ans,choice1_sim_verb) in choice1_set_of_sim_ans:
                    know_sent_tokens = [ws_choice1 if token.lower()==choice1_sim_ans.lower() else token for token in know_sent_tokens]
                    know_sent = " ".join(know_sent_tokens)
                    ent_comparisons_for_choice1.add(("TFF",new_sent,know_sent))
                    psl["context"].append(prob['choice1']+'$$'+choice1_sim_ans+'$$'+'K'+know_no)
                    psl["context"].append(prob['pronoun']+'$$'+choice1_sim_ans+'$$'+'K'+know_no)
                psl["case"].append("TFF"+'$$'+'K'+know_no)
        else:
            # If q1=F, q2=F, q3=T
            if len(choice1_set_of_sim_ans)==0 and len(choice2_set_of_sim_ans)==0:
                sent_tokens = ws_sent.split(" ")
                sent_tokens = [ws_choice1 if token.lower()==ws_pronoun.lower() else token for token in sent_tokens]
                new_sent = " ".join(sent_tokens)

                know_sent_tokens = know_sent.split(" ")
                know_sent_tokens = [i_pronouns[0] if token.lower() in i_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [he_pronouns[0] if token.lower() in he_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [she_pronouns[0] if token.lower() in she_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [you_pronouns[0] if token.lower() in you_pronouns else token for token in know_sent_tokens]

                know_sent = " ".join(know_sent_tokens)
                ent_comparisons_for_choice1.add(("FFT",new_sent,know_sent))

                sent_tokens = ws_sent.split(" ")
                sent_tokens = [ws_choice2 if token.lower()==ws_pronoun.lower() else token for token in sent_tokens]
                new_sent = " ".join(sent_tokens)
                ent_comparisons_for_choice2.add(("FFT",new_sent,know_sent))
                psl["case"].append("FFT"+'$$'+'K'+know_no)
            # If q1=F, q2=T, q3=T
            if len(choice1_set_of_sim_ans) == 0 and len(choice2_set_of_sim_ans)>0:
                sent_tokens = ws_sent.split(" ")
                sent_tokens = [ws_choice2 if token.lower()==ws_pronoun.lower() else token for token in sent_tokens]
                new_sent = " ".join(sent_tokens)

                know_sent_tokens = know_sent.split(" ")
                know_sent_tokens = [i_pronouns[0] if token.lower() in i_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [he_pronouns[0] if token.lower() in he_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [she_pronouns[0] if token.lower() in she_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [you_pronouns[0] if token.lower() in you_pronouns else token for token in know_sent_tokens]

                for (choice2_sim_ans,choice2_sim_verb) in choice2_set_of_sim_ans:
                    know_sent_tokens = [ws_choice2 if token.lower()==choice2_sim_ans.lower() else token for token in know_sent_tokens]
                    know_sent = " ".join(know_sent_tokens)
                    ent_comparisons_for_choice2.add(("FTT",new_sent,know_sent))
                    psl["context"].append(prob['choice2']+'$$'+choice2_sim_ans+'$$'+'K'+know_no)
                    psl["context"].append(prob['pronoun']+'$$'+choice2_sim_ans+'$$'+'K'+know_no)
                     
                get_similarity_score(choice2_set_of_sim_ans, k_ans_list, psl, know_no)
                psl["case"].append("FTT"+'$$'+'K'+know_no)
            # If q1=T, q2=F, q3=T
            elif len(choice1_set_of_sim_ans) > 0 and len(choice2_set_of_sim_ans)== 0:
                sent_tokens = ws_sent.split(" ")
                sent_tokens = [ws_choice1 if token.lower() == ws_pronoun.lower() else token for token in sent_tokens]
                new_sent = " ".join(sent_tokens)

                know_sent_tokens = know_sent.split(" ")
                know_sent_tokens = [i_pronouns[0] if token.lower() in i_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [he_pronouns[0] if token.lower() in he_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [she_pronouns[0] if token.lower() in she_pronouns else token for token in know_sent_tokens]
                know_sent_tokens = [you_pronouns[0] if token.lower() in you_pronouns else token for token in know_sent_tokens]

                for (choice1_sim_ans, choice1_sim_verb) in choice1_set_of_sim_ans:
                    know_sent_tokens = [ws_choice1 if token.lower()==choice1_sim_ans.lower() else token for token in know_sent_tokens]
                    know_sent = " ".join(know_sent_tokens)
                    ent_comparisons_for_choice1.add(("TFT",new_sent,know_sent))
                    psl["context"].append(prob['choice1']+'$$'+choice1_sim_ans+'$$'+'K'+know_no)
                    psl["context"].append(prob['pronoun']+'$$'+choice1_sim_ans+'$$'+'K'+know_no)
                
                get_similarity_score(choice1_set_of_sim_ans, k_ans_list, psl, know_no)
                
                psl["case"].append("TFT"+'$$'+'K'+know_no)
            # If q1=T, q2=T, q3=Ts
            elif len(choice1_set_of_sim_ans) > 0 and len(choice2_set_of_sim_ans) > 0:
                ans_tokens = ans.split(" ")
                psl["case"].append("TTT"+'$$'+'K'+know_no)
                anss_wrt_choice1 = []
                for (know_choice,verb2) in choice1_set_of_sim_ans:
                    new_ans = ""
                    psl["context"].append(prob['choice1']+'$$'+know_choice+'$$'+'K'+know_no)
                    for ans_token in ans_tokens:
                        if ans_token==ws_pronoun:
                            new_ans += " " + know_choice
                        else:
                            new_ans += " " + ans_token
                    anss_wrt_choice1.append(new_ans.strip())

                anss_wrt_choice2 = []
                for (know_choice,verb2) in choice2_set_of_sim_ans:
                    psl["context"].append(prob['choice2']+'$$'+know_choice+'$$'+'K'+know_no)
                    if know_choice == "":
                        continue
                    new_ans = ""
                    for ans_token in ans_tokens:
                        if ans_token==ws_pronoun:
                            new_ans += " " + know_choice
                        else:
                            new_ans += " " + ans_token
                    anss_wrt_choice2.append(new_ans.strip())
                
                get_similarity_score(choice1_set_of_sim_ans, k_ans_list, psl, know_no)
                get_similarity_score(choice2_set_of_sim_ans, k_ans_list, psl, know_no)
                 
                for (k_ans,k_verb) in k_ans_list:
                    psl["context"].append(prob['pronoun']+'$$'+k_ans+'$$'+'K'+know_no)
                    if k_ans=="":
                        continue
                    for ans_wrt_choice1 in anss_wrt_choice1:
                        ent_comparisons_for_choice1.add(("TTT",k_ans,ans_wrt_choice1))
                    for ans_wrt_choice2 in anss_wrt_choice2:
                        ent_comparisons_for_choice2.add(("TTT",k_ans,ans_wrt_choice2))

    ttt_bucket_ch1 = []
    ttf_bucket_ch1 = []
    tff_bucket_ch1 = []
    fff_bucket_ch1 = []
    for (t,h,p) in ent_comparisons_for_choice1:
        if are_same(h.lower(),p.lower()):
            ent_score = 1.0
            cntr_score = 0.0
        elif are_not_same(h.lower(),p.lower()):
            ent_score = 0.0
            cntr_score = 1.0
        else:
            if use_predictor:
                score = predictor.predict(hypothesis=h,premise=p)
                ent_score = score["label_probs"][0]
                cntr_score = score["label_probs"][1]
                if len(psl["context"]) == 0:
                    psl["entailment"].append(prob['choice1']+'$$'+prob['pronoun']+'$$'+'K'+know_no+'$$'+str(ent_score))
                else:
                    psl["similarity"].append(prob['choice1']+'$$'+prob['pronoun']+'$$'+'K'+know_no+'$$'+str(ent_score))
            else:
                ent_score = 0.0
                cntr_score = 0.0

        if t=="TTT":
            ttt_bucket_ch1.append((ent_score,cntr_score))
        elif t=="TFT" or t=="FTT" or t=="TTF":
            ttf_bucket_ch1.append((ent_score,cntr_score))
        elif t=="FFT" or t=="FTF" or t=="TFF":
            tff_bucket_ch1.append((ent_score,cntr_score))
        else:
            fff_bucket_ch1.append((ent_score,cntr_score))

        if verbose_output:
            print("TYPE: ",t,"; COMPARE 1: ",h," --WITH-- ",p)
            print("SCORES: ",(ent_score,cntr_score))

        choice1_ent_contr_scores.append((ent_score,cntr_score))

    ttt_bucket_ch2 = []
    ttf_bucket_ch2 = []
    tff_bucket_ch2 = []
    fff_bucket_ch2 = []
    for (t,h,p) in ent_comparisons_for_choice2:
        if are_same(h.lower(),p.lower()):
            ent_score = 1.0
            cntr_score = 0.0
        elif are_not_same(h.lower(),p.lower()):
            ent_score = 0.0
            cntr_score = 1.0
        else:
            if use_predictor:
                score = predictor.predict(hypothesis=h,premise=p)
                ent_score = score["label_probs"][0]
                cntr_score = score["label_probs"][1]
                if len(psl["context"]) == 0:
                    psl["entailment"].append(prob['choice2']+'$$'+prob['pronoun']+'$$'+'K'+know_no+'$$'+str(ent_score))
                else:
                    psl["similarity"].append(prob['choice1']+'$$'+prob['pronoun']+'$$'+'K'+know_no+'$$'+str(ent_score))
            else:
                ent_score = 0.0
                cntr_score = 0.0

        if t=="TTT":
            ttt_bucket_ch2.append((ent_score,cntr_score))
        elif t=="TFT" or t=="FTT" or t=="TTF":
            ttf_bucket_ch2.append((ent_score,cntr_score))
        elif t=="FFT" or t=="FTF" or t=="TFF":
            tff_bucket_ch2.append((ent_score,cntr_score))
        else:
            fff_bucket_ch2.append((ent_score,cntr_score))

        if verbose_output:
            print("TYPE: ",t,"; COMPARE 2: ",h," --WITH-- ",p)
            print("SCORES: ",(ent_score,cntr_score))

        choice2_ent_contr_scores.append((ent_score,cntr_score))

    print("FOR TTT BUCKET: ")
    choice1_conf,choice2_conf = get_conf(ttt_bucket_ch1,ttt_bucket_ch2)
    #print("HIII: ",(ttt_bucket_ch1,ttt_bucket_ch2))
    #print("HELLO",choice1_conf,choice2_conf)
    if choice1_conf==choice2_conf:
        print("FOR TTF BUCKET: ")
        choice1_conf,choice2_conf = get_conf(ttf_bucket_ch1,ttf_bucket_ch2)

    if choice1_conf==choice2_conf:
        print("FOR TFF BUCKET: ")
        choice1_conf,choice2_conf = get_conf(tff_bucket_ch1,tff_bucket_ch2)

    if choice1_conf==choice2_conf:
        print("FOR FFF BUCKET: ")
        choice1_conf,choice2_conf = get_conf(fff_bucket_ch1,fff_bucket_ch2)

    result = "unknown"
    if ans_is_choice1:
        if choice1_conf > choice2_conf:
            result = "correct"
        elif choice2_conf > choice1_conf:
            result = "incorrect"
    else:
        if choice2_conf > choice1_conf:
            result = "correct"
        elif choice1_conf > choice2_conf:
            result = "incorrect"

    if verbose_output:
        print("RESULT: ",result)
    return result

def process_qasrl_output(qasrl_output, pronoun):
    qa_pairs_array = []
    json_obj = qasrl_output
    sent_tokens = json_obj["words"]
    verbs_objs = json_obj["verbs"]
    for verbs_obj in verbs_objs:
        verb = verbs_obj["verb"]
        qa_pairs = verbs_obj["qa_pairs"]
        for qa_pair in qa_pairs:
            ques = qa_pair["question"]

            new_ques_tokens = []
            ques_last_char = ques[-1]
            if ques_last_char=="?":
                ques = ques[0:-1] + " ?"
            ques_tokens = ques.split(" ")
            for ques_token in ques_tokens:
                if ques_token.lower()==verb.lower() or wn_lemmatizer.lemmatize(ques_token.lower(),pos='v')==wn_lemmatizer.lemmatize(verb.lower(),pos='v'):
                    new_ques_tokens.append("V")
                else:
                    new_ques_tokens.append(ques_token)
            new_ques = " ".join(new_ques_tokens)
            answers = qa_pair["spans"]
            ans_is_pronoun = False
            if pronoun is not None:
                for ans in answers:
                    ans_text = ans["text"]
                    if ans_text==pronoun:
                        ans_is_pronoun = True
                        break
            if ans_is_pronoun:
                qa_pair = {'ques':new_ques, 'ans':pronoun, 'verb':verb}#wn_lemmatizer.lemmatize(verb, pos='v')}
                qa_pairs_array.append(qa_pair)
            else:
                for ans in answers:
                    qa_pair = {'ques':new_ques, 'ans':ans["text"], 'verb':verb}#wn_lemmatizer.lemmatize(verb, pos='v')}
                    qa_pairs_array.append(qa_pair)

    return qa_pairs_array

def choose_max_entailment(psl):
    ent = {}
    for each in psl["entailment"]:
        tokens = each.split("$$")
        cur = tokens[0]+"$$"+tokens[1]+'$$'+tokens[2]
        if cur not in ent:
            max = 0.0
        else:
            max = ent[cur]
        if max < float(tokens[3]):
            max = float(tokens[3])
        ent[cur] = max
    psl["entailment"] = []
    
    for (key, value) in ent.items():
        psl["entailment"].append(key+"$$"+str(value))
        

def choose_max_similarity(psl):
    ent = {}
    for each in psl["similarity"]:
        tokens = each.split("$$")
        cur = tokens[0]+"$$"+tokens[1]+'$$'+tokens[2]
        if cur not in ent:
            max = 0.0
        else:
            max = ent[cur]
        if max < float(tokens[3]):
            max = float(tokens[3])
        ent[cur] = max
    psl["similarity"] = []
    
    for (key, value) in ent.items():
        psl["similarity"].append(key+"$$"+str(value))

def process(psl):
    #if "case" in psl and len(psl["case"]) == 1 and psl["case"][0] == "TTF":
    #    context = []
    #    for each in psl["case"]:
    #        if each != "TTF":
    #            return     
    if len(psl["entailment"]) < 2:
        return
    context = []
    
    for each in psl["context"]:
        
        choice1 = psl["entailment"][0].split("$$")
        choice2 = psl["entailment"][1].split("$$")
        if float(choice1[-1]) < float(choice2[-1]):
            if prob['choice1']+'$$' not in each:
                context.append(each)
        else:
            if prob['choice2']+'$$' not in each:
                context.append(each)
    psl["context"] = context

if __name__=="__main__":
    populate_ques_type_list()
    
    #load all the wsc problems
    all_probs_file = "inputs/test_problems_file.json"
    #all_probs_file = "final_problems.json"
    
    f = open(all_probs_file,"r")
    all_probs = f.read()
    probs = json.loads(all_probs)

    #load QASRL output for all WSC problems
    qasrl_wsc_output_dict = {}
    qasrl_ws_sent_file = "inputs/ws_sents_and_qasrl_out.txt"
    bert_wsc_json = "inputs/ws_sents_and_qasrl_out.txt"
    
    f = open(qasrl_ws_sent_file,"r")
    for line in f:
        sent_and_qasrl = line.rstrip().strip().split("$$$$") 
        json_obj = json.loads(sent_and_qasrl[1].strip())
        sentence = sent_and_qasrl[0].strip()
        qasrl_wsc_output_dict[sentence] = json_obj

    
    qasrl_know_sent_file = "inputs/auto/combined_qasrl_final.json"
    f = open(qasrl_know_sent_file,"r")
    qasrl_know_output_dict = json.loads(f.read())
   
    wsc_know_sent_file = "inputs/auto/sorted_filtered_ksents_10.json"
    f = open(wsc_know_sent_file,"r")
    temp_list = json.loads(f.read());
    wsc_know_sent_dict = {}
    
    for each in temp_list:
        wsc_know_sent_dict[each["ws_sent"]] = each
    
    
    know_not_parsed = 0
    wssent_not_parsed = 0
    correct = 0
    incorrect = 0
    unknown = 0
    know_unk_list = []
    psl_output = []
    
    for i in range(0,len(probs)):
        prob = probs[i]
        psl = {}
        ws_sent = prob["ws_sent"]
        ws_sent = ws_sent.strip()
        psl["ws_sent"]= ws_sent;
        psl["ans"]= prob['ans'];
        psl["pronoun"] = prob["pronoun"]
        psl["domain"] = [prob["choice1"], prob["choice2"], prob["pronoun"]]
        psl["coref_target"] = [prob["choice1"]+'$$'+prob["pronoun"], prob["choice2"]+'$$'+prob["pronoun"],
                                prob["pronoun"]+'$$'+prob["choice1"], prob["pronoun"]+'$$'+prob["choice2"]]
    
        value1 = 0
        value2 = 0
        if prob["ans"].lower() == prob["choice1"].lower():
            value1 = 1
        else:
            value2 = 1
        psl["coref_target_truth"] = [prob["choice1"]+'$$'+prob["pronoun"]+'$$'+str(value1), 
                                    prob["choice2"]+'$$'+prob["pronoun"]+'$$'+str(value2),
                                    prob["pronoun"]+'$$'+prob["choice1"]+'$$'+str(value1), 
                                    prob["pronoun"]+'$$'+prob["choice2"]+'$$'+str(value2)] 
        #scr_choice1 = prob["scr_score"]["choice1"] / (prob["scr_score"]["choice1"] + prob["scr_score"]["choice2"])
        #scr_choice2 = prob["scr_score"]["choice2"] / (prob["scr_score"]["choice1"] + prob["scr_score"]["choice2"])
        psl["scr_score"] = [prob["choice1"]+'$$'+prob['pronoun']+'$$'+str(prob["scr_score"]["choice1"]), 
                            prob["choice2"]+'$$'+prob['pronoun']+'$$'+str(prob["scr_score"]["choice2"])] 
        if ws_sent in qasrl_wsc_output_dict:
            ws_sent_qasrl_pairs = qasrl_wsc_output_dict[ws_sent]
            if ws_sent in wsc_know_sent_dict:
                knowledge_sent = wsc_know_sent_dict[ws_sent]["k_sents"]
                know_no = 1
                psl["context"] = []
                psl["case"] = []
                psl["entailment"] = []
                psl["similarity"] = []
                for know in knowledge_sent:
                    know_s = know['k_sent'].rstrip()
                    know_s = know_s.strip()
                    
                    if know_s in qasrl_know_output_dict:
    
                        know_sent_qasrl_pairs = qasrl_know_output_dict[know_s]
                        pronoun = prob["pronoun"]
                        
                        
                        ws_sent_qa_pairs = process_qasrl_output(ws_sent_qasrl_pairs,pronoun)
                        know_sent_qa_pairs = process_qasrl_output(know_sent_qasrl_pairs, None)
                       
                        if verbose_output:
                            print("************************************************************")
                        
                        result = main(prob, ws_sent_qa_pairs, know_sent_qa_pairs, psl, str(know_no))
                        psl["context"] = list(set(psl["context"]))
                        choose_max_entailment(psl)
                        choose_max_similarity(psl)
                        #process(psl)
                        if result=="correct":
                            correct += 1
                        elif result=="incorrect":
                            incorrect += 1
                        else:
                            unknown += 1
                        if verbose_output:
                            print("************************************************************")
    
                    else: 
                        #if know_s=="NA":
                        #    print("NA: ",ws_sent)
                        #else:
                        #    print("NOT NA: ",ws_sent)
                        if know_s!="NA":
                            print("KNOW SENT NOT FOUND: ",know_s)
                        know_not_parsed += 1
                        know_unk_list.append(prob)
                        
                    know_no = know_no + 1
        else:
            print("WS SENT NOT FOUND: ",ws_sent)
            wssent_not_parsed+=1
        psl_output.append(psl)
    with open('new_psl_context.json', 'w') as outfile:
        json.dump(psl_output, outfile)
    
    with open('similar_pair.json', 'w') as outfile:
        json.dump(similar_pair, outfile)
        
    print("CORRECT: ",correct)
    print("INCORRECT: ",incorrect)
    print("UNKNOWN: ",unknown)
    print("KNOW SENT NOT FOUND", know_not_parsed)
    print(know_unk_list)
    print("************************************************************")
