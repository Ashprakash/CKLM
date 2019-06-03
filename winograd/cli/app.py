# -*- coding: utf-8 -*-

import json
import shlex, subprocess
import sys
import math

parent_dir = '/Users/ash/Documents/Study/Research/psl-examples/winograd/'
PIPE= '/Users/ash/Documents/Study/Research/psl-examples/winograd/log/run.txt'
def create_psl_exec_files(coref_txt, coref_truth_txt, context_pair_txt, domain_txt, entailment_txt, commonsense_txt, similarity_txt):

    coref_file = open('../data/coref_targets.txt', 'w')
    coref_file.write(coref_txt)

    coref_truth_file = open('../data/coref_truth.txt', 'w')
    coref_truth_file.write(coref_truth_txt)

    context_file = open('../data/context_obs.txt', 'w')
    context_file.write(context_pair_txt)

    domain_file = open('../data/domain_obs.txt', 'w')
    domain_file.write(domain_txt)

    entailment_file = open('../data/entailment_obs.txt', 'w')
    entailment_file.write(entailment_txt)

    commonsense_file = open('../data/commonsense_obs.txt', 'w')
    commonsense_file.write(commonsense_txt)

    similarity_file = open('../data/similar_obs.txt', 'w')
    similarity_file.write(similarity_txt)

    coref_file.close()
    coref_truth_file.close()
    context_file.close()
    domain_file.close()
    entailment_file.close()
    commonsense_file.close()

def run_psl():
    process = subprocess.Popen(['/Users/ash/Documents/Study/Research/psl-examples/winograd/cli/run.sh'])
    #process = subprocess.Popen(['/home/apraka23/Winograd/WSC_with_knowledge/winograd/cli/run.sh'])
    process.wait()

def softmax(s1, s2):
    """Compute softmax values for each sets of scores in x."""
    sc1 = math.pow(math.e, s1);
    sc2 = math.pow(math.e, s2);
    return sc1/(sc1+sc2), sc2/(sc1+sc2) 

def get_ans(prob, bert):
    coref_file = open('inferred-predicates/COREF.txt', 'r')
    inferred_predicate = coref_file.read()
    inferred = inferred_predicate.split('\n')
    print("inferred", inferred)
    
    bert_isChoice1 = False;
    if 'bert_choice1' not in prob: #if PSL file is not having bert score then pick from the bert file
        score1 = bert["choice1_score"] 
        score2 = bert["choice2_score"] 
    else:
        score1 = prob["bert_choice1"]
        score2 = prob["bert_choice2"]
        
    if score1 > score2:
        bert_isChoice1 = True

    score1 = 0.0
    score2 = 0.0
    for each in inferred:
        coref_score = each.split('\t')
        if len(coref_score) < 2:
            continue

        coref_score[0] = coref_score[0][1:-1]
        coref_score[1] = coref_score[1][1:-1]
        coref_score[0] = coref_score[0].replace("\\", "")
        coref_score[1] = coref_score[1].replace("\\", "")
        if coref_score[0] == prob['domain'][0]:
            score1 = coref_score[2]
        elif coref_score[0] ==  prob['domain'][1]:
            score2 = coref_score[2]

    # if not (inferred_ans.lower() == bert['choice1'].lower() and not bert_isChoice1 and bert['ans'].lower() == bert['choice1'].lower()):
    #     inferred_ans = bert['choice2']

    # if not (bert['ans'].lower() ==  bert['choice2'].lower() and bert_isChoice1 and inferred_ans.lower() == bert['choice2'].lower()):
    #     inferred_ans = bert['choice1']

    return score1, score2

def get_normalized_prob(ch1_score, ch2_score):
    while ch1_score < 0.5 and ch2_score < 0.5:
        if ch1_score < 0.01 and ch2_score < 0.01:
            ch1_score = ch1_score * 100
            ch2_score = ch2_score * 100
        elif ch1_score < 0.1 and ch2_score < 0.1:
            ch1_score = ch1_score * 10
            ch2_score = ch2_score * 10
        elif ch1_score < 0.3 and ch2_score < 0.3:
            ch1_score = ch1_score * 3
            ch2_score = ch2_score * 3
        elif ch1_score < 0.5 and ch2_score < 0.5:
            ch1_score = ch1_score * 2
            ch2_score = ch2_score * 2
    return ch1_score, ch2_score;

def main():

    #probs_with_context_file = "../data/wsc_problem_psl.json"
    probs_with_context_file = "../data/new_psl_problems.json"
    bert_scores_file = open("../data/bert_par_scores.json", "r")
    bert_scores = json.loads(bert_scores_file.read())
    #probs_with_context_file = "../data/test.json"
    f = open(probs_with_context_file,"r")
    all_probs = f.read()
    probs_and_context = json.loads(all_probs)

    correct = 0
    incorrect = 0
    total = 0
    isCommonsense = True

    for i in range(0, len(probs_and_context)):
        each = probs_and_context[i]
        bert = bert_scores[i]
        coref_target = each['coref_target']
        coref_target_truth = each['coref_target_truth']
        if 'similarity' in each:
            similar = each['similarity']
        else:
            similar = []
        if 'context' in each:
            context = each['context']
        else:
            context = []
        if 'entailment' in each:
            entailment = each['entailment']
        else:
            entailment = []

        domain = each['domain']
        commonsense = each['scr_score']
        coref_txt=''
        coref_truth_txt=''
        context_pair_txt= ''
        domain_txt = ''
        entailment_txt = ''
        commonsense_txt = ''
        similarity_txt = ''

        for coref in coref_target:
            token = coref.split('$$')
            coref_txt = coref_txt+token[0]+'\t'+token[1]+'\n'

        for coref_truth in coref_target_truth:
            token = coref_truth.split('$$')
            coref_truth_txt = coref_truth_txt+token[0]+'\t'+token[1]+'\t'+token[2]+'\n'

        for con in context:
            token = con.split('$$')
            context_pair_txt = context_pair_txt+token[0]+'\t'+token[1]+'\t'+token[2]+'\n'

        for sim in similar:
            token = sim.split('$$')
            similarity_txt = similarity_txt+token[0]+'\t'+token[1]+'\t'+token[2]+'\t'+token[3]+'\n'

        domain_txt = domain_txt+domain[0]+'\t'+'can'+'\n'
        domain_txt = domain_txt+domain[1]+'\t'+'can'+'\n'
        domain_txt = domain_txt+domain[2]+'\t'+'p'+'\n'

        if isCommonsense:
            choice1 = ''
            choice2 = ''
            ctoken1 = commonsense[0].split('$$')
            ctoken2 = commonsense[1].split('$$')
            if 'bert_choice1' not in each: #if PSL file is not having bert score then pick from the bert file
                score1 = bert["choice1_score"] / (bert["choice1_score"] + bert["choice2_score"])
                score2 = bert["choice2_score"] / (bert["choice1_score"] + bert["choice2_score"])
                choice1 = bert["choice1"]
                choice2 = bert["choice2"]
            else:
                score1 = each["bert_choice1"] / (each["bert_choice1"] + each["bert_choice2"])
                score2 = each["bert_choice2"] / (each["bert_choice1"] + each["bert_choice2"])
                choice1 = ctoken1[0]
                choice2 = ctoken2[0]

            commonsense_txt = commonsense_txt+choice1+'\t'+each["pronoun"]+'\t'+str(score1)+'\n'
            commonsense_txt = commonsense_txt+choice2+'\t'+each["pronoun"]+'\t'+str(score2)+'\n'

        if len(context) > 1:
            know_entailment = {}
            for ent in entailment:
                token = ent.split('$$')
                if token[2] not in know_entailment:
                    know_entailment[token[2]] = []
                know_entailment[token[2]].append(token)

            for key, value in know_entailment.items():
                if len(know_entailment[key]) == 2:
                    entail = know_entailment[key]
                    ch1_tokens = entail[0]
                    ch2_tokens = entail[1]
                    ch1_score = float(ch1_tokens[3])
                    ch2_score = float(ch2_tokens[3])
                    ch1_score, ch2_score = get_normalized_prob(ch1_score, ch2_score)
                    entailment_txt = entailment_txt+ch1_tokens[0]+'\t'+ch1_tokens[1]+'\t'+ch1_tokens[2]+'\t'+str(ch1_score)+'\n'
                    entailment_txt = entailment_txt+ch2_tokens[0]+'\t'+ch2_tokens[1]+'\t'+ch1_tokens[2]+'\t'+str(ch2_score)+'\n'
                else:
                    for ent in know_entailment[key]:
                        token = ent
                        if float(token[3]) < 0.1:
                            entailment_txt = entailment_txt+token[0]+'\t'+token[1]+'\t'+token[2]+'\t'+token[3]+'\n'

        # for com in commonsense:
        #     token = com.split('$$')
        #     commonsense_txt = commonsense_txt+token[0]+'\t'+token[1]+'\t'+token[2]+'\n'

        create_psl_exec_files(coref_txt, coref_truth_txt, context_pair_txt, domain_txt, entailment_txt, commonsense_txt, similarity_txt)
        run_psl();
        psl_score1, psl_score2 = get_ans(each, bert);
        if psl_score1 > psl_score2:
            inferred_ans = each['domain'][0]
        else:
            inferred_ans = each['domain'][1]

        print('INFERRED_ANS: ', inferred_ans)
        if inferred_ans.lower() == each['ans'].lower():
            correct = correct + 1
            each["predicted"] = "CORRECT"
        else:
            incorrect = incorrect + 1
            each["predicted"] = "INCORRECT"
        each['bert_choice1'] = bert["choice1_score"]
        each['bert_choice2'] = bert["choice2_score"]
        each['psl_score1'] = psl_score1
        each['psl_score2'] = psl_score2

        print("WS_SENT: "+each["ws_sent"])
        total = total + 1

    print("Correct : ", correct)
    print("Incorrect : ", incorrect)
    print("Total: ", total)
    with open('../data/new_psl_problems_scores.json', 'w') as outfile:
        json.dump(probs_and_context, outfile)

if __name__=="__main__":
    main()
