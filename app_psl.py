# -*- coding: utf-8 -*-

import json
import shlex, subprocess
import sys;

parent_dir = '/Users/ash/Documents/Study/Research/psl-examples/winograd/'
PIPE= '/Users/ash/Documents/Study/Research/psl-examples/winograd/log/run.txt'
def create_psl_exec_files(coref_txt, coref_truth_txt, context_pair_txt):

    coref_file = open('../data/coref_targets.txt', 'w')
    coref_file.write(coref_txt)

    coref_truth_file = open('../data/coref_truth.txt', 'w')
    coref_truth_file.write(coref_truth_txt)

    context_file = open('../data/context_obs.txt', 'w')
    context_file.write(context_pair_txt)

    coref_file.close()
    coref_truth_file.close()
    context_file.close()

def run_psl():
    process = subprocess.Popen(['/Users/ash/Documents/Study/Research/psl-examples/winograd/cli/run.sh'])
    process.wait()

def get_ans(prob):
    coref_file = open('inferred-predicates/COREF.txt', 'r')
    inferred_predicate = coref_file.read()
    inferred = inferred_predicate.split('\n')
    max = sys.float_info.min
    inferred_ans = ''
    for each in inferred:
        print(each)
        coref_score = each.split('\t')
        if max < float(coref_score[2]):
            if prob['pronoun'] == coref_score[0]:
                inferred_ans = coref_score[1]
            else:
                inferred_ans = coref_score[0]
    return inferred_ans, max

def main():

    probs_with_context_file = "../data/wsc_problem_psl.json"
    f = open(probs_with_context_file,"r")
    all_probs = f.read()
    probs_and_context = json.loads(all_probs)

    correct = 0
    incorrect = 0
    for i in range(0, len(probs_and_context)):
        each = probs_and_context[i]
        coref_target = each['coref_target']
        coref_target_truth = each['coref_target_truth']
        context = each['context']
        coref_txt=''
        coref_truth_txt=''
        context_pair_txt= ''
        for coref in coref_target:
            token = coref.split('$$')
            coref_txt = coref_txt+token[0]+'\t'+token[1]+'\n'

        for coref_truth in coref_target_truth:
            token = coref_truth.split('$$')
            coref_truth_txt = coref_truth_txt+token[0]+'\t'+token[1]+'\t'+token[2]+'\n'

        for con in context:
            token = con.split('$$')
            context_pair_txt = context_pair_txt+token[0]+'\t'+token[1]+'\n'

        create_psl_exec_files(coref_txt, coref_truth_txt, context_pair_txt)
        run_psl();
        inferred_ans, max = get_ans(each);
        if inferred_ans == each['ans']:
            correct = correct + 1
        else:
            incorrect = incorrect + 1

    print("Correct : ", correct)
    print("Incorrect : ", incorrect)
    print("Total: ", len(probs_and_context))

if __name__=="__main__":
    main()
