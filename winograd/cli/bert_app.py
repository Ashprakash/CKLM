import json
import math
import matplotlib.pyplot as plt


def parse_wsc_out():
    split_str = "************************************************************"
    wsc_out_file = open("../data/output_wsc_nov.json", "r") 
    output_str = wsc_out_file.read()
    output_str_arr = output_str.split(split_str) 
    wsc_output = []
    for each in output_str_arr:
        obj = {}
        values = each.split('\n')
        if len(values) > 7:
            obj['ws_sent'] = values[1].replace('SENT:  ', '') 
            obj['pronoun'] = values[2].replace('PRONOUN:  ', '')
            obj['ans'] = values[4].replace('ANSWER:  ', '')
            obj['choice1'] = values[5].replace('CHOICE1:  ', '')
            obj['choice2'] = values[6].replace('CHOICE2:  ', '')
            obj["result"] = values[-2].replace('RESULT:  ', '')
            wsc_output.append(obj)
    with open('../data/wsc_prev_output.json', 'w') as outfile:
        json.dump(wsc_output, outfile)
    return wsc_output


def diff_array(diff, array_list):
    if diff > 0.5:
        array_list[5] = array_list[5] + 1
    elif diff > 0.4:
        array_list[4] = array_list[4] + 1
    elif diff > 0.3:
        array_list[3] = array_list[3] + 1
    elif diff > 0.2:
        array_list[2] = array_list[2] + 1
    elif diff > 0.00000001:
        array_list[1] = array_list[1] + 1
    else:
        array_list[0] = array_list[0] + 1

def softmax(s1, s2):
    """Compute softmax values for each sets of scores in x."""
    sc1 = math.pow(math.e, s1);
    sc2 = math.pow(math.e, s2);
    return sc1/(sc1+sc2), sc2/(sc1+sc2) 

def calculate_bert_scores(psl_scores):
    bert_scores_file = open("../data/bert_par_scores.json", "r") 
    bert_scores = json.loads(bert_scores_file.read())
    correct = 0
    incorrect = 0
    total = 0
    correct_diff = []
    incorrect_diff = []
    correct_grt_1 = 0
    correct_lt_1 = 0
    incorrect_grt_1 = 0
    incorrect_lt_1 = 0

    # diff = ['>0.5', '>0.4', '>0.3', '>0.2', '>0.1', '<0.1']
    diff = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    bert_list = [0, 0, 0, 0, 0, 0]
    psl_correct_list = [0, 0, 0, 0, 0, 0]
    scr_correct_list = [0, 0, 0, 0, 0, 0]
    for i in range(0, len(bert_scores)):
        bert = bert_scores[i]
        psl = psl_scores[i]
        diff = 0.0
        print(bert)
        isBert_correct, bert_choice1, bert_choice2 = check_Bert_correct(bert)
        isSCR_correct, scr_score1, scr_score2 = check_SCR_correct(psl, bert)
        scr_scores = softmax(scr_score1, scr_score2)
        scr_score1 = scr_scores[0]
        scr_score2 = scr_scores[1]

        scores = softmax(bert_choice1, bert_choice2)
        score1 = scores[0]
        score2 = scores[1]

        # if score1 > score2:
        #     diff = score1 - score2
        #     if bert['choice1'].lower() == bert['ans'].lower():
        #         correct_diff.append([i, diff]);
        #         if diff < 1.0:
        #             correct_lt_1 = correct_lt_1 + 1 
        #         else:
        #             correct_grt_1 = correct_grt_1 + 1
        #     else:
        #         incorrect_diff.append([i, diff]);
        #         if diff < 1.0:
        #             incorrect_lt_1 = incorrect_lt_1 + 1 
        #         else:
        #             incorrect_grt_1 = incorrect_grt_1 + 1

        # else:
        #     diff = score2 - score1
        #     if bert['choice2'].lower() == bert['ans'].lower():
        #         correct_diff.append([i, diff]);
        #         if diff < 1.0:
        #             correct_lt_1 = correct_lt_1 + 1 
        #         else:
        #             correct_grt_1 = correct_grt_1 + 1
        #     else:
        #         incorrect_diff.append([i, diff]);
        #         if diff < 1.0:
        #             incorrect_lt_1 = incorrect_lt_1 + 1 
        #         else:
        #             incorrect_grt_1 = incorrect_grt_1 + 1

        if isBert_correct:
            diff_array(abs(score1 - score2), bert_list)
        if isSCR_correct:
            diff_array(abs(scr_score1 - scr_score2), scr_correct_list)
        if psl['predicted'] == 'CORRECT':
            diff_array(abs(float(psl['psl_score1']) - float(psl['psl_score2'])), psl_correct_list)
        # print("WS_SENT: "+bert["question"])
        total = total + 1

    print("bert Diff List: ", bert_list)
    print("psl Diff List: ", psl_correct_list)
    print("scr Diff List: ", scr_correct_list)
    print("Correct : ", correct)
    print("Incorrect : ", incorrect)
    print("Total: ", total)

    # plt.bar(diff+0.00, psl_correct_list, width=0.2, color='g')
    # plt.bar(diff+0.25, bert_list, width=0.2,color='b')
    # plt.bar(diff+0.5, scr_correct_list, width=0.2,color='r')
    # plt.show()
    # correct = np.array(correct_diff)
    # incorrect = np.array(incorrect_diff)
    # print(correct_lt_1)
    # print(correct_grt_1)
    # print(incorrect_lt_1)
    # print(incorrect_grt_1)
    # print("Correct Lesser than 1", str(correct_lt_1 / (correct_lt_1 + incorrect_lt_1)))
    # print("Correct Greater than 1", str(correct_grt_1 / (correct_grt_1 + incorrect_grt_1)))
    # print("Incorrect Lesser than 1", str(incorrect_lt_1 / (correct_lt_1 + incorrect_lt_1)))
    # print("Incorrect Greater than 1", str(incorrect_grt_1 / (correct_grt_1 + incorrect_grt_1)))
    # c_x, c_y = correct.T
    # i_x, i_y = incorrect.T
    # plt.scatter(c_x, c_y, c='b', marker='x', label='correct')
    # plt.scatter(i_x, i_y, c='r', marker='s', label='incorrect')
    # plt.legend(loc='upper left')
    # plt.show()


def check_Bert_correct(bert):
    isBert_correct = False
    choice1 = bert['choice1_score']
    choice2 = bert['choice2_score']
    # choice1, choice2 = softmax(choice1, choice2)
    if bert['ans'].lower() == bert['choice1'].lower() and choice1 > choice2:
        isBert_correct = True
    if bert['ans'].lower() == bert['choice2'].lower() and choice2 > choice1:
        isBert_correct = True
    return isBert_correct, choice1, choice2

def check_SCR_correct(psl, bert):
    isSCR_correct = False
    scr_choice1 = float(psl['scr_score'][0].split('$$')[2])
    scr_choice2 = float(psl['scr_score'][1].split('$$')[2])
    if psl['ans'].lower() == bert['choice1'].lower() and scr_choice1 > scr_choice2:
        isSCR_correct = True
    if psl['ans'].lower() == bert['choice2'].lower() and scr_choice2 > scr_choice1:
        isSCR_correct = True
    return isSCR_correct, scr_choice1, scr_choice2

def compare_psl_wsc_prev(wsc_output_scores, psl_scores, bert_scores):
    wsc_output_map = {}
    bert_output_map = {}
    for wsc_each in wsc_output_scores:
        wsc_output_map[wsc_each['ws_sent']] = wsc_each

    for i in range(0, len(bert_scores)):
        bert_output_map[psl_scores[i]['ws_sent']] = bert_scores[i]

    for psl in psl_scores:

        if psl['ws_sent'] in bert_output_map: 
            bert_out = bert_output_map[psl['ws_sent']]
            wsc_out = None
            if psl['ws_sent'] in wsc_output_map:
                wsc_out = wsc_output_map[psl['ws_sent']]
            isBert_correct, b_choice1, b_choice2 = check_Bert_correct(bert_out)
            isSCR_correct, scr_choice1, scr_choice2 = check_SCR_correct(psl, bert_out)    
            # if psl['predicted'] == 'INCORRECT' and isBert_correct:
            #     if 'context' in psl and len(psl['context']) == 0:
            #          print('PSL no context')
            #     print('PSL INCORRECT, BERT CORRECT: '+psl['ws_sent'])
            # if psl['predicted'] == 'CORRECT' and not isBert_correct:
            #     print('PSL CORRECT, BERT INCORRECT: '+psl['ws_sent'])

            # if psl['predicted'] == 'INCORRECT' and not isBert_correct and wsc_out and wsc_out['result'] == 'correct' and not isSCR_correct:
            #     print('Previous correct, PSL INCORRECT, SCR and BERT INCORRECT: '+psl['ws_sent'])
            
            # # if psl['predicted'] == 'INCORRECT' and isSCR_correct:
            # #     if 'context' in psl and len(psl['context']) == 0:
            # #          print('PSL no context')
            # #     print('PSL INCORRECT, SCR CORRECT: '+psl['ws_sent'])

            # if psl['predicted'] == 'CORRECT' and not isSCR_correct:
            #     print('PSL CORRECT, SCR INCORRECT: '+psl['ws_sent'])

            # if psl['predicted'] == 'INCORRECT' and not isBert_correct and not isSCR_correct:
            #     if 'context' in psl and len(psl['context']) == 0:
            #          print('PSL no context')
            #     print('PSL INCORRECT, BERT and SCR INCORRECT: '+psl['ws_sent'])

        if psl['ws_sent'] in wsc_output_map: 
            wsc_out = wsc_output_map[psl['ws_sent']]
            
            # if psl['predicted'] == 'CORRECT' and wsc_out['result'] == 'incorrect':
            #     if 'context' in psl and len(psl['context']) == 0:
            #         print('PSL no context')
            #     print('PSL CORRECT, PREV_SYS INCORRECT: '+wsc_out['ws_sent'])
            if psl['predicted'] == 'INCORRECT' and wsc_out['result'] == 'correct':
                if 'context' in psl and len(psl['context']) == 0:
                    print('PSL no context')
                print('PSL INCORRECT, PREV_SYS CORRECT: '+wsc_out['ws_sent'])

        # else:
        #     if psl['predicted'] == 'CORRECT':
        #         if 'context' in psl and len(psl['context']) == 0:
        #             print('PSL no context')
        #         print('PSL CORRECT: Knowledge doesnt exist: '+psl['ws_sent'])

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
    bert_scores_file = open("../data/bert_par_scores.json", "r") 
    bert_scores = json.loads(bert_scores_file.read())

    wsc_output_file = open("../data/wsc_prev_output.json", "r") 
    wsc_output_scores = json.loads(wsc_output_file.read())

    #psl_scores_file = open("../backup/psl_multiple_know_out_197_BERT.json", "r") 
    psl_scores_file = open("../data/new_psl_problems_scores.json", "r") 
    psl_scores = json.loads(psl_scores_file.read())

    compare_psl_wsc_prev(wsc_output_scores, psl_scores, bert_scores)

    #calculate_bert_scores(psl_scores);
    correct = 0
    incorrect = 0

    # for i in range(0, len(bert_scores)):
    #     bert = bert_scores[i]
    #     bertCorrect, c1, c2 = check_Bert_correct(bert)
        
    #     if c2 < c1:
    #         correct = correct + 1
    #     else:
    #         print('*******')
    #         incorrect = incorrect + 1

    for i in range(0, len(psl_scores)):
        prob = psl_scores[i]
        if prob['predicted'] == 'CORRECT':
            correct = correct + 1
        else:
            incorrect = incorrect + 1

    print("correct : "+str(correct))
    print("incorrect : "+str(incorrect))

if __name__=="__main__":
    main()
