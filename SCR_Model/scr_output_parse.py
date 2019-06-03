import json
import ast

def merge_prob_problem():
    with open('scr_wsc_prob.json') as f:
        wsc_proabablities = json.load(f)

    with open('scr_wsc285.json') as f:
        problems = json.load(f)

    output = []
    for i in range(0,len(problems)): 
        prob = problems[i]
        obj = {}
        obj['substitution'] = prob["substitution"]
        obj['correctness'] = prob["correctness"]
        obj['question_id'] = prob["question_id"]
        obj['score'] = wsc_proabablities[i]
        output.append(obj)
    

    with open('wsc_scr_scores.json', 'w') as outfile:
        json.dump(output, outfile)

def merge_probability_finalprob():
    all_probs_file = 'inputs/wsc_problems_final.json'
    f = open(all_probs_file,"r")
    all_probs = f.read()
    problems = ast.literal_eval(all_probs)
    
    with open('scr_wsc_prob.json') as f:
        scr_scores = json.load(f)
    
    j = 0
    for i in range(0,len(problems)): 
        prob = problems[i]
        prob["scr_score"] ={'choice1': scr_scores[j], 'choice2': scr_scores[j+1]}
        j = j+2
        
    with open('final_problems.json', 'w') as outfile:
        json.dump(problems, outfile)
    
def calculate_285():
    with open('../final_problems.json') as f:
        problems = json.load(f)
        
    with open('../new_psl_context.json') as f:
        psl_problems = json.load(f)
        
    correct = 0
    incorrect = 0
    for i in range(0,len(problems)): 
        prob = problems[i]
        psl = psl_problems[i]
        isChoice1 = False
        isChoice2 = False
        
        if prob['choice1'].lower() == prob['ans'].lower():
            isChoice1 = True
        else:
            isChoice2 = True
            
        token1 = psl['scr_score'][0].split('$$')
        token2 = psl['scr_score'][1].split('$$')
        
        if float(token1[2]) < float(token2[2]):
            if isChoice2:
                correct = correct + 1
            else:
                incorrect = incorrect + 1
        else:
            if isChoice1:
                correct = correct + 1
            else:
                incorrect = incorrect + 1
    
    print("Correct : ", correct)
    print("Incorrect : ", incorrect)
    print("Total: ", len(problems))

def calculate_285_corrected():
    with open('../final_problems.json') as f:
        problems = json.load(f)
    
    with open('corrected_wsc_scores.json') as f:
        scores = json.load(f)
    correct = 0
    incorrect = 0
    j = 0
    for i in range(0,len(problems)): 
        prob = problems[i]
        isChoice1 = False
        isChoice2 = False
        
        if prob['choice1'].lower() == prob['ans'].lower():
            isChoice1 = True
        else:
            isChoice2 = True
        prob['scr_score']['choice1'] = scores[j]['score']
        prob['scr_score']['choice2'] = scores[j+1]['score']
        
        if scores[j]['score'] < scores[j + 1]['score']:
            if isChoice2:
                correct = correct + 1
            else:
                incorrect = incorrect + 1
        else:
            if isChoice1:
                correct = correct + 1
            else:
                incorrect = incorrect + 1
        j = j + 2
    
    with open('../final_problems.json', 'w') as f:
        json.dump(problems, f)
        
    print("Correct : ", correct)
    print("Incorrect : ", incorrect)
    print("Total: ", len(problems))
   
def calculate_273():
    with open('wsc273_scr_scores.json') as f:
        problems = json.load(f)
        
    correct = 0
    incorrect = 0
    j = 0
    for i in range(0, len(problems)): 
        if j > len(problems) - 1:
            break
        prob_choice1 = problems[j]
        prob_choice2 = problems[j+1]
        isChoice1 = prob_choice1["correctness"]
        isChoice2 = prob_choice2["correctness"]
            
        if prob_choice1['score'] < prob_choice2['score']:
            if isChoice2:
                correct = correct + 1
            else:
                incorrect = incorrect + 1
        else:
            if isChoice1:
                correct = correct + 1
            else:
                incorrect = incorrect + 1
        j = j+2
        
    
    print("Correct : ", correct)
    print("Incorrect : ", incorrect)
    print("Total: ", len(problems)/2)
    
    
def compare_273_285():
    with open('wsc273.json') as f:
        prob_273 = json.load(f)
        
    with open('wsc285.json') as f:
        prob_285 = json.load(f)
    
    with open('wsc273_par_probs.json') as f:
        prob_273_scores = json.load(f)
    
    with open('wsc285_par_probs.json') as f:
        prob_285_scores = json.load(f)
    
    prob_273_dict = {}
    prob_285_dict = {}
    for each in prob_273:
        key = each['substitution'].lower().replace(' ', '')
        if key[-1] != '.':
            key = key+'.'
        prob_273_dict[key] = each
        
    for each in prob_285:
        prob_285_dict[each['substitution'].lower().replace(' ', '')] = each
    
    count = 0
    i = 0
    for key, value in prob_285_dict.items():
        if key not in prob_273_dict:
            print(prob_285_dict[key]['substitution'])
            print(prob_285_dict[key]['question_id'])
            count = count + 1
            prob_285_dict[key]['score'] = prob_285_scores[i] 
        else:
            prob_285_dict[key]['score'] = prob_273_scores[i] 
        i = i + 1
    output = []
    for key, value in prob_285_dict.items():
        output.append(value)
        
    with open('corrected_wsc_scores.json', 'w') as outfile:
        json.dump(output, outfile)
    
def update_wsc_psl():
    with open('../final_problems.json') as f:
        problems = json.load(f)
        
    with open('../new_psl_context.json') as f:
        psl_problems = json.load(f)
    
    for i in range(0, len(problems)): 
        prob = problems[i]
        psl = psl_problems[i]
        
        token1 = psl['scr_score'][0].split('$$')
        token2 = psl['scr_score'][1].split('$$')
        psl['scr_score'][0] = token1[0]+'$$'+token1[1]+'$$'+str(prob['scr_score']['choice1'])
        psl['scr_score'][1] = token2[0]+'$$'+token2[1]+'$$'+str(prob['scr_score']['choice2'])
   
    with open('../new_psl_context.json', 'w') as outfile:
        json.dump(psl_problems, outfile)
    
if __name__ == "__main__":    
    calculate_285()
    