import json
import ast 

all_probs_file = "inputs/wsc_problems_final.json"
f = open(all_probs_file,"r")
all_probs = f.read()
probs = ast.literal_eval(all_probs)

wsc_bert_output = [];
for i in range(0,len(probs)):
        prob = probs[i]
        ws_sent = prob["ws_sent"]
        ws_sent = ws_sent.strip()
        obj = {}
        obj['question'] = ws_sent;
        obj['pronoun'] = prob["pronoun"];
        obj['choice1'] = prob["choice1"];
        obj['choice2'] = prob["choice2"];
        obj['ans'] = prob["ans"];
        obj['cand_ques'] = [];
        obj['cand_ques'].append(ws_sent.replace(' '+prob["pronoun"]+' ', ' '+prob["choice1"].lower()+' '))
        obj['cand_ques'].append(ws_sent.replace(' '+prob["pronoun"]+' ', ' '+prob["choice2"].lower()+' '));
        wsc_bert_output.append(obj);


with open('scr_model_out.json', 'w') as outfile:
    json.dump(wsc_bert_output, outfile)

        
