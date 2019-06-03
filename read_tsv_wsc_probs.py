import json

def main(file_path):
    list_of_probs = []
    f = open(file_path,"r")
    for line in f:
        items = line.rstrip().split("\t")
        if len(items)==7:
            ws_sent = items[0]
            ans = items[1]
            pronoun = items[2]
            choices = items[3]
            #print(choices)
            choices_list = choices.split(":")
            choice1 = choices_list[0]
            choice2 = choices_list[1]
            search_query = items[4]
            know_sent = items[5]
            know_url = items[6]
            ws_prob = {}
            ws_prob["ws_sent"] = ws_sent
            ws_prob["pronoun"] = pronoun
            ws_prob["ans"] = ans
            ws_prob["choice1"] = choice1
            ws_prob["choice2"] = choice2
            ws_prob["know_sent"] = know_sent
            ws_prob["search_query"] = search_query
            ws_prob["know_url"] = know_url
            list_of_probs.append(ws_prob)
        elif len(items)==4:
            ws_sent = items[0]
            ans = items[1]
            pronoun = items[2]
            choices = items[3]
            choices_list = choices.split(":")
            choice1 = choices_list[0]
            choice2 = choices_list[1]
            ws_prob = {}
            ws_prob["ws_sent"] = ws_sent
            ws_prob["pronoun"] = pronoun
            ws_prob["ans"] = ans
            ws_prob["choice1"] = choice1
            ws_prob["choice2"] = choice2
            list_of_probs.append(ws_prob)
        else:
            print(items)
    
    f.close()
    return list_of_probs            

if  __name__=="__main__":
    out_json_file = "wsc_problems_file.json"
    #file_path = "wsc_probs_tsv.tsv"
    #list_of_probs = main(file_path)
    #with open(out_json_file, 'w') as outfile:
    #    json.dump(list_of_probs, outfile)
    
    with open(out_json_file,'r') as outfile:
        data = json.loads(outfile.read())
        print(len(data))
        count = 0
        for prob in data:
            if "know_sent" in prob:
                count+=1
        print(count)
   
    
    
