#import nltk
import xml.etree.ElementTree as ET 
import re
from pprint import pprint
import ast
import json

def process1(text):
    #text = text.replace("."," .")
    #text = text.replace(","," ,")
    #text = text.replace(";"," ;")
    #text = text.replace("\"Dibs!\"","Dibs")
    #text = text.replace("\"Check\"","\" Check \"")
    #text = text.replace("20 ,000","20,000")
    #text = text.replace("couldn't","could n't")
    #text = text.replace("didn't","did n't")
    #text = text.replace("doesn't","does n't")
    #text = text.replace("wasn't","was n't")
    #text = text.replace("can't","ca n't")
    #text = text.replace("don't","do n't")
    #text = text.replace("hadn't","had n't")
    #text = text.replace("won't","wo n't")
    #text = text.replace("wouldn't","would n't")
    text = text.replace("Sam's","Sam 's")
    text = text.replace("Tina's","Tina 's")
    text = text.replace("Ann's","Ann 's")
    text = text.replace("Joe's","Joe 's")
    text = text.replace("Charlie's","Charlie 's")
    text = text.replace("Cooper's","Cooper 's")
    text = text.replace("Yakutsk's","Yakutsk 's")
    text = text.replace("he's","he 's")
    text = text.replace("Fred's","Fred 's")
    text = text.replace("Goodman's","Goodman 's")
    text = text.replace("Emma's","Emma 's")
    text = text.replace("Susan's","Susan 's")
    text = text.replace("Pam's","Pam 's")
    text = text.replace("Mark's","Mark 's")
    text = text.replace("Amy's","Amy 's")
    text = text.replace("Paul's","Paul 's")
    text = text.replace("I'm","I 'm")
    re.sub( '\s+', ' ', text ).strip()
    return text


def process(text):
    if text[-1]!=".":
        text += "."
    text = text.replace("."," .")
    text = text.replace(","," ,")
    text = text.replace(";"," ;")
    text = text.replace("\"Dibs!\"","Dibs")
    text = text.replace("\"Check\"","Check")
    text = text.replace("20 ,000","20,000")
    text = text.replace("couldn't","could not")
    text = text.replace("didn't","did not")
    text = text.replace("doesn't","does not")
    text = text.replace("wasn't","was not")
    text = text.replace("can't","can not")
    text = text.replace("don't","do not")
    text = text.replace("hadn't","had not")
    text = text.replace("won't","will not")
    text = text.replace("wouldn't","would not")
    #text = text.replace("Sam's","Sam 's")
    #text = text.replace("Tina's","Tina 's")
    #text = text.replace("Ann's","Ann 's")
    #text = text.replace("Joe's","Joe 's")
    #text = text.replace("Charlie's","Charlie 's")
    #text = text.replace("Cooper's","Cooper 's")
    #text = text.replace("Yakutsk's","Yakutsk 's")
    text = text.replace("he's","he is")
    #text = text.replace("Fred's","Fred 's")
    #text = text.replace("Goodman's","Goodman 's")
    #text = text.replace("Emma's","Emma 's")
    #text = text.replace("Susan's","Susan 's")
    #text = text.replace("Pam's","Pam 's")
    #text = text.replace("Mark's","Mark 's")
    #text = text.replace("Amy's","Amy 's")
    #text = text.replace("Paul's","Paul 's")
    text = text.replace("I'm","I am")
    re.sub( '\s+', ' ', text ).strip()
    return text


def read_wscxml_file(xml_file_path):
    list_of_wsc_probs = []
    
    all_comparisons = set()
    # create element tree object 
    tree = ET.parse(xml_file_path) 
    # get root element 
    root = tree.getroot() 
    # create empty list for news items 
#    data_points = [] 
    # iterate news items 
    for item in root.findall('./schema'):
        sent = ""
        pronoun = ""
        answers = []
        corr_ans = ""
        for child in item:
            if child.tag=="text":
                for sent_child in child:
                    sent += sent_child.text.strip().rstrip().replace("\n"," ") + " "
                for child1 in child:
                    if child1.tag=="pron":
                        pronoun = child1.text.replace("\n"," ")
            if child.tag=="answers":
                for ans_child in child:
                    answers.append(ans_child.text.replace("\n"," "))
            if child.tag=="correctAnswer":
                ans = child.text
                if ans.strip()=="A":
                    corr_ans = answers[0]
                elif ans.strip()=="B":
                    corr_ans = answers[1]
                
        prob = {}
        sent = re.sub(' +',' ',sent).strip()
        pronoun = re.sub(' +',' ',pronoun).strip()
        ans = re.sub(' +',' ',corr_ans).strip()
        choice1 = re.sub(' +',' ',answers[0]).strip()
        choice2 = re.sub(' +',' ',answers[1]).strip()
        prob["ws_sent"] = sent
        prob["pronoun"] = pronoun
        prob["ans"] = ans
        prob["choice1"] = choice1
        prob["choice2"] = choice2
        prob["know_sent"] = "NA"
        prob["search_query"] = "NA"
        prob["know_url"] = "NA"

        list_of_wsc_probs.append(prob)

    return list_of_wsc_probs            

def update_wsc_probs_json(tsv_probs_file, wsc_probs_file, output_wsc_probs_file):
    wsc_probs_dict = {}
    f1 = open(tsv_probs_file,'r')
    for line in f1:
        parts = line.split("\t")
        ws_sent = parts[0]
        know_sent = parts[1]
        know_url = parts[2]
        search_query = parts[3]
        tmp_dict = {}
        tmp_dict["ws_sent"] = ws_sent
        tmp_dict["know_sent"] = know_sent
        tmp_dict["know_url"] = know_url
        tmp_dict["search_query"] = search_query
        wsc_probs_dict[ws_sent] = tmp_dict

    f2 = open(wsc_probs_file,'r')
    all_probs_data = f2.read()
    all_wsc_probs = ast.literal_eval(all_probs_data)
    for wsc_prob in all_wsc_probs:
        if wsc_prob["ws_sent"] in wsc_probs_dict.keys():
            tmp_dict = wsc_probs_dict[wsc_prob["ws_sent"]]
            wsc_prob["know_sent"] = tmp_dict["know_sent"]
            wsc_prob["know_url"] = tmp_dict["know_url"]
            wsc_prob["search_query"] = tmp_dict["search_query"]

    with open(output_wsc_probs_file, 'wt') as out:
        pprint(all_wsc_probs, stream=out)

def combine_sent_qasrl(sents_file,qasrl_file,sent_and_qasrl_file):
    sents = []
    f1 = open(sents_file,'r')
    for line in f1:
        sents.append(line)
    
    qasrls = []
    f2 = open(qasrl_file,'r')
    for line in f2:
        qasrls.append(line)

    with open(sent_and_qasrl_file, 'a') as the_file:
        for i in range(0,len(sents)):
            the_file.write(sents[i].rstrip()+"$$$$"+qasrls[i].rstrip()+"\n")

def check_probs_without_know_sent(wsc_probs_file):
    f2 = open(wsc_probs_file,'r')
    all_probs_data = f2.read()
    all_wsc_probs = ast.literal_eval(all_probs_data)
    count=0
    for wsc_prob in all_wsc_probs:
        if wsc_prob["know_sent"]=="NA":
            count += 1
    print(count)

if __name__=="__main__":
    #xml_data_file_path = "./WSCollection.xml"
    #wsc_data = read_wscxml_file(xml_data_file_path)
    #with open('wsc_problems.json', 'wt') as out:
    #    pprint(wsc_data, stream=out)

    #tsv_probs_file = "newer_wsc_probs_1.tsv"
    #wsc_probs_json_file = "../inputs/wsc_problems_final.json"
#    output_wsc_probs_file = "wsc_problems_final.json"
    #update_wsc_probs_json(tsv_probs_file, wsc_probs_json_file, output_wsc_probs_file)
    
#    check_probs_without_know_sent(output_wsc_probs_file)
   
    know_sents_file = "/home/ASUAD/asharm73/nrl-qasrl/group24_know_sents_to_parse"
    know_qasrl_file = "/home/ASUAD/asharm73/nrl-qasrl/group24_know_parsed"
    know_sents_and_qasrl_file = "/home/ASUAD/asharm73/WSC_with_knowledge/inputs/group24/know_sents_and_qasrl_out.txt"
    combine_sent_qasrl(know_sents_file,know_qasrl_file,know_sents_and_qasrl_file)
 
    '''
    know_sents_file = ""
    know_qasrl_file = ""
    know_sents_and_qasrl_file = ""
    combine_sent_qasrl(know_sents_file,know_qasrl_file,know_sents_and_qasrl_file)

    ws_sents_file = ""
    ws_qasrl_file = ""
    ws_sents_and_qasrl_file = ""
    combine_sent_qasrl(ws_sents_file,ws_qasrl_file,ws_sents_and_qasrl_file)
    '''



