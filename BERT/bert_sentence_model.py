# -*- coding: utf-8 -*-

import json
from pprint import pprint
import torch
import logging
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForNextSentencePrediction
import numpy as np

#logging.basicConfig(format = '%(asctime)s %(message)s',
#                    datefmt = '%m/%d/%Y %H:%M:%S',
#                    level = logging.DEBUG)
#logger=logging.getLogger()

def predict(tokens_tensor, segments_tensors, device, n_gpu):
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')
    #model.to(device)
    model.eval()
    # Predict hidden states features for each layer
    encoded_layers, _ = model(tokens_tensor, segments_tensors)
    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    assert len(encoded_layers) == 12

    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    #model.to(device)
    model.eval()
    predictions = model(tokens_tensor, segments_tensors)
    return predictions


def get_segment_ids(tokenized_text):
    question_encountered = False
    segmend_ids = []
    for token in tokenized_text:
        if token == '?':
            question_encountered = True
            segmend_ids.append(0)
            continue

        if question_encountered:
            segmend_ids.append(1)
        else:
            segmend_ids.append(0)
    return segmend_ids

def main():
    data_dir = '/home/apraka23/Winograd/pytorch-pretrained-BERT/wsc_problems.json'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    with open(data_dir) as f:
        wsc_problems = json.load(f)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    correct = 0
    incorrect = 0
    for i in range(0, len(wsc_problems)):
        if i == 20:
            break
        prob = wsc_problems[i]
        candidates = prob["cand_ques"]
        choice1_ques = candidates[0].strip()
        choice2_ques = candidates[1].strip()

        tokenized_choice_1 = tokenizer.tokenize(choice1_ques)
        tokenized_choice_2 = tokenizer.tokenize(choice2_ques)
        segment_ids_ch1 = get_segment_ids(tokenized_choice_1)
        segment_ids_ch2 = get_segment_ids(tokenized_choice_2)

        print(tokenized_choice_1)
        print(segment_ids_ch1)
        print(tokenized_choice_2)
        print(segment_ids_ch2)
        indexed_token1_ch1 = tokenizer.convert_tokens_to_ids(tokenized_choice_1)
        indexed_token1_ch2 = tokenizer.convert_tokens_to_ids(tokenized_choice_2)

        tokens_tensor_ch1 = torch.tensor([indexed_token1_ch1])
        segments_tensors_ch1 = torch.tensor([segment_ids_ch1])

        tokens_tensor_ch2 = torch.tensor([indexed_token1_ch2])
        segments_tensors_ch2 = torch.tensor([segment_ids_ch2])

        choice1_pred = predict(tokens_tensor_ch1, segments_tensors_ch1, device, n_gpu)
        choice2_pred = predict(tokens_tensor_ch2, segments_tensors_ch2, device, n_gpu)
        choice1_list = choice1_pred.detach().numpy()
        choice2_list = choice2_pred.detach().numpy()

        if(np.amax(choice1_list) > np.amax(choice2_list)):
            logging.debug('choice1')
            if prob["ans"] == prob["choice1"]:
                correct = correct + 1;
            else:
                incorrect = incorrect + 1;
        else:
            if prob["ans"] == prob["choice2"]:
                correct = correct + 1;
            else:
                incorrect = incorrect + 1;
            print('choice2')
        print(choice1_pred)
        print(choice2_pred)
        print("Correct Inside Loop: ", correct)
        print("Incorrect Inside Loop: ", incorrect)
        print(i+1, "of ", len(wsc_problems))

    print("Correct", correct)
    print("Incorrect", incorrect)
    print("Total", len(wsc_problems))

if __name__=="__main__":
    main()