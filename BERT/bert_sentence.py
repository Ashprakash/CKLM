# -*- coding: utf-8 -*-

import json
from pprint import pprint
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForNextSentencePrediction

with open('wsc_problems.json') as f:
    wsc_problems = json.load(f)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
for i in range(0,len(probs)):
    candidates = prob["cand_ques"]
    choice1_ques = candidates[0].strip()
    choice2_ques = candidates[1].strip()

    tokenized_choice_1 = tokenizer.tokenize(choice1_ques)
    tokenized_choice_2 = tokenizer.tokenize(choice2_ques)
    segment_ids_ch1 = get_segment_ids(tokenized_choice_1)
    segment_ids_ch2 = get_segment_ids(tokenized_choice_2)
    
    indexed_token1_ch1 = tokenizer.convert_tokens_to_ids(tokenized_choice_1)
    indexed_token1_ch2 = tokenizer.convert_tokens_to_ids(tokenized_choice_2)
    
    tokens_tensor_ch1 = torch.tensor([indexed_token1_ch1])
    segments_tensors_ch1 = torch.tensor([segment_ids_ch1])
    
    tokens_tensor_ch2 = torch.tensor([indexed_token1_ch2])
    segments_tensors_ch2 = torch.tensor([segment_ids_ch2])
    
    choice1_pred = predict(tokens_tensor_ch1, segments_tensors_ch1)
    choice2_pred = predict(tokens_tensor_ch2, segments_tensors_ch2)
    print(choice1_pred)
    print(choice2_pred)
    

def predict(tokens_tensor, segments_tensors):
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    s
    # Predict hidden states features for each layer
    encoded_layers, _ = model(tokens_tensor, segments_tensors)
    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    assert len(encoded_layers) == 12
    
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    model.eval()
    predictions = model(tokens_tensor, segments_tensors)
    return predictions
    

def get_segment_ids(tokenized_text):
    question_encountered = False
    segmend_ids = []
    for token in tokenized_choice_1:
        if token == '?':
            segmend_ids.append(0)
            question_encountered = True
        if question_encountered:
            segmend_ids.append(1)
    return segmend_ids
            