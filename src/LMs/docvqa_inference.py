import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import json


def inference(opt, model, mydata, save_to_file):
    # 1 load dataset
    # test_dataset = mydata.test_dataset
    loader_test = DataLoader(mydata.test_dataset, batch_size=opt.batch_size)

    model.eval()
    res = []
    # 2. iterate and inference
    # for row in test_dataset:
    for batch in loader_test:
        questionId = batch['questionId']
        input_ids = batch['input_ids'].to(opt.device)
        position_ids = batch['position_ids'].to(opt.device)
        attention_mask = batch['attention_mask'].to(opt.device)
        token_type_ids = batch['token_type_ids'].to(opt.device)
        bbox = batch['bbox'].to(opt.device)
        # start_positions = row['ans_start_positions'].to(opt.device)
        # end_positions = row['ans_end_positions'].to(opt.device)
        pixel_values = batch['pixel_values'].to(opt.device)

        with torch.no_grad():
            outputs = model(input_ids = input_ids, position_ids=position_ids,
                    token_type_ids=token_type_ids, bbox = bbox, attention_mask = attention_mask, 
                    pixel_values = pixel_values)
            # 3.1 get start_logits and end_logits
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # 3.2: get largest logit for both
            for idx, (s,e) in enumerate(zip(start_logits,end_logits)):
                predicted_start_idx = s.argmax(-1).item()    # item() is to get scalars
                predicted_end_idx = e.argmax(-1).item()
                # print("Predicted start idx:", predicted_start_idx)
                # print("Predicted end idx:", predicted_end_idx)
                # 4. get the text answer
                answer = mydata.tokenizer.decode(input_ids[idx][predicted_start_idx:predicted_end_idx+1]) 
                res.append({"questionId":questionId[idx].item(), "answer":answer})
    # save it
    res = json.dumps(res)
    save_res(save_to_file,res)
    return res

def save_res(path,performance_str):
    with open(path, 'w') as f:
        f.write(str(performance_str))

