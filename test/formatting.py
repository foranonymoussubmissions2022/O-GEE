import json
import ast
import pandas as pd
import csv
from collections import Counter
import re
from sklearn.model_selection import train_test_split

all_samples = []
with open("../full_wikidata.jsonl", 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    all_samples.append(json.loads(json_str))


indices = list(range(len(all_samples)))
train_idx, test_idx = train_test_split(indices, test_size=0.3)
dev_idx, test_idx = train_test_split(test_idx, test_size=0.5)
train=[all_samples[i] for i in train_idx]
test=[all_samples[i] for i in test_idx]
dev=[all_samples[i] for i in dev_idx]

with open("re_data/eq_train.json","w") as f:
    for entry in train:
        json.dump(entry,f)
        f.write("\n")

with open("re_data/eq_test.json","w") as f:
    for entry in test:
        json.dump(entry,f)
        f.write("\n")

with open("re_data/eq_dev.json","w") as f:
    for entry in dev:
        json.dump(entry,f)
        f.write("\n")


def create_sparse_full_data(path, output_path):
    data = []
    samples = []
    with open("wde_unlabelled_event.schema","r") as f:
        for o, line in enumerate(f):
            if o ==2:
                event2role =  ast.literal_eval(line)

    with open(path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        samples.append(json.loads(json_str))


    k = 0

    for t, sample in enumerate(samples):
        if t % 1000==0:
            print("%d out of %d completed"%(t, len(samples)))
        data.append({"title":str(t), "paragraphs":[]})
        u_sentence = sample["text"]
        for i, t_sentence in enumerate(sample["sentences"]):
            data[-1]["paragraphs"].append({"qas":[], "context":u_sentence})
            tmp = {}
            for event in sample["events"][i]:
                token_start = event[0][0] 
                token_end = event[0][1]
                event_type = event[0][2] 
                trigger = " ".join(t_sentence[token_start:token_end+1]) 
                if event_type not in tmp:
                    tmp[event_type] = {"triggers":[], "arguments":{}}
                tmp[event_type]["triggers"].append(trigger)


                for event_part in event[1:]: 
                    token_start = event_part[0]
                    token_end = event_part[1]
                    role_type = event_part[2]
                    if role_type not in tmp[event_type]["arguments"]:
                        tmp[event_type]["arguments"][role_type] = []
                    start_index, end_index = get_correct_indices(u_sentence, [token for sent in sample["sentences"] for token in sent], token_start, token_end)
                    
                    arg_string = u_sentence[start_index:end_index]
                    tmp[event_type]["arguments"][role_type].append(arg_string)
                event_type = event_type
                for property in event2role[event_type]:      
                    k+=1
                    question = event_type + "," + property
                    if not event_type in tmp or not tmp[event_type]["arguments"]:
                        data[-1]["paragraphs"][-1]["qas"].append({"question":question, "id":str(k), "answers":[], "is_impossible":True})
                
                    elif event_type in tmp and tmp[event_type]["arguments"]:
                        if property not in tmp[event_type]["arguments"]:
                            data[-1]["paragraphs"][-1]["qas"].append({"question":question, "id":str(k), "answers":[], "is_impossible":True})
                        elif property in tmp[event_type]["arguments"]:
                            answ = {}
                            answrs = []
                            for argument_string in tmp[event_type]["arguments"][property]:
                                answ["text"] = argument_string
                                answ["answer_start"] = u_sentence.index(argument_string)
                                answrs.append(answ)
                            data[-1]["paragraphs"][-1]["qas"].append({"question":question, "id":str(k), "answers":answrs, "is_impossible":False})
    return data


def get_correct_indices(detokenized_sentence, tokenized_sentence, token_start, token_end):
    minimum = 100
    start_indices = [m.start(0) for m in re.finditer(re.escape(tokenized_sentence[token_start]), detokenized_sentence)]
    end_indices = [m.end(0) for m in re.finditer(re.escape(tokenized_sentence[token_end]), detokenized_sentence)]

    for end_index in end_indices:
        for start_index in start_indices:
            if start_index > end_index:
                continue
            if end_index-start_index < minimum and end_index-start_index>=len(tokenized_sentence[token_start]):
                minimum = end_index-start_index
                best_start_index = start_index
                best_end_index = end_index
    return best_start_index, best_end_index
        
def create_training_data():
    train = create_sparse_full_data("re_data/eq_train.json", "re_data/re_train.json")
    train_dataset = {"version":"v2.0", "data":train}
    with open('re_data/sparse_re_train.json', 'w') as f:
        json.dump(train_dataset, f)
    train_dataset = {}
    train = {}

    test = create_sparse_full_data("re_data/eq_test.json", "re_data/re_test.json")
    test_dataset = {"version":"v2.0", "data":test}
    with open('re_data/sparse_re_test.json', 'w') as f:
        json.dump(test_dataset, f)
    test_dataset={}
    test = {}

    dev = create_sparse_full_data("re_data/eq_dev.json", "re_data/re_dev.json")
    dev_dataset = {"version":"v2.0", "data":dev}
    with open('re_data/sparse_re_dev.json', 'w') as f:
        json.dump(dev_dataset, f)
    dev_dataset = {}
    dev = {}

create_training_data()

all_keys = []
class_counter =  Counter()
data = []

def load_data(part = "train"):
    c = Counter()
    data = []
    samples = []
    event_classes = set()
    if part == "dev":
        t = "val"
    else:
        t = part
    with open("re_data/eq_"+part+".json", 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        samples.append(json.loads(json_str))

    for sample in samples:
        event_types = []
        for i, sentence in enumerate(sample["sentences"]):
            for event in sample["events"][i]:
                token_start = event[0][0] 
                token_end = event[0][1] 
                event_type = event[0][2] 
                c.update({event_type:1})
                event_classes.update({event_type})
                event_types.append(event_type)
                class_counter.update({event_type})
        data.append({"label":event_types, "text": sample["text"]})

    return data, list(event_classes), class_counter.most_common(1000)


# read from schema all event types
with open("wde_unlabelled_event.schema","r") as f:
    lines = [line.rstrip() for line in f]
event_classes = ast.literal_eval(lines[0])
event_properties = ast.literal_eval(lines[1])
event_class_properties = ast.literal_eval(lines[2])
a = []

for part in ["train","test", "dev"]:
    data, event_classes, cc = load_data(part)
    a.append(event_classes)
    with open("mlc_data/multilabel_"+part+".csv", "w") as csvfile:
        fieldnames = ["id","text"]+a[0]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, entry in enumerate(data):
            x = dict()
            x["id"] = i
            x["text"] = entry["text"]
            for t in a[0]:
                if t not in entry["label"]:
                    x[t] = 0
                else:
                    x[t] = 1
            writer.writerow(x)


