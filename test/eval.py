import json
import csv 
import pandas as pd
import numpy as np
from collections import Counter
import json
import csv 
import jsonlines
from collections import Counter
import ast
import copy
from rem import run_relation_extraction

def predict(mode, extra="_with_focal_loss"):
    all_ground_samples=[]
    path_to_ground_data = "re_data/eq_test.json"
    x= 0
    write_samples = []

    classes = list(pd.read_csv("mlc_data/multilabel_test.csv").columns)[2:]
    with open("unlabelled_event.schema","r") as f:
        for o, line in enumerate(f):
            if o ==2:
                classprop =  ast.literal_eval(line)

    with open(path_to_ground_data, 'r') as json_file:
        json_list = list(json_file)
    
    for json_str in json_list:
        all_ground_samples.append(json.loads(json_str))

    with open(extra+".csv", "r") as f, jsonlines.open('re_data/eq_test.json',"r") as f2:
        output_reader = csv.reader(f, delimiter="\t")
        ground = [o for o in f2]
        for i, (output_row, sample) in enumerate(zip(output_reader, all_ground_samples)):
            ground = get_ground_dict_from_sample(sample, mode)
            queries = []
            output = {}
            context = output_row[1]
            if i%100==0:
                print("%d out of %d"%(i, len(all_ground_samples)))
            predicted_classes = [cl for cl, prediction in zip(classes, output_row[2:]) if prediction!='False']           
            for cl in predicted_classes:
                output[cl] = {}
                for prop in classprop[cl]:
                    queries.append(cl+","+prop)
            results = run_relation_extraction(queries, context)

            if not any(results):
                x+=1
            tmp1 = []
            tmp2 = []
            # sort queries and results by probability
            for res, query in zip(results, queries):
                if res:
                    tmp1.append(res)
                    tmp2.append(query)
                    
            results = tmp1
            results = [(ordered_dict[0]["text"], ordered_dict[0]["probability"]) for ordered_dict in results]
            queries = tmp2

            queries = [i for _,i in sorted(zip(results,queries), key=lambda x: x[0][1], reverse=True)]

            results = sorted(results, key=lambda x: x[1], reverse=True)


            results, queries = [answ for answ in results if answ[0]!="empty"],[q for q, answ in zip(queries, results) if answ[0]!="empty"]
            for res_answer, query in zip(results, queries):
                cl = query.split(",")[0]
                prop = query.split(",")[1]
                answer = res_answer[0]
                output[cl][prop] = answer
                
            output = {key:value for key,value in output.items() if key!=None}
            preserve_output = copy.deepcopy(output)
            preserve_ground = copy.deepcopy(ground)
            write_sample={"text":context, "predicted":preserve_output, "ground":preserve_ground}
            write_samples.append(write_sample)

    with jsonlines.open(extra+"_predictions.json","w") as f:
        f.write_all(write_samples)

def get_ground_dict_from_sample(sample, data="dbp"):
    groundtruth = sample["event"]
    groundtruth_events = groundtruth.split("<extra_id_1><extra_id_1><extra_id_0>")
    ground_dict = {}
    for en, event in enumerate(groundtruth_events):
        if en==0:
            the_split = event.split("<extra_id_0>")[2:]
            groundtruth_tokenized = [i.replace("<extra_id_1>","") if "<extra_id_1>" in i else i for i in the_split]
        else:
            the_split = event.split("<extra_id_0>")
            groundtruth_tokenized = [i.replace("<extra_id_1>","") if "<extra_id_1>" in i else i for i in the_split]
        if data=="wde":
            for token in groundtruth_tokenized:
                if  token[0]=="Q":#token[0].isupper():# for wikidata token[0]=="Q":
                    ground_event_type = token.split()[0]
                if ground_event_type not in ground_dict:
                    ground_dict[ground_event_type] = {}
                if token[0] == "P":#not token[0].isupper():# for wikidata token[0] == "P":
                    argument_type = token.split()[0]
                    argument_string = " ".join(token.split()[1:])
                    if argument_type not in ground_dict[ground_event_type]:
                        ground_dict[ground_event_type][argument_type] = []
                    ground_dict[ground_event_type][argument_type].append(argument_string)
        else:
            for token in groundtruth_tokenized:
                if  token[0].isupper():# for wikidata token[0]=="Q":
                    ground_event_type = token.split()[0]
                if ground_event_type not in ground_dict:
                    ground_dict[ground_event_type] = {}
                if not token[0].isupper():# for wikidata token[0] == "P":
                    argument_type = token.split()[0]
                    argument_string = " ".join(token.split()[1:])
                    if argument_type not in ground_dict[ground_event_type]:
                        ground_dict[ground_event_type][argument_type] = []
                    ground_dict[ground_event_type][argument_type].append(argument_string)
    return ground_dict

def eval_scores(path, Approach="Text2Event"):
    macroP = 0
    macroR = 0
    macroF1 = 0
    totalTP = 0
    totalFP = 0
    totalFN = 0
    
    with open(path,"r") as f:
        scores = json.load(f)
    scoresF = {}
    for cl in scores:
        totalTP += scores[cl]["tp"]
        totalFP += scores[cl]["fp"]
        totalFN += scores[cl]["fn"]
        if scores[cl]["tp"] == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = scores[cl]["tp"]/(scores[cl]["tp"]+scores[cl]["fp"])
            recall = scores[cl]["tp"]/(scores[cl]["tp"]+scores[cl]["fn"])
            f1 = 2*precision*recall/(precision+recall)
        scoresF[cl] = (precision,recall,f1)

    microP = totalTP/(totalTP+totalFP)
    microR = totalTP/(totalTP+totalFN)
    microF1 = microP*2*microR/(microP+microR)
    for cl in scoresF:
        macroP += scoresF[cl][0]
        macroR += scoresF[cl][1]
        macroF1 += scoresF[cl][2]
    macroP /= len(scoresF)
    macroR /= len(scoresF)
    macroF1 /= len(scoresF) 
    #macroF1 = macroP*2*macroR/(macroP+macroR)

    print(Approach+" &","%.2f" % macroP, "&", "%.2f" % macroR, "&", "%.2f" % macroF1, "&","%.2f" % microP, "&","%.2f" % microR, "&","%.2f" % microF1, " \\")

def load_t2e(sample, mode):
    # if t2e predicted multiple events of the same type in one sample, merge their properties into one event isntance,
    # if it predicted multiple properties of the same type for the same event type, overwrite them
    ground_dict = get_ground_dict_from_sample(sample, mode)
    output = sample["t2e"][0]
    output_dict = {}
    for event in output:
        cl = event["type"]
        if cl not in output_dict:
            output_dict[cl] = {}
        for role in event["roles"]:
            property = role[1]
            arg_string = role[2]
            if property not in output_dict[cl]:
                output_dict[cl][property] = arg_string
    return ground_dict, output_dict

def load_dygiepp(sample):
    groundtruth = sample["events"]
    ground_dict = {}
    for gg, sentence_events in enumerate(groundtruth):
        for event in sentence_events:
            for event_part in event:
                if len(event_part) == 2:
                    event_type = event_part[1]
                    ground_dict[event_type] = {}
                elif event_part[2] not in ground_dict[event_type]:
                    ground_dict[event_type][event_part[2]]=[]
                    start = event_part[0]
                    end = event_part[1]
                    merged_sentence = [token for sentence in sample["sentences"] for token in sentence]
                    ground_dict[event_type][event_part[2]].append(merged_sentence[start:end+1])

    output = sample["predicted_events"]
    output_dict = {}
    for gg, sentence_events in enumerate(output):
        for event in sentence_events:
            for event_part in event:
                if len(event_part) == 4:
                    event_type = event_part[1]
                    output_dict[event_type] = {}
                elif event_part[2] not in output_dict[event_type]:
                    output_dict[event_type][event_part[2]]=[]
                    start = event_part[0]
                    end = event_part[1]
                    merged_sentence = [token for sentence in sample["sentences"] for token in sentence]
                    output_dict[event_type][event_part[2]].append(" ".join(merged_sentence[start:end+1]))
    
    new_ground_dict =copy.deepcopy(ground_dict)
    new_output_dict = copy.deepcopy(output_dict)

    for cl in ground_dict:
        for prop in ground_dict[cl]:
            new_ground_dict[cl][prop] = [" ".join(ground_dict[cl][prop][0])]

    for cl in output_dict:
        for prop in output_dict[cl]:
            new_output_dict[cl][prop] = output_dict[cl][prop][0]
    
    return new_ground_dict, new_output_dict

def exact_eval(path_to_output_data, mode,):
    scores = {}
    property_scores = {}
    for cl in all_event_types:
        scores[cl] = {"tp":0,"fp":0,"tn":0,"fn":0}
    for prop in all_properties:
        property_scores[prop] = {"tp":0,"fp":0,"tn":0,"fn":0}

    version1_tp = 0
    version1_fp = 0
    version1_tn = 0
    version1_fn = 0

    all_output_samples = []
    with open(path_to_output_data, 'r') as json_file:
        json_list = list(json_file)
    
    for json_str in json_list:
        all_output_samples.append(json.loads(json_str))
 

    for i, sample in enumerate(all_output_samples):
        #print("%d out of %d complete"%(i, len(all_output_samples)))
        if "t2e" in path_to_output_data:
            ground_dict, output_dict = load_t2e(sample, mode)
        elif "dygiepp" in path_to_output_data:
            ground_dict, output_dict = load_dygiepp(sample)
        else:
            ground_dict, output_dict = sample["ground"], sample["predicted"]
            for cl in ground_dict:
                for prop in ground_dict[cl]:
                    ground_dict[cl][prop] = [ground_dict[cl][prop]]

        done_already = []
        while ground_dict != {} or output_dict != {}:
            for cl in scores:
                if cl in ground_dict and cl in output_dict:
                    scores[cl]["tp"] +=1
                    for property in class_prop[cl]:
                        if property in ground_dict[cl] and property in output_dict[cl]:
                            output_value = output_dict[cl][property]
                            ground_value = ground_dict[cl][property][0]
                            while isinstance(output_value, list):
                                output_value = output_value[0]
                            while isinstance(ground_value, list):
                                ground_value = ground_value[0]
                            
                            if ground_value==output_value or ground_value in output_value or output_value in ground_value:
                            #if ground_dict[cl][property][0] == output_dict[cl][property]:
                                property_scores[property]["tp"]+=1
                                version1_tp += 1
                            else:
                                property_scores[property]["fp"]+=1
                                property_scores[property]["fn"]+=1
                                version1_fp += 1
                                version1_fn += 1
                        elif property not in ground_dict[cl] and property in output_dict[cl]:
                            property_scores[property]["fp"]+=1
                            version1_fp += 1
                        elif property in ground_dict[cl] and property not in output_dict[cl]:
                            property_scores[property]["fn"]+=1
                            version1_fn += 1
                        elif property not in ground_dict[cl] and property not in output_dict[cl]:
                            property_scores[property]["tn"]+=1
                            version1_tn += 1
                        if property in ground_dict[cl]:
                            del ground_dict[cl][property]
                        if property in output_dict[cl]:
                            del output_dict[cl][property]
                elif cl in ground_dict and cl not in output_dict:
                    scores[cl]["fn"] +=1
                elif cl not in ground_dict and cl in output_dict:
                    scores[cl]["fp"] +=1
                elif cl not in ground_dict and cl not in output_dict and cl not in done_already:
                    scores[cl]["tn"] +=1
                done_already.append(cl)
                if cl in ground_dict :
                    del ground_dict[cl]
                if cl in output_dict:
                    del output_dict[cl]   
    property_r = version1_tp/(version1_tp+version1_fn)
    property_p = version1_tp/(version1_tp+version1_fp)
    prop_f1 = (2 * (property_p * property_r) / (property_p + property_r))

    with open(mode+"_class_scores.json","w") as f:
        json.dump(scores, f)   
    with open(mode+"_property_scores.json","w") as f:
        json.dump(property_scores, f)

if __name__=="__main__":
    with open("unlabelled_event.schema","r") as f:
        lines = [line.rstrip() for line in f]
    all_event_types = ast.literal_eval(lines[0])
    all_properties = ast.literal_eval(lines[1])
    class_prop = ast.literal_eval(lines[2])

    mode = "wde"
    extra = "_with_focal_loss"
    predict(mode,extra)
    print("classes:")
    exact_eval(mode+"_"+extra+"_predictions.json", mode)
    eval_scores(mode+"_class_scores.json", "O-GEE")

    print("properties:")
    eval_scores(mode+"_property_scores.json", "O-GEE")

    



    



