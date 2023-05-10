import sys, os
import ast
import os
from classes import * 
import ndjson
from collections import Counter,  OrderedDict
import shelve
import time
import json
import datetime
import datefinder
import shelve
import csv
import re
rx = r"\.(?=\S)"
import copy
from sklearn.model_selection import train_test_split
import jsonlines
import spacy  # version 3.0.6'

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("entityLinker")

def remove_non_ascii(string):
    return ''.join(char for char in string if ord(char) < 256)


dbpedia_ids_to_wikidata_ids = shelve.open('../data/shelves/dbpedia_to_wikidata_en')
wikidata_ids_to_dbpedia_ids = shelve.open('../data/shelves/wikidata_to_dbpedia_en')
types_dbo = shelve.open('../data/shelves/types_dbo2')
types_wd = shelve.open('../data/shelves/types_wd')
instance_types = shelve.open("../data/shelves/instance_types")


with open("event_subevents.json","r") as f:
    events = json.load(f)


def first_stage_eval_data():
    directory = "../data/raw_data/"
    dataset = {}
    dygiepp_data = []
    i = 0
    k = 0
    mistakes = 0
    event_wd_not_in_events = 0
    d = []
    all = 0
    e = time.time()
    context_window = 0
    for u, filename in enumerate(os.listdir(directory)):
        if ".gitignore" in filename:
            continue
        print("%d out of %d completed"%(u, len(os.listdir(directory))))
        with open(directory+filename, "r") as f:
            print(filename)
            json_list = ndjson.load(f)
        for g,js in enumerate(json_list):
            all+=1
            Article_obj = Article(js)
            parent_event_link = ("_".join(Article_obj.article_name.split(" ")))
            try:
                parent_event_wd_id = parent_event_link
            except (KeyError):
                mistakes+=1
                continue
            if parent_event_wd_id not in events:
                event_wd_not_in_events+=1

            for input_paragraph in Article_obj.input_paragraphs:
                last_sentence_subevents = []
                for sentence in input_paragraph.sentences:
                    all_sentence_link_types = [link["types"] for link in sentence.links]
                    all_sentence_link_types = [link for links in all_sentence_link_types for link in links]
                    for link in sentence.links:
                        if "Event" in all_sentence_link_types:
                            if "Event" in link["types"]:
                                subevent_link = link["target"]
                                try:
                                    subevent_wd_id = subevent_link
                                except (KeyError):
                                    d.append(subevent_wd_id)
                                    continue
                                # if subevent doesnt have partof event
                                i+=1
                                if i%10000==0:
                                    print(i)
                                last_sentence_subevents.append((subevent_wd_id, subevent_link))
                                if parent_event_wd_id not in dataset:
                                    dataset[parent_event_wd_id] = {}
                                if subevent_wd_id +"@" + subevent_link not in dataset[parent_event_wd_id]:
                                    dataset[parent_event_wd_id][subevent_wd_id +"@" + subevent_link] = {"samples":[], "groundtruth":{}}
                                dataset[parent_event_wd_id][subevent_wd_id +"@" + subevent_link]["samples"].append((link, sentence.text, sentence.links))
                                dygiepp_data.append(sentence.text)
                        elif context_window:
                            context_window = 0
                            for subevent in last_sentence_subevents:
                                subevent_wd_id, subevent_link = subevent
                                link, text, links = dataset[parent_event_wd_id][subevent_wd_id +"@" + subevent_link]["samples"][-1]
                                text += sentence.text
                                links += sentence.links
                                dataset[parent_event_wd_id][subevent_wd_id +"@" + subevent_link]["samples"][-1] = (link, text, links)
                                dygiepp_data.append(sentence.text)
                            last_sentence_subevents = [] 
    print("all articles:", all)
    print("mistakes:", mistakes)
    print("print not in events:",event_wd_not_in_events)
    print("missed_subevents:", len(d))
    with open('dbp_eval_data_with_context_window.txt', 'w') as f:
        json.dump(dataset,f)
    
first_stage_eval_data()



with open('dbp_eval_data_with_context_window.txt', 'r') as f:
    dataset = json.load(f)
    
done_already = []
with open("dbp_full_event_list2.txt","w") as f:
    for parent_event in dataset:
        subevents = dataset[parent_event]
        for key in subevents.keys():
            x = key.split("@")[0]+"\n"
            if x not in done_already:
                done_already.append(x)
                f.write(x) 
            else:
                continue   
print("First stage done")

subevents_properties = {}
with open("dbp_all_subevent_properties.csv", "r") as f:
    lines = f.readlines()[1:]
    for row in lines:
        row = row.rstrip().split("\t")
        #subevent, event = row.split("\t")[0], row.split("\t")[1]
        event, property, value = row[0], row[1], row[2]
        if event not in subevents_properties:
            subevents_properties[event] = {}
            
        if property not in subevents_properties[event]:
            subevents_properties[event][property] = []
        subevents_properties[event][property].append(value)

d = []
for event_wd_id in dataset:
    for subevent in dataset[event_wd_id]:
        number_of_sentences = len(dataset[event_wd_id][subevent]["samples"]) 
        try:
            subevent_wd_id, subevent_link = subevent.split("@")
        except ValueError:
            print(subevent)
        if subevent_wd_id not in subevents_properties:
            d.append(subevent_wd_id)
            continue
        subevent_properties = subevents_properties[subevent_wd_id]
        dataset[event_wd_id][subevent]["groundtruth"] = subevent_properties
print(len(d))

with open('dbp_all_data.txt', 'w') as f:
    json.dump(dataset,f)


with open('dbp_all_data.txt', 'r') as f:
    dataset = json.load(f)


new_dataset = {}
tpe = 0
no_type = 0
total_subevents = []
subevents_missed_because_no_prop_or_no_instance_type = []
d2 = []
subevent_types_removed_because_they_have_no_properties = []
subevents_removed_because_they_have_no_types_remaining = []
d5 = []
number_of_samples = []
number_of_participant_properties = 0
number_of_time_properties = 0
missed_subevents = 0
ggg = 0
total_failure = 0
all_event_types = set()
for num, event_wd_id in enumerate(dataset):
    print("%d out %d"%(num, len(dataset)))
    for subevent in dataset[event_wd_id]:
        total_subevents.append(subevent)
        try:
            subevent_wd_id, subevent_link = subevent.split("@")
        except ValueError:
            total_failure+=1
            continue
        if "World_War_II" in subevent_wd_id:
            print()
        if subevent_wd_id in types_dbo:
            try:
                subevent_types = types_dbo[subevent_wd_id]
            except(KeyError):
                d2.append(subevent_link)
        else:
            full_url = "<http://dbpedia.org/resource/"+subevent_wd_id+">"
            if full_url in types_dbo:
                try:
                    subevent_types = types_dbo[full_url]
                except(KeyError):
                    d2.append(subevent_link)
            else:
                subevents_missed_because_no_prop_or_no_instance_type.append(subevent_wd_id)
                continue
        subevent_types = list(subevent_types)
        groundtruth_properties = dataset[event_wd_id][subevent]["groundtruth"]
        tmp = subevent_types
        #ignore events which are types as "event"
        all_event_types.update(subevent_types)
        if "SocietalEvent" in subevent_types: #event
            subevent_types.remove("SocietalEvent") #event
        if "Event" in subevent_types:
            subevent_types.remove("Event") 
        if not subevent_types:
            no_type+=1
            continue
        groundtruth_properties = {k:v for k, v in groundtruth_properties.items()}
        groundtruth_values = {wd:k for k,v in groundtruth_properties.items() for wd in v}

        if groundtruth_values == {}:
            tpe+=1
            continue

        completed = set()
        for sample in dataset[event_wd_id][subevent]["samples"]:
            new_groundtruth = {}
            sample_subevent = sample[0]
            if subevent_wd_id not in subevents_properties:
                missed_subevents+=1
                continue
            subevent_class_type = subevent_types
            sample_subevent["types"] = subevent_class_type
            
            
            sample_context = re.sub(rx, ". ", sample[1])
            sample_context = sample_context.replace("-"," - ")
            sample_context = sample_context.replace("–"," – ")
            sample_context = sample_context.replace("  ", " ")
            # avoid short bulletpoints
            if "*" in sample_context and len(sample_context)<100:
                continue

            if len(sample_context)<30:
                continue
            sample_links = sample[2]
            try: 
                subevent_anchor = nlp(sample_subevent["anchor_text"])   
                doc = nlp(sample_context)
            except:
                print("missed one")
                continue
            total = []
            first = False
            subevent_list_of_words = [t.text for t in subevent_anchor]

            if sample_subevent["anchor_text"] in doc.text:
                start = -1
                end = -1
                found = False
                start_char = doc.text.index(sample_subevent["anchor_text"])
                end_char = start_char + len(sample_subevent["anchor_text"])
                sample_subevent["character_range"] = (start_char,end_char)
                for t in doc:
                    if t.text == sample_subevent["anchor_text"]:
                        start = t.i
                        end = t.i
                        sample_subevent["token_range"] = (start,end)
                        found = True
                        break
                    elif not first and t.text in sample_subevent["anchor_text"]:
                        start = t.i
                        end = t.i
                        total.append(t.text)
                        first = True
                    elif total != subevent_anchor.text.split() and t.text not in sample_subevent["anchor_text"]:
                        first = False
                        total = []
                    elif t.text in sample_subevent["anchor_text"]:
                        end = t.i
                        total.append(t.text)
                    if total==subevent_list_of_words:
                        found = True
                        end = t.i
                        sample_subevent["token_range"] = (start,end)
                        break
            else:
                continue
            if not found:
                continue


            times = datefinder.find_dates(sample_context,base_date=datetime.datetime(9999, 1, 1, 0, 0), source="True")
            times = [(time_expression[0].isoformat(),time_expression[1])  for time_expression in times]#+[dateparser_time]
            new_times = []
            for tm in times:
                if  "." in tm[1]:
                    new_time = (tm[0],tm[1].split(".")[0])
                    new_times.append(new_time)
                else:
                    new_times.append(tm)
            if new_times:
                times = new_times
            if sample_subevent["anchor_text"] not in sample_context:
                    continue
            for ttime in times:
                if ttime[0] not in groundtruth_values:
                    continue
                else:
                    property = groundtruth_values[ttime[0]]
                    first = 1
                    # if the time is exactly accurate, store it and remove from groundtruth_values
                    for t in doc:
                        if first and t.text in ttime[1].split():
                            start = t.i
                            end = t.i
                            start_char = sample_context.index(t.text)
                            end_char = sample_context.index(t.text)+len(t.text)
                            first = 0
                        elif t.text in ttime[1].split(): # .split() so that a piece of word is not found in a bigger word
                            end = t.i
                            end_char = sample_context.index(t.text)+len(t.text)
                        elif not first:
                            break
                    if property not in new_groundtruth:
                        new_groundtruth[property] = []
                    new_groundtruth[property].append([start, end, start_char, end_char, property, ttime[1]]) # it used to be just ttime
                    del groundtruth_values[ttime[0]]
            
            updated_times =[]
            for t in times:
                if t:
                    tmp1 = t[0].split("-")
                    tmp2 = [i.split(":") for i in tmp1]
                    tmp3 = [i for j in tmp2 for i in j]
                    updated_times.append((tmp3,t[1]))
            # If a year,month../data correct sequence was found but the rest doesn't match, store the sequence as happening on the 1st
            ground_base = ["9999","01","01T00","00","00Z"]
            for property_value in groundtruth_values:
                if not updated_times:
                    break
                ground_time = []

                if isinstance(property_value, str) and len(property_value.split("-")) == 3 and property_value[-1]=="Z" and property_value[-3:-1].isdigit():
                    tmp1 = property_value.split("-")
                    tmp2 = [i.split(":") for i in tmp1]
                    arranged_property_value = [i for j in tmp2 for i in j]

                    found = False
                    for updated_time in updated_times:
                        start = -1
                        end = -1
                        # if the year is not correct just skip
                        if updated_time[0][0]!=arranged_property_value[0]:
                            continue
                        if found:
                            break
                        else:
                            ground_time.append(arranged_property_value[0])
                            if updated_time[0][1] == arranged_property_value[1]:
                                ground_time.append(arranged_property_value[1])
                            else:
                                ground_time.append(ground_base[1])
                            if updated_time[0][2] == arranged_property_value[2]:
                                ground_time.append(arranged_property_value[2])
                            else:
                                ground_time.append(ground_base[2])
                            if updated_time[0][3] == arranged_property_value[3]:
                                ground_time.append(arranged_property_value[3])
                            else:
                                ground_time.append(ground_base[3])
                            if updated_time[0][4] == arranged_property_value[4]:
                                ground_time.append(arranged_property_value[4])
                            else:
                                ground_time.append(ground_base[4])
                            if updated_time[1] not in sample_context:
                                ground_time = []
                                continue
                            start_char = sample_context.index(updated_time[1])
                            end_char = start_char+len(updated_time[1])
                            sc = 0
                            for t in doc:
                                if t.text in updated_time[1]:
                                    sc = sample_context.index(t.text, sc)
                                    if sc<=start_char and sc+len(t.text)>=start_char:
                                        start = t.i
                                    if sc<=end_char and sc+len(t.text)>=end_char:
                                        end = t.i
                                    if start>=0 and end>=0:
                                        break
                            if any(ground_time) and start>=0 and end>=0:
                                found = True
                                property = groundtruth_values[property_value]
                                if property not in new_groundtruth:
                                    new_groundtruth[property] = []
                                a,b,c,d,e = ground_time[0],ground_time[1],ground_time[2],ground_time[3],ground_time[4]
                                a,b,c,d,e = int(a),int(b), int(c[:2]), int(d), int(e[:-1])
                                ground_time = datetime.datetime(a,b,c,d,e)
                                if ground_time not in new_groundtruth[property]:
                                    new_groundtruth[property].append([start, end, start_char, end_char, property, updated_time[1]]) # it used to be ground_time.isoformat()
                                ground_time = []
                                
            for ent in doc._.linkedEntities:
                qid = "Q"+str(ent.identifier)
                try:
                    dbp_id = wikidata_ids_to_dbpedia_ids[qid]
                except:
                    continue
                if dbp_id in groundtruth_values:
                    property = groundtruth_values[dbp_id]
                    if property not in new_groundtruth:
                        new_groundtruth[property] = []
                    new_groundtruth[property].append([ent.span.start, ent.span.end-1, ent.span.start_char, ent.span.end_char, property, ent.span.text, dbp_id])

            # once we have gone through entities, we want to look for literals: quantities and time expressions. 
            count_sum = 0
            
            # skip if there are no integer values in property values
            start = -1
            if [val for val in list(groundtruth_values.keys()) if val.isdigit()]:
                # check each digit in the context if it fits any of the property values. Also add up numbers in case the property value is the sum of integers
                for i in doc:
                    txt = i.text
                    formatting_i = txt.replace(",","")
                    formatting_i = formatting_i.replace(".","")
                    if txt.isdigit():
                        if start == -1:
                            start = i.i
                        end = i.i
                        start_char = sample_context.index(txt)
                        end_char = sample_context.index(txt)+len(txt)
                        try:
                            count_sum+=int(txt)
                        except:
                            print(txt)
                    elif formatting_i.isdigit():
                        count_sum+=int(formatting_i)
                        if start == -1:
                            start = i.i
                        end = i.i
                        start_char = sample_context.index(txt)
                        end_char = sample_context.index(txt)+len(txt)
                        #txt=formatting_i
                    if  formatting_i in groundtruth_values:
                        property = groundtruth_values[formatting_i]
                        if property not in new_groundtruth:
                            new_groundtruth[property] = []
                        new_groundtruth[property].append([i.i, i.i, sample_context.index(txt), sample_context.index(txt)+len(txt), property, txt])
                if str(count_sum) in groundtruth_values:
                    if property not in new_groundtruth:
                        new_groundtruth[property] = []
                    new_groundtruth[property].append([start, end, start_char, end_char, property, doc.text[start_char:end_char]]) # count_sum
                

            if event_wd_id not in new_dataset:
                new_dataset[event_wd_id] = {}
            if subevent not in new_dataset[event_wd_id]:
                new_dataset[event_wd_id][subevent] = []
            if new_groundtruth:
                new_sample = {"sample_event_link":sample_subevent, "untokenized_context":sample_context,"tokenized_context":[t.text for t in doc], "sample_groundtruth": new_groundtruth, "ground_event_types": subevent_types}
                new_dataset[event_wd_id][subevent].append(new_sample)
    if num%100==0:
        with open("the_data/dbp_data"+str(num)+".json","w") as f:
            json.dump(new_dataset,f)
        new_dataset = {}

with open("the_data/dbp_data"+str(num)+".json","w") as f:
    json.dump(new_dataset,f)

dataset = {}

for subdir, dirs, files in os.walk("the_data"):
    for file in files:
        filepath = subdir + os.sep + file
        print(filepath)
        with open("the_data/"+file, "r") as f:
            dataset.update(json.load(f))

with open("the_data/dbp__full_data.json","w") as f:
    json.dump(dataset, f)


event_class_counter = Counter()


with open("the_data/dbp__full_data.json","r") as f:
    dataset = json.load(dataset)

tmp_dataset = copy.deepcopy(dataset)
for key in tmp_dataset:
    for subkey in tmp_dataset[key]:
        for j, sample in enumerate(tmp_dataset[key][subkey]):
            for i, prop in enumerate(sample["sample_groundtruth"].keys()):
                new_prop_label = prop.replace("http://dbpedia.org/ontology/","").replace(">","").replace("<","")
                dataset[key][subkey][j]["sample_groundtruth"][new_prop_label] = dataset[key][subkey][j]["sample_groundtruth"][prop]
                del dataset[key][subkey][j]["sample_groundtruth"][prop]
            for i, event_type in enumerate(sample["ground_event_types"]):
                    dataset[key][subkey][j]["ground_event_types"][i] = event_type.replace("<http://dbpedia.org/ontology/","").replace(">","")
            for i, event_type in enumerate(sample["sample_event_link"]["types"]):
                    dataset[key][subkey][j]["sample_event_link"]["types"][i] = event_type.replace("<http://dbpedia.org/ontology/","").replace(">","")

country_specific_event_classes = ['Q7864918', 'Q17496410', 'Q1128324', 'Q22333900', 'Q319496', 'Q849095', 'Q1141795',\
 'Q24333627', 'Q9102', 'Q24397514', 'Q56185179', 'Q17317594', 'Q22162827', 'Q123577', 'Q22284407', 'Q15283424', 'Q890055', 'Q25080094', 'Q47566',\
  'Q107540719', 'Q7870', 'Q248952', 'Q22696407', 'Q7961', 'Q9208', 'Q60874', 'Q8036']
bad_event_classes = ['Q1944346','Q2425489', 'Q2705481', 'Q148578', 'Q854845', 'Q15893266', 'Q928667', 'Q763288', 'Q1378139', 'Q26540', 'Q149918', 'Q7217761', 'Q27949697','Q27949697']
bad_properties = ['P4342', 'P9307', 'P9346', 'P935', 'P373', 'P2002']



clean_dataset = {}
property_counter = Counter()

for key in dataset:
    t = dataset[key]
    if not isinstance(dataset[key], dict):
        continue
    for subkey in dataset[key]:
        for sample in dataset[key][subkey]:
            if sample and "token_range" in sample["sample_event_link"]:
                if key not in clean_dataset:
                    clean_dataset[key] = {}
                if subkey not in clean_dataset[key]:
                    clean_dataset[key][subkey] = []
                clean_dataset[key][subkey].append(sample)

                property_counter.update(list(sample["sample_groundtruth"].keys()))
                event_class_counter.update(sample["ground_event_types"])

with open("dbpe__clean_data.json","w") as f:
    json.dump(clean_dataset, f)

with open("dbpe__clean_data.json","r") as f:
    clean_dataset = json.load(f)



def filter_clean_data(filter_size=50):
    filtered_dataset = {}
    for key in clean_dataset:
        for subkey in clean_dataset[key]:
            for sample in clean_dataset[key][subkey]:
                new_sample = copy.deepcopy(sample)
                for type in sample["sample_event_link"]["types"]:
                    if type in country_specific_event_classes + bad_event_classes:
                        new_sample["sample_event_link"]["types"].remove(type)
                if "type" in sample["sample_groundtruth"]:
                    del new_sample["sample_groundtruth"]["type"]
                for event_type in sample["ground_event_types"]:
                    if event_class_counter[event_type]<filter_size or event_type in country_specific_event_classes + bad_event_classes:
                        new_sample["ground_event_types"].remove(event_type)

                if not new_sample["ground_event_types"]:
                    continue

                for property in sample["sample_groundtruth"]:
                    if property_counter[property] < filter_size:
                        del new_sample["sample_groundtruth"][property]
                    elif property in bad_properties:
                        del new_sample["sample_groundtruth"][property]

                if not new_sample["sample_groundtruth"]:
                    continue

                if key not in filtered_dataset:
                    filtered_dataset[key] = {}
                if subkey not in filtered_dataset[key]:
                    filtered_dataset[key][subkey] = []
                filtered_dataset[key][subkey].append(new_sample)

    with open("dbpe__filtered_data"+str(filter_size)+".json","w") as f:
        json.dump(filtered_dataset, f)

def second_pass(data, class_filter, prop_filter, label ="2nd"):
    with open(data,"r") as f:
        d = json.load(f)
    c1 = Counter()
    c2 = Counter()
    filtered_dataset = {}


    for key in d:
        for subkey in d[key]:
            for sample in d[key][subkey]:
                s = set()
                s.update(sample["ground_event_types"])
                s.update(sample["sample_event_link"]["types"])
                s = list(s)
                c1.update(s)
                c2.update(list(sample["sample_groundtruth"].keys()))
    
    for key in d:
        for subkey in d[key]:
            for sample in d[key][subkey]:
                new_sample = copy.deepcopy(sample)
                for event_type in sample["ground_event_types"]:
                    if c1[event_type]<class_filter:
                        new_sample["ground_event_types"].remove(event_type)

                if not new_sample["ground_event_types"]:
                    continue

                for property in sample["sample_groundtruth"]:
                    if c2[property] < prop_filter:
                        del new_sample["sample_groundtruth"][property]

                if not new_sample["sample_groundtruth"]:
                    continue

                if key not in filtered_dataset:
                    filtered_dataset[key] = {}
                if subkey not in filtered_dataset[key]:
                    filtered_dataset[key][subkey] = []
                filtered_dataset[key][subkey].append(new_sample)

    with open("dbpe__"+label+"_pass_filtered_data.json","w") as f:
        json.dump(filtered_dataset, f)

    with open("dbpe_class_counter.json", "w") as f:
        json.dump(c1,f)
    
    with open("dbpe_property_counter.json", "w") as f:
        json.dump(c2,f)

filter_clean_data(100)

second_pass("dbpe__filtered_data100.json", class_filter = 100, prop_filter = 50, label="3rd")


with open("dbpe__class_counter.json", "r") as f:
    C = json.load(f)


with open("dbpe__3rd_pass_filtered_data.json","r") as f:
    dataset = json.load(f)

n_events = 0
n_context = 0
n_links = 0
for key in dataset:
    context = ""
    for subkey in dataset[key]:
        n_events+=1
        for link in dataset[key][subkey]:
            if link["sample_groundtruth"]:
                n_links+=1
            if link["untokenized_context"] != context:
                context = link["untokenized_context"]
                n_context +=1


#with open("dbp_label_class.json", "r") as f:
    #class_labels = json.load(f)

#with open("dbp_label_prop.json", "r") as f:
    #prop_labels = json.load(f)

all_samples = []
new_cl_counter = Counter()
new_prop_counter = Counter()

all_event_classes = set()
for event in dataset:
    for subevent in dataset[event]:
        for sample in dataset[event][subevent]:
            types = list(set(sample["sample_event_link"]["types"]+sample["ground_event_types"]))
            for c in types:
                all_event_classes.update({c})

with open("dbpe_all_event_classes.txt","w") as f:
    for line in list(all_event_classes):
        f.write(f"{line}\n")

#all_event_classes = set()


#choose minority class
def select_minority_classes(dataset):
    all_samples = []
    for event in dataset:
        for subevent in dataset[event]:
            for sample in dataset[event][subevent]:
                types = list(set(sample["sample_event_link"]["types"]+sample["ground_event_types"]))
                min = 1000*1000
                #all_event_classes.update(types)
                for c in types:
                    if C[c] < min:
                        min = C[c]
                        min_cl = c
                del sample["sample_event_link"]["types"]
                sample["sample_event_link"]["event_class"] = min_cl
                new_cl_counter.update({min_cl:1})
                new_prop_counter.update(list(sample["sample_groundtruth"].keys()))
                all_samples.append(sample)
    return all_samples

all_samples = select_minority_classes(dataset)

n = len(all_samples)
def third_pass(all_samples, class_counter, property_counter, class_filter = 100, prop_filter=50):
    new_samples = []
    for sample in all_samples:
        new_sample = copy.deepcopy(sample)
        sample_class = sample["sample_event_link"]["event_class"] 
        if class_counter[sample_class]<class_filter:
            continue
        else:
            for property in sample["sample_groundtruth"]:
                if property_counter[property] < prop_filter:
                    del new_sample["sample_groundtruth"][property]

            if not new_sample["sample_groundtruth"]:
                continue
            
            new_samples.append(new_sample)

    return new_samples

all_samples = third_pass(all_samples, new_cl_counter, new_prop_counter)

event_classes = set()
all_event_classes = set()
event_properties = set()
event_class_properties = {}
for sample in all_samples:
    sample_cl = sample["sample_event_link"]["event_class"]
    event_classes.update({sample_cl})
    properties = list(sample["sample_groundtruth"].keys())
    event_properties.update(properties)
    if sample_cl not in event_class_properties:
        event_class_properties[sample_cl] = set()
    event_class_properties[sample_cl].update(properties) 

event_classes = list(event_classes)
event_properties = list(event_properties)
all_event_classes = list(all_event_classes)
for cl in event_class_properties:
    event_class_properties[cl] = list(event_class_properties[cl])


with open("../data/training/t2e/dbpe_unlabelled_event.schema","w") as f:
    f.write(f"{list(event_classes)}\n")
    f.write(f"{list(event_properties)}\n")
    f.write(f"{event_class_properties}\n")

event_classes = [cl for cl in event_classes]
event_properties = [prop for prop in event_properties]
for cl in event_class_properties:
    event_class_properties[cl] = [prop for prop in event_class_properties[cl]]
event_class_properties = {cl:v for cl,v in event_class_properties.items()}

with open("../data/training/t2e/dbpe_labelled_event.schema","w") as f:
    f.write(f"{list(event_classes)}\n")
    f.write(f"{list(event_properties)}\n")
    f.write(f"{event_class_properties}\n")

#

tmp = copy.deepcopy(all_samples)
tmp2 = {}
j = 0
for i, sample in enumerate(tmp):
    if "untokenized_context_with_links" in sample:
        print(j)
        j+=1
        context = sample["untokenized_context_with_links"]
    else:
        context = sample["untokenized_context"]
        tmp[i]["untokenized_context_with_links"] = context
    #context = sample["untokenized_context"]
    if context not in tmp2:
        tmp2[context] = [sample]
    else:
        if tmp2[context] == [sample]:
            continue
        else:
            event_classes = []
            for samp in tmp2[context]:
                event_classes.append(samp["sample_event_link"]["event_class"])
            if sample["sample_event_link"]["event_class"] in event_classes:
                continue
            else:
                tmp2[context].append(sample)

all_samples = []
for i, context in enumerate(tmp2):
    new_sample = {}
    tmp_sample = tmp2[context]
    if len(tmp_sample)==1:
        cl = tmp_sample[0]["sample_event_link"]["event_class"]
        props = {prop:v for prop, v in tmp_sample[0]["sample_groundtruth"].items()}
        new_sample["untokenized_context"] = tmp_sample[0]["untokenized_context_with_links"]
        #new_sample["untokenized_context"] = tmp_sample[0]["untokenized_context"]
        doc = nlp(new_sample["untokenized_context"])
        new_sample["tokenized_context"]= [t.text for t in doc]
        #new_sample["tokenized_context"]= tmp_sample[0]["tokenized_context"]
        new_sample["groundtruth_events"] = [{cl : {"sample_event_link":tmp_sample[0]["sample_event_link"], "sample_groundtruth":props}}]
        all_samples.append(new_sample)
    else:
        for i, sample in enumerate(tmp2[context]):
            cl = sample["sample_event_link"]["event_class"]
            props = {prop:v for prop, v in sample["sample_groundtruth"].items()}
            new_sample["untokenized_context"] = sample["untokenized_context_with_links"]
            #new_sample["untokenized_context"] = sample["untokenized_context"]
            doc = nlp(new_sample["untokenized_context"])
            new_sample["tokenized_context"]= [t.text for t in doc]
            #new_sample["tokenized_context"]= sample["tokenized_context"]
            if "groundtruth_events" not in new_sample:
                new_sample["groundtruth_events"] = [{cl : {"sample_event_link":tmp_sample[0]["sample_event_link"], "sample_groundtruth":props}}]
            else:
                new_sample["groundtruth_events"].append({cl : {"sample_event_link":tmp_sample[0]["sample_event_link"], "sample_groundtruth":props}})
            all_samples.append(new_sample)



def create_baseline_training_data(all_samples, path = "../data/training"):
    t2e_training_samples = []
    relation_extraction_samples = []
    dygiepp_training_samples = []
    for sample_num, sample in enumerate(all_samples):
        if sample_num % 100 == 0:
            print("%d / %d" %(sample_num, len(all_samples)))
        sentence_tokenized_context = []
        sentence_tokenized_events = []
        re_tokenized_events = []
        doc = nlp(sample["untokenized_context"])
        sync = dict()
        for sent in doc.sents:
            sentence_tokenized_context.append([])
            sentence_tokenized_events.append([])
            re_tokenized_events.append([])
            for t in sent:
                sentence_tokenized_context[-1].append(t.text)
                for event in sample["groundtruth_events"]:
                    for event_class in event:
                        if t.i == event[event_class]["sample_event_link"]["token_range"][0]:
                            sentence_tokenized_events[-1].append([[t.i, event_class]])#sample["groundtruth_events"][event_class]["sample_event_link"]["event_class"]]])
                            re_tokenized_events[-1].append([[t.i,event[event_class]["sample_event_link"]["token_range"][1],event_class]])#sample["groundtruth_events"][event_class]["sample_event_link"]["event_class"]]])
                            for t2 in sent:
                                for property in event[event_class]["sample_groundtruth"]:
                                    if isinstance(event[event_class]["sample_groundtruth"][property][0], list):
                                        if event[event_class]["sample_groundtruth"][property][0][0] == t2.i:
                                            if event_class not in sync:
                                                sync[event_class] = []
                                            sync[event_class].append(property)
                                            sentence_tokenized_events[-1][-1].append([event[event_class]["sample_groundtruth"][property][0][0],event[event_class]["sample_groundtruth"][property][0][1],property])
                                            re_tokenized_events[-1][-1].append([event[event_class]["sample_groundtruth"][property][0][0],event[event_class]["sample_groundtruth"][property][0][1],property])
                                    else:
                                        if event[event_class]["sample_groundtruth"][property][0] == t2.i:
                                            if event_class not in sync:
                                                sync[event_class] = []
                                            sync[event_class].append(property)
                                            sentence_tokenized_events[-1][-1].append([event[event_class]["sample_groundtruth"][property][0],event[event_class]["sample_groundtruth"][property][1],property])
                                            re_tokenized_events[-1][-1].append([event[event_class]["sample_groundtruth"][property][0],event[event_class]["sample_groundtruth"][property][1],property])
        X = "<extra_id_0>"
        for event in sample["groundtruth_events"]:
            for event_class in event :
                #event_class = sample["groundtruth_events"]["sample_event_link"]["event_class"]
                event_trigger = event[event_class]["sample_event_link"]["anchor_text"]
                X += "<extra_id_0>"+event_class+" "+ event_trigger
                groundtruth = event[event_class]["sample_groundtruth"]
                for property in groundtruth: #
                        if event_class not in sync or property not in sync[event_class]:
                            continue
                        X += "<extra_id_0>"
                        if isinstance(groundtruth[property][0], list):
                            X += property+" "+ str(groundtruth[property][0][5])
                        else:
                            X += property+" "+ groundtruth[property][5]
                        X += "<extra_id_1>" #
                X += "<extra_id_1>"
        X += "<extra_id_1>"
        t2e_training_sample = {}
        t2e_training_sample["text"] = sample["untokenized_context"]
        t2e_training_sample["event"] = X
        t2e_training_samples.append(t2e_training_sample)

        dygiepp_training_sample = {}
        dygiepp_training_sample["doc_key"] = sample_num
        dygiepp_training_sample["dataset"] = "event extraction"
        dygiepp_training_sample["sentences"] = sentence_tokenized_context
        dygiepp_training_sample["events"] =  sentence_tokenized_events
        dygiepp_training_samples.append(dygiepp_training_sample)

        re_sample = {}
        re_sample["doc_key"] = sample_num
        re_sample["dataset"] = "event extraction"
        re_sample["sentences"] = sentence_tokenized_context
        re_sample["events"] =  re_tokenized_events
        relation_extraction_samples.append(re_sample)

    indices = list(range(len(dygiepp_training_samples)))
    train_idx, test_idx = train_test_split(indices, test_size=0.3)
    dev_idx, test_idx = train_test_split(test_idx, test_size=0.5)
    #train, test = train_test_split(new_dygiepp_samples, test_size=0.3)
    #dev, test = train_test_split(test, test_size=0.5)
    train=[dygiepp_training_samples[i] for i in train_idx]
    test=[dygiepp_training_samples[i] for i in test_idx]
    dev=[dygiepp_training_samples[i] for i in dev_idx]

    with jsonlines.open(path+"/dygiepp/dbpe_eq_train.json","w") as f:
        f.write_all(train)

    with jsonlines.open(path+"/dygiepp/dbpe_eq_test.json","w") as f:
        f.write_all(test)

    with jsonlines.open(path+"/dygiepp/dbpe_eq_dev.json","w") as f:
        f.write_all(dev)

    train=[t2e_training_samples[i] for i in train_idx]
    test=[t2e_training_samples[i] for i in test_idx]
    dev=[t2e_training_samples[i] for i in dev_idx]

    with jsonlines.open(path+"/t2e/dbpe_eq_train.json","w") as f:
        f.write_all(train)

    with jsonlines.open(path+"/t2e/dbpe_eq_test.json","w") as f:
        f.write_all(test)

    with jsonlines.open(path+"/t2e/dbpe_eq_val.json","w") as f:
        f.write_all(dev)

    train=[relation_extraction_samples[i] for i in train_idx]
    test=[relation_extraction_samples[i] for i in test_idx]
    dev=[relation_extraction_samples[i] for i in dev_idx]

    with jsonlines.open(path+"/re/dbpe_eq_train.json","w") as f:
        f.write_all(train)

    with jsonlines.open(path+"/re/dbpe_eq_test.json","w") as f:
        f.write_all(test)

    with jsonlines.open(path+"/re/dbpe_eq_dev.json","w") as f:
        f.write_all(dev)

create_baseline_training_data(all_samples, path = "../data/training")

def create_sparse_full_data(tokenized_path, untokenized_path, output_path):
    data = []
    tokenized_samples = []
    untokenized_samples = []
    with open("../data/training/t2e/dbpe_unlabelled_event.schema","r") as f:
        for o, line in enumerate(f):
            if o ==2:
                event2role =  ast.literal_eval(line)

    with open(tokenized_path, 'r') as json_file1, open(untokenized_path, "r") as json_file2:
        json_list1 = list(json_file1)
        json_list2 = list(json_file2)

    for json_str1, json_str2 in zip(json_list1, json_list2):
        tokenized_samples.append(json.loads(json_str1))
        untokenized_samples.append(json.loads(json_str2))

    k = 0

    for t, (t_sample, u_sample) in enumerate(zip(tokenized_samples, untokenized_samples)):
        if t % 1000==0:
            print("%d out of %d completed"%(t, len(tokenized_samples)))
        data.append({"title":str(t), "paragraphs":[]})
        u_sentence = u_sample["text"]
        for i, t_sentence in enumerate(t_sample["sentences"]):
            #data[-1]["paragraphs"] = []
            #data[-1]["paragraphs"]["context"] = detokenized_sentence
            data[-1]["paragraphs"].append({"qas":[], "context":u_sentence})
            tmp = {}
            for event in t_sample["events"][i]:
                token_start = event[0][0] 
                token_end = event[0][1]
                event_type = event[0][2] 
                trigger = " ".join(t_sentence[token_start:token_end+1]) 
                if event_type not in tmp:
                    tmp[event_type] = {"triggers":[], "arguments":{}}
                tmp[event_type]["triggers"].append(trigger)
                    #tmp[class_label_uri[event_type]] = {"triggers":[], "arguments":{}}
                #tmp[class_label_uri[event_type]]["triggers"].append(trigger)

                for event_part in event[1:]: 
                    token_start = event_part[0]
                    token_end = event_part[1]
                    role_type = event_part[2]
                    # assumption of one event per event type in a sample
                    #if prop_label_uri[role_type] not in tmp[class_label_uri[event_type]]["arguments"]:
                        #tmp[class_label_uri[event_etype]]["arguments"][prop_label_uri[role_type]] = []
                    if role_type not in tmp[event_type]["arguments"]:
                        tmp[event_type]["arguments"][role_type] = []
                    start_index, end_index = get_correct_indices(u_sentence, [token for sent in t_sample["sentences"] for token in sent], token_start, token_end)
                    
                    arg_string = u_sentence[start_index:end_index]
                    #tmp[class_label_uri[event_type]]["arguments"][prop_label_uri[role_type]].append(arg_string)
                    tmp[event_type]["arguments"][role_type].append(arg_string)
                #event_type = class_label_uri[event_type]
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
    try:
        start_indices = [m.start(0) for m in re.finditer(re.escape(tokenized_sentence[token_start]), detokenized_sentence)]
        end_indices = [m.end(0) for m in re.finditer(re.escape(tokenized_sentence[token_end]), detokenized_sentence)]
    except:
        print(1)
        return 0,0

    for end_index in end_indices:
        for start_index in start_indices:
            if start_index > end_index:
                continue
            if end_index-start_index < minimum and end_index-start_index>=len(tokenized_sentence[token_start]):
                minimum = end_index-start_index
                best_start_index = start_index
                best_end_index = end_index
    try:
        return best_start_index, best_end_index
    except:
        print(1)
        return 0,0


def create_training_data(path):

    train = create_sparse_full_data(path+"re/dbpe_eq_train.json", path+"t2e/dbpe_eq_train.json", path+"re/dbpe_re_train.json")
    train_dataset = {"version":"v2.0", "data":train}
    with open(path+'re/dbpe2_sparse_re_train.json', 'w') as f:
        json.dump(train_dataset, f)
    train_dataset = {}
    train = {}

    test = create_sparse_full_data(path+"re/dbpe_eq_test.json", path+"t2e/dbpe_eq_test.json", path+"re/dbpe_re_test.json")
    test_dataset = {"version":"v2.0", "data":test}
    with open(path+'re/dbpe2_sparse_re_test.json', 'w') as f:
        json.dump(test_dataset, f)
    test_dataset={}
    test = {}

    dev = create_sparse_full_data(path+"re/dbpe_eq_dev.json", path+"t2e/dbpe_eq_val.json", path+"re/dbpe_re_dev.json")
    dev_dataset = {"version":"v2.0", "data":dev}
    with open(path+'re/dbpe2_sparse_re_dev.json', 'w') as f:
        json.dump(dev_dataset, f)
    dev_dataset = {}
    dev = {}

create_training_data("../data/training/")



