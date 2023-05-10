from re import X
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from torch import cuda
from transformers import DistilBertTokenizer, DistilBertModel
import json
from torch.nn import Module
import networkx as nx
import csv
from torch import nn
print("start")

        
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
            self.tokenizer = tokenizer
            self.data = dataframe
            self.text = dataframe.text
            self.targets = self.data.list
            self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 88)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        return torch.mean(focal_loss)

def loss_fn(outputs, targets):
    return FocalLoss(gamma=2)(outputs, targets)


def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)
        sigm = torch.sigmoid(outputs)
        sigm = sigm.tolist()    
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)#/max(min(sigm))

        if _%100==0:
            print("Epoch: %d" %(epoch)+" , Loss: %f" %(loss))
        
        loss.backward()
        optimizer.step()


def validation(epoch, mode="dev"):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    if mode=="dev":
        loader = dev_loader
    else:
        loader = testing_loader
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

if __name__=="__main__":
    
    device = 'cuda' if cuda.is_available() else 'cpu'

    # Creating the dataset and dataloader for the neural network
    df = pd.read_csv('../data/training/mlc_data/wde_multilabel_train.csv')
    header = list(df.head())

    df['list'] = df[df.columns[2:]].values.tolist()
    new_df = df[['text', 'list']].copy()
    new_df.head()


    MAX_LEN = 256
    TRAIN_BATCH_SIZE = 12
    VALID_BATCH_SIZE = 6
    EPOCHS = 30
    LEARNING_RATE = 1e-05
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_size = 1

    train_dataset = new_df

    df2 = pd.read_csv('../data/training/mlc_data/wde_multilabel_test.csv')
    df2['list'] = df2[df2.columns[2:]].values.tolist()
    new_df = df2[['text', 'list']].copy()
    new_df.head()




    test_dataset = new_df


    df3 = pd.read_csv('../data/training/mlc_data/wde_multilabel_dev.csv')
    df3['list'] = df3[df3.columns[2:]].values.tolist()
    new_df = df3[['text', 'list']].copy()
    new_df.head()

    dev_dataset = new_df


    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("DEV Dataset: {}".format(new_df.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)
    dev_set = CustomDataset(dev_dataset, tokenizer, MAX_LEN)



    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0
                    }

    dev_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0
                    }
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    dev_loader = DataLoader(dev_set, **test_params)

    model = BERTClass()
    model= nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
    all_ground = 0
    plus = 0
    all_predicted = 0
    max_score = 0
    for epoch in range(EPOCHS):
        train(epoch)
        outputs, targets = validation(epoch)
        outputs = np.array(outputs)
        indices = np.argmax(outputs, axis = 1)
        #for j, i in enumerate(indices):
            #outputs[j][i] = 1.0
        outputs = outputs >= 0.5
        targets = np.array(targets)

        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        f1_score_weighted= metrics.f1_score(targets, outputs, average='weighted')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
        print(f"F1 Score (weighted) = {f1_score_weighted}")
        if f1_score_micro > max_score:
            max_score = f1_score_micro

            torch.save(model.state_dict(), "wd_model")
            print("Test results:")
            outputs, targets = validation(epoch, mode="test")
            outputs = np.array(outputs)
            indices = np.argmax(outputs, axis = 1)
            #for j, i in enumerate(indices):
                #outputs[j][i] = 1.0

            outputs = outputs >= 0.5
            targets = np.array(targets)
            classes = list(pd.read_csv("../data/training/mlc_data/wde_multilabel_test.csv").columns)[2:]
            tp = 0
            fp = 0
            fn = 0
            for sentence_target, sentence_output in zip(targets,outputs):
                ground_classes = [cl for cl, target in zip(classes, sentence_target) if target!=False] 
                predicted_classes = [cl for cl, target in zip(classes, sentence_output) if target!=False] 
                for cl in ground_classes:
                    if cl not in predicted_classes:
                        fn+=1
                    else:
                        tp+=1
                        predicted_classes.remove(cl)
                for cl in predicted_classes:
                    if cl not in ground_classes:
                        fp+=1
            p = tp/(tp+fp)
            r = tp/(tp+fn)
            f1 = 2*p*r/(p+r)
            

            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
            f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
            f1_score_weighted= metrics.f1_score(targets, outputs, average='weighted')
            print(f"Accuracy Score = {accuracy}")
            print(f"F1 Score (Micro) = {f1_score_micro}")
            print(f"F1 Score (Macro) = {f1_score_macro}")
            print(f"F1 Score (weighted) = {f1_score_weighted}")

            print("my f1:", f1)
            #best_wd2 with loss division
            with open("../evaluation/minority_classes/mlc_output/wde_with_focal_loss.csv", "w") as f, open('../data/training/mlc_data/wde_multilabel_test.csv',"r") as f2:
                writer = csv.DictWriter(f, fieldnames=header, delimiter = '\t',  quoting=csv.QUOTE_NONE, quotechar='')
                reader = csv.reader(f2)
                next(reader, None)
                for output, input, target, row  in zip(outputs, test_dataset["text"], targets, reader):
                    d = {}
                    for o, h in zip(output,header[2:]):
                        d[h] = o
                    d["id"] = row[0]
                    d["text"] = row[1]
                    writer.writerow(d)


