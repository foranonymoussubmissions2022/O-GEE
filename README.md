# O-GEE

This is the code for O-GEE (Ontology-Guided Event Extraction), based on the paper: "Transformer-based Ontology-Guided Event Extraction"

![alt text](https://github.com/foranonymoussubmissions2022/O-GEE/blob/main/figs/example.png)


## Steps
To recreate the dataset creation, training and evaluation as described in the paper follow the steps as described below. 
### Starting from scratch
* To recreate the dataset from the beginning start [here](#Data-creation-and-preprocessing).
#### Data creation and preprocessing
* Follow instructions in the [EventTextWikipediaDumper](https://github.com/foranonymoussubmissions2022/EventTextWikipediaDumper) to run [MWDumper](https://www.mediawiki.org/wiki/Manual:MWDumper) and get Wikipedia articles of events in .ndjson file format. Place the resulting files into data\raw in the project folder.
* Run ```data/data_download.sh``` to prepare Wikidata and Dbpedia dumps and redirects.
* Run ```data/create_shelves.py``` to assure quick access to the dumps.
* Run ```processing/create_training_data.py```.This step will create training data for the baselines and the multilabel classification step of our approach.
* To train and evaluate specific baselines follow instructions in their respective subfolders in the baselines folder. 
### Using our complete data
#### Training
* To train the multilabel classification model first format the multilabel training, validation and testing data by running  ```mlc/format_mlc_data.py```. Then follow up by running ```mlc/mlc.py```. This will generate the saved model and the output of the model inside the ```/evaluation/minority_classes/mlc_output/```.
* To train the Relation Extraction Model first generate the appropriate format for the data by running ```python mlc/REM/convert_data.py```, then run ```mlc/REM/train.sh```.
#### Evaluation
* Finally to evaluate the perfomance of our model and the baselines,and generate the output of our model run ```evaluation/evaluation.py```.




| **Approach** | **Event Classification** | | | | | | **Relation Extraction** | | | | | |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **Macro** | | | **Micro** | | | **Macro** | | | **Micro** | | |
| | **P** | **R** | **$F_1$** | **P** | **R** | **$F_1$** | **P** | **R** | **$F_1$** | **P** | **R** | **$F_1$** |
| | | | | | | | | | | | | |
| **DBpedia** |
| ASP | - | - | - | - | - | - | 0.46 | 0.42 | 0.42 | 0.57 | 0.41 | 0.48 |
| T2E | **0.80** | **0.79** | **0.80** | **0.94** | **0.94** | **0.94** | 0.70 | 0.66 | 0.67 | 0.79 | 0.78 | 0.79  |
| DyGIE++ | 0.70 | 0.73 | 0.71 | 0.90 | 0.89 | 0.89  | 0.63 | 0.58 | 0.59 | 0.78 | 0.75 | 0.76 |
| O-GEE | 0.77 | 0.74 | 0.74 | 0.92 | 0.91 | 0.92 | **0.72** | **0.68** | **0.70** | **0.82** | **0.81** | **0.81**   |
| | | | | | | | | | | | | |
|**Wikidata**|  
| ASP | - | - | - | - | - | - | 0.50 | 0.39 | 0.41 | 0.71 | 0.40 | 0.51 |
| T2E  | 0.84 | 0.86 | 0.85 | 0.87 | 0.89 | 0.88 | 0.77 | 0.73 | 0.74 | 0.87 | **0.89** | 0.88  |
| DyGIE++ | 0.45 | 0.39 | 0.41 | 0.80 | 0.63 | 0.71 | 0.41 | 0.40 | 0.39 | 0.83 | 0.62 | 0.71 |
| O-GEE | **0.86** | **0.87** | **0.86** | **0.89** | **0.90** | **0.89** | **0.85** | **0.79** | **0.81** | **0.90** | **0.89** | **0.90** |

## Working with a new dataset
To test our approach on other events ontologies or datasets follow the steps as described below.

![alt text](https://github.com/foranonymoussubmissions2022/O-GEE/blob/main/figs/pipeline.png)

* Format the dataset as follows. Each line represents a json file containing text, it's sentences, and events in their respective sentences:


```
{"doc_key": 36206, 
"dataset": "event extraction", 
"sentences": [["The", "2020", "United", "States", "presidential", "election", "in", "Missouri", "was", "held", "on", "Tuesday", ",", "November", "3", ",", "2020", ",", "as", "part", "of", "the", "2020", "United", "States", "presidential", "election", "in", "which", "all", "50", "states", "plus", "the", "District", "of", "Columbia", "participated", "."]],
"events": [[[[1, 5, "Election"], [1, 1, "startDate"]]]]}
```

The "events" field should contain a list containing a sublist for each sentence in the "sentences" field. Each of these sublists contains another sublist per event.
An event with N arguments will be written as a list of the form:

```
[
  [trigger_token_start_index, trigger_token_end_index, event_type], 
    [argument_token_start1_index, argument_token_end_index1, arg1_type], 
    [argument_token_start2_index, argument_token_end_index2, arg2_type], 
    ...,  
    [argument_token_startN_index, argument_token_end_indexN, argN_type]
]
```


* run  ```test/formatting.py``` to create necessary event classification and relation extraction training splits.
* run ```test/convert_data.py```for further processing of relation extraction data.
* run  ```test/mlc.py``` to train the event classification model and produce an output on the testing data.
* run  ```test/train.sh``` to train the relation extraction model.
* run  ```test/evaluation``` to get the final evaluation scores of the approach on the new dataset.
