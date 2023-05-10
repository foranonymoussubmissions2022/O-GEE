# SERT

This is the code for SERT (Semantic Event Representations from Transformers), based on the paper: "Extraction of Fine-grained Semantic Event Representations from Transformers"




## Steps
To recreate the dataset creation, training and evaluation as described in the paper follow the steps as described below. 
### Data creation and preprocessing
* Follow instructions in the [EventTextWikipediaDumper](https://github.com/foranonymoussubmissions2022/EventTextWikipediaDumper) to run [MWDumper](https://www.mediawiki.org/wiki/Manual:MWDumper) and get Wikipedia articles of events in .ndjson file format. Place the resulting files into data\raw in the project folder.
* Run ```data/data_download.sh``` to prepare Wikidata and Dbpedia dumps and redirects.
* Run ```data/create_shelves.py``` to assure quick access to the dumps.
* Run ```processing/create_training_data.py```.This step will create training data for the baselines and the multilabel classification step of our approach.
* To train and evaluate specific baselines follow instructions in their respective subfolders in the baselines folder. 
### Training
* To train the multilabel classification model first format the multilabel training, validation and testing data by running  ```mlc/format_mlc_data.py```. Then follow up by running ```mlc/mlc.py```. This will generate the saved model and the output of the model inside the ```/evaluation/minority_classes/mlc_output/```.
* To train the Relation Extraction Model first generate the appropriate format for the data by running ```python mlc/REM/convert_data.py```, then run ```mlc/REM/train.sh```.
### Evaluation
* Finally to evaluate the perfomance of our model and the baselines,and generate the output of our model run ```evaluation/evaluation.py```.

## Working with a new dataset
To test our approach on other events ontologies or datasets follow the steps as described below
* Format the dataset as shown ```test/full_dataset.jsonl```. Each line represents a json file containing text, it's sentences, and events in their respective sentences. Where an event with `N` arguments will be written as a list of the form `[[trigger_tok, event_type], [start_tok_arg1, end_tok_arg1, arg1_type], [start_tok_arg2, end_tok_arg2, arg2_type], ..., [start_tok_argN, end_tok_argN, argN_type]]`. For instance,
  ```json
  [
    [],
    [],
    [
      [
        [15, "Event Class"],
        [13, 13, "Event argument1"]
        [17, 17, "Event argument2"]
      ]
    ]
  ]
  ```

* run  ```test/formatting.py``` to create necessary event classification and relation extraction training splits.
* run ```test/convert_data.py```for further processing of relation extraction data.
* run  ```test/mlc.py``` to train the event classification model and produce an output on the testing data.
* run  ```test/train.sh``` to train the relation extraction model.
* run  ```test/evaluation``` to get the final evaluation scores of the approach on the new dataset.
