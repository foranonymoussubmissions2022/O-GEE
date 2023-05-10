# Datasets for extraction of semantic event representation

We introduce two new large-scale datasets for the extraction of semantic event representations based on DBpedia and Wikidata. The two datasets contain consist of $43,291$ and $72,649$ samples ,respectively.

We derive the datasets from all Wikipedia articles of events. Given this set of Wikipedia articles, for each Wikipedia link that occurs in the text of an article, it is mapped to a corresponding Wikidata and DBpedia event entry.   

Both datasets follow the same formatting, similar to that of DyGie++. They are .jsonl files where each line contains a json like the one below:
```
{"doc_key": 36206, 
"dataset": "event extraction", 
"sentences": [["The", "2020", "United", "States", "presidential", "election", "in", "Missouri", "was", "held", "on", "Tuesday", ",", "November", "3", ",", "2020", ",", "as", "part", "of", "the", "2020", "United", "States", "presidential", "election", "in", "which", "all", "50", "states", "plus", "the", "District", "of", "Columbia", "participated", "."]],
"events": [[[[1, 5, "Election"], [1, 1, "startDate"]]]]}
```
The "events" field is a list containing a sublist for each sentence in the "sentences" field. Each of these sublists contains another sublist per event.
An event with N arguments will be written as a list of the form:
  ```
[[trigger_token_start_index, trigger_token_end_index, event_type], 
[argument_token_start1_index, argument_token_end_index1, arg1_type], 
[argument_token_start2_index, argument_token_end_index2, arg2_type], 
..., [argument_token_startN_index, argument_token_end_indexN, argN_type]]. 
```


The   ```event_ontology_distribution.json``` file contains a dictionary describing the distribution of  event and property labels across the two datasets.

