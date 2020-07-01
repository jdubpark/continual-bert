
## Continual BERT: Continual Learning for Adaptive Extractive Summarization of COVID-19 Literature

### Introduction

The medical community has been relentlessly working on releasing new research and information on the novel coronavirus, SARS-CoV-2 (COVID-19). Many of those research are promptly published without peer-review as it is critical for the community to share the latest findings on the deadly virus. Furthermore, there aren't just enough time and resources for researchers to peer-review each and every papers. Hence, some of the papers might report abnormal or controversial findings without strong evidence that could have been mitigated with peer-review.

### Our Approach

Here, we present an approach leveraging state-of-the-art NLP models to classify sections and extract critical data from research papers. We introduce some modifications to the existing SOTA models to bolster their sequential learning while preventing catastrophic forgetting and to enhance their adaptability to the flow of new information.

We note that it is critical for NLP models to have a rigid sequential learning system for real-world application where data flows in time-sensitive manners. Furthermore, it is also essential for NLP models to rapidly adapt to the new information while still retaining its old learnings.

### Starting with the code

#### Preprocess data
Requirements
- Stanford CoreNLP

```bash
# define paths
export CLASSPATH=/path/to/stanford-corenlp/stanford-corenlp-v.jar # Stanford CoreNLP jar
export RAW_PATH=data/raw/
export TOKENIZED_PATH=data/tokenized/
export JSON_PATH=data/to_json/
export BERT_DATA_PATH=data/to_bert/
# split and tokenize raw documents
python preprocess.py --tokenize --raw_path RAW_PATH --save_path TOKENIZED_PATH
# format to simpler json files
python preprocess.py --to-lines --raw_path RAW_PATH --save_path JSON_PATH --map_path MAP_PATH --lower
# bert tokenizer with [SEP] and [CLS] for every sentence
python preprocess.py --to-bert --raw_path JSON_PATH --save_path BERT_DATA_PATH --oracle_mode greedy/combination
```
Oracle Mode: combination is more accurate but at slower speed
