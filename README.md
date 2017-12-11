# CNNs vs RNNs for Sentence Classification
## Models (WIP)
General idea is the same:
1) Encode source sentence to get single a vector sentence representation
2) Project this representation onto desired classes

## Results
| Model/Dataset | KSAI | SST1 | SST2 |
| --- | --- | ---| --- | 
| boe-baseline | 0.823 | x | 0.82 |
| Elman-RNN-baseline | 0.785 | x | 0.82 |
| Elman-RNN-bi | 0.807 | x | 0.83 | 
| Elman-RNN-bi-avg | 0.815 | x | 0.84 | 
| LSTM-bi-avg | 0.780 | x | 0.86 |
| GRU-bi-avg-rand | 0.817 | x | 0.85 | 
| GRU-bi-avg-static | 0.757 | x | 0.83 |   
| GRU-bi-avg-nonstatic | 0.823 | x | 0.84 | 
| GRU-bi-avg-charseg | 0.824 | x | 0.85 |      
| CNN-rand (Kim, 2014) | 0.814 | x | 0.83 |   
| CNN-static (Kim, 2014) | 0.753 | x | 0.87 |    
| CNN-nonstatic (Kim, 2014) | 0.822 | x | 0.87 |
*KSAI stands for Kaggle Spooky Author Identification
*the metric is __accuracy__ <br>
*you can find __configs__ for all the models in __experiments_configs__ folder (names align)


"cnn_paper_dataset": true
    "train_data_path": "data_cnn_paper/stsa.binary.phrases.train",
    "validation_data_path": "data_cnn_paper/stsa.binary.dev",
    "test_data_path": "data_cnn_paper/stsa.binary.test",
    "evaluate_on_test": true,

Different approaches are different in a way we encode the sentence <br>
Models I experimented with: <br> 
- __boe-baseline__: just avarage over word embeddings 
- __Elman-RNN-baseline__: simple Elman RNN; starting point for future experiments; see "Hyperparameters and Training" section for details
- __Elman-RNN-bi__: the same as RNNs above, but bidirectional 
- __Elman-RNN-bi-avg__: use average over RNN hidden states instead of just last hidden state 
- __GRU-bi-avg__: use GRU cells
- __LSTM-bi-avg__: use LSTM cells
- __GRU-bi-avg-static__: embeddings are initialized with vectors from _glove_ and fixed  
- __GRU-bi-avg-rand__:  embeddings are randomly initialized
- __CNN-rand__:  word embeddings are randomly initialized
- __CNN-static__: word embeddings are initialized with vectors from _glove_ and fixed  
- __CNN-nonstatic__: same as CNN-static, but pre trained vectors are fine tuned
- __GRU-bi-avg-charseq__: word embeddings plus CNN based word characters embeddings  
Note: words embeddings are nonstatic unless otherwise noted

## Hyperparameters and Training
__experiments_configs__ folder containes hyperparameters for all the experiments (and models) (they are composed in a way to hold "all things being equal" property) 

## Spooky Author Identification using Neural Networks

Competition goal is to predict the author of excerpts from horror stories by Edgar Allan Poe, Mary Shelley, and HP Lovecraft

However, the goal of my experiments was not to get the best score possible, but to experience myself how several neural networks variants react to the sentence representation options. 


Competition link: [link](https://www.kaggle.com/c/spooky-author-identification) <br>
All the models are implemented using [allennlp](https://github.com/allenai/allennlp) lib

To reproduce experiments, fetch allennlp's commit specified at requirements.txt.   

[Competiton rules](https://www.kaggle.com/c/spooky-author-identification/rules) allow for use of the data for academic goals, 
so I attached prepared data to this repo (/data folder): `preprocessed-train.txt`, `preprocessed-dev.txt`, `tc-tok-test_public_X.txt`
## Steps to reproduce (Kaggle)
1) Download glove embeddings and put to to the data/glove folder: <br>
`https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz`
2) Train a model: <br>
`python3 -m run train experiments_configs/authors_classifier.json -s experiments_models/pilot`<br>
(you can choose another .json config to train another model, or choose another save folder)
3) [Optional] Compute metrics values for your model on dev set: <br>
`python3 -m run evaluate --archive_file experiments_models/pilot/model.tar.gz --evaluation_data_file data/preprocessed-dev.txt --cuda_device -1`
4) Convert test set to the json lines format: <br>
`python3 data_utils/predict_utils.py data/tc-tok-test_public_X.txt data/tc-tok-test_public_X.jsonl`
5) Translate the test set: <br>
`python3 -m run predict experiments_models/pilot/model.tar.gz data/tc-tok-test_public_X.jsonl --output-file data/submission-pilot.jsonl`
6) Convert test set to the Kaggle submision format: <br>
`python3 data_utils/predict_utils.py data/submission-pilot.jsonl data/submission-pilot.csv`  <br>
(assumes python3 points to python 3.6)
Note: you can reproduce for other datasets by changing dataset path in configs and following almost the same procedure as above

## Data preprocessing
The data is first split into dev (3000 points) and train sets (code is in the /data_utils folder), and then truecased and tokenized using [moses scripts](https://github.com/marian-nmt/moses-scripts)    