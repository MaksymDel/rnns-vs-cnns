# CNNs vs RNNs for Sentence Classification
This project takes as background [CNNs for sentence classification](https://arxiv.org/pdf/1408.5882.pdf) paper and does following: 
1) Reimplements models from the paper
2) Adds a new dataset to verify how conclusion stated in the paper generalize to a new domain
3) Implements several recurrent architectures and evaluates them on both new and paper's data

## Models (WIP)
General idea is the same:
1) Encode a source sentence in some way (RNN / CNN) to get single a vector sentence representation
2) Project this representation onto desired classes
The table below shows experiments results. See following section for models descriptions.

## Results
| Model/Dataset | KSAI | SST1 | SST2 |
| --- | --- | ---| --- | 
| boe-baseline | __0.823__ | 0.23 | 0.82 |
| Elman-RNN-baseline | 0.785 | 0.39 | 0.82 |
| Elman-RNN-bi | 0.807 | 0.37 | 0.83 | 
| Elman-RNN-bi-avg | 0.815 | 0.40 | 0.84 | 
| LSTM-bi-avg | 0.780 | 0.43 | 0.86 |
| GRU-bi-avg-rand | 0.817 | 0.41 | __0.85__ | 
| GRU-bi-avg-static | 0.757 | 0.42 | 0.83 |   
| GRU-bi-avg-nonstatic | __0.823__ | 0.42 | 0.84 | 
| GRU-bi-avg-charseg | __0.824__ | 0.43 | __0.85__ |      
| CNN-rand (Kim, 2014) | 0.814 | 45.0 | 0.83 |   
| CNN-static (Kim, 2014) | 0.753 | 45.5 | __0.87__ |    
| CNN-nonstatic (Kim, 2014) | 0.822 | __48.0__ | __0.87__ |

*KSAI stands for Kaggle Spooky Author Identification
*the metric is __accuracy__ <br>
*you can find __configs__ for all the models in __experiments_configs__ folder (names align)
*hyperparameters for RNNs were tuned on KSAI  

## Modes descriptions:
- __boe-baseline__: just avarage over word embeddings 
- __Elman-RNN-baseline__: simple Elman RNN; starting point for future experiments; see "Hyperparameters and Training" section for details
- __Elman-RNN-bi__: the same as RNNs above, but bidirectional 
- __Elman-RNN-bi-avg__: use average over RNN hidden states instead of just last hidden state 
- __GRU-bi-avg__: use GRU cells
- __LSTM-bi-avg__: use LSTM cells
- __GRU-bi-avg-static__: embeddings are initialized with _glove_ vectors and fixed  
- __GRU-bi-avg-rand__:  embeddings are randomly initialized
- __CNN-rand__:  embeddings are randomly initialized
- __CNN-static__: embeddings are initialized with vectors from _glove_ and fixed  
- __CNN-nonstatic__: same as CNN-static, but pre trained vectors are fine tuned
- __GRU-bi-avg-charseq__: word embeddings plus CNN based word characters embeddings, and bidirectional GRU on top  
Note: words embeddings are nonstatic unless otherwise noted

## Hyperparameters and Training
__experiments_configs__ folder containes hyperparameters for all the experiments (and models) (they are composed in a way to hold "all things being equal" property) 

## Reproduce (SST1, SST2)
1) Download glove embeddings and put to to the data/glove folder: <br>
`https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz`
2) Train a model (it will do early stopping and be evaluated on the test set automaticaly at the end): <br>
`python3 -m run train experiments_configs/gru-bi-agv.json -s experiments_models/gru-bi-agv`<br>
(you can choose another .json config to train another model, or choose another save folder)

## Reproduce (Kaggle)
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

### Spooky Author Identification using Neural Networks

Competition goal is to predict the author of excerpts from horror stories by Edgar Allan Poe, Mary Shelley, and HP Lovecraft

However, the goal of my experiments was not to get the best score possible, but to experience myself how several neural networks variants react to the sentence representation options. 

Competition link: [link](https://www.kaggle.com/c/spooky-author-identification) <br>
All the models are implemented using [allennlp](https://github.com/allenai/allennlp) lib

To reproduce experiments, fetch allennlp's commit specified at requirements.txt.   

[Competiton rules](https://www.kaggle.com/c/spooky-author-identification/rules) allow for use of the data for academic goals, 
so I attached prepared data to this repo (/data folder): `preprocessed-train.txt`, `preprocessed-dev.txt`, `tc-tok-test_public_X.txt`

### Data preprocessing
The data is first split into dev (3000 points) and train sets (code is in the /data_utils folder), and then truecased and tokenized using [moses scripts](https://github.com/marian-nmt/moses-scripts)    