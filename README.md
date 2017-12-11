# Spooky Author Identification using Neural Networks


Competition goal is to predict the author of excerpts from horror stories by Edgar Allan Poe, Mary Shelley, and HP Lovecraft

However, the goal of my experiments was not to get the best score possible, but to experience myself how several neural networks variants react to the sentence representation options. 


Competition link: [link](https://www.kaggle.com/c/spooky-author-identification) <br>
All the models are implemented using [allennlp](https://github.com/allenai/allennlp) lib

To reproduce experiments, fetch allennlp's commit specified at requirements.txt.   

[Competiton rules](https://www.kaggle.com/c/spooky-author-identification/rules) allow for use of the data for academic goals, 
so I attached prepared data to this repo (/data folder): `preprocessed-train.txt`, `preprocessed-dev.txt`, `tc-tok-test_public_X.txt`
## Steps to reproduce
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

## Data preprocessing
The data is first split into dev (3000 points) and train sets (code is in the /data_utils folder), and then truecased and tokenized using [moses scripts](https://github.com/marian-nmt/moses-scripts)    

## Models (WIP)
General idea is the same:
1) Encode source sentence to get single vector sentence representation
2) Project this representation onto desired classes

Different approaches differ in a way the encode the sentence <br>
Sentence encoding variations I experimented with: <br> 
- __dummy-baseline__: just always predict most common author
- __boe-baseline__: just avarage over word embeddings 
- __RNN-baseline__: tuned baseline RNN; starting point for future experiments; see "Hyperparameters and Training" section for details 
- __Elman-RNN__: use simple Elman RNN cells
- __GRU__: use GRU cells
- __LSTM__: use LSTM cells
- __RNN-uni__: the same as RNNs above, but unidirectional
- __RNN-avg-of-states__: use average over RNN hidden states instead of just last hidden state
- __RNN-rand__:  embeddings are randomly initialized
- __RNN-static__: embeddings are initialized with vectors from _glove_ and fixed  
- __RNN-non-static__: same as RNN-static, but pre trained vectors are fine tuned
- __CNN-rand__:  word embeddings are randomly initialized
- __CNN-static__: word embeddings are initialized with vectors from _glove_ and fixed  
- __CNN-non-static__: same as CNN-static, but pre trained vectors are fine tuned
- __RNN/CNN-word-char-CNN__: word embeddings plus CNN based word characters embeddings  
- __RNN/CNN-word-char-RNN__: word embeddings plus CNN based word characters embeddings  
- __RNN/CNN-ling__: use linguistic tags to augment RNN 
- __RNN-char-seg__: RNN over word characters with explicit word segmentation
- __RNN-char__: RNN over characters without explicit word segmentation
- __CNN-char-seg__: CNN over word characters with explicit word segmentation
- __CNN-char__: CNN over characters without explicit word segmentation

## Future work
It is interesting also to see how rubust the results are across different datasets. 