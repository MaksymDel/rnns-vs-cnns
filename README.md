# Spooky Author Identification using Neural Networks


Competition goal is to predict the author of excerpts from horror stories by Edgar Allan Poe, Mary Shelley, and HP Lovecraft

However, the goal of my experiments was not to get the best score possible, but to experience it myself how do several neural networks variants react to the sentence representation options. 


Competition link: [link](https://www.kaggle.com/c/spooky-author-identification) <br>
All the models are implemented using [allennlp](https://github.com/allenai/allennlp) lib

To reproduce experiments, fetch allennlp's commit specified at requirements.txt.   

## Reproduce
1) [Competiton rules](https://www.kaggle.com/c/spooky-author-identification/rules) allow for use of the data for academic goals, so I attached prepared data to this repo (/data folder). 
2) The data is first split into dev (3000 points) and train sets (code is in the /data_utils folder), and then truecased and tokenized using [moses scripts](https://github.com/marian-nmt/moses-scripts)    

## Method
General idea is the same:
1) Encode source sentence to get single vector sentence representation
2) Project this representation onto desired classes

Different approaches differ in a way the encode the sentence <br>
Sentence encoding variations I experimented with: <br> 
- RNN over word embeddings -> take last encoder output 
- RNN over word embeddings -> concatenate all encoder outputs
- CNN over word embeddings 

- char-RNN with explicit word segmentation
- char-CNN with explicit word segmentation

- char-RNN without explicit word segmentation
- char-CNN without explicit word segmentation

- RNN over word embeddings + CNN based word characters embeddings
- RNN over word embeddings + linguistic tegs

- RNN over word embeddings + linguistic tegs + CNN based word characters embeddings

## Future work
It is interesting also to see how rubust are results across different datasets. 