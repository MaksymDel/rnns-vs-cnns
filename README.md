# Spooky Author Identification using Neural Networks
My experiments with Kaggle's Spooky Author Identification dataset.

Competition goal is to predict the author of excerpts from horror stories by Edgar Allan Poe, Mary Shelley, and HP Lovecraft

However, the goal of my experiments was not to get the best score possible, but to experience it myself how does neural networks perform. 


Competition link: [link](https://www.kaggle.com/c/spooky-author-identification) <br>
All the models are implemented using [allennlp](https://github.com/allenai/allennlp) lib

To reproduce experiments, fetch allennlp's commit specified at requirements.txt.   

## Method
General idea is the same:
1) Encode source sentence to get single vector sentence representation
2) Project this representation onto desired classes

Different approaches differ in a way the encode the sentence <br>
Encoders I experimented with: <br> 
- RNN over word embeddings 
- CNN over word embeddings

- char-RNN with explicit word segmentation
- char-CNN with explicit word segmentation

- char-RNN without explicit word segmentation
- char-CNN without explicit word segmentation

- RNN over word embeddings + CNN based word characters embeddings
- RNN over word embeddings + linguistic tegs

- RNN over word embeddings + linguistic tegs + CNN based word characters embeddings

