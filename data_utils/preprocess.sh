MOSES_SCRIPTS="moses-scripts/scripts"
DATA="../data"

# tokenize
for f in {train_X,dev_X,test_public_X}.txt
do
    $MOSES_SCRIPTS/tokenizer/tokenizer.perl < $DATA/$f > $DATA/tok-$f
done

# train truecaser model
$MOSES_SCRIPTS/recaser/train-truecaser.perl --model truecaser-model.mdl --corpus $DATA/tok-train_X.txt

# truecase
for f in tok-{train_X,dev_X,test_public_X}.txt
do
    $MOSES_SCRIPTS/recaser/truecase.perl --model truecaser-model.mdl < $DATA/$f > $DATA/tc-$f
done

# from datasets for model training and validation
paste -d @@@ $DATA/tc-tok-train_X.txt $DATA/train_y.txt > $DATA/preprocessed-train.txt
paste -d @@@ $DATA/tc-tok-dev_X.txt $DATA/dev_y.txt > $DATA/preprocessed-dev.txt
