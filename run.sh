export TRAINING_DATA=input/train_preprocessed.csv
export TEST_DATA=input/test_preprocessed.csv
export MODEL=$1

FOLD=0 python -m script.training
FOLD=1 python -m script.training
FOLD=2 python -m script.training
FOLD=3 python -m script.training
FOLD=4 python -m script.training

python -m script.predict