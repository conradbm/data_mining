# Author: Blake Conrad
# Purpose: run everything we will need for bayes classification training and testing

python Q1_bayes_train_write_bmconrad.py data/1fold_train.txt
python Q2_bayes_read_test_bmconrad.py data/1fold_test.txt

python Q1_bayes_train_write_bmconrad.py data/2fold_train.txt
python Q2_bayes_read_test_bmconrad.py data/2fold_test.txt

python Q1_bayes_train_write_bmconrad.py data/3fold_train.txt
python Q2_bayes_read_test_bmconrad.py data/3fold_test.txt
