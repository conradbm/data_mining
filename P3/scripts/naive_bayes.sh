# Author: Blake Conrad
# Purpose: run everything we will need for naive bayes classification
#           training and testing


python Q1_naive_bayes_train_write_bmconrad.py 1fold_train.txt
python Q2_naive_bayes_read_test_bmconrad.py 1fold_test.txt

python Q1_naive_bayes_train_write_bmconrad.py 2fold_train.txt
python Q2_naive_bayes_read_test_bmconrad.py 2fold_test.txt

python Q1_naive_bayes_train_write_bmconrad.py 3fold_train.txt
python Q2_naive_bayes_read_test_bmconrad.py 3fold_test.txt
