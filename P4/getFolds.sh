sed -n 1,327p CLEANED-house-votes-84.data.txt > train.txt
sed -n 328,435p CLEANED-house-votes-84.data.txt > validate.txt

sed -n 1,109p train.txt > test1.txt
sed -n 110,327p train.txt > train1.txt

sed -n 110,218p train.txt > test2.txt
sed -n 1,109p train.txt >> train2.txt
sed -n 219,327p train.txt >> train2.txt

sed -n 219,327p train.txt > test3.txt
sed -n 1,218p train.txt > train3.txt

