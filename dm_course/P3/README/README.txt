Author: Blake Conrad
Contents: How to run and execute the code for bayes/naive-bayes on the 
iris-shuffled.txt data given for HW3.

EASY TESTING: Move into the main folder, HW3_bmconrad/ then run the following command:

>> sh scripts/bayes.sh // 3 Runs of bases on correctly split data
>> sh scripts/naive_bayes.sh // 3 Runs of naive bayes on correctly split data

EASY CHECKING: My actual results are in the folder titled results/ which contains a PDF of my averaged 3 runs of each of these models, testing accuracy, precision, recall, and f-measure. 

(sh scripts/getFolds.sh)
1. I chopped up the data using a small script called getFolds.sh located 
at scripts/getFolds.sh. This script chops my data into 3 different 
chunks; the training and testing sets for each fold. The first fold will 
contain the first 50 observations (out of 150) as the test data (or held 
out data) while the last 100 observations we train on. The second fold 
will hold out the observations from 50-100, and training on 1-50 & 
100-150, and lastly the third fold will hold out observations 100-150, 
and train on observations from 1-100. These files are automatically 
chopped, placed, and generated (without reordering observations) in the 
same folder the script is ran (assuming there exists the file named: 
iris-shuffled.txt).

(sh scripts/bayes.sh)
(sh scripts/naive_bayes.sh)
2. Next, by being inside of the P3/ directory, one can run another 1 of 
2 scripts; scripts/bayes.sh or scripts/naive_bayes.sh, these two shells 
will go run the necessary 4 Python Scripts due for this assignment, run 
them once each for their cooresponding folds (fold1,fold2, and fold3) 
and will print out the results for the confusion matrix, accuracy, 
precision, recall, and f-measure for each to the console. An example 
would be: cd HW3_bmconrad/; sh scripts/bayes.sh >> yourFile.txt; This 
would assume you are in the folder with your folds, and original 
dataset, and the 4 python scripts for bayes and naive bayes (containg 
the actual meat of the project), and will append the output into a text 
file named yourFile.txt if it doesn't already exist it will be created 
for you. 

// Model Files Structure
3. The model objects created are simply flat files named, 
bayes_model.txt and naive_bayes_model.txt, which follow a fairly 
intuitive format:
// Description of the next line(s) and what they are
line2
line3
line4
// Another description of the next line(s) and what thye are
line6
// More detailed descriptions
line7

By simply reading the first line, you should be able to intuitively 
parse the following lines based on its english statement. 


(py Q1_bayes_train_write_bmconrad.py)
(py Q2_bayes_read_test_bmconrad.py)
(py Q1_naive_bayes_train_write_bmconrad.py)
(py Q2_naive_bayes_read_test_bmconrad.py)
3. As I mentioned before, the meat of the project are in the above 4 
python scripts. 

If you have any questions my contact is at: bmconrad@iupui.edu
