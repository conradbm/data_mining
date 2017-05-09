Author: Blake Conrad
Contents: HW2 Kmeans Clustering

::::::::::::::::::::::::::::::::::::::
::: Important Info Regarding Files :::
::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::
:::::: Main Program Run Example ::::::
::::::::::::::::::::::::::::::::::::::

// Program assumes the iris.txt to be a .csv file with no header row
// The user can decide any K in place of the 5
// The program will only give Purity Scores if k=3
1. python asign2_bmconrad.py iris.txt 5

// The user can pick ROW VALUES inside of iris.txt specified in centroids.txt which the 
program will choose for the user. If those indices do not exist, random ones will be chosen. 
// If your K value and the number of lines in the centroids.txt file do not match, the K will be set to the number of rows in your centroids.txt file. 
// The program will allow for 1 additional 
new line character at the end of the file to be removed, but if there is a ton of new line 
characters at the end of the file, it will continue to count them as NON INDEX VALID points and choose random ones for you, rather than just breaking.  

2. python asign2_bmconrad.py iris.txt 6 centroids.txt

::::::::::::::::::::::::::::::::::::::
:::::::: Get 10 Purity Scores ::::::::
::::::::::::::::::::::::::::::::::::::

// the 3 is the command line argument the script is accepting
// the results can be stored in any file, but this is how I got them

1. sh runKmeans10Times.sh 3 | grep 'purity' >> asign2_bmconrad_best_purity_results.txt

::::::::::::::::::::::::::::::::::::::::
:::::: My Best Purity Score Result :::::
::::::::::::::::::::::::::::::::::::::::

1. Located in the same directory this one was opened in under: 
asign2_bmconrad_best_purity_results.txt
