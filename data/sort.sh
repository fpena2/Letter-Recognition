#!/bin/sh
sort KNN.txt -o KNN.txt;
sort ANN.txt -o ANN.txt;
sort GAUS.txt -o GAUS.txt;
sort SVC.txt -o SVC.txt;

sort KNN_PCA.txt -o KNN_PCA.txt;
sort ANN_PCA.txt -o ANN_PCA.txt;
sort GAUS_PCA.txt -o GAUS_PCA.txt;
sort SVC_PCA.txt -o SVC_PCA.txt;