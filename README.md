# Decision Tree Binary Classifier (from scratch implementation)

Description:

This code learns a decision tree with a specified maximum depth, predicts the labels of the training and testing examples, and calculates training and testing errors. It uses ID3 algorithm for node selection.

How to Run:

Command - python decisionTree.py [args...]

Where above [args...] is a placeholder for six command-line arguments: <train input> <test input> <max depth> <train out> <test out> <metrics out>. These arguments are described in detail below:
1. <train input>: path to the training input .tsv file (see Section 2.1)
2. <test input>: path to the test input .tsv file (see Section 2.1)
3. <max depth>: maximum depth to which the tree should be built
4. <train out>: path of output .labels file to which the predictions on the training data should be written
5. <test out>: path of output .labels file to which the predictions on the test data should be written
6. <metrics out>: path of the output .txt file to which metrics such as train and test error should be written
