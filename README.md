# decision_tree
Python program for training the decision tree for people detection (based on LBP - Local Binary Patterns) and converting the tree to hardware (FPGA) implementation.

## Acknowledgment

This code is used in the project entitled:

*"NEW CONCEPT OF THE NETWORK OF SMART CAMERAS WITH ENHANCED AUTONOMY FOR AUTOMATIC SURVEILLANCE SYSTEM"*

More information about the project can be found: http://www.vision.put.poznan.pl/?page_id=237

## Description

The code provided allows preparing training and testing data for decision tree based classifier alongside the ability to convert the created decision tree to hardware implementation (VHDL code for FPGA device). Classifier in current implementation allows detecting human silhouette. The INRIA Person Dataset (http://pascal.inrialpes.fr/data/human/) is used as a image source.

## Languages and tools used

The project is written in Python 2.7.10 and uses following libraries (+ all the prequisitions for those):
* scikit-learn - for decision tree training and testing,
* scikit-image - for image loading, lbp calculation,
* numpy - histogram calculation,
* matplotlib - visualization.

## Modules description
* Tree.py - classes for storing tree structure and convert it to VHDL code.
* analyse_classifier.py - functions for extracting decisions from scikit-learn trees (code adapted from some stackoverflow question).
* prepare_data.py - code for calcualting lbp region descriptors for images and saving them as prepared data for learning and testing the decision tree.
* decision_tree.py - main file that loads prepared data, trains the decision tree and tests it.

## Running the code
Use prepare\_data.py script for preparing the data from the image dataset. Then run decision\_tree.py script for learning/testing the decision tree and converting it to VHDL code.

## License

MIT
