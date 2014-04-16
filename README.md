HOG
===

Requires:
Opencv >= 2.4.x
Liblinear >= 1.9.x

Running each of the executables displays help information.

train:
1) Computes the HOG descriptors of the given directory of images and trains a linear svm from them.

2) To facilitate faster reading of images from the directory, opencv's yml data storage is used.  Consequently, for large datasets, memory consumption (~8G) is to be expected.

detect:
1) Performs a sliding-window detection using the trained svm on the given directory of images.

2) The scores are saved in text files such that each detection is denoted in the format: [x,y,w,h,score].

generateYML:
1) Creates and saves the images of the given directory in Opencv's yml datastorage.


