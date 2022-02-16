## Introduction
*XSVMC-Lib* is an open source library that implements algorithms proposed to make support vector machine (SVM) predictions interpretable.

## Requirements
*XSVMC-Lib* requires Python 3.8+. Since *XSVMC-Lib* has been implemented as an extension of *sklearn.svm.SVC*, it also requires the [SciKit-Learn](https://scikit-learn.org) package (python3 -m pip install sklearn).

Although they are not required by *XSVMC-Lib*, the following packages are necessary for running the examples:

- [Numpy](https://numpy.org) (```python3 -m pip install numpy```)
- [Matploplib](https://matplotlib.org) (```python3 -m pip install matploplib```)
- [OpenCV](https://opencv.org) (```python3 -m pip install opencv-python```)

## Examples

To run an example, say *digits_explanation.py*, you may use the following commands:

```
cd /path/to/xsvmc-lib
python3 examples/digits_explanation.py
```

## Datasets
Parts of the following datasets are used in several examples that illustrate the use of *XSVMC-Lib*.

- [The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)
- [The MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/)


## License
*XSVMC-Lib* is released under the [Apache License, Version 2.0](LICENSE).

