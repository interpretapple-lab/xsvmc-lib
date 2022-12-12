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

## Technical Information
The mathematical foundation of *XSVMC-Lib* can be found in 

M. Loor and G. De Tré, "[Contextualizing Support Vector Machine Predictions](https://doi.org/10.2991/ijcis.d.200910.002)," International Journal of Computational Intelligence Systems, Volume 13, Issue 1, 2020, Pages 1483 - 1497, doi: 10.2991/ijcis.d.200910.002.


## License
*XSVMC-Lib* is released under the [Apache License, Version 2.0](LICENSE).

## Citing
If you use *XSVMC-Lib*, please cite the following article:

M. Loor, A. Tapia-Rosero and G. De Tré, "[An Open-Source Software Library for Explainable Support Vector Machine Classification](https://doi.org/10.1109/FUZZ-IEEE55066.2022.9882731)," 2022 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), 2022, pp. 1-7, doi: 10.1109/FUZZ-IEEE55066.2022.9882731.


### BibTeX

```
@INPROCEEDINGS{
    xsvmlib,  
    author={Loor, Marcelo and Tapia-Rosero, Ana and {De Tr\'{e}}, Guy},  
    booktitle={2022 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)},   
    title={An Open-Source Software Library for Explainable Support Vector Machine Classification},   
    year={2022}, pages={1-7},  
    doi={10.1109/FUZZ-IEEE55066.2022.9882731}
}
```