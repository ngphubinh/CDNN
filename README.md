# CDNN
Centroid Displacement based k-NN (CDNN) Algorithm
The CDNN algorithm is proposed in the paper [Robust Biometric Recognition From Palm Depth Images for Gloved Hands](https://ieeexplore.ieee.org/document/7161357). To accelerate future projects using our algorithm and also evaluate CDNN against other K-NN algorithms. We develop and open-source CDNN as a Python library for performing multi-label classification tasks, based on the \texttt{scikit-learn} API. We also aim to modify and extend CDNN for the study of regression in the near future. 


The repositry includes:
- Native Python implementations of CDNN alongside a flexible framework for adapting different distance metrics.
- Examples of using CDNN
- A comparision between CDNN and tradditional k-NN algorithm on some sample datasets
- A comparision of using different distance metrics with CDNN

Please refer to example.ipynb for examples.

A sample result will look like this:
```
Testing with k = 21

---------------Digits dataset------------------
Loading data.....
Done loading data!

Number of classes: 10
Data dimension: 64
Number of training samples: 1437
Number of testing samples: 360

Predict time for CDNN: 0.091s
Accuracy for CDNN with k = 21: 0.992

Predict time for kNN with uniform weights: 0.032s
Accuracy for kNN with k = 21 and uniform weights: 0.978

Predict time for kNN with distance weights: 0.025s
Accuracy for kNN with k = 21 and distance weights: 0.983
```

## Citation
If you use this code or CDNN algorithm for your research, please cite this paper.
```
@article{nguyen2015robust,
  title={Robust biometric recognition from palm depth images for gloved hands},
  author={Nguyen, Binh P and Tay, Wei-Liang and Chui, Chee-Kong},
  journal={IEEE Transactions on Human-Machine Systems},
  volume={45},
  number={6},
  pages={799--804},
  year={2015},
  publisher={IEEE}
}
```
