## Forecasting the Disturbance Storm Time Index with Bayesian Deep Learning<br>
[![DOI](https://github.com/ccsc-tools/zenodo_icons/blob/main/icons/dst.svg)](https://zenodo.org/record/7529705#.Y8ApDBXMJD8)


## Authors
Yasser Abduallah, Jason T. L. Wang, Prianka Bose, Genwei Zhang, Firas Gerges, and Haimin Wang

## Abstract

The disturbance storm time (Dst) index is an important and useful measurement in space weather research. It has been used to characterize the size and intensity of a geomagnetic storm. A negative Dst value means that the Earth’s magnetic field is weakened, which happens during storms. Here, we present a novel deep learning method, called the Dst Transformer (or DSTT for short), to perform short-term, 1-6 hour ahead, forecasting of the Dst index based on the solar wind parameters provided by the NASA Space Science Data Coordinated Archive. The Dst Transformer combines a multi-head attention layer with Bayesian inference, which is capable of quantifying both aleatoric uncertainty and epistemic uncertainty when making Dst predictions. Experimental results show that the proposed Dst Transformer outperforms related machine learning methods in terms of the root mean square error and R-squared. Furthermore, the Dst Transformer can produce both data and model uncertainty quantification results, which can not be done by the existing methods. To our knowledge, this is the first time that Bayesian deep learning has been used for Dst index forecasting.


For the latest updates of the tool refer to https://github.com/deepsuncode/Dst-prediction

## Installation on local machine
To install TensorFlow with pip refer to https://www.tensorflow.org/install/pip

Tested on Python 3.9.16 and the following version of libraries
|Library | Version   | Description  |
|---|---|---|
|keras| 2.10.0 | Deep learning API|
|numpy| 1.24.2| Array manipulation|
|scikit-learn| 1.2.1| Machine learning|
|matplotlib| 3.6.3| Visutalization tool|
| pandas|1.5.3| Data loading and manipulation|
| seaborn | 0.12.2| Visualization tool|
| scipy|1.10.0| Provides algorithms for optimization and statistics|
| tensorboard| 2.10.1 | Provides the visualization and tooling needed for machine learning|
| tensorflow| 2.10.1| Deep learning tool for high performance computation |
|tensorflow-probability | 0.17.0| For probabilistic models|
