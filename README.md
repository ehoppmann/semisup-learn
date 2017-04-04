Semi-supervised learning frameworks for Python
===============

This project contains Python implementations for semi-supervised
learning, made compatible with scikit-learn, including

- **Contrastive Pessimistic Likelihood Estimation (CPLE)** (based on - but not equivalent to - [Loog, 2015](http://arxiv.org/abs/1503.00269)), a `safe' framework applicable for all classifiers which can yield prediction probabilities
(safe here means that the model trained on both labelled and unlabelled data should not be worse than models trained only on the labelled data)

- Self learning (self training), a naive semi-supervised learning framework applicable for any classifier (iteratively labelling the unlabelled instances using a trained classifier, and then re-training it on the resulting dataset - see e.g. http://pages.cs.wisc.edu/~jerryzhu/pub/sslicml07.pdf )

- Semi-Supervised Support Vector Machine (S3VM) - a simple scikit-learn compatible wrapper for the QN-S3VM code developed by 
Fabian Gieseke, Antti Airola, Tapio Pahikkala, Oliver Kramer (see http://www.fabiangieseke.de/index.php/code/qns3vm ) 
This method was included for comparison

The first method is a novel extension of [Loog, 2015](http://arxiv.org/abs/1503.00269) for any discriminative classifier (the differences to the original CPLE are explained below). The last two methods are only included for comparison. 

 
The advantages of the CPLE framework compared to other semi-supervised learning approaches include  

- it is a **generally applicable framework (works with scikit-learn classifiers which allow per-sample weights)**

- it needs low memory (as opposed to e.g. Label Spreading which needs O(n^2)), and 

- it makes no additional assumptions except for the ones made by the choice of classifier 

The main disadvantage is high computational complexity. Note: **this is an early stage research project, and work in progress** (it is by no means efficient or well tested)!

If you need faster results, try the Self Learning framework (which is a naive approach but much faster):

```python
from frameworks.SelfLearning import *

any_scikitlearn_classifier = SVC()
ssmodel = SelfLearningModel(any_scikitlearn_classifier)
ssmodel.fit(X, y)
```

For details consult [the documentation](docs/README.md)