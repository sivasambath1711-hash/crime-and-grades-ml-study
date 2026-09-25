# crime-and-grades-ml-study
Comparative Study of Regularised Learning Methods
A rigorous, cross-validated comparison of six model families — ordinary least squares, ridge, lasso and elastic-net regression, a support vector machine, and a multilayer perceptron — applied to two contrasting prediction problems in R. The project answers a practical question: which method is right, and when?

Key findings
Regularisation is conditional — no measurable gain on the low-dimensional dataset (13 predictors), consistent improvement on the high-dimensional one (39 predictors).
Accuracy vs interpretability — the neural network predicted best but explained nothing; lasso predicted nearly as well while revealing the drivers of each outcome.
A single metric can hide a broken model — on imbalanced data, two models reached ~85% accuracy by predicting the majority class for every case, catching 0% of at-risk students; balanced accuracy and recall exposed this.
