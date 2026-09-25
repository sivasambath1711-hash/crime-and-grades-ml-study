# ---- Setup ----
library(MASS)              # provides the Boston data set

# Quick look at the data: dimensions and variable names
dim(Boston)                # how many rows (observations) and columns (variables)?
names(Boston)              # the 14 column names; crim is our response
?Boston                    # opens the help page describing each variable

# ---- Part (b): multiple linear regression, crim on all predictors ----
# crim ~ . means: regress crim on every other column in Boston.
lm.all <- lm(crim ~ ., data = Boston)

# summary() gives the per-predictor t-tests (the p-values we need for
# H0: beta_j = 0), plus the overall F-test, R^2 and residual error.
summary(lm.all)
# ---- Part (b) extension: multicollinearity check ----
# VIF (Variance Inflation Factor) measures how much each predictor is
# explained by the OTHER predictors. High VIF = redundant information,
# which inflates standard errors and can make a real predictor look
# insignificant. Rule of thumb: VIF > 5 is worth noting, > 10 is serious.
library(car)
vif(lm.all)

# Method: glmnet, with 10-fold cross-validation to pick lambda
# ============================================================
library(glmnet)
x <- model.matrix(crim ~ ., data = Boston)[, -1]
y <- Boston$crim

set.seed(1)
foldid <- sample(rep(1:10, length.out = nrow(x)))   # ONE fold assignment for everyone

# Same foldid passed to every glmnet model -> identical folds -> fair comparison
cv.ridge <- cv.glmnet(x, y, alpha = 0,   foldid = foldid)
cv.lasso <- cv.glmnet(x, y, alpha = 1,   foldid = foldid)
cv.enet  <- cv.glmnet(x, y, alpha = 0.5, foldid = foldid)

# OLS on the same folds, by hand
ols.err <- sapply(1:10, function(k){
  tr <- Boston[foldid != k, ]; te <- Boston[foldid == k, ]
  mean((te$crim - predict(lm(crim ~ ., data = tr), te))^2)
})

# Tidy results table
results <- data.frame(
  Model  = c("OLS", "Ridge", "Lasso", "ElasticNet"),
  CV_MSE = round(c(mean(ols.err), min(cv.ridge$cvm),
                   min(cv.lasso$cvm), min(cv.enet$cvm)), 3)
)
print(results)

# SVM section — convert crim to a binary class first
# ============================================================
library(e1071)

# crim01 = 1 if crime is above the median (high-crime area), else 0.
# Median chosen so the two classes are balanced (~50/50), making
# accuracy a meaningful metric and avoiding a lazy majority-class model.
crim01 <- factor(ifelse(Boston$crim > median(Boston$crim), 1, 0))

# Build a modelling frame: all the predictors EXCEPT crim, plus the new label.
dat <- data.frame(Boston[, !names(Boston) %in% "crim"], crim01)

table(crim01)              # confirm the 253/253 balanced split

# ---- Standard linear SVM (e1071), 10-fold cross-validated ----
# This is the "basic" SVM (L2-regularised by default — see note below).
set.seed(1)
foldid <- sample(rep(1:10, length.out = nrow(dat)))   # same 10-fold scheme idea

acc <- sapply(1:10, function(k){
  tr <- dat[foldid != k, ]; te <- dat[foldid == k, ]
  m  <- svm(crim01 ~ ., data = tr, kernel = "linear", cost = 1)  # fit on train
  mean(predict(m, te) == te$crim01)                              # accuracy on test
})
cat("Linear SVM  CV-accuracy:", round(mean(acc), 4), "\n")

# Regularised SVM: L1 (lasso) and elastic-net, via sparseSVM
# sparseSVM does L1/elastic-net penalties for binary classification,
# which e1071 cannot — that's why we use it here.
# ============================================================
library(sparseSVM)

# sparseSVM needs x as a numeric matrix and y as a 0/1 vector.
x.svm <- model.matrix(crim01 ~ ., data = dat)[, -1]   # predictors, no intercept
y.svm <- as.numeric(as.character(crim01))             # 0/1 labels as numbers

# cv.sparseSVM does its own k-fold CV and tunes the penalty strength (lambda).
# alpha = 1   -> pure L1 (lasso SVM)
# alpha = 0.5 -> elastic net SVM (blend of L1 and L2)
set.seed(1)
cv.l1   <- cv.sparseSVM(x.svm, y.svm, alpha = 1,   nfolds = 10)
set.seed(1)
cv.enet <- cv.sparseSVM(x.svm, y.svm, alpha = 0.5, nfolds = 10)

# cv.sparseSVM reports misclassification ERROR at the best lambda.
# Accuracy = 1 - error, so we convert for a fair comparison with the e1071 SVM.
cat("L1  SVM  CV-accuracy:", round(1 - min(cv.l1$cve),   4), "\n")
cat("Enet SVM CV-accuracy:", round(1 - min(cv.enet$cve), 4), "\n")

# Which predictors did the L1 SVM keep vs zero out? (feature selection)
coef(cv.l1$fit, lambda = cv.l1$lambda.min)

#   decay = 0   -> unregularised MLP
#   decay = 0.1 -> L2-regularised MLP
# Inputs are standardised (MLPs are sensitive to feature scale).
# ============================================================
library(nnet)

crim01 <- factor(ifelse(Boston$crim > median(Boston$crim), 1, 0))
preds  <- Boston[, !names(Boston) %in% "crim"]   # 13 predictors, no crim

set.seed(1)
foldid <- sample(rep(1:10, length.out = nrow(Boston)))

# Helper: 10-fold CV accuracy for a given decay (regularisation strength)
cv_mlp <- function(decay_val){
  acc <- sapply(1:10, function(k){
    tr_x <- preds[foldid != k, ]; te_x <- preds[foldid == k, ]
    # Standardise using TRAIN means/sds only, then apply to test (avoids leakage)
    mu <- colMeans(tr_x); sdv <- apply(tr_x, 2, sd)
    tr <- data.frame(scale(tr_x, mu, sdv), crim01 = crim01[foldid != k])
    te <- data.frame(scale(te_x, mu, sdv), crim01 = crim01[foldid == k])
    # size = 5 hidden units; maxit = max training iterations
    m <- nnet(crim01 ~ ., data = tr, size = 5, decay = decay_val,
              maxit = 300, trace = FALSE)
    mean(predict(m, te, type = "class") == te$crim01)   # test-fold accuracy
  })
  mean(acc)
}

cat("MLP unregularised (decay=0)    CV-accuracy:", round(cv_mlp(0),   4), "\n")
cat("MLP L2-regularised (decay=0.1) CV-accuracy:", round(cv_mlp(0.1), 4), "\n")
 