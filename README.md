# Credit Risk Analysis - Supervised Machine Learning

## Challenge - Overview

In this look at Supervised Machine Learning we use Oversampling and Undersampling techniques and various modelling methods to assess credit card risk. In the real world this is an unbalanced classification problem, so we need to look at the best way to get fair samples to train our models and compare the results. The dataset contained our target variable `loan_status` which splits the data into `high_risk` and `low_risk`. The goal of the machine learning algorithms is to predict, based on similar data provided, which class an application falls into.

For oversampling we use the `RandomOverSampler` and `SMOTE` algorithms, and for undersampling we use `ClusterCentroids`. Combining these methods, we also make use of `SMOTEENN`. Using the newly resampled data we use `LogisticRegression` modelling to compare them against each other, by producing a `balanced_accuracy_score`, a `confusion_matrix`, and a `classification_report`. We explored the machine learning models `BalancedRandomForestClassifier` and `EasyEnsembleClassifier` after this, passing in split data and comparing them in the same way as the sampling examples.


## Challenge - Results

As describe in the overview, we take the same metrics of each model we used during the challenge: `balanced_accuracy_score`, `confusion_matrix`, and `classification_report`. This allows us to compare each sampling and modelling technique we worked on.

* **Oversampling** - using the `RandomOverSampler` sampling and `LogisticRegression` algorithm. This produced the following result output:

  - Balanced Accuracy Score:

    ```
    0.6318837601879046
    ```

  - Confusion Matrix:

    ```
    [[   49    38]
    [ 5126 11992]]
    ```

  - Classification Report:

    ```
                       pre       rec       spe        f1       geo       iba       sup

      high_risk       0.01      0.56      0.70      0.02      0.63      0.39        87
       low_risk       1.00      0.70      0.56      0.82      0.63      0.40     17118

    avg / total       0.99      0.70      0.56      0.82      0.63      0.40     17205
    ```

* **Oversampling** - using the `SMOTE` sampling and `LogisticRegression` algorithms.

  - Balanced Accuracy Score:

    ```
    0.6392101881060872
    ```

  - Confusion Matrix:

    ```
    [[   58    29]
    [ 6646 10472]]
    ```

  - Classification Report:

    ```
                       pre       rec       spe        f1       geo       iba       sup

      high_risk       0.01      0.67      0.61      0.02      0.64      0.41        87
       low_risk       1.00      0.61      0.67      0.76      0.64      0.41     17118

    avg / total       0.99      0.61      0.67      0.75      0.64      0.41     17205
    ```

* **Undersampling** - using the `ClusterCentroids` sampling and `LogisticRegression` algorithm.

  - Balanced Accuracy Score:

    ```
    0.5177570695899859
    ```

  - Confusion Matrix:

    ```
    [[  50   37]
    [9230 7888]]
    ```

  - Classification Report:

    ```
                       pre       rec       spe        f1       geo       iba       sup

      high_risk       0.01      0.57      0.46      0.01      0.51      0.27        87
       low_risk       1.00      0.46      0.57      0.63      0.51      0.26     17118

    avg / total       0.99      0.46      0.57      0.63      0.51      0.26     17205
    ```

* **Combination Sampling** - using the `SMOTEENN` sampling and `LogisticRegression` algorithm.

  - Balanced Accuracy Score:

    ```
    0.6289296203633199
    ```

  - Confusion Matrix:

    ```
    [[  62   25]
    [7785 9333]]
    ```

  - Classification Report:

    ```
                       pre       rec       spe        f1       geo       iba       sup

      high_risk       0.01      0.71      0.55      0.02      0.62      0.40        87
       low_risk       1.00      0.55      0.71      0.71      0.62      0.38     17118

    avg / total       0.99      0.55      0.71      0.70      0.62      0.38     17205
    ```

* **Modelling** - using the `BalancedRandomForestClassifier` algorithm.

  - Balanced Accuracy Score:

    ```
    0.7885466545953005
    ```

  - Confusion Matrix:

    ```
    [[   71,    30],
    [ 2153, 14951]]
    ```

  - Classification Report:

    ```
                       pre       rec       spe        f1       geo       iba       sup

      high_risk       0.03      0.70      0.87      0.06      0.78      0.60       101
       low_risk       1.00      0.87      0.70      0.93      0.78      0.62     17104

    avg / total       0.99      0.87      0.70      0.93      0.78      0.62     17205
    ```

* **Modelling** - using the `EasyEnsembleClassifier` algorithm.

  - Balanced Accuracy Score:

    ```
    0.9316600714093861
    ```

  - Confusion Matrix:

    ```
    [[   93,     8],
    [  983, 16121]]
    ```

  - Classification Report:

    ```
                       pre       rec       spe        f1       geo       iba       sup

      high_risk       0.09      0.92      0.94      0.16      0.93      0.87       101
       low_risk       1.00      0.94      0.92      0.97      0.93      0.87     17104

    avg / total       0.99      0.94      0.92      0.97      0.93      0.87     17205
    ```

## Challenge - Summary

The first metric, the balanced accuracy score, takes into account that the two classes are heavily weighted in one direction, and produces a value based on the **sensitivity** (true positive rate) and **specificity** (true negative rate), and how well the model predicts them. From the results section we can see, based on the balanced accuracy score, the final model, `EasyEnsembleClassifier`, performs the best with `0.9316600714093861`. This means the model produced true positives and true negatives correctly 93% of the time.

The second metric, the confusion matrix, gives us a layout of the true positive, false negative, false positive, and true negative. This metric is more a visual aid, as the data is better understood in the third metric, the classification report.

The classification report takes the information shown in the confusion matrix and produces values for the **precision/positive predictive value**, **sensitivity/recall/true positive rate**, **specificity/true negative rate**, and **F1 score/harmonic mean**, as well as others. Depending on the goal of the model, which values are important is a determination made by the data scientist.

As we are predicting risk for a bank, we want to reduce the number of defaults on loans to as low as possible. In an example with twenty people, if two of them are high-risk applicants we would rather mark five of them as high-risk, covering the two, and only lend to the fifteen low-risk. This is a better alternative to loaning to nineteen of the twenty and having one of the high-risk customers default on the loan. This means we want to have high sensitivity/recall for high-risk customers - we want to catch as many of the positive high-risk cases as possible, and we favour that over catching ***just*** high-risk cases. While precision can still be important, it is not our focus when choosing a good model.

All the models we looked at had low precision in catching high-risk customers, which we know is okay, so we can put that statistic aside and concentrate on recall/sensitivity. The `EasyEnsembleClassifier` model produced the best recall value for high-risk, with 92%, a good margin above the same metric from the other models. **For this reason, I would recommend the East Ensemble Classification model** from the six, for a bank looking to predict high-risk versus low-risk for credit applications.

## Context

This is the Challenge Repo for Module 18 of the University of Toronto School of Continuing Studies Data Analysis Bootcamp Course - **Supervised Machine Learning** - Python, Machine Learning, Scikit-Learn, Regression, and Classification. Following the guidance of the module we end up pushing this selection of files to GitHub.
