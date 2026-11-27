# SVC Model: Pulsar Star Classification

A support vector machine (SVM) classifier that predicts whether a radio signal comes from a pulsar star, built in Python with scikit-learn and pandas.

Course project for [course name], California State University, San Bernardino (2023).

## Dataset

`pulsar_stars.csv` (HTRU2): radio signal candidates described by 8 numeric features (statistics of the integrated pulse profile and the DM-SNR curve), with a `target_class` label of 1 for pulsar and 0 for non-pulsar. Pulsars are a small minority of the samples, so accuracy alone can be misleading. Precision, recall, and specificity are reported as well.

## Approach

1. Load the data with pandas and inspect its size, columns, and class distribution
2. Split into 80% training and 20% test sets
3. Standardize features with `StandardScaler` (fit on the training set only)
4. Train an SVM with an RBF kernel (`C=10`, `gamma=0.3`)
5. Evaluate on the unseen test set: accuracy, confusion matrix, precision, recall, and specificity
6. Save the trained model with `pickle` as `pulsar_svm_model.sav`

## Results

| Metric | Score |
|---|---|
| Accuracy | [X]% |
| Precision (pulsar) | [X]% |
| Recall (pulsar) | [X]% |
| Specificity | [X]% |

## How to run

```bash
pip install pandas scikit-learn
python [file_name].py
```

## Tech

Python, pandas, scikit-learn (SVC, StandardScaler, train_test_split, metrics), pickle
