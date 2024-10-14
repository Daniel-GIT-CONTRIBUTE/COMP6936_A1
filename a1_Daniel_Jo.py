import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit
from pygam import LinearGAM, LogisticGAM, s, f
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, roc_auc_score, precision_score
import matplotlib.pyplot as plt
##sites consulted:
#https://www.geeksforgeeks.org/generalized-additive-model-in-python/
#https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html


data = pd.read_csv('heart.csv')
X = data.drop(columns="target")
y = data.target

#Define GAMs, CV, and necessary steps so it can be used outside of the training loop
five_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
GAMs = None
AUC_SCORE = []
F1_SCORE = []
FifthFold = None

# Train and test loop using 5 Fold CV
for fold, (train_index, test_index) in enumerate(five_fold.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    GAMs = LogisticGAM(s(0) + f(1) + f(2) + s(3) + s(4) + f(5) + f(6) + s(7) + f(8) + s(9) + f(10) + f(11) + f(12)).fit(
        X_train, y_train)
    y_pred = GAMs.predict(X_test)
    y_pred_prob = GAMs.predict_mu(X_test)
    f1 = f1_score(y_test, y_pred)
    auc_of_fold = roc_auc_score(y_test, y_pred)
    AUC_SCORE.append(auc_of_fold)
    F1_SCORE.append(f1)

    if (fold == 4):
        precision = precision_score(y_test, y_pred)



## Generate partial dependence for each feature

# graph prep
fig, axs = plt.subplots(2, 6, figsize=(20, 7))
fig.subplots_adjust(hspace=0.5)
axs = axs.flatten()

## get the feature names
feature_names = X.columns.tolist()

#graph each fig
for i, ax in enumerate(axs):
    XX = GAMs.generate_X_grid(term=i)
    ax.plot(XX[:, i], GAMs.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], GAMs.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
    if i == 0:
        ax.set_ylim(-30,30)
    ax.set_title(feature_names[i]);
    if i == 1:
        ax.set_xlabel("0: Female / 1: Male")



##Question: I thought males had more heart disease. Next steps is to investigate if the graph above was wrong.
##Presupposition: Males have a higher likely chance of heart disease

males = data[data["sex"] == 1].reset_index()
females = data[data["sex"] == 0].reset_index()
print("total males", len(males), "/", "males with heart disease", males.target.sum())
print("total females", len(females), "/", "females with heart disease", females.target.sum())


## I was wrong. For this set of data, females are more likely to have heart disease

## recycled draw code from when I took 3202
def aurocDraw(my_model, model_name_string, fig, ax):
    # Get probability scores for the positive class
    model = my_model

    if model_name_string == "LR":
        y_pred = model.decision_function(X_test)
    elif model_name_string == "FOREST":
        y_pred = model.predict_proba(X_test)[:, 1]
    elif model_name_string == "KNN":
        y_pred = model.predict_proba(X_test)[:, 1]
    elif model_name_string == "GAMs":
        y_pred = model.predict_proba(X_test)
    else:
        raise ValueError("Invalid model_name_string")

    fpr, tpr, threshold = roc_curve(y_test, y_pred)

    auc_ = auc(fpr, tpr)

    if (model_name_string == "LR"):
        my_label = ("logistic (auc = %0.3f)" % auc_)
    if (model_name_string == "FOREST"):
        my_label = ("Random Forest (auc = %0.3f)" % auc_)
    if (model_name_string == "KNN"):
        my_label = ("KNN (auc = %0.3f)" % auc_)

    if (model_name_string == "GAMs"):
        my_label = ("GAMs (auc = %0.3f)" % auc_)

    ax.plot(fpr, tpr, marker=".", label=my_label)
    plt.fill_between(fpr, tpr, 0, alpha=0.2)  # Fills the area under the line


## Draw Area Under the Curve
fig, ax = plt.subplots(figsize = (5,5))
aurocDraw(GAMs, "GAMs", fig, ax )


print("mean F1 : ", np.mean(F1_SCORE))
print("mean AUC: ", np.mean(AUC_SCORE))