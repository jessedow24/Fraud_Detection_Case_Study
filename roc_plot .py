import matplotlib.pyplot as plt
import numpy as np
from cleaned_data import clean_data
from logistic_regression import get_xy_arrays, sk_log_reg_probs
from sklearn.cross_validation import train_test_split
​
def get_rates(y_true, y_pred, thresh=1):
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	y_pred = y_pred >= thresh
	TP = ((y_pred == y_true) & (y_true == 1)).sum()
	FP = ((y_pred != y_true) & (y_true == 0)).sum()
	FN = ((y_pred != y_true) & (y_true == 1)).sum()
	TN = ((y_pred == y_true) & (y_true == 0)).sum()
	TPR = float(TP)/(TP + FN)
	FPR = float(FP)/(FP + TN)
	return TPR, FPR
​
def plot_roc(y_true, y_pred):
    tprs, fprs = [], []
    for thresh in sorted(y_pred):
        TPR, FPR = get_rates(y_test, probs, thresh=thresh)
        tprs.append(TPR)
        fprs.append(FPR)
	plt.clf()
    plt.plot(fprs, tprs)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve 'Roll Tide!'")
    plt.savefig('roc_plt.png')
​
if __name__ == '__main__':
    X, y = get_xy_arrays()
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    probs = sk_log_reg_probs(X_train, y_train, X_test, y_test)
    plot_roc(y_test, probs)
