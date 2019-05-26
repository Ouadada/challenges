
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import time

from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import binarize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

sns.set(style='white', context='notebook', palette='deep')

RANDOM_SEED = 213
np.random.seed(RANDOM_SEED)

RANDOM_SEED = 213
np.random.seed(RANDOM_SEED)
train.head()

X = train.drop(["id", "target"], axis=1)
y = train["target"]

del train

#Transform target to -1 and 1 -> no effect on LogReg
lb = preprocessing.LabelBinarizer(neg_label=-1, pos_label=1, sparse_output=False)
lb.fit([0,1])
lb.classes_
#y = lb.transform(y).reshape(250)

g = sns.countplot(y)

# Checking for missing values
X.isnull().any().describe()
y.isnull().any()

# Descriptive statistics
print('Distributions of first 8 columns')
plt.figure(figsize=(26, 24))
for i, col in enumerate(list(X.columns)[0:8]):
    plt.subplot(7, 4, i + 1)
    plt.hist(X[col])
    plt.title(col)

sns.pairplot(X.iloc[:,0:5])

X[X.columns[:]].mean().plot('hist');
plt.title('Distribution of means of all columns');

X[X.columns[:]].std().plot('hist');
plt.title('Distribution of stds of all columns');

corr = X.iloc[:,0:80].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)


# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_SEED, stratify = y)


# Modeling

#display all the metrics available for the scoring. POssible to use user defined scorers
import sklearn.metrics
sorted(sklearn.metrics.SCORERS.keys())

from sklearn.base import BaseEstimator, TransformerMixin

class addStats(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mean_ = 0.
        self.std_ = 0.
    #pass
        
    def fit(self, X, y=None):
        self.mean_ = X.mean(axis = 1)
        self.std_ = X.std(axis = 1)
        return self
            
    def transform(self, X, y=None):
        self.mean_ = X.mean(axis = 1).reshape(-1,1)
        self.std_ = X.std(axis = 1).reshape(-1,1)
        X = np.append(X, self.mean_, 1)
        X = np.append(X, self.std_, 1)
        
        return X
        
    def fit_transform(self, X, y=None):
        X = self.fit(X, y).transform(X)
        return X
    
    
class addNoise(BaseEstimator, TransformerMixin):

    def __init__(self, noise_std):
        self.noise_std = noise_std
        
    def fit(self, X, y=None):
        return self
            
    def transform(self, X, y=None):
        X += np.random.normal(0, noise_std, X.shape)
        return X
        
    def fit_transform(self, X, y=None):
        X = self.fit(X, y).transform(X)
        return X
    
    
class smote_transformer(BaseEstimator, TransformerMixin):

    def __init__(self, ratio):
        self.ratio = ratio
        
    def fit(self, X, y):
        return self
            
    def transform(self, X, y):
        smote = SMOTE(ratio=self.ratio, n_jobs=-1)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res
        
    def fit_transform(self, X, y):
        X_res, y_res = self.fit(X, y).transform(X,y)
        return X_res, y_res


sm = smote_transformer(ratio='minority')
a, b = sm.fit_transform(X_test, y_test)
print(X_test.shape)
print(a.shape)
print(b.shape)
print(y_test.shape)


#smote = SMOTE(ratio='minority', n_jobs=-1)
NOISE_STD = 0.02

pipeline = Pipeline([('scale', StandardScaler()),
                     ('scaler', RobustScaler()),
                     ('shuffle', shuffle()),
                     ('addNoise', addNoise(noise_std=NOISE_STD)),
                     #('oversampling', smote),
                     #('oversampling', smote_transformer(ratio='minority')),
                     #('QuantileTransformer', QuantileTransformer()), Moins bon score que le standardScaler
                     #('Imputer', Imputer(missing_values='NaN', strategy='mean', axis=0)),
                     #('poly_features', PolynomialFeatures(2)), #Overfit à balle
                     #('addStats', addStats()),
                     #('feature_selection', SelectFromModel(LogisticRegression(C=0.1, penalty='l1',class_weight='balanced', random_state=RANDOM_SEED,
                     #                               solver='liblinear'))),
                     #('features_selection', RFE(LogisticRegression(C=0.1, penalty='l1',class_weight='balanced', random_state=RANDOM_SEED,
                     #                               solver='liblinear'))),
                     #('features_selection', RFE(SVC(kernel="linear", C=1))),
                     #('features_selection', SelectKBest(f_classif)),
                     #('LogisticRegression', LogisticRegression())
                     ('bagging', BaggingClassifier(LogisticRegression()))
])

#pipeline.set_params(LogisticRegression__penalty='l1',
#                    LogisticRegression__class_weight = 'balanced',
#                    LogisticRegression__random_state = RANDOM_SEED,
#                    LogisticRegression__solver = 'liblinear')
                    
pipeline.set_params(bagging__base_estimator__penalty='l1',
                    bagging__base_estimator__class_weight='balanced',
                    bagging__base_estimator__random_state=RANDOM_SEED,
                    bagging__base_estimator__solver='liblinear')

scoring = {'AUC': 'roc_auc', 
           'Accuracy' : make_scorer(accuracy_score)} #, 'AUC': 'roc_auc'

n_folds = 10


#param_grid = {'LogisticRegression__C': 10**np.linspace(-1.5,-0.5,10)[:]}
              #'features_selection__k': np.linspace(1,300,20)[:].astype(int)}
param_grid = {'bagging__base_estimator__C':10**np.linspace(-1,-0.5,5)[:],
              'bagging__n_estimators':np.linspace(6, 10, 5).astype(int),
              'bagging__max_samples':np.linspace(0.5, 1.0, 5),
              'bagging__max_features':np.linspace(0.5, 1.0, 5)}


clf = GridSearchCV(pipeline, 
                                 param_grid = param_grid,
                                 cv = StratifiedKFold(n_splits = n_folds),
                                 scoring = scoring, 
                                 refit = 'AUC',
                                 return_train_score = True,
                                 n_jobs = 6,
                                 verbose = 2)
#To get the name of the parameters in the pipeline :
#for param in clf.get_params().keys():
#    print(param)


clf.fit(X_train, y_train)


# PIck best hyper parameter
results = clf.cv_results_
for scorer in zip(sorted(scoring)):
    print()
    print(scorer)
    print('Best parameter(for scorer in refit parameter): ',clf.best_params_)
    print('Best score (for scorer in refit parameter): ',clf.best_score_)
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true,y_pred))
    print()
    print("Grid scores on development set:")

    means = results['mean_test_%s' % scorer]
    stds = results['std_test_%s' % scorer]
    for mean, std, params in zip(means, stds, results['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
print()


print()

print('Best estimator: ',clf.best_estimator_)

results = clf.cv_results_
print(results)
#Récupère la valeur des paramètres et change le type du nparray qui pose un problème dans matplotlib
params = results['param_bagging__base_estimator__C'].data.astype('str').astype('float')

fig, ax = plt.subplots(figsize=(13, 13))
ax.set_title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)
ax.set_xlabel("min_samples_split")
ax.set_ylabel("CV score +/- Std")


for scorer, color in zip(sorted(scoring), ['g', 'k']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        
        ax.fill_between(params, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        
        ax.plot(params, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7, 
                label='%s (%s)' % (scorer, sample))
        
    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]
    best_param = params[best_index]
        
    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([params[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score and the best parameter for that scorer
    ax.annotate("%0.2f" % best_score,
                (best_param, best_score + 0.005))
    
    ax.annotate("%0.2f" % best_param,
                (best_param, 0.005))


plt.legend(loc="best")
plt.grid(False)
plt.show()

y_score = clf.best_estimator_.predict_proba(X_test)[:, 1]

threshold = 0.5
y_score_class = binarize(y_score.reshape(-1, 1), threshold)
confusion_mtx = confusion_matrix(y_true, y_score_class)

sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap="Blues")

# histogram of predicted probabilities

# 8 bins
plt.hist(y_score, bins=8)

# x-axis limit from 0 to 1
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability')
plt.ylabel('Frequency')


# Choix du meilleur seuil -----------------------------------
#Coefficients du meilleur modèle obtenu par CV
best_estimator_coef = clf.best_estimator_.named_steps['LogisticRegression'].coef_
best_estimator_coef.shape

#pd.DataFrame(dict(zip(X.columns,best_estimator_coef[0]),index=[0])).T

def plot_precision_recall_curve(y_test, y_score):
    
    plt.figure(figsize=(8, 8))
    
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', step = 'post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    
def plot_precision_recall_accuracy_vs_threshold(y_test, y_score):

    plt.figure(figsize=(10, 10))
    plt.title("Precision, Recall and Accuracy Scores as a function of the decision threshold")
     
    #Plot accuracy
    thresholds = np.linspace(0, 1, 41)
    #precision, recall, thresholds = precision_recall_curve(y_test, y_score)

    accuracy = []
    for t in thresholds:
        y_score_class = binarize(y_score.reshape(-1, 1), t)
        accuracy.append(accuracy_score(y_test, y_score_class))
        
    best_accuracy = max(accuracy)
    best_threshold = thresholds[accuracy.index(best_accuracy)]
    plt.plot(thresholds, accuracy, linewidth=2, label='Accuracy (max = %0.3f)' % best_accuracy )
    
    #Plot recall and precision
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    plt.plot(thresholds, precision[:-1], "b--", label="Precision")
    plt.plot(thresholds, recall[:-1], "g-", label="Recall")
    
    # Plot a dotted vertical line at the best score for that scorer marked by x
    plt.plot([best_threshold, ] * 2, [0, best_accuracy],
            linestyle=':', color=color, marker='.', markeredgewidth=3, ms=8)

    # Annotate the best parameter for accuracy    
    plt.annotate("%0.2f" % best_threshold,
                (best_threshold, 0.005))
    
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    
def best_accuracy_best_threshold(y_test, y_score):
    thresholds = np.linspace(0, 1, 41)
    #precision, recall, thresholds = precision_recall_curve(y_test, y_score)

    accuracy = []
    for t in thresholds:
        y_score_class = binarize(y_score.reshape(-1, 1), t)
        accuracy.append(accuracy_score(y_test, y_score_class))
        
    best_accuracy = max(accuracy)
    best_threshold = thresholds[accuracy.index(best_accuracy)]

    return best_accuracy, best_threshold
    
def plot_roc_curve(y_test, y_score, label=None):
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 10))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label='AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')


best_accuracy, best_threshold = best_accuracy_best_threshold(y_test, y_score)
print(best_accuracy)
print(best_threshold)

plot_precision_recall_curve(y_test, y_score)

plot_precision_recall_accuracy_vs_threshold(y_test, y_score)

plot_roc_curve(y_test, y_score, label=None)


# LEarning best model on all data
#smote = SMOTE(ratio='minority', n_jobs=-1)
NOISE_STD = 0.02

pipeline = Pipeline([('scaler', RobustScaler()),
                     ('shuffle', shuffle()),
                     ('addNoise', addNoise(noise_std=NOISE_STD)),
                     ('bagging', BaggingClassifier(LogisticRegression()))
])

best_C = clf.best_params_['bagging__base_estimator__C']
best_n = clf.best_params_['bagging__n_estimators']
best_ms = clf.best_params_['bagging__max_samples']
best_mf = clf.best_params_['bagging__max_features']

print(best_C, best_n, best_ms, best_mf)

pipeline.set_params(bagging__base_estimator__penalty='l1',
                    bagging__base_estimator__class_weight='balanced',
                    bagging__base_estimator__random_state=RANDOM_SEED,
                    bagging__base_estimator__solver='liblinear',
                   bagging__base_estimator__C=best_C,
                   bagging__n_estimators=best_n,
                   bagging__max_samples=best_ms,
                   bagging__max_features=best_mf)


pipeline = Pipeline([('scaler', StandardScaler()),
                    #('features_selection', SelectKBest(f_classif)),
                     #('addStats', addStats()),
                     ('LogisticRegression', LogisticRegression())
])

best_C = clf.best_params_['LogisticRegression__C']
#best_k = clf.best_params_['features_selection__k']
print(best_C)


pipeline.set_params(LogisticRegression__C = best_C,
                    LogisticRegression__penalty='l2',
                    LogisticRegression__class_weight = 'balanced',
                    LogisticRegression__random_state = RANDOM_SEED,
                    LogisticRegression__solver = 'liblinear')


pipeline.fit(X, y)

test_id = test[["id"]]
test = test.drop(["id"], axis=1)

# predict results
y_test_pred = pipeline.predict_proba(test)[:,1]
results = pd.Series(y_test_pred,name="target")

submission = pd.concat([test_id,results],axis = 1)

submission.to_csv(f"dno_lasso_{time()}.csv",index=False)