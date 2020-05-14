import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# %%
# Load data
df = pd.read_csv('npf_train.csv')
df['date'] = pd.to_datetime(df.date)
# Overview data
df.head()
df.describe(include='all')

df['event'].value_counts()
# predicting IA might be a problem

# Drop useless columns
df.drop(['id', 'partlybad'], axis=1, inplace=True)

# Make season from date
df['spring'] = [1 if month in [2, 3, 4] else 0 for month in df.date.dt.month]
df['summer'] = [1 if month in [5, 6, 7] else 0 for month in df.date.dt.month]
df['autumn'] = [1 if month in [8, 9, 10] else 0 for month in df.date.dt.month]
df['winter'] = [1 if month in [11, 12, 1] else 0 for month in df.date.dt.month]
df.drop(['date'], axis=1, inplace=True)
# %%
# Calculate correlation matrix
cor_matrix = df.corr().abs()

# Remove duplication
for i in range(cor_matrix.shape[0]):
    for j in range(i+1):
        cor_matrix.iloc[i, j] = np.nan
cor = cor_matrix.unstack()
cor = cor.sort_values(ascending=False)

# highly correlated pairs
cor[cor > 0.9]

# Do some manual removal of columns

# %%
X = df.drop('event', axis=1)
y = df.event
# Encode y like this
np.sort(y.unique())
y = y.astype('category').cat.codes

# %%
pca = PCA()
X_pca = pca.fit_transform(X)

# %%
fig, ax = plt.subplots()
ax.plot(np.cumsum(pca.explained_variance_ratio_))

# %%
# Use first 10 components for prediction
# Split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X_pca[:, :10], y,
                                                    random_state=7)

# K-fold on train data
K = 5    # number of folds/rounds/splits
kf = KFold(n_splits=K, shuffle=False)
kf = kf.split(X_train)
kf = list(kf)

# %%
test_acc_cv = np.zeros(5, 3)


def train_model(model_instance):
    model_instance = model_instance.fit(X_train[train_indices, :], y_train[train_indices])
    y_pred_val = model_instance.predict(X_train[test_indices, :])
    return accuracy_score(y_train[test_indices], y_pred_val)


# %%
for i, (train_indices, test_indices) in enumerate(kf):
    kneighbor = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    test_acc_cv[i, 0] = train_model(kneighbor)

    rf = RandomForestClassifier()
    test_acc_cv[i, 1] = train_model(rf)

    logis = LogisticRegression()
    test_acc_cv[i, 2] = train_model(logis)

acc_val = np.mean(test_acc_cv, axis=0)   # compute the mean of validation acc

# %%
# Better way by using grid search to find best parameters
# across default cross validation with k=5
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

svm = GridSearchCV(SVC(), tuned_parameters)
svm.fit(X_train, y_train)
print('Best parameters found on cross-validation')
print(svm.best_params_)
print('Accuracy SVM')
print(f'Mean: {svm.cv_results_['mean_test_score']}, std: {svm.cv_results_['mean_test_score']}')

# %%
tuned_parameters = [{'penalty': ['l2', 'l1', 'elasticnet', 'none'],
                     'C': [1, 10, 100]}, {'penalty': ['none']}]

logis = GridSearchCV(LogisticRegression(), tuned_parameters)
logis.fit(X_train, y_train)
print('Best parameters found on cross-validation')
print(logis.best_params_)
print('Accuracy Logistic regression')
print(f'Mean: {logis.cv_results_['mean_test_score']}, std: {logis.cv_results_['mean_test_score']}')

# %%
tuned_parameters = [{'n_estimators': [10, 50, 100],
                     'max_depth': [5, 10, 20, None]}]

rf = GridSearchCV(RandomForestClassifier(), tuned_parameters)
rf.fit(X_train, y_train)
print('Best parameters found on cross-validation')
print(rf.best_params_)
print('Accuracy Logistic regression')
print(f'Mean: {rf.cv_results_['mean_test_score']}, std: {rf.cv_results_['mean_test_score']}')

# %%
tuned_parameters = [{'n_neighbors': [5, 10, 15]]


kneighbor = GridSearchCV(KNeighborsClassifier(), tuned_parameters)
kneighbor.fit(X_train, y_train)
print('Best parameters found on cross-validation')
print(kneighbor.best_params_)
print('Accuracy Logistic regression')
print(f'Mean: {kneighbor.cv_results_['mean_test_score']}, std: {kneighbor.cv_results_['mean_test_score']}')
