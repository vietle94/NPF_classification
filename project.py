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
from sklearn.preprocessing import StandardScaler
import seaborn as sns
# %%
# Load data
df = pd.read_csv('npf_train.csv')
df['date'] = pd.to_datetime(df.date)
# Overview data
df.head()
df.tail()
df.describe(include='all')
df['event'].value_counts()
# predicting IA might be a problem
# np.unique([i.split('.')[1] for i in df.columns if '.mean' in i])

# %%
# Make season from date
df['spring'] = [1 if month in [4, 5] else 0 for month in df.date.dt.month]
df['summer'] = [1 if month in [6, 7, 8, 9] else 0 for month in df.date.dt.month]
df['autumn'] = [1 if month in [10] else 0 for month in df.date.dt.month]
df['winter'] = [1 if month in [11, 12, 1, 2, 3] else 0 for month in df.date.dt.month]
df.drop(['date'], axis=1, inplace=True)

# %%
name = ['CO2168', 'Glob', 'H2O168', 'NO168', 'NOx168',
        'O3168', 'PTG', 'Pamb0', 'RHIRGA168',
        'RPAR', 'SO2168', 'SWS', 'T168', 'UV_B', 'CS']
name_ = ['HYY_META.' + i + '.mean' if 'CS' not in i else i + '.mean' for i in name]
name_ += ['spring', 'summer', 'winter', 'autumn', 'event']
name += ['spring', 'summer', 'winter', 'autumn', 'event']
df = df.loc[:, name_]
df.columns = name
df.columns
# %%
# Calculate correlation matrix
cor_matrix = df.corr().abs()

# Remove duplication
for i in range(cor_matrix.shape[0]):
    for j in range(i+1):
        cor_matrix.iloc[i, j] = np.nan
cor = cor_matrix.unstack()
cor = cor.sort_values(ascending=False)
cor[cor > 0.9]

# %%
fig, ax = plt.subplots(figsize=(18, 12))
sns.heatmap(cor_matrix, ax=ax, annot=True)
ax.set_title('Correlation matrix of all variables',
             size=22, weight='bold')
# Do some manual removal of columns if we want to

# %%
X = df.drop('event', axis=1)
y = df.event
# Encode y like this
# np.sort(y.unique())
# y = y.astype('category').cat.codes
y = [0 if i == 'nonevent' else 1 for i in y]
y = np.array(y)
y
# %%
# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

# %%
# Scale data first
scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)

#
pca = PCA()
pca.fit(X_train_scaled)

# %%
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(np.cumsum(pca.explained_variance_ratio_))
ax.set_title('Cummulative explained variance', size=22, weight='bold')
ax.set_xlabel('Number of PCs')

# %%
# K-fold on train data
K = 5    # number of folds/rounds/splits
kf = KFold(n_splits=K, shuffle=False)
kf = kf.split(X_train)
kf = list(kf)

# %%
test_acc_cv = np.zeros([5, 3])


def train_model(model_instance):
    model_instance = model_instance.fit(X_train_scaled[train_indices, :],
                                        y_train[train_indices])
    y_pred_val = model_instance.predict(X_train_scaled[test_indices, :])
    return accuracy_score(y_train[test_indices], y_pred_val)


# %%
for i, (train_indices, test_indices) in enumerate(kf):
    kneighbor = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    test_acc_cv[i, 0] = train_model(kneighbor)

    rf = RandomForestClassifier()
    test_acc_cv[i, 1] = train_model(rf)

    logis = LogisticRegression(max_iter=1000)
    test_acc_cv[i, 2] = train_model(logis)

acc_val = np.mean(test_acc_cv, axis=0)   # compute the mean of validation acc
acc_val

# %%
# Better way by using grid search to find best parameters
# across default cross validation with k=5
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

svm = GridSearchCV(SVC(), tuned_parameters)
svm.fit(X_train_scaled, y_train)
print('Best parameters found on cross-validation')
print(svm.best_params_)
print('Accuracy SVM')
print(f"Mean: {svm.cv_results_['mean_test_score']}, std: {svm.cv_results_['mean_test_score']}")

# %%
tuned_parameters = [{'C': [1, 10, 100]}, {'penalty': ['none']}]

logis = GridSearchCV(LogisticRegression(max_iter=1000), tuned_parameters)
logis.fit(X_train_scaled, y_train)
print('Best parameters found on cross-validation')
print(logis.best_params_)
print('Accuracy Logistic regression')
print(f"Mean: {logis.cv_results_['mean_test_score']}, std: {logis.cv_results_['mean_test_score']}")

# %%
tuned_parameters = [{'n_estimators': [10, 50, 100],
                     'max_depth': [5, 10, 20, None]}]

rf = GridSearchCV(RandomForestClassifier(), tuned_parameters)
rf.fit(X_train_scaled, y_train)
print('Best parameters found on cross-validation')
print(rf.best_params_)
print('Accuracy Random Forest')
print(f"Mean: {rf.cv_results_['mean_test_score']}, std: {rf.cv_results_['mean_test_score']}")

# %%
fig, ax = plt.subplots(figsize=(12, 9))
features = df.columns[:-1]
importances = rf.best_estimator_.feature_importances_
indices = np.argsort(importances)
ax.barh([features[i] for i in indices], importances[indices])
ax.set_title('Feature Importance', weight='bold', size=22)
ax.set_xlabel('Relative Importance')


# %%
tuned_parameters = [{'n_neighbors': [5, 10, 15]}]

kneighbor = GridSearchCV(KNeighborsClassifier(), tuned_parameters)
kneighbor.fit(X_train_scaled, y_train)
print('Best parameters found on cross-validation')
print(kneighbor.best_params_)
print('Accuracy K-neighbors')
print(
    f"Mean: {kneighbor.cv_results_['mean_test_score']}, std: {kneighbor.cv_results_['mean_test_score']}")
