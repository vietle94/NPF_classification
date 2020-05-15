import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.decomposition import PCA

# %%
# Load data
df = pd.read_csv('npf_train.csv')
df['date'] = pd.to_datetime(df.date)
# Drop useless columns
df.drop(['id', 'partlybad'], axis=1, inplace=True)

# Make season from date
df['spring'] = [1 if month in [2, 3, 4] else 0 for month in df.date.dt.month]
df['summer'] = [1 if month in [5, 6, 7] else 0 for month in df.date.dt.month]
df['autumn'] = [1 if month in [8, 9, 10] else 0 for month in df.date.dt.month]
df['winter'] = [1 if month in [11, 12, 1] else 0 for month in df.date.dt.month]
df.drop(['date'], axis=1, inplace=True)

X = df.drop('event', axis=1)
y = df.event

# Encode y like this
np.sort(y.unique())
y = y.astype('category').cat.codes

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.75, random_state=42)

# %%
# PCA
# %%
pca = PCA()
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

tpot = TPOTClassifier(verbosity=2, random_state=42)
tpot.fit(X_train_pca, y_train)
print(tpot.score(X_test_pca, y_test))
tpot.export('tpot_project_pipeline.py')
