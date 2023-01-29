import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics

# ================================================================================== #

# Load data                                                    (1)
data = pd.read_csv('assignment2_bmd.csv')

# Deal with missing values                                     (2)
print(data.isna().sum())
print("\n there is no NULL values in the data")

# strings encoding                                             (3)
columns = ['sex', 'fracture', 'medication']


def feature_encoder(x, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(x[c].values))
        x[c] = lbl.transform(list(x[c].values))
    return data


feature_encoder(data, columns)

# ================================================================================== #

# Feature Selection                                              (4)
X = data.iloc[:, 1:40]
Y = data['bmd']

# Get the correlation between the features                       (5)
corr = data.corr()
# Top 50% Correlation training features with the Value           (6)
top_feature = corr.index[abs(corr['bmd']) > 0.5]
# Correlation plot        (7)
plt.subplots(figsize=(8, 6))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)
X = X[top_feature]

# ================================================================================== #

# using the train test split function                              (7)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=100, test_size=0.20, shuffle=True)


# Apply Multiple Linear Regression on the selected features         (8)
cls = linear_model.LinearRegression()
cls.fit(X_train, y_train)
prediction = cls.predict(X_test)
MMSE = metrics.mean_squared_error(np.asarray(y_test), prediction)
print(' \n Multiple Mean Square Error : ', MMSE)

# ================================================================================== #

poly_features = PolynomialFeatures(degree=3)
# transforms the existing features to higher degree features.        (9)
X_train_poly = poly_features.fit_transform(X_train)


# fit the transformed features to Linear Regression                  (10)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set                                     (11)
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set                                         (12)
prediction = poly_model.predict(
            poly_features.fit_transform(X_test))
PMSE = metrics.mean_squared_error(y_test, prediction)
print('\n Polynomial Mean Square Error : ', PMSE)

if MMSE > PMSE:
    print("\n *$*" + " Polynomial Mean Square Error is LESS THAN Multiple Mean Square Error " + "*$*")
else:
    print("\n *$*" + " Multiple Mean Square Error is LESS THAN Polynomial Mean Square Error " + "*$*")
