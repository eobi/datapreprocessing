# datapreprocessing
Optimal machine learning data pre-processing skills

This doc is focused on machine learning using python.

#1 Data Imports
numpy -- Library used to handle mathematical computations
matplotlib.pyplot  -- Libray used for plotting charts in python
pandas -- Library used for handling data imports and management of the imported dataset

X -- used to denote independent variable
y -- used to denote dependent variable

#2 Handling missing data (several ways to handle missing data..this is just one of it)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])  #Spcify the column to update
X[:, 1:3] = imputer.transform(X[:, 1:3])   #Fill d missing data by the mean o the column

#3 Categorical Variable handling
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#4 Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#5 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)



The above steps are to be considered during the data preprocessing stage of your project.

#Best of Luck
