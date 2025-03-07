# Salary Prediction using Simple Linear Regression

This notebook implements a simple linear regression model to predict salaries based on years of experience.

## 1. Data Loading

We will load the dataset from the CSV file.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp

data_set = pd.read_csv("Salary_Data.csv")
data_set.describe().T
data_set.head()
```

## 2. Data Preprocessing

We will prepare the data for training the model.

```python
x = data_set.iloc[:, :-1].values   # Features
y = data_set.iloc[:, 1].values      # Target variable
```

## 3. Splitting the Dataset

We will split the dataset into training and test sets.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
```

## 4. Model Training

We will fit the simple linear regression model to the training dataset.

```python
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
```

## 5. Predictions

We will make predictions on both the training and test sets.

```python
y_pred = regressor.predict(x_test)
x_pred = regressor.predict(x_train)
```

## 6. Visualization

We will visualize the results of the model.

```python
mtp.scatter(x_train, y_train, color="green")
mtp.plot(x_train, x_pred, color="red")
mtp.title("Salary vs Experience (Training Dataset)")
mtp.xlabel("Years of Experience")
mtp.ylabel("Salary (In Rupees)")
mtp.grid()
mtp.show()
```

### Observations

In the above plot, we can see the real values observations in green dots and predicted values covered by the red regression line. The regression line shows a correlation between the dependent and independent variable.

```python
mtp.scatter(x_test, y_test, color="blue")
mtp.plot(x_train, x_pred, color="red")
mtp.title("Salary vs Experience (Test Dataset)")
mtp.xlabel("Years of Experience")
mtp.ylabel("Salary (In Rupees)")
mtp.grid()
mtp.show()
```

### Conclusion

The model appears to be a good fit for the training set and can make reasonable predictions on the test set.
