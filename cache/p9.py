import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)
x = np.random.rand(100) * 100
z= np.random.normal(0, 25, 100)
y = 2.5 * x +z

data = pd.DataFrame({'X': x, 'Y': y})

pearson_corr = data.corr(method='pearson')
spearman_corr, _ = spearmanr(data['X'], data['Y'])
print(_)

X = data['X'].values.reshape(-1, 1)
Y = data['Y'].values
model = LinearRegression().fit(X, Y)
Y_pred = model.predict(X)
regression_coeff = model.coef_[0] 
regression_intercept = model.intercept_ 
mse = mean_squared_error(Y, Y_pred)

print("Pearson Correlation Coefficient Matrix:")
print(pearson_corr)
print("\nSpearman Rank Correlation Coefficient:", spearman_corr)
print("\nLinear Regression Equation: Y = {:.2f}X + {:.2f}".format(regression_coeff, regression_intercept))
print("Mean Squared Error (MSE):", mse)

plt.figure(figsize=(8, 6))
plt.scatter(data['X'], data['Y'], color='blue', label='Data Points')
plt.plot(data['X'], Y_pred, color='red', label='Regression Line')
plt.title('X-Y Scatter Plot with Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Correlation Matrix')
plt.show()
