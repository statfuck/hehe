# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

data = load_iris()
X = data.data 
y = data.target 
target_names = data.target_names  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)  
X_lda = lda.fit_transform(X_scaled, y)

lda_df = pd.DataFrame(X_lda, columns=['LDA1', 'LDA2'])
lda_df['Target'] = y

plt.figure(figsize=(8, 6))
sns.scatterplot(data=lda_df, x='LDA1', y='LDA2', hue='Target', palette='Set1', style='Target', s=100)
plt.title('LDA of Iris Dataset')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.legend(title='Class', labels=target_names)
plt.grid()
plt.show()

print("Linear Discriminant Analysis (LDA) Results")
print("--------------------------------------------------")
print("Explained Variance Ratio by LDA Components:")
for i, ratio in enumerate(lda.explained_variance_ratio_, start=1):
    print(f"  LDA{i}: {ratio:.4f}")
