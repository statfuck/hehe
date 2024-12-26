import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
np.random.seed(10)
data_one_way = pd.DataFrame({
    "Group": np.repeat(['A', 'B', 'C'], 10),
    "Response": np.concatenate([
        np.random.normal(50, 5,10),
        np.random.normal(55, 5,10),
        np.random.normal(60, 5,10)
    ])
})
print(data_one_way)

data_two_way = pd.DataFrame({
    "Factor1": np.repeat(['Low', 'Medium', 'High'], 6),
    "Factor2": np.tile(['Type1', 'Type2'], 9),
    "Response": np.concatenate([
        np.random.normal(50, 5, 6),
        np.random.normal(55, 5, 6),
        np.random.normal(60, 5, 6)
    ])
})
print(data_two_way)
def one_way_anova(data, groups, response): 
    grouped_data = [group[response].values for _, group in data.groupby(groups)]
    f_stat, p_value = f_oneway(*grouped_data)
    print("\nOne-way ANOVA Results:")
    print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Reject the null hypothesis: Significant difference among group means.")
    else:
        print("Fail to reject the null hypothesis: No significant difference among group means.")

def two_way_anova(data,  factor1, factor2,response):
 
    formula = f"{response} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
    model = ols(formula, data).fit()
    print("\nTwo-way ANOVA Results:")
    print(sm.stats.anova_lm(model, typ=2))
    
   
one_way_anova(data_one_way, groups="Group", response="Response")
two_way_anova(data_two_way,  factor1="Factor1", factor2="Factor2",response="Response",)