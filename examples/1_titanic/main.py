from analysis_tools.common import *
from analysis_tools.eda import *
from sklearn.datasets import fetch_openml


X, y = fetch_openml('titanic', version=1, return_X_y=True)
data = pd.concat([X, y], axis=1)

plot_corr(data)