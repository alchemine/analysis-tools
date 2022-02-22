# Analysis tools for Machine learning projects

## 1. Install
```bash
$ pip install analysis-tools
```

## 2. Tutorial
```python
from analysis_tools.common import *
from analysis_tools.eda import *
from sklearn.datasets import fetch_openml


data   = fetch_openml('titanic', version=1, as_frame=True, data_home='.')
target = 'survived'

num_features       = ['age', 'sibsp', 'parch', 'fare']
cat_features       = data.columns.drop(num_features)
data[num_features] = data[num_features].astype(np.float32)
data[cat_features] = data[cat_features].astype('category')

plot_missing_value(data)
plot_features(data)
plot_corr(data)
plot_features_target(data, target)
```


자세한 내용은 [examples/titanic/main.ipynb](https://github.com/djy-git/analysis-tools/blob/main/examples/titanic/main.ipynb) 를 참고
