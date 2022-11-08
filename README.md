# Analysis tools for Machine learning projects

## 1. Usage
```bash
$ pip install analysis-tools
```

## 2. Tutorial
[examples/titanic/eda.ipynb](https://github.com/djy-git/analysis-tools/blob/main/examples/titanic/eda.ipynb)를 참고

```python
from analysis_tools import eda, metrics

data   = pd.DataFrame(..)
target = 'survived'

num_features       = ['age', 'sibsp', 'parch', 'fare']
cat_features       = data.columns.drop(num_features)
data[num_features] = data[num_features].astype('float32')
data[cat_features] = data[cat_features].astype('string')

eda.plot_missing_value(data)
eda.plot_features(data)
eda.plot_features_target(data, target)
eda.plot_corr(data.corr())
metrics.get_feature_importance(data, target)
```

![](https://github.com/djy-git/analysis-tools/blob/main/examples/titanic/visualization/Missing%20value_1.png?raw=true)
![](https://github.com/djy-git/analysis-tools/blob/main/examples/titanic/visualization/Features_1.png?raw=true)
![](https://github.com/djy-git/analysis-tools/blob/main/examples/titanic/visualization/Features%20vs%20Target_1.png?raw=true)
![](https://github.com/djy-git/analysis-tools/blob/main/examples/titanic/visualization/Correlation%20matrix_1.png?raw=true)
![](https://github.com/djy-git/analysis-tools/blob/main/examples/titanic/visualization/Feature%20importance_1.png?raw=true)
