from analysis_tools.common import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OrdinalEncoder


## Missing value
def plot_missing_value(data, figsize=figsize(3, 1), dir_path=PATH.RESULT, plot=PLOT):
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    with FigProcessor(fig, dir_path, plot, "Missing values"):
        msno.matrix(data, ax=axes[0])
        ms = data.isnull().sum()
        sns.barplot(ms.index, ms, ax=axes[1])
        axes[1].bar_label(axes[1].containers[0])
        axes[1].set_xticklabels([])


## Feature exploration
### Single feature
def _plot_on_ax(plot_fn, ax, figsize, dir_path, plot, title):
    if ax is not None:
        plot_fn(ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        with FigProcessor(fig, dir_path, plot, title):
            plot_fn(ax)
def plot_num_feature(data_f, bins=BINS, ax=None, figsize=figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    def plot_fn(ax):
        sns.histplot(data_f, bins=bins, ax=ax, kde=True, stat='density')
        ax.set_xlabel(None)
    _plot_on_ax(plot_fn, ax, figsize, dir_path, plot, data_f.name)
def plot_cat_feature(data_f, ax=None, figsize=figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    def plot_fn(ax):
        density = data_f.value_counts().sort_index() / len(data_f)
        sns.barplot(density.index, density.values, ax=ax)
    _plot_on_ax(plot_fn, ax, figsize, dir_path, plot, data_f.name)


### Multiple features
def plot_features(data, bins=BINS, n_cols=5, figsize=figsize(3, 3), dir_path=PATH.RESULT, plot=PLOT):
    n_features = len(data.columns)
    n_rows     = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    with FigProcessor(fig, dir_path, plot, "Feature distribution"):
        for ax, f in zip(axes.flat, data):
            data_f_notnull = data[f].dropna()
            ax.set_title(f)
            if data_f_notnull.nunique() > bins:
                ## Numerical feature or categorical feature
                try:
                    ax.hist(data_f_notnull, bins=bins, density=True, color='olive', alpha=0.5)
                except Exception as e:
                    print(f"[{f}]: {e}")
            else:
                ## Categorical feature
                cnts = data[f].value_counts().sort_index() / len(data[f])
                ax.bar(cnts.index, cnts.values, width=0.5, alpha=0.5)
                ax.set_xticks(cnts.index)
def plot_num_num_features(data, f1, f2, bins=BINS, ax=None, figsize=figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    def plot_fn(ax):
        sns.histplot(x=data[f1], y=data[f2], bins=bins, ax=ax)
        ax.set_xlabel(None);  ax.set_ylabel(None)
    _plot_on_ax(plot_fn, ax, figsize, dir_path, plot, f"{f1} vs {f2}")
def plot_num_cat_features(data, f1, f2, n_classes=N_CLASSES_PLOT, ax=None, figsize=figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    def plot_fn(ax):
        selected_classes = data[f2].value_counts().index[:n_classes]
        idxs_selected    = data[f2][data[f2].isin(selected_classes)].index
        data_f1, data_f2 = data[f1][idxs_selected], data[f2][idxs_selected]
        sns.violinplot(x=data_f1, y=data_f2, ax=ax, orient='h', order=reversed(sorted(selected_classes)), cut=0)
        ax.set_xlabel(None);  ax.set_ylabel(None)
    _plot_on_ax(plot_fn, ax, figsize, dir_path, plot, f"{f1} vs {f2}")
def plot_cat_num_features(data, f1, f2, n_classes=N_CLASSES_PLOT, ax=None, figsize=figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    def plot_fn(ax):
        selected_classes = data[f1].value_counts().index[:n_classes]
        idxs_selected    = data[f1][data[f1].isin(selected_classes)].index
        data_f1, data_f2 = data[f1][idxs_selected], data[f2][idxs_selected]
        sns.violinplot(x=data_f1, y=data_f2, ax=ax, orient='v', order=sorted(selected_classes), cut=0)
        ax.set_xlabel(None);  ax.set_ylabel(None)
    _plot_on_ax(plot_fn, ax, figsize, dir_path, plot, f"{f1} vs {f2}")
def plot_cat_cat_features(data, f1, f2, n_classes=N_CLASSES_PLOT, ax=None, figsize=figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    def plot_fn(ax):
        ratio = pd.crosstab(data[f2], data[f1])
        ratio /= ratio.sum(axis=0)
        ratio.sort_index(inplace=True, ascending=False)  # sort by index
        ratio = ratio[sorted(ratio)]    # sort by column
        ratio = ratio.iloc[:n_classes, :n_classes]
        sns.heatmap(ratio, ax=ax, annot=True, fmt=".2f", cmap=sns.light_palette('firebrick', as_cmap=True), cbar=False)
        ax.set_xlabel(None);  ax.set_ylabel(None)
    _plot_on_ax(plot_fn, ax, figsize, dir_path, plot, f"{f1} vs {f2}")
def plot_features_target(data, target, n_cols=5, figsize=figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    num_features = data.select_dtypes('number').columns
    target_type  = 'num' if target in num_features else 'cat'
    n_rows       = int(np.ceil(len(data.columns)/n_cols))
    fig, axes    = plt.subplots(n_rows, n_cols, figsize=figsize)
    with FigProcessor(fig, dir_path, plot, "Features vs Target"):
        for ax, f in zip(axes.flat, data.columns.drop(target)):
            ax.set_title(f"{f} vs {target}")
            f_type = 'num' if f in num_features else 'cat'
            eval(f"plot_{f_type}_{target_type}_features")(data, f, target, ax=ax)
def plot_corr(data, figsize=figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=figsize)
    with FigProcessor(fig, dir_path, plot, "Correlation matrix"):
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt=".2f", cmap='coolwarm', cbar=False)


## Feature importance
def get_feature_importance(data, target, problem='classification', bins=BINS, figsize=figsize(1, 1), dir_path=PATH.RESULT, plot=PLOT):
    ## 1. Split data into X, y
    data               = data.dropna()
    cat_features       = data.select_dtypes('category').columns
    data[cat_features] = data[cat_features].apply(OrdinalEncoder().fit_transform)
    X, y = data.drop(columns=target), data[target]

    ## 2. Model
    model = RandomForestClassifier(n_jobs=-1) if problem == 'classification' else RandomForestRegressor(n_jobs=-1)
    model.fit(X, y)

    ## 3. Get feature importance
    MDI_importance  = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    perm_importance = pd.Series(permutation_importance(model, X, y).importances_mean, index=X.columns).sort_values(ascending=False)

    ## 4. Mean importance
    fi1     = pd.Series(range(len(MDI_importance)), index=MDI_importance.index)
    fi2     = pd.Series(range(len(perm_importance)), index=perm_importance.index)
    mean_fi = ((fi1 + fi2)/2).sort_values()

    ## 5. Plot
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    with FigProcessor(fig, dir_path, plot, "Feature importance"):
        for ax, data, ylabel, title in zip(axes,
                                          [MDI_importance.head(bins), perm_importance.head(bins), mean_fi.head(bins)],
                                          ["Mean decrease in impurity", "Mean accuracy decrease", "Mean rank"],
                                          ["Feature importance using MDI", "Feature importance using permutation on full model", "Feature importance using MDI, permutation on full model"]):
            sns.barplot(data.index, data, ax=ax)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=30)

    # return MDI_importance, perm_importance, mean_fi
    return mean_fi
