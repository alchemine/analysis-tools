"""Exploratory Data Analysis tools

This module contains functions and classes for exploratory data analysis.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>
from analysis_tools.common import *


# Utility function
def plot_on_ax(plot_fn, suptitle, ax=None, label=None,                              dir_path=None, figsize=None, show_plot=None):
    """Plot on a single axis if ax is not None. Otherwise, plot on a new figure.

    Parameters
    ----------
    plot_fn : function
        Function to plot.

    suptitle : str
        Title of the plot.

    ax : matplotlib.axes.Axes or list of matplotlib.axes.Axes
        Axis to plot.

    label : bool
        Whether to show label.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.
    """
    if ax is not None:
        plot_fn(ax)
        context = contextlib.nullcontext()
    else:
        fig, ax = plt.subplots(figsize=PLOT_PARAMS.get(figsize, 'figsize'))
        context = FigProcessor(fig, dir_path, show_plot, suptitle)
    with context:
        plot_fn(ax)
        if label is False:
            ax.set_xlabel(None);  ax.set_ylabel(None)


# Missing value
def plot_missing_value(data,                                                        dir_path=None, figsize=None, show_plot=None):
    """Plot counts of missing values of each feature.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, None], 'b': [1, None, 3, 4, 5], 'c': [None, 2, 3, 4, 5]})
    >>> eda.plot_missing_value(data, dir_path='.')
    """
    fig, axes = plt.subplots(2, 1, figsize=PLOT_PARAMS.get(figsize, 'figsize'))
    with FigProcessor(fig, dir_path, show_plot, "Missing value"):
        msno.matrix(data, ax=axes[0])
        ms = data.isnull().sum()
        sns.barplot(ms.index, ms, ax=axes[1])
        axes[1].bar_label(axes[1].containers[0])
        axes[1].set_xticklabels([])


# Features
def plot_features(data1, data2=None, bins=None, n_cols=None, title=None,            dir_path=None, figsize=None, show_plot=None):
    """Plot histogram or bar for all features.

    Parameters
    ----------
    data1 : pandas.DataFrame
        DataFrame to be analyzed.

    data2 : pandas.DataFrame
        DataFrame to be analyzed.

    bins : int
        Number of bins.

    n_cols : int
        Number of columns.

    title : str
        Title

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_features(data, dir_path='.')
    """
    num_features = data1.select_dtypes('number').columns
    bins   = PLOT_PARAMS.get(bins, 'bins')
    n_cols = PLOT_PARAMS.get(n_cols, 'n_cols')
    n_features = len(data1.columns)
    n_rows     = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=PLOT_PARAMS.get(figsize, 'figsize'))
    with FigProcessor(fig, dir_path, show_plot, title if title else "Features"):
        for ax in axes.flat[n_features:]:
            ax.axis('off')
        datas  = [data1] if data2 is None else [data1, data2]
        colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
        for data, color in zip(datas, colors):
            for ax, f in zip(axes.flat, data1):
                ax.set_title(f)
                data_f_notnull = data[f].dropna()
                if len(data_f_notnull) == 0:
                    continue
                if f in num_features:
                    ax.hist(data_f_notnull, bins=bins, density=True, color=color, alpha=0.5)
                    '''
                    # easy to understand but, too slow
                    sns.histplot(data_f_notnull, stat='probability', color=color, alpha=0.5, ax=ax)  
                    ax.set_ylabel(None)
                    '''
                else:
                    # normalize including NaN, sample 100 unique values
                    density = data_f_notnull.value_counts(normalize=True).sort_index()[:100]
                    sns.barplot(density.index, density.values, alpha=0.5, color=color, ax=ax)
                    if data_f_notnull.nunique() > 20:
                        ax.set_xticklabels([])
def plot_features_target(data, target, n_cols=None, target_type='auto', title=None, dir_path=None, figsize=None, show_plot=None):
    """Plot features vs target.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.
        Dtypes of numerical features should be `number`(`numpy.float32` is recommended).
        Dtypes of categorical features should be `string` or `category`.

    target : str
        Target feature.

    dir_path : str
        Directory path to save the plot.

    n_cols : int
        Number of columns.

    target_type : str
        Type of target.
        target_type should be 'auto' or 'num', 'cat'.
        target_type is inferred automatically when 'auto' is set.

    title : str
        Title

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 1, 2], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': ['a10', 'b22', 'c11', 'a10', 'b22']})
    >>> num_features = ['a']
    >>> cat_features = data.columns.drop(num_features)
    >>> data[num_features] = data[num_features].astype('float32')
    >>> data[cat_features] = data[cat_features].astype('string')
    >>> eda.plot_features_target(data, 'a', dir_path='.')
    """
    num_features = data.select_dtypes('number').columns
    if target_type == 'auto':
        target_type = 'num' if target in num_features else 'cat'
    n_features = len(data.columns)-1  # -1: except target
    n_cols     = PLOT_PARAMS.get(n_cols, 'n_cols')
    n_rows     = int(np.ceil(n_features/n_cols))
    fig, axes  = plt.subplots(n_rows, n_cols, figsize=PLOT_PARAMS.get(figsize, 'figsize'))
    with FigProcessor(fig, dir_path, show_plot, title if title else f"Features vs Target({target})"):
        for ax, f in zip(axes.flat, data.columns.drop(target)):
            ax.set_title(f"{f} vs {target}")
            f_type = 'num' if f in num_features else 'cat'
            eval(f"plot_{f_type}_{target_type}_features")(data, f, target, ax=ax)
def plot_corr(corr1, corr2=None, annot=True, mask=True,                             dir_path=None, figsize=None, show_plot=None):
    """Plot correlation matrix.

    Parameters
    ----------
    corr1 : pandas.DataFrame
        Correlation matrix.

    corr2 : pandas.DataFrame
        Correlation matrix.

    annot : bool
        Whether to show values.

    mask : bool
        Whether to show only lower triangular matrix.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 1, 2], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [10, 20, 30, 10, 20]})
    >>> eda.plot_corr(corr, dir_path='.')
    """
    if corr2 is None:
        def plot_fn(ax):
            mask_mat = np.eye(len(corr1), dtype=bool)
            mask_mat[np.triu_indices_from(mask_mat, k=1)] = mask
            sns.heatmap(corr1, mask=mask_mat, ax=ax, annot=annot, fmt=".2f", cmap='coolwarm', center=0)
        plot_on_ax(plot_fn, "Correlation matrix", None, None, dir_path, figsize, show_plot)
    else:
        figsize = PLOT_PARAMS.get(figsize, 'figsize')
        fig, axes = plt.subplots(1, 2, figsize=(2*figsize[0], figsize[1]))
        with FigProcessor(fig, dir_path, show_plot, "Correlation matrix"):
            for ax, corr in zip(axes.flat, (corr1, corr2)):
                mask_mat = np.eye(len(corr), dtype=bool)
                mask_mat[np.triu_indices_from(mask_mat, k=1)] = mask
                sns.heatmap(corr, mask=mask_mat, ax=ax, annot=annot, fmt=".2f", cmap='coolwarm', center=0)


def plot_num_feature(data_f, bins=None, ax=None,                                    dir_path=None, figsize=None, show_plot=None):
    """Plot histogram of a numeric feature.

    Parameters
    ----------
    data_f : pandas.Series
        Series to be analyzed.

    bins : int
        Number of bins.

    ax : matplotlib.axes.Axes
        Axis to plot.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_num_feature(data['a'], dir_path='.')
    """
    def plot_fn(ax):
        sns.histplot(data_f, bins=PLOT_PARAMS.get(bins, 'bins'), ax=ax, kde=True, stat='density')
        ax.set_xlabel(None)
    plot_on_ax(plot_fn, data_f.name, ax, None, dir_path, figsize, show_plot)
def plot_cat_feature(data_f, ax=None,                                               dir_path=None, figsize=None, show_plot=None):
    """Plot bar of a categorical feature.

    Parameters
    ----------
    data_f : pandas.Series
        Series to be analyzed.

    ax : matplotlib.axes.Axes
        Axis to plot.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_cat_feature(data['b'], dir_path='.')
    """
    def plot_fn(ax):
        density = data_f.value_counts(normalize=True).sort_index()
        sns.barplot(density.index, density.values, ax=ax)
    plot_on_ax(plot_fn, data_f.name, ax, None, dir_path, figsize, show_plot)
def plot_num_num_features(data, f1, f2, bins=None, ax=None, label=False,            dir_path=None, figsize=None, show_plot=None):
    """Plot histogram of two numeric features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    f1 : str
        First numerical feature.

    f2 : str
        Second numerical feature.

    bins : int
        Number of bins.

    ax : matplotlib.axes.Axes
        Axis to plot.

    label : bool
        Whether to show x, y label.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_num_num_features(data, 'a', 'c', dir_path='.')
    """
    def plot_fn(ax):
        if len(data[f1]) > 10000:
            sns.histplot(x=data[f1], y=data[f2], bins=PLOT_PARAMS.get(bins, 'bins'), ax=ax)  # fast
        else:
            sns.scatterplot(x=data[f1], y=data[f2], alpha=0.5, ax=ax)  # slow
    plot_on_ax(plot_fn, f"{f1} vs {f2}", ax, label, dir_path, figsize, show_plot)
def plot_num_cat_features(data, f1, f2, n_classes=None, ax=None, label=False,       dir_path=None, figsize=None, show_plot=None):
    """Plot violinplot of categorical, numerical features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    f1 : str
        First numerical feature.

    f2 : str
        Second categorical feature.

    n_classes : int
        Number of classes to plot.

    ax : matplotlib.axes.Axes
        Axis to plot.

    label : bool
        Whether to show x, y label.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_num_cat_features(data, 'a', 'b', dir_path='.')
    """
    def plot_fn(ax):
        selected_classes = data[f2].value_counts().index[:PLOT_PARAMS.get(n_classes, 'n_classes')]
        idxs_selected    = data[f2][data[f2].isin(selected_classes)].index
        data_f1, data_f2 = data[f1][idxs_selected], data[f2][idxs_selected]
        sns.violinplot(x=data_f1, y=data_f2, ax=ax, orient='h', order=reversed(sorted(selected_classes)), cut=0, scale='width')
    plot_on_ax(plot_fn, f"{f1} vs {f2}", ax, label, dir_path, figsize, show_plot)
def plot_cat_num_features(data, f1, f2, n_classes=None, ax=None, label=False,       dir_path=None, figsize=None, show_plot=None):
    """Plot violinplot of categorical, numerical features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    f1 : str
        First categorical feature.

    f2 : str
        Second numerical feature.

    n_classes : int
        Number of classes to plot.

    ax : matplotlib.axes.Axes
        Axis to plot.

    label : bool
        Whether to show x, y label.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_cat_num_features(data, 'b', 'a', dir_path='.')
    """
    def plot_fn(ax):
        selected_classes = data[f1].value_counts().index[:PLOT_PARAMS.get(n_classes, 'n_classes')]
        idxs_selected    = data[f1][data[f1].isin(selected_classes)].index
        data_f1, data_f2 = data[f1][idxs_selected], data[f2][idxs_selected]
        sns.violinplot(x=data_f1, y=data_f2, ax=ax, orient='v', order=sorted(selected_classes), cut=0, scale='width')
    plot_on_ax(plot_fn, f"{f1} vs {f2}", ax, label, dir_path, figsize, show_plot)
def plot_cat_cat_features(data, f1, f2, n_classes=None, ax=None, label=False,       dir_path=None, figsize=None, show_plot=None):
    """Plot heatmap of two categorical features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    f1 : str
        First categorical feature.

    f2 : str
        Second categorical feature.

    n_classes : int
        Number of classes to plot.

    ax : matplotlib.axes.Axes
        Axis to plot.

    label : bool
        Whether to show x, y label.

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': ['a10', 'b22', 'c11', 'a10', 'b22']})
    >>> eda.plot_cat_cat_features(data, 'b', 'c', dir_path='.')
    """
    def plot_fn(ax):
        ratio = pd.crosstab(data[f2], data[f1], normalize='columns')
        ratio.sort_index(inplace=True, ascending=False)  # sort by index
        ratio = ratio[sorted(ratio)]                     # sort by column
        ratio = ratio.iloc[:PLOT_PARAMS.get(n_classes, 'n_classes'), :PLOT_PARAMS.get(n_classes, 'n_classes')]
        sns.heatmap(ratio, ax=ax, annot=True, fmt=".2f", cmap=sns.light_palette('firebrick', as_cmap=True), cbar=False)
    plot_on_ax(plot_fn, f"{f1} vs {f2}", ax, label, dir_path, figsize, show_plot)
def plot_pair(data1, data2=None, sample=1000, subplot=True,                         dir_path=None, figsize=None, show_plot=None):
    """Plot pair plot for all numerical features.

    Parameters
    ----------
    data1 : pandas.DataFrame
        DataFrame to be analyzed.

    data2 : pandas.DataFrame
        DataFrame to be analyzed.

    sample : int
        Number of samples

    subplot : bool
        Whether to split figure

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_pair(data, dir_path='.')
    """
    figsize = PLOT_PARAMS.get(figsize, 'figsize')
    if data2 is None:
        sample = min(sample, len(data1))
        data1  = data1.sample(sample)

        g = sns.PairGrid(data1, x_vars=list(data1), y_vars=list(data1))
        fig, axes = g.fig, g.axes
        with FigProcessor(fig, dir_path, show_plot, suptitle='Pairplot', tight_layout=False):
            fig.set_size_inches(figsize)
            for row, f1 in enumerate(data1):
                for col, f2 in enumerate(data1):
                    f1_type = 'num' if data1[f1].nunique() > 10 else 'cat'
                    f2_type = 'num' if data1[f2].nunique() > 10 else 'cat'
                    if (f1 == f2) or ((f1_type == 'cat') and (f2_type == 'cat')):
                        continue
                    eval(f"plot_{f2_type}_{f1_type}_features")(data1, f2, f1, ax=axes[row, col], label=True)

            g.map_diag(sns.histplot)
            g.fig.set_size_inches(figsize)
    else:
        sample = np.min([sample, len(data1), len(data2)])
        if subplot:
            n_cols = 1 if data2 is None else 2
            grids  = gridspec.GridSpec(1, n_cols)
            fig    = plt.figure(figsize=(n_cols*figsize[0], figsize[1]))
            with FigProcessor(fig, dir_path, show_plot, suptitle='Pairplot', tight_layout=False):
                for grid, data in zip(grids, (data1, data2)):
                    g = sns.PairGrid(data, x_vars=list(data), y_vars=list(data))
                    axes = g.axes
                    for row, f1 in enumerate(data):
                        for col, f2 in enumerate(data):
                            f1_type = 'num' if data[f1].nunique() > 10 else 'cat'
                            f2_type = 'num' if data[f2].nunique() > 10 else 'cat'
                            if (f1 == f2) or ((f1_type == 'cat') and (f2_type == 'cat')):
                                continue
                            eval(f"plot_{f2_type}_{f1_type}_features")(data, f2, f1, ax=axes[row, col], label=True)
                    g.map_diag(sns.histplot)
                    g.fig.set_size_inches(figsize)
                    SeabornFig2Grid(g, fig, grid)
                grids.tight_layout(fig)
        else:
            data1 = data1.sample(sample)
            data1['ID'] = 'First'
            data2 = data2.sample(sample)
            data2['ID'] = 'Second'
            data  = pd.concat([data1, data2], ignore_index=True)
            fig = sns.pairplot(data, hue='ID', plot_kws={'alpha': alpha, 's': s}, markers=['o', 'D'], diag_kind='hist').fig
            with FigProcessor(fig, dir_path, show_plot, suptitle='Pairplot', tight_layout=False):
                fig.set_size_inches(figsize)
def plot_pair2(data1, data2=None, sample=1000, alpha=0.3, s=20, subplot=True,       dir_path=None, figsize=None, show_plot=None):
    """Plot pair plot for all numerical features.

    Parameters
    ----------
    data1 : pandas.DataFrame
        DataFrame to be analyzed.

    data2 : pandas.DataFrame
        DataFrame to be analyzed.

    sample : int
        Number of samples

    alpha : float
        Transparency ratio

    s : float
        Size of points

    subplot : bool
        Whether to split figure

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_pair(data, dir_path='.')
    """
    figsize = PLOT_PARAMS.get(figsize, 'figsize')
    if data2 is None:
        sample = min(sample, len(data1))
        fig = sns.pairplot(data1.sample(sample), plot_kws={'alpha': alpha, 's': s}).fig
        with FigProcessor(fig, dir_path, show_plot, suptitle='Pairplot', tight_layout=False):
            fig.set_size_inches(figsize)
    else:
        sample = np.min([sample, len(data1), len(data2)])
        if subplot:
            n_cols = 1 if data2 is None else 2
            grids  = gridspec.GridSpec(1, n_cols)
            fig    = plt.figure(figsize=(n_cols*figsize[0], figsize[1]))
            with FigProcessor(fig, dir_path, show_plot, suptitle='Pairplot', tight_layout=False):
                colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
                for grid, data, color in zip(grids, (data1, data2), colors):
                    # g = sns.PairGrid(data.sample(sample))
                    # g.map_diag(sns.histplot, color=color)  # TODO: why not working?
                    # g.map_offdiag(sns.scatterplot, s=s, alpha=alpha, color=color)
                    g = sns.pairplot(data.sample(sample), plot_kws={'alpha': alpha, 's': s, 'color': color})
                    SeabornFig2Grid(g, fig, grid)
                grids.tight_layout(fig)
        else:
            data1 = data1.sample(sample)
            data1['ID'] = 'First'
            data2 = data2.sample(sample)
            data2['ID'] = 'Second'
            data  = pd.concat([data1, data2], ignore_index=True)
            fig = sns.pairplot(data, hue='ID', plot_kws={'alpha': alpha, 's': s}, markers=['o', 'D'], diag_kind='hist').fig
            with FigProcessor(fig, dir_path, show_plot, suptitle='Pairplot', tight_layout=False):
                fig.set_size_inches(figsize)


# Time series features
def plot_ts_features(data, title='Features',                                        dir_path=None, figsize=None, show_plot=None):
    """Plot time series line plot for all numerical features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    title : str
        Title

    dir_path : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import analysis_tools.eda as eda
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e'], 'c': [1.2, 2.3, 3.4, 4.5, 5.6]})
    >>> eda.plot_ts_features(data, dir_path='.')
    """
    plot_on_ax(lambda ax: data.select_dtypes('number').plot(subplots=True, ax=ax, sharex=True), title, None, dir_path, figsize, show_plot)
