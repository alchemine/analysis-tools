"""Exploratory Data Analysis tools

This module contains functions and classes for exploratory data analysis.
"""

# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common import *


# Utility function
def plot_on_ax(plot_fn, suptitle,                       ax=None, save_dir=None, figsize=None, show_plot=None, **plot_kws):
    """Plot on a single axis if ax is not None. Otherwise, plot on a new figure.

    Parameters
    ----------
    plot_fn : function
        Function to plot.

    suptitle : str
        Title of the plot.

    ax : matplotlib.axes.Axes
        Axis to plot.

    save_dir : str
        Directory path to save the plot.

    figsize : tuple
        Figure size.

    show_plot : bool
        Whether to show the plot.
    """
    if ax is not None:
        cm = contextlib.nullcontext()
    else:
        fig, ax = plt.subplots(figsize=PLOT_PARAMS.get('figsize', figsize))
        cm = FigProcessor(fig, save_dir, show_plot, suptitle)
    with cm:
        plot_fn(ax)
        if plot_kws.get('xlabel', False) is not True:
            ax.set_xlabel(None)
        if plot_kws.get('ylabel', False) is not True:
            ax.set_ylabel(None)


# Missing value
def plot_missing_value(data, show_df=False,                      save_dir=None, figsize=None, show_plot=None, **plot_kws):
    """Plot counts of missing values of each feature.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    show_df : bool
        Whether to show value counts dataframe

    save_dir : str
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
    >>> eda.plot_missing_value(data, save_dir='.')
    """
    fig, axes = plt.subplots(2, 1, figsize=PLOT_PARAMS.get('figsize', figsize))
    with FigProcessor(fig, save_dir, show_plot, "Missing value"):
        msno.matrix(data, ax=axes[0])
        ms = data.isnull().sum()
        sns.barplot(ms.index, ms, ax=axes[1])
        axes[1].bar_label(axes[1].containers[0])
        axes[1].set_xticklabels([])

    if show_df:
        data_null = data[data.isna().any(1)]
        for f in data_null:
            if data_null[f].notnull().sum() > 0:
                display(data_null.value_counts(f).sort_index().to_frame().T.style.background_gradient(axis=1))

# Features
def plot_fn_num(data_f, ax, **plot_kws):
    sns.histplot(data_f, bins=PLOT_PARAMS.get('bins', plot_kws), ax=ax, kde=True, stat='density', color=plot_kws.get('color', None))
def plot_fn_cat(data_f, ax, **plot_kws):
    flag_datetime = is_datetime_format(data_f.unique()[0])
    if flag_datetime:
        data_f = pd.to_datetime(data_f)
        sns.histplot(data_f, bins=PLOT_PARAMS.get('bins', plot_kws), ax=ax, kde=True, stat='density', color=plot_kws.get('color', None))
        ax.tick_params(axis='x', labelrotation=30)
    else:
        cnts = data_f.value_counts(normalize=True).sort_index()
        sns.barplot(cnts.index, cnts.values, order=cnts.index, ax=ax)
        xticklabels = cnts.sort_values()[-PLOT_PARAMS.get('n_classes', plot_kws):].index
        ax.set_xticks(lmap(lambda l: cnts.index.get_loc(l), xticklabels))
        ax.set_xticklabels(xticklabels, rotation=30, ha='right', rotation_mode='anchor')

def plot_features(data1, data2=None, title='Features',           save_dir=None, figsize=None, show_plot=None, **plot_kws):
    """Plot histogram or bar for all features.

    Parameters
    ----------
    data1 : pandas.DataFrame
        DataFrame to be analyzed.

    data2 : pandas.DataFrame
        DataFrame to be analyzed.

    title : str
        Title

    save_dir : str
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
    >>> eda.plot_features(data, save_dir='.')
    """
    n_cols = PLOT_PARAMS.get('n_cols', plot_kws)
    n_features = len(data1.columns)
    n_rows     = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=PLOT_PARAMS.get('figsize', figsize))
    with FigProcessor(fig, save_dir, show_plot, title):
        for ax in axes.flat[n_features:]:
            ax.axis('off')
        for ax, f in zip(axes.flat, data1):
            datas  = [data1] if data2 is None else [data1, data2]
            colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
            for data, color in zip(datas, colors):
                plot_kws['color'] = color
                data_f_notnull = data[f].dropna()
                if is_numeric_dtype(data_f_notnull):
                    plot_fn_num(data_f_notnull, ax, **plot_kws)
                    # ax.hist(data_f_notnull, bins=bins, density=True, color=color, alpha=0.5)
                else:
                    plot_fn_cat(data_f_notnull, ax, **plot_kws)
            ax.set_title(f)
            ax.set_xlabel(None);  ax.set_ylabel(None)

def plot_features_target(data, target, target_type='auto',       save_dir=None, figsize=None, show_plot=None, **plot_kws):
    """Plot features vs target.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.
        Dtypes of numerical features should be `number`(`numpy.float32` is recommended).
        Dtypes of categorical features should be `string` or `category`.

    target : str
        Target feature.

    target_type : str
        Type of target.
        target_type should be 'auto' or 'num', 'cat'.
        target_type is inferred automatically when 'auto' is set.

    save_dir : str
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
    >>> eda.plot_features_target(data, 'a', save_dir='.')
    """
    n_cols = PLOT_PARAMS.get('n_cols', plot_kws)
    if target_type == 'auto':
        target_type = dtype(data[target])
    n_features = len(data.columns) - 1  # -1: except target
    n_rows     = int(np.ceil(n_features/n_cols))
    fig, axes  = plt.subplots(n_rows, n_cols, figsize=PLOT_PARAMS.get('figsize', figsize))
    axes       = np.array(axes) if n_rows*n_cols == 1 else axes
    with FigProcessor(fig, save_dir, show_plot, "Features vs Target"):
        for ax in axes.flat[n_features:]:
            ax.axis('off')
        for ax, f in zip(axes.flat, data.columns.drop(target)):
            eval(f"plot_{dtype(data[f])}_{target_type}_features")(data, f, target, ax=ax, **plot_kws)
            ax.set_title(f"{f} vs {target}")
            ax.set_xlabel(None);  ax.set_ylabel(None)

def plot_two_features(data, f1, f2, title=None,         ax=None, save_dir=None, figsize=None, show_plot=None, **plot_kws):
    """Plot joint distribution of two features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    f1 : str
        Feature 1.

    f2 : str
        Feature 2.

    title : str
        Title.

    ax : matplotlib.axes.Axes
        Axis to plot.

    save_dir : str
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
    >>> eda.plot_two_features(data, 'a', 'b', save_dir='.')
    """
    plot_fn = lambda ax: eval(f"plot_{dtype(data[f1])}_{dtype(data[f2])}_features")(data, f1, f2, ax=ax, **plot_kws)
    plot_on_ax(plot_fn, f"{f1} vs {f2}" if title is None else title, ax, save_dir, figsize, show_plot, **plot_kws)

def plot_corr(corr1, corr2=None, annot=True, mask=True,          save_dir=None, figsize=(15, 15), show_plot=None, **plot_kws):
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

    save_dir : str
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
    >>> eda.plot_corr(corr, save_dir='.')
    """
    if corr2 is None:
        def plot_fn(ax):
            mask_mat = np.eye(len(corr1), dtype=bool)
            mask_mat[np.triu_indices_from(mask_mat, k=1)] = mask
            sns.heatmap(corr1, mask=mask_mat, ax=ax, annot=annot, fmt=".2f", cmap='coolwarm', center=0)
        plot_on_ax(plot_fn, "Correlation matrix", None, save_dir, figsize, show_plot)
    else:
        figsize = PLOT_PARAMS.get('figsize', figsize)
        fig, axes = plt.subplots(1, 2, figsize=(2*figsize[0], figsize[1]))
        with FigProcessor(fig, save_dir, show_plot, "Correlation matrix"):
            for ax, corr in zip(axes.flat, (corr1, corr2)):
                mask_mat = np.eye(len(corr), dtype=bool)
                mask_mat[np.triu_indices_from(mask_mat, k=1)] = mask
                sns.heatmap(corr, mask=mask_mat, ax=ax, annot=annot, fmt=".2f", cmap='coolwarm', center=0)

def plot_num_feature(data_f,                            ax=None, save_dir=None, figsize=None, show_plot=None, **plot_kws):
    """Plot histogram of a numeric feature.

    Parameters
    ----------
    data_f : pandas.Series
        Series to be analyzed.

    ax : matplotlib.axes.Axes
        Axis to plot.

    save_dir : str
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
    >>> eda.plot_num_feature(data['a'], save_dir='.')
    """
    def plot_fn(ax):
        plot_fn_num(data_f, ax, **plot_kws)
    plot_on_ax(plot_fn, data_f.name, ax, save_dir, figsize, show_plot, **plot_kws)
def plot_cat_feature(data_f,                            ax=None, save_dir=None, figsize=None, show_plot=None, **plot_kws):
    """Plot bar of a categorical feature.

    Parameters
    ----------
    data_f : pandas.Series
        Series to be analyzed.

    ax : matplotlib.axes.Axes
        Axis to plot.

    save_dir : str
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
    >>> eda.plot_cat_feature(data['b'], save_dir='.')
    """
    def plot_fn(ax):
        plot_fn_cat(data_f, ax, **plot_kws)
    plot_on_ax(plot_fn, data_f.name, ax, save_dir, figsize, show_plot, **plot_kws)
def plot_num_num_features(data, f1, f2, sample=100_000, ax=None, save_dir=None, figsize=None, show_plot=None, **plot_kws):
    """Plot scatter plot of two numeric features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    f1 : str
        First numerical feature.

    f2 : str
        Second numerical feature.

    sample : int
        Number of samples.
        If sample is None or False, then use entire data.

    ax : matplotlib.axes.Axes
        Axis to plot.

    save_dir : str
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
    >>> eda.plot_num_num_features(data, 'a', 'c', save_dir='.')
    """
    def plot_fn(ax):
        sns.scatterplot(x=data[f1], y=data[f2], alpha=PLOT_PARAMS.get('alpha', plot_kws), ax=ax, color=plot_kws.get('color', None))
    if sample and (len(data) > sample):
        data = data.sample(sample)
    plot_on_ax(plot_fn, f"{f1} vs {f2}", ax, save_dir, figsize, show_plot, **plot_kws)
def plot_num_cat_features(data, f1, f2,                 ax=None, save_dir=None, figsize=None, show_plot=None, **plot_kws):
    """Plot violinplot of categorical, numerical features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    f1 : str
        First numerical feature.

    f2 : str
        Second categorical feature.

    ax : matplotlib.axes.Axes
        Axis to plot.

    save_dir : str
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
    >>> eda.plot_num_cat_features(data, 'a', 'b', save_dir='.')
    """
    def plot_fn(ax):
        selected_classes = data[f2].value_counts().index[:PLOT_PARAMS.get('n_classes', plot_kws)]
        idxs_selected    = data[f2][data[f2].isin(selected_classes)].index
        data_f1, data_f2 = data[f1][idxs_selected], data[f2][idxs_selected]
        sns.violinplot(x=data_f1, y=data_f2, ax=ax, orient='h', order=reversed(sorted(selected_classes)), cut=0)
    plot_on_ax(plot_fn, f"{f1} vs {f2}", ax, save_dir, figsize, show_plot, **plot_kws)
def plot_cat_num_features(data, f1, f2,                 ax=None, save_dir=None, figsize=None, show_plot=None, **plot_kws):
    """Plot violinplot of categorical, numerical features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    f1 : str
        First categorical feature.

    f2 : str
        Second numerical feature.

    ax : matplotlib.axes.Axes
        Axis to plot.

    save_dir : str
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
    >>> eda.plot_cat_num_features(data, 'b', 'a', save_dir='.')
    """
    def plot_fn(ax):
        selected_classes = data[f1].value_counts().index[:PLOT_PARAMS.get('n_classes', plot_kws)]
        idxs_selected    = data[f1][data[f1].isin(selected_classes)].index
        data_f1, data_f2 = data[f1][idxs_selected], data[f2][idxs_selected]
        sns.violinplot(x=data_f1, y=data_f2, ax=ax, orient='v', order=sorted(selected_classes), cut=0)
    plot_on_ax(plot_fn, f"{f1} vs {f2}", ax, save_dir, figsize, show_plot, **plot_kws)
def plot_cat_cat_features(data, f1, f2,                 ax=None, save_dir=None, figsize=None, show_plot=None, **plot_kws):
    """Plot heatmap of two categorical features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    f1 : str
        First categorical feature.

    f2 : str
        Second categorical feature.

    ax : matplotlib.axes.Axes
        Axis to plot.

    save_dir : str
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
    >>> eda.plot_cat_cat_features(data, 'b', 'c', save_dir='.')
    """
    def plot_fn(ax):
        ratio = pd.crosstab(data[f2], data[f1], normalize='columns')
        ratio.sort_index(inplace=True, ascending=False)  # sort by index
        ratio = ratio[sorted(ratio)]                     # sort by column
        n_classes = PLOT_PARAMS.get('n_classes', plot_kws)
        ratio = ratio.iloc[:n_classes, :n_classes]
        sns.heatmap(ratio, ax=ax, annot=True, fmt=".2f", cmap=sns.light_palette('firebrick', as_cmap=True), cbar=False)
    plot_on_ax(plot_fn, f"{f1} vs {f2}", ax, save_dir, figsize, show_plot, **plot_kws)
def plot_pair(data1, data2=None, subplot=True,                   save_dir=None, figsize=(20, 20), show_plot=None, **plot_kws):
    """Plot pair plot for all numerical features.

    Parameters
    ----------
    data1 : pandas.DataFrame
        DataFrame to be analyzed.

    data2 : pandas.DataFrame
        DataFrame to be analyzed.

    plot_kws : dict
        Plot parameters.

    subplot : bool
        Whether to split figure

    save_dir : str
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
    >>> eda.plot_pair(data, save_dir='.')
    """
    def plot_custom_pair(data, g):
        fs = data.columns
        for row, col in permutations(range(len(fs)), 2):
            if (row == col) or (dtype(data[fs[col]]) == 'cat' and dtype(data[fs[row]]) == 'cat'):
                # TODO: cat - cat features
                g.axes[row, col].axis('off')
            else:
                plot_two_features(data, fs[col], fs[row], ax=g.axes[row, col], **plot_kws)
        g.map_diag(sns.histplot)

    figsize = PLOT_PARAMS.get('figsize', figsize)
    if data2 is None:
        # fig = sns.pairplot(data1, plot_kws={'alpha': PLOT_PARAMS.get('alpha', plot_kws), 's': PLOT_PARAMS.get('s', plot_kws)}).fig
        fs = data1.columns
        g  = sns.PairGrid(data1, diag_sharey=False, x_vars=fs, y_vars=fs)
        fig, axes = g.fig, g.axes
        with FigProcessor(fig, save_dir, show_plot, suptitle='Pairplot', tight_layout=False):
            plot_custom_pair(data1, g)
            fig.set_size_inches(figsize)
    else:
        if subplot:
            n_cols = 1 if data2 is None else 2
            grids  = gridspec.GridSpec(1, n_cols)
            fig    = plt.figure(figsize=(n_cols*figsize[0], figsize[1]))
            with FigProcessor(fig, save_dir, show_plot, suptitle='Pairplot', tight_layout=False):
                colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
                for grid, data, color in zip(grids, (data1, data2), colors):
                    plot_kws['color'] = color
                    fs = data.columns
                    g  = sns.PairGrid(data, diag_sharey=False, x_vars=fs, y_vars=fs)
                    plot_custom_pair(data, g)
                    SeabornFig2Grid(g, fig, grid)
                grids.tight_layout(fig)
        else:
            data1['ID'], data2['ID'] = 'First', 'Second'
            data = pd.concat([data1, data2], ignore_index=True)
            fig  = sns.pairplot(data, hue='ID', plot_kws={'alpha': PLOT_PARAMS.get('alpha', plot_kws), 's': PLOT_PARAMS.get('marker_size', plot_kws), 'markers': ['o', 'D'], 'diag_kind': 'hist'}).fig
            with FigProcessor(fig, save_dir, show_plot, suptitle='Pairplot', tight_layout=False):
                fig.set_size_inches(figsize)


# Time series features
def plot_ts_features(data, title='Features',                     save_dir=None, figsize=None, show_plot=None, **plot_kws):
    """Plot time series line plot for all numerical features.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame to be analyzed.

    title : str
        Title

    save_dir : str
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
    >>> eda.plot_ts_features(data, save_dir='.')
    """
    plot_on_ax(lambda ax: data.select_dtypes('number').plot(subplots=True, ax=ax, sharex=True),
               title, None, save_dir, figsize, show_plot)
