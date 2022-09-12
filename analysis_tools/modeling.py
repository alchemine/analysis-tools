"""Modeling analysis tools

Modeling wrapping functions or classes are defined here.
"""
# Author: Dongjin Yoon <djyoon0223@gmail.com>


from analysis_tools.common import *


def get_scaled_model(model, scaler=None):
    """
    Creates a pipeline that applies the given scaler to the given model.

    Parameters
    ----------
    model : sklearn model
        sklearn model.

    scaler : sklearn scaler
        sklearn scaler.

    Returns
    -------
    scaled sklearn model
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler() if scaler is None else scaler
    return make_pipeline(scaler, model)


def save_tree_visualization(fitted_model, X, y, file_path, feature_names=None, class_names=None, orientation='LR', test_sample=None):
    """
    Save a dtreeviz visualization of the given model.

    Parameters
    ----------
    fitted_model : sklearn model
        sklearn model fitted.

    X : pandas.dataframe or numpy.array
        Feature array

    y : pandas.series or numpy.array
        Target array

    file_path : string
        Path to save the dtreeviz visualization. file_path must end with '.svg'.

    feature_names : list of strings
        List of feature names.

    class_names : list of strings
        List of class names.

    orientation : string
        Orientation of the tree.
        'LR' for left to right, 'TB' for top to bottom.

    test_sample : pandas.series or numpy.array
        One sample of test data

    Examples
    --------
    >>> from analysis_tools.modeling import *
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier

    >>> iris = load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> model = DecisionTreeClassifier(max_depth=3)
    >>> model.fit(X, y)

    >>> save_tree_visualization(model, X, y, 'iris_tree.svg', feature_names=iris.feature_names, class_names=list(iris.target_names), test_sample=X[0])
    """
    from dtreeviz.trees import dtreeviz

    viz = dtreeviz(fitted_model, X, y, feature_names=feature_names, class_names=class_names, orientation=orientation, X=test_sample)
    assert file_path.endswith('.svg'), 'file_path must end with .svg'
    viz.save(file_path)


def generate_dataset_tf(X, y=None, batch_size=32, shuffle=True):
    """Generate TensorFlow dataset from array-like data (X, y)

    Parameters
    ----------
    X : array-like
        Input data

    y : numpy.ndarray (default=None)
        Output data

    batch_size : int (default=32)
        Batch size

    shuffle : bool (default=True)
        Whether to shuffle data

    Returns
    -------
    TensorFlow dataset : tf.data.Dataset
        Generated dataset
    """
    import tensorflow as tf

    # 1. Generate TensorFlow dataset
    X = np.array(X, dtype=np.float32)
    if y is None:
        y = np.zeros(len(X), dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    # 2. Options
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    return ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
def generate_dataloader_torch(X, y=None, batch_size=32, shuffle=True, device=None):
    """Generate TensorFlow dataset from array-like data (X, y)

    Parameters
    ----------
    X : array-like
        Input data

    y : numpy.ndarray (default=None)
        Output data

    batch_size : int (default=32)
        Batch size

    shuffle : bool (default=True)
        Whether to shuffle data

    device : str (default=None)
        Device to load the data

    Returns
    -------
    PyTorch dataloader : torch.utils.data.DataLoader
        Generated dataloader
    """
    import torch
    from torch.utils.data import Dataset, DataLoader

    class CustomDataset(Dataset):
        def __init__(self, X, y=None):
            self.X = torch.tensor(X, device=device, dtype=torch.float32)
            if y is None:
                y = torch.zeros(len(self.X), device=device, dtype=torch.float32)  # dummy
            self.y = torch.tensor(y, device=device, dtype=torch.float32)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    return DataLoader(CustomDataset(X, y), batch_size=batch_size, shuffle=shuffle)
