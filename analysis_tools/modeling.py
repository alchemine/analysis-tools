from analysis_tools.common import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_scaling_model(model, scaler=StandardScaler()):
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
    return make_pipeline(scaler, model)
