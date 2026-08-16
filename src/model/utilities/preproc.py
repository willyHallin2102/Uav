"""
    src / models / utilities / preproc.py
    -------------------------------------

"""
from __future__ import annotations

import numpy as np
from functools import singledispatch
from typing import Any, Dict, Type, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

preproc = Union[StandardScaler, OneHotEncoder, MinMaxScaler]


# ======================================================================
#   Serialization
# ======================================================================

@singledispatch
def _serialize(proc: Any) -> Dict[str, Any]:
    """
    Serialize a supported scikit-learn preprocessing object into a JSON - compatible
    dictionary representation. This is the generic dispatch entry - point. Concrete
    implementations are registered per supported preprocessors - type (e.g., 
    `StandardScaler`, `MinMaxScaler`, and `MinMaxEncoder`).

    -----
    Args:
    proc: A fitted preprocessing object.
    --------
    Returns:
    Dictionary containing minimal fitted state required to construct the preprocessor.
    -------
    Raises:
    TypeError: If the provided object type is not supported
    -------
    """
    raise TypeError(f"Unsupported preprocessing type: `{type(proc).__name__}`")


@_serialize.register(StandardScaler)
def _(proc: StandardScaler) -> Dict[str, Any]:
    """
    Extract the fitted state of a `StandardScaler` into a minimal, JSON-serializable 
    dictionary. The returned dictionary contains only the learned statistics required 
    to perform future transformers, excluding any non-essential or derived attributes.
    """
    return {
        "mean_": proc.mean_.tolist(), "scale_": proc.scale_.tolist(),
        "var_": proc.var_.tolist(), "n_samples_seen_": int(proc.n_samples_seen_),
    }


@_serialize.register(MinMaxScaler)
def _(proc: MinMaxScaler) -> Dict[str, Any]:
    """
    Extract the fitted state of a MinMaxScaler into a compact, JSON-serializable 
    dictionary. Only the learned scaling parameters and feature range configuration
    necessary for reconstructing transformation behavior are included.
    """
    return {
        "data_min_": proc.data_min_.tolist(), "data_max_": proc.data_max_.tolist(),
        "data_range_": proc.data_range_.tolist(), "scale_": proc.scale_.tolist(),
        "min_": proc.min_.tolist(), "feature_range": tuple(proc.feature_range),
    }


@_serialize.register(OneHotEncoder)
def _(proc: OneHotEncoder) -> Dict[str, Any]:
    """
    Serialize a fitted OneHotEncoder into a JSON-safe dictionary.
    """
    # Convert numpy arrays to lists, preserving string types
    categories_serialized = []
    for category in proc.categories_:
        if category.dtype.kind in ('U', 'S'):  # String type
            categories_serialized.append(category.tolist())
        else:
            # Convert to float to maintain numerical precision
            categories_serialized.append(category.astype(float).tolist())
    
    return {
        "categories_": categories_serialized,
        "drop": proc.drop, 
        "handle_unknown": proc.handle_unknown,
        "dtype": str(proc.dtype),
        "sparse_output": getattr(proc, "sparse_output", None),
        "sparse": getattr(proc, "sparse", None), 
        "min_frequency": getattr(proc, "min_frequency", None),
        "max_categories": getattr(proc, "max_categories", None),
    }


# ======================================================================
#   Deserialization
# ======================================================================

@singledispatch
def _deserialize(cls: Any, params: Dict[str, Any]) -> Preproc:
    """
    Reconstruct a supported preprocessing object from a serialized parameter 
    dictionary. This is the generic dispatch entry point. Concrete implementations 
    are registered for each supported preprocessor class.

    -----
    Args:
    cls: The preprocessor class to reconstruct.
    params: Serialized state produced by `_serialized`
    --------
    Returns:
    A fitted preprocessing instance with restored internal state
    -------
    Raises:
    If the provided class is not supported for deserialization.
    """
    raise TypeError(f"Unsupported preprocessor class: {cls}")


@_deserialize.register(type(StandardScaler()))
def _(cls: Type[StandardScaler], p: Dict[str, Any]) -> StandardScaler:
    """
    Reconstruct a fitted StandardScaler from serialized parameters. The returned 
    instance has its learned statistics restored and is ready for transformation 
    without requiring a refit.
    """
    proc = StandardScaler()
    
    proc.mean_ = np.asarray(p["mean_"], dtype=float)
    proc.scale_ = np.asarray(p["scale_"], dtype=float)
    proc.var_ = np.asarray(p["var_"], dtype=float)
    proc.n_samples_seen_ = int(p["n_samples_seen_"])
    proc.n_features_in_ = proc.mean_.shape[0]
    
    return proc


@_deserialize.register(type(MinMaxScaler()))
def _(cls: Type[MinMaxScaler], p: Dict[str, Any]) -> MinMaxScaler:
    """
    Reconstruct a fitted MinMaxScaler from serialized parameters. All learned 
    scaling attributes and feature metadata are restored so that the instance 
    behaves identically to the original fitted scaler.
    """
    proc = MinMaxScaler(feature_range=tuple(p["feature_range"]))
    
    proc.data_min_ = np.asarray(p["data_min_"], dtype=float)
    proc.data_max_ = np.asarray(p["data_max_"], dtype=float)
    proc.data_range_ = np.asarray(p["data_range_"], dtype=float)
    proc.scale_ = np.asarray(p["scale_"], dtype=float)
    proc.min_ = np.asarray(p["min_"], dtype=float)
    proc.n_features_in_ = proc.data_min_.shape[0]
    
    return proc


@_deserialize.register(type(OneHotEncoder()))
def _(cls: Type[OneHotEncoder], p: Dict[str, Any]) -> OneHotEncoder:
    """
    Reconstruct a fitted OneHotEncoder from serialized parameters.
    """
    # Convert categories to numpy arrays with appropriate dtype
    categories = []
    for category in p["categories_"]:
        # Determine if categories are strings or numbers
        if category and isinstance(category[0], str):
            categories.append(np.array(category, dtype=str))
        else:

            # Try to convert to float, fallback to object
            try:
                categories.append(np.array(category, dtype=float))
            
            except (ValueError, TypeError):
                categories.append(np.array(category, dtype=object))
    
    kwargs = {
        "categories": categories,
        "drop": p["drop"], 
        "handle_unknown": p["handle_unknown"],
    }

    if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
        kwargs["sparse_output"] = bool(p.get("sparse_output", False))
        
    else:
        kwargs["sparse"] = bool(p.get("sparse", False))

    if p.get("min_frequency") is not None:
        kwargs["min_frequency"] = p["min_frequency"]
    
    if p.get("max_categories") is not None:
        kwargs["max_categories"] = p["max_categories"]

    # Parse dtype string back to actual dtype
    dtype_str = p.get("dtype", "float64")
    dtype_map = {
        "float64": np.float64, "float32": np.float32,
        "int64": np.int64, "int32": np.int32,
        "str": str, "object": object
    }
    kwargs["dtype"] = dtype_map.get(dtype_str, np.float64)
    proc = OneHotEncoder(**kwargs)

    # Set fitted attributes
    proc.categories_ = categories
    proc.n_features_in_ = len(proc.categories_)
    proc._n_features_outs = [len(c) for c in proc.categories_]
    proc._feature_indices = np.cumsum([0] + proc._n_features_outs)
    
    # Set infrequent handling attributes
    proc._infrequent_enabled = False
    proc._drop_idx_after_grouping = None

    return proc


# ======================================================================
#   Function Calls or API for preproc <-> params
# ======================================================================

def serialize_preproc(proc: Preproc) -> Dict[str, Any]:
    """
    Convert a fitted preprocessing object into a structured, JSON-safe representation. 
    The output dictionary includes 

        - The preprocessor type identifier
        - A minimal serialization parameter state.
    
    This representation is designed for lightweight storage and reconstruction 
    without persisting the full scikit-learn object.
    """
    return {"type": type(proc).__name__, "params": _serialize(proc),}


def deserialize_preproc(data: Dict[str, Any]) -> Preproc:
    """
    Reconstruct a fitted preprocessing object from its serialized form. The function 
    resolves the stored type identifier to the appropriate scikit-learn class and 
    restores the learned state using the internal deserialization dispatch mechanism.

    -----
    Args:
    data : Dictionary produced by `serialize_preproc`.
    --------
    Returns:
    A fully reconstructed preprocessing instance ready for use.
    -------
    Raises:
    ValueError: If the stored preprocessor type cannot be resolved.
    """
    # Try to get the class by its string name first
    cls_name = data["type"]
    cls_map: Dict[str, Type[Preproc]] = {
        "StandardScaler": StandardScaler, 
        "MinMaxScaler": MinMaxScaler,
        "OneHotEncoder": OneHotEncoder,
    }
    internal_cls_map: Dict[str, Type[Preproc]] = {
        "_data.StandardScaler": StandardScaler,
        "_encoders.OneHotEncoder": OneHotEncoder,
        "_data.MinMaxScaler": MinMaxScaler,
    }
    cls = cls_map.get(cls_name)
    if cls is None:
        for key, value in internal_cls_map.items():
            if key in cls_name:
                cls = value
                break
    
    if cls is None: 
        raise ValueError(f"Unknown preprocessor type: {cls_name}")
    
    return _deserialize.dispatch(type(cls()))(cls, data["params"])
