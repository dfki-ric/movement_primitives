"""Input and output from and to files of movement primitives."""
import inspect
import json
import pickle
import yaml
import numpy as np


basic_types = (int, float, bool, str, type(None))
basic_types_and_sequences = (int, float, bool, str, list, tuple, type(None))


def write_pickle(filename, obj):
    """Write object to pickle format.

    Parameters
    ----------
    filename : str
        Output file.

    obj : object
        Any object.
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def read_pickle(filename):
    """Read object from pickle format.

    Parameters
    ----------
    filename : str
        Input file.

    Returns
    -------
    obj : object
        Python object.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def write_yaml(filename, obj):
    """Write object to YAML format.

    Parameters
    ----------
    filename : str
        Output file.

    obj : object
        Any custom object that is a hierarchical composition of basic data
        types and numpy arrays.
    """
    export = _recursive_to_dict(obj, True)
    with open(filename, "w") as f:
        yaml.dump(export, f)


def read_yaml(filename):
    """Read object from YAML format.

    Parameters
    ----------
    filename : str
        Input file.

    Returns
    -------
    obj : object
        Python object.
    """
    with open(filename, "r") as f:
        export = yaml.safe_load(f)
    return _dict_to_object(export)


def write_json(filename, obj):
    """Write object to JSON format.

    Parameters
    ----------
    filename : str
        Output file.

    obj : object
        Any custom object that is a hierarchical composition of basic data
        types and numpy arrays.
    """
    export = _recursive_to_dict(obj)
    with open(filename, "w") as f:
        json.dump(export, f)


def read_json(filename):
    """Read object from JSON format.

    Parameters
    ----------
    filename : str
        Input file.

    Returns
    -------
    obj : object
        Python object.
    """
    with open(filename, "r") as f:
        export = json.load(f)
    return _dict_to_object(export)


def _recursive_to_dict(obj, convert_tuple=False):
    result = {"module": obj.__module__, "class": obj.__class__.__name__}
    for k, v in obj.__dict__.items():
        if convert_tuple and isinstance(v, tuple):
            result[k] = list(v)
        elif isinstance(v, basic_types_and_sequences):
            result[k] = v
        elif isinstance(v, np.ndarray):
            result[k] = v.tolist()
        else:
            result[k] = _recursive_to_dict(v)
    return result


def _recursive_from_dict(obj, export):
    for k, v in export.items():
        if isinstance(v, basic_types):
            setattr(obj, k, v)
        elif isinstance(v, (tuple, list)):
            if isinstance(obj.__dict__[k], np.ndarray):
                obj.__dict__[k] = np.array(v)
            else:
                setattr(obj, k, v)
        else:
            _recursive_from_dict(getattr(obj, k), v)


def _dict_to_object(export):
    module_name = export.pop("module")
    module = __import__(module_name, {}, {}, fromlist=["dummy"], level=0)
    class_dict = dict(inspect.getmembers(module))
    class_name = export.pop("class")
    if class_name not in class_dict:
        raise ImportError(f"cannot import name '{class_name}' from '{module}'")
    clazz = class_dict[class_name]

    argspec = inspect.getfullargspec(clazz)
    ctor_kwargs = {}
    for arg in argspec.args:
        if arg in export:
            ctor_kwargs[arg] = export.pop(arg)
    obj = clazz(**ctor_kwargs)

    _recursive_from_dict(obj, export)
    return obj
