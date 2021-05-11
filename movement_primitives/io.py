import pprint
import numpy as np
import yaml
import inspect


basic_types = (int, float, bool, str, type(None))
basic_types_and_sequences = (int, float, bool, str, list, tuple, type(None))


def write_yaml(filename, mp, verbose=0):
    export = _recursive_to_dict(mp)
    if verbose:
        pprint.pprint(export)
    with open(filename, "w") as f:
        yaml.dump(export, f)


def _recursive_to_dict(obj):
    result = {"module": obj.__module__, "class": obj.__class__.__name__}
    for k, v in obj.__dict__.items():
        if isinstance(v, basic_types_and_sequences):
            result[k] = v
        elif isinstance(v, np.ndarray):
            result[k] = v.tolist()
        else:
            result[k] = _recursive_to_dict(v)
    return result


def read_yaml(filename):
    with open(filename, "r") as f:
        export = yaml.safe_load(f)
    module = __import__(export.pop("module"), {}, {}, fromlist=["dummy"], level=0)
    class_dict = dict(inspect.getmembers(module))
    clazz = class_dict[export.pop("class")]
    argspec = inspect.getfullargspec(clazz)
    ctor_kwargs = {}
    for arg in argspec.args:
        if arg in export:
            ctor_kwargs[arg] = export.pop(arg)
    obj = clazz(**ctor_kwargs)
    _recursive_from_dict(obj, export)
    return obj


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
