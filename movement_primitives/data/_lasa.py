import os
import zipfile
import io
import scipy.io
import numpy as np
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen


LASA_URL = ("http://bitbucket.org/khansari/lasahandwritingdataset/get/"
            "38304f7c0ac4.zip")


def load_lasa(shape_idx):
    """Load demonstrations from LASA dataset.

    The LASA dataset contains 2D handwriting motions recorded from a
    Tablet-PC. It can be found `here
    <https://bitbucket.org/khansari/lasahandwritingdataset>`_
    Take a look at the `detailed explanation
    <http://cs.stanford.edu/people/khansari/DSMotions#SEDS_Benchmark_Dataset>`_
    for more information.

    The following plot shows multiple demonstrations for the same shape.

    .. plot::

        import matplotlib.pyplot as plt
        from movement_primitives.data import load_lasa
        X, Xd, Xdd, dt, shape_name = load_lasa(0)
        plt.figure()
        plt.title(shape_name)
        plt.plot(X[:, :, 0].T, X[:, :, 1].T)
        plt.show()

    Parameters
    ----------
    shape_idx : int
        Choose demonstrated shape, must be within range(30).

    Returns
    -------
    T : array, shape (n_demos, n_steps)
        Times

    X : array, shape (n_demos, n_steps, n_dims)
        Positions

    Xd : array, shape (n_demos, n_steps, n_dims)
        Velocities

    Xdd : array, shape (n_demos, n_steps, n_dims)
        Accelerations

    dt : float
        Time between steps

    shape_name : string
        Name of the Matlab file from which we load the demonstrations
        (without suffix).
    """
    dataset_path = get_common_dataset_path()
    if not os.path.isdir(dataset_path + "lasa_data"):  # pragma: no cover
        url = urlopen(LASA_URL)
        z = zipfile.ZipFile(io.BytesIO(url.read()))
        z.extractall(dataset_path)
        os.rename(dataset_path + z.namelist()[0],
                  dataset_path + "lasa_data" + os.sep)

    dataset_path += "lasa_data" + os.sep + "DataSet" + os.sep
    demos, shape_name = _load_from_matlab_file(dataset_path, shape_idx)
    X, Xd, Xdd, dt = _convert_demonstrations(demos)
    t = np.linspace(0, X.shape[1], X.shape[1])
    T = np.tile(t, (X.shape[0], 1))
    return T, X, Xd, Xdd, dt, shape_name


def _load_from_matlab_file(dataset_path, shape_idx):
    """Load demonstrations from Matlab files.

    Parameters
    ----------
    dataset_path : str
        Path where external datasets are stored.

    shape_idx : int
        Index of the shape in the LASA dataset.

    Returns
    -------
    demos : array
        Demonstrations.

    shape_name : string
        Name of the Matlab file from which we load the demonstrations
        (without suffix).
    """
    file_name = sorted(os.listdir(dataset_path))[shape_idx]
    return (scipy.io.loadmat(dataset_path + file_name)["demos"][0],
            file_name[:-4])


def _convert_demonstrations(demos):
    """Convert Matlab struct to numpy arrays.

    Parameters
    ----------
    demos : array
        Demonstrations.

    Returns
    -------
    X : array, shape (n_demos, n_steps, n_dims)
        Positions

    Xd : array, shape (n_demos, n_steps, n_dims)
        Velocities

    Xdd : array, shape (n_demos, n_steps, n_dims)
        Accelerations

    dt : float
        Time between steps
    """
    tmp = []
    for demo_idx in range(demos.shape[0]):
        # The Matlab format is strange...
        demo = demos[demo_idx][0, 0]
        # Positions, velocities and accelerations
        tmp.append((demo[0], demo[2], demo[3]))

    X = np.transpose([P for P, _, _ in tmp], [0, 2, 1])
    Xd = np.transpose([V for _, V, _ in tmp], [0, 2, 1])
    Xdd = np.transpose([A for _, _, A in tmp], [0, 2, 1])

    dt = float(demos[0][0, 0][4])

    return X, Xd, Xdd, dt


def get_common_dataset_path():
    """Returns the path where all external datasets are stored.

    Returns
    -------
    dataset_path : str
        Path where external datasets are stored.
    """
    dataset_path = os.path.expanduser("~")
    dataset_path += os.sep + ".movement_primitive_data" + os.sep
    return dataset_path
