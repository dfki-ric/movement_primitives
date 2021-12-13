#!/usr/bin/env python
import os
import warnings
from setuptools import setup, find_packages


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    with open(os.path.join("movement_primitives", "_version.py")) as f:
        version = f.read().strip().split('"')[1]
    setup_config = dict(
        name="movement_primitives",
        version=version,
        author="Alexander Fabisch",
        author_email="alexander.fabisch@dfki.de",
        url="https://github.com/dfki-ric/movement_primitives",
        description="Movement primitives",
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=["Programming Language :: Python :: 3"],
        license="BSD-3-clause",
        packages=find_packages(),
        install_requires=[],
        extras_require={
            "all": ["pytransform3d", "cython", "numpy", "scipy", "matplotlib",
                    "open3d", "tqdm", "gmr", "PyYAML", "numba", "pybullet"],
            "doc": ["pdoc3"],
            "test": ["nose", "coverage"]
        }
    )
    try:
        from Cython.Build import cythonize
        import numpy
        cython_config = dict(
            ext_modules=cythonize("movement_primitives/dmp_fast.pyx"),
            zip_safe=False,
            compiler_directives={"language_level": "3"},
            include_dirs=[numpy.get_include()],
            extra_compile_args=[
                "-O3",
                "-Wno-cpp", "-Wno-unused-function"
            ]
        )
        setup_config.update(cython_config)
    except ImportError:
        warnings.warn("Cython or NumPy is not available. "
                      "Install it if you want fast DMPs.")

    setup(**setup_config)
