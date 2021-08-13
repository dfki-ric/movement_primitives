#!/usr/bin/env python
from setuptools import setup
import warnings
try:
    from Cython.Build import cythonize
    cython_available = True
except ImportError:
    cython_available = False


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    setup_config = dict(
        name="movement_primitives",
        version="0.0.dev",
        author="Alexander Fabisch",
        author_email="alexander.fabisch@dfki.de",
        url="https://git.hb.dfki.de/dfki-learning/movement_primitives",
        description="Movement primitives",
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=["Programming Language :: Python :: 3"],
        license="no license",
        packages=["movement_primitives"],
        install_requires=[],
        extras_require={
            "all": ["pytransform3d", "cython", "numpy", "scipy", "matplotlib",
                    "open3d", "tqdm", "gmr", "PyYAML"],
            "doc": ["pdoc3"],
            "test": ["nose", "coverage"]
        }
    )
    if cython_available:
        import numpy
        cython_config = dict(
            ext_modules=cythonize("dmp_fast.pyx"),
            zip_safe=False,
            compiler_directives={'language_level': "3"},
            include_dirs=[numpy.get_include()],
            extra_compile_args=[
                "-O3",
                "-Wno-cpp", "-Wno-unused-function"
            ]
        )
        setup_config.update(cython_config)
    else:
        warnings.warn("Cython is not available. Install it if you want fast DMPs.")

    setup(**setup_config)
