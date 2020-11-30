from setuptools import setup
import warnings
try:
    from Cython.Build import cythonize
    cython_available = True
except ImportError:
    cython_available = False


if __name__ == "__main__":
    if cython_available:
        setup(
            name="dmps",
            ext_modules=cythonize("dmp_fast.pyx"),
            zip_safe=False,
            compiler_directives={'language_level': "3"},
            extra_compile_args=[
                "-O3",
                # disable warnings caused by Cython using the deprecated
                # NumPy C-API
                "-Wno-cpp", "-Wno-unused-function"
            ]
        )
    else:
        warnings.warn("Cython is not available. Please install it.")