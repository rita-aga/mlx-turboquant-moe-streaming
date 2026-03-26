from setuptools import setup, Extension
import numpy as np

setup(
    name="fast_pread",
    ext_modules=[
        Extension(
            "fast_pread",
            sources=["fast_pread.c"],
            include_dirs=[np.get_include()],
        ),
    ],
)
