from setuptools import setup, Extension
import numpy as np

setup(
    name="mlx_streaming_extensions",
    ext_modules=[
        Extension(
            "fast_pread",
            sources=["fast_pread.c"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "fast_streaming",
            sources=["fast_streaming.c"],
            include_dirs=[np.get_include()],
        ),
    ],
)
