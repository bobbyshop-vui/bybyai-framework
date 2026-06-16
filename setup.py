from setuptools import setup, find_packages

setup(
    name="metal_ai",
    version="0.1.0",
    author="bobbydeveloper2014",
    description="macOS Metal GPU framework for Transformer LM training and inference",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    py_modules=["metal_ai"],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "tinygrad",
        "pyobjc-framework-Foundation",
        "pyobjc-framework-Metal",
        "pyobjc-core",
        "torch",
        "safetensors",
        "tqdm",
    ],
    extras_require={
        "dev": ["cython", "setuptools"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
