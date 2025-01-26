from setuptools import setup, find_packages

setup(
    name="superweights",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "clize>=5.0.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    author="Codeium",
    description="A package for analyzing and manipulating transformer model weights",
    python_requires=">=3.8",
)
