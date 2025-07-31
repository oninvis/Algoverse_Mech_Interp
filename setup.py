from setuptools import setup, find_packages

setup(
    name="algo-neutrality",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "datasets",
        "huggingface-hub",
        "ipython",
        "jupyter",
        "plotly",
        "transformer-lens",
        "pandas",
        "torch",
    ],
    python_requires=">=3.12",
) 