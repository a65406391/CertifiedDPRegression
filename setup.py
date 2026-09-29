from setuptools import find_packages, setup

setup(
    name="private_regression_agt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "tqdm",
    ],
)
