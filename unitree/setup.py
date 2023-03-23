"""this file is used to describe your package, its dependencies, and other details."""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="unitree",
    version="0.0.1",
    author="Keith Siilats",
    author_email="keith@siilats.com",
    description="Unitree Z1 robot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/siilats/unitree",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        # List your package dependencies here
        "numpy>=1.19.5",
        "pandas>=1.3.0",
        "matplotlib>=3.4.2",
        "scikit",
        "opencv-python>=4.7"]
)