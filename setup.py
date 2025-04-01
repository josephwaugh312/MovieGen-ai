"""
Setup script for MovieGen package.
"""
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="MovieGen",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A movie recommendation system using the MovieLens dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/MovieGen",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
) 