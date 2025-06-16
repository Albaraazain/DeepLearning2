"""
Setup script for the Deep Learning project
"""

from setuptools import setup, find_packages

setup(
    name="deep-learning-the2",
    version="1.0.0",
    description="Deep Learning Course Project - THE2 (CENG403)",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "dl-project=main:main",
        ],
    },
)