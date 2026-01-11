"""
Setup script for Chemical Engineering Computational Package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if it exists
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="chemeng",
    version="0.1.0",
    author="Siddharth Srinivasan",
    author_email="",
    description="A comprehensive computational chemical engineering package for calculations and modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chemeng",
    packages=find_packages(exclude=["__pycache__", "*.pyc", "*__pycache__*", "*.pyc"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies - currently only using standard library
        # Add numpy, scipy, etc. if needed in the future
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "all": [
            "numpy>=1.20.0",
            "scipy>=1.7.0",
            "matplotlib>=3.3.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
    ],
    keywords=[
        "chemical engineering",
        "thermodynamics",
        "reaction engineering",
        "heat transfer",
        "mass transfer",
        "fluid mechanics",
        "process control",
        "process economics",
        "transport phenomena",
    ],
)
