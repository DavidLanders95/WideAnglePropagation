from pathlib import Path

from setuptools import find_packages, setup

description = "Wide-angle wave propagation methods for electron microscopy"
readme_path = Path(__file__).with_name("README.md")
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else description
)

setup(
    name="wide-angle-propagation",
    version="0.1.0",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "jax",
        "jaxlib",
        "abtem",
        "ase",
        "matplotlib",
        "tqdm",
        "scipy",
        "diffrax",
    ],
    extras_require={
        "dev": [
            "pytest",
            "jupyter",
            "notebook",
        ],
    },
    include_package_data=True,
)
