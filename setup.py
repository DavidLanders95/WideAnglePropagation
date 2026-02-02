from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wide-angle-propagation",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Wide-angle wave propagation methods for electron microscopy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wide-angle-propagation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "jax",
        "jaxlib",
        "abtem",
        "ase",
        "matplotlib",
        "tqdm",
        "scipy",
    ],
    extras_require={
        "dev": [
            "pytest",
            "jupyter",
            "notebook",
        ],
    },
    package_data={
        "wide_angle_propagation": ["*.cu"],
    },
    include_package_data=True,
)