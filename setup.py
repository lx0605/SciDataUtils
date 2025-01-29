from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mypyfuns",  # Package name
    version="0.1.0",  # Initial version
    author="Xiao Luo",
    author_email="luoxiaoustc@outlook.com",
    description="A Python package for scientific computing and data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lx0605/SciDataUtils",  # Your GitHub repo URL
    project_urls={
        "Bug Tracker": "https://github.com/lx0605/SciDataUtils/issues",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "plotly",
        "pyvista",
        "scipy",
        "scikit-learn",
        "torch",
        "h5py",
    ],
    extras_require={
        "dev": ["pytest", "black", "twine"],
    },
)