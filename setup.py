import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="excolor",
    version="0.1.0",
    author="Tim Pyrkov",
    author_email="tim.pyrkov@gmail.com",
    description="Extended colors for python",
    long_description=read("README.md"),
    license = "MIT License",
    long_description_content_type="text/markdown",
    url="https://github.com/timpyrkov/excolor",
    packages=find_packages(exclude=("docs")),
    package_dir={"": "."},
    package_data={
        "excolor": ["*.py", "*.pyi"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Artistic Software",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "numpy",
        "pillow",
        "matplotlib",
        "opencv-python",
        "pythonperlin",
        "requests",
        "seaborn",
    ],
)

