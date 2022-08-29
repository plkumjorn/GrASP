from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="grasptext",
    version="0.0.2",
    description="GReedy Augmented Sequential Patterns: an algorithm for extracting patterns from text data",
    py_modules=["grasptext"],
    package_dir={"": "grasptext"},
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/plkumjorn/GrASP",
    author="Piyawat Lertvittayakumjorn",
    author_email="plkumjorn@gmail.com",

    install_requires = [
        "numpy >= 1.16.3",
        "scikit-learn >= 0.23.2",
        "nltk >= 3.2.4",
        "spacy >= 2.0.12",
        "termcolor >= 1.1.0",
        "tqdm >= 4.46.0",
        "Flask >= 0.12.2",
        "patool",
    ],

    extras_require = {
        "dev": [
            "pytest >= 3.7",
            "check-manifest",
            "twine",
        ],
    },
)