from setuptools import setup
from setuptools import find_packages

with open("./requirements.txt", "r", encoding = "utf-8") as f:
    install_requires = f.read()

with open("./README.md", "r", encoding = "utf-8") as f:
    long_description = f.read()

setup(
    name = "cltask",
    version = "0.1.0",
    author = "AllenYolk (Huang Yifan, from Peking University)",
    author_email = "huang2627099045@gmail.com",
    description = ("Continual learning tasks implemented by Pytorch."),
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/AllenYolk/continual-learning-tasks",
    packages = find_packages(),
    install_requires = install_requires,
    classifiers = [
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Science/Engineering :: Artificial Intelligence",
    ],
    python_requires = ">=3.6",
)