"""
Setup script for PulmoFoundation package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pulmofoundation",
    version="1.0.0",
    author="Zhengrui Guo",
    author_email="zguobc@connect.ust.hk",
    description="A foundation model for lung pathology whole-slide image analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dddavid4real/PulmoFoundation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="pathology, medical imaging, deep learning, foundation model, lung cancer",
)

