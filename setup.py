"""
Setup script for the Developmental Trajectories package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="head-trajectories",
    version="1.0.0",
    author="Abderahmane Ainouche",
    author_email="abderahmane.ainouche.ai@gmail.com",
    description="Developmental Trajectories of Attention Heads - Mechanistic Interpretability Research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abderahmane-ai/head-trajectories",
    packages=find_packages(exclude=["tests", "tests.*", "modal_jobs", "modal_jobs.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-probing=run_probing:main",
            "run-analysis=run_analysis:main",
            "run-single-experiment=run_single_experiment:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
