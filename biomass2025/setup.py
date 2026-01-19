"""Setup configuration for Biomass Estimation"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="biomass-estimation",
    version="1.0.0",
    author="Nguyen Van Quy and Nguyen Hong Hai",
    author_email="support@biomass-estimation.com",
    description="Interactive ML dashboard for above-ground biomass prediction using satellite data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MichaelTheAnalyst/BioVision-Analytics-Hub",
    project_urls={
        "Bug Tracker": "https://github.com/MichaelTheAnalyst/BioVision-Analytics-Hub/issues",
        "Documentation": "https://github.com/MichaelTheAnalyst/BioVision-Analytics-Hub/blob/main/docs/",
        "Source Code": "https://github.com/MichaelTheAnalyst/BioVision-Analytics-Hub",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "biovision-dashboard=dashboard.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "machine-learning",
        "biomass-prediction",
        "satellite-data",
        "gedi",
        "sentinel",
        "remote-sensing",
        "streamlit",
        "data-science",
    ],
)

