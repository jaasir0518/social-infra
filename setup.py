from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="social-infrastructure-prediction",
    version="0.1.0",
    author="Social Infrastructure Team",
    author_email="team@socialinfra.ai",
    description="Machine learning system for predicting infrastructure maintenance needs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/social-infrastructure-prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.931",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0",
            "torch-gpu>=1.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "social-infra-train=scripts.training.train_models:main",
            "social-infra-predict=scripts.deployment.batch_prediction:main",
            "social-infra-api=api.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv"],
    },
)