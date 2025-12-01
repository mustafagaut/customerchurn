from setuptools import setup, find_packages

setup(
    name="telco-churn-ml",
    version="0.1.0",
    description="Telco Customer Churn Prediction Model",
    author="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "category_encoders>=2.5.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "jupyter>=1.0.0",
        ]
    },
)
