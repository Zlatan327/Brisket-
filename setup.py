from setuptools import setup, find_packages

setup(
    name="nba-prediction-tool",
    version="1.0.0",
    description="Machine learning system for predicting NBA game winners",
    author="NBA Analytics Team",
    packages=find_packages(),
    install_requires=[
        "xgboost>=3.2.0",
        "scikit-learn>=1.8.0",
        "fastapi>=0.128.0",
        "uvicorn>=0.40.0",
        "pandas>=3.0.0",
        "numpy>=2.4.0",
        "nba-api>=1.11.0",
        "requests>=2.32.0",
        "beautifulsoup4>=4.14.0",
        "python-dotenv>=1.2.0",
        "pydantic>=2.12.0",
        "sqlalchemy>=2.0.40"
    ],
    python_requires=">=3.11",
)
