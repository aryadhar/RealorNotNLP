"""The setup script."""

from setuptools import setup, find_packages

project_name = "RealOrNot"  # for PyPI listing

requirements = [
    "psycopg2-binary",
    "pandas==1.1.0",
    "loguru",
    "nltk",
    "sklearn",
    "matplotlib",
    "requests",
    "seaborn",
    "gender_guesser",
    "spacy==2.3.2",
    "pyLDAvis==2.1.2",
    "gensim==3.8.3",
    "flask",
    "gunicorn",
    "scikit-image",
    "waitress",
    "wordcloud",
    "marshmallow",
    "marshmallow_jsonapi",
    "plotly",
    "nbformat",
    "dictdiffer",
    "statsmodels",
]
setup(
    author="Arya Dha",
    author_email="...",
    python_requires=">=3.5",
    description="Real or Not using NLP",
    install_requires=requirements,
    name=project_name,
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/aryadhar/RealorNotNLP",
    version="0.1.0",
    zip_safe=False,
)
