import io
import os
from setuptools import setup, find_packages

# ------------------------
# Package Metadata
# ------------------------

PACKAGE_NAME = "rag-assisted-chatbot"      # pip name
MODULE_NAME = "rag_assisted_chatbot"       # import name

DESCRIPTION = (
    "RAG Assisted Chatbot to query GitHub profile and Resume for "
    "intelligent HR screening and question answering."
)

AUTHOR = "Vijay Dipak Takbhate"
AUTHOR_EMAIL = "vijay.takbhate@incred.com"
URL = "https://github.com/Vijay-Takbhate-incred/simulator.git"
REQUIRES_PYTHON = ">=3.9"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


# ------------------------
# Helpers
# ------------------------

def load_requirements(filename="requirements.txt"):
    try:
        with open(os.path.join(BASE_DIR, filename), encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    except FileNotFoundError:
        return []


def load_readme():
    try:
        with io.open(os.path.join(BASE_DIR, "README.md"), encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return DESCRIPTION


# ------------------------
# Setup
# ------------------------

setup(
    name=PACKAGE_NAME,
    version="4.0.0",
    description="RAG Assisted Chatbot Package",
    long_description=load_readme(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    python_requires=REQUIRES_PYTHON,

    packages=find_packages(),               # finds rag_assisted_chatbot
    install_requires=load_requirements(),

    include_package_data=True,
    license="MIT",
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
