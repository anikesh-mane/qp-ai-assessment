from setuptools import setup, find_
packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qp-chatbot",  
    version="0.1.0",  
    author="Anikesh Mane",  
    author_email="anikeshmane@example.com",  # Replace with your email
    description="A chatbot for question answering",  # Replace with a short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anikesh-mane/qp-ai-assessment",  # Replace with your repository URL
    packages=find_packages(where='src'),  
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "python-dotenv",
        "fastapi",
        "uvicorn",
        "pymilvus",
        "pymilvus[model]",
        "openai",
        "PyPDF2",
        "langchain",
        "langchain_community",
        "langchain_huggingface",
        "langchain-milvus",
        "FlagEmbedding>=1.2.2",
        "datasets",
        "peft",
        "nltk",
    ],  # Copied from requirements.txt
    python_requires=">=3.10",  
    entry_points={
        'console_scripts': [
            'qp-chatbot=main:app',  
        ],
    },
)
