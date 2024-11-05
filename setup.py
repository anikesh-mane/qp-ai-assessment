from setuptools import setup, find_packages

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
        "License :: OSI Approved :: GNU General Public Licence v3.0",  # Replace with your license
        "Operating System :: OS Independent",
    ],
    install_requires=open('requirements.txt').read().splitlines(),
    python_requires=">=3.10",  
)
