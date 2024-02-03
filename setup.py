from setuptools import setup, find_packages

setup(
    name="SRGAN",
    version="0.1.0",
    description="A deep learning project that is build for Super Resolution GAN for the Mnist digit dataset",
    author="Atikul Islam Sajib",
    author_email="atikul.sajib@ptb.de",
    url="https://github.com/atikul-islam-sajib/SRCGAN.git",  # Update with your project's GitHub repository URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="SRCGAN machine-learning",
    project_urls={
        "Bug Tracker": "https://github.com/atikul-islam-sajib/SRCGAN.git/issues",
        "Documentation": "https://atikul-islam-sajib.github.io/SRCGAN-deploy/",
        "Source Code": "https://github.com/atikul-islam-sajib/SRCGAN.git",
    },
)
