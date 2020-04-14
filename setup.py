import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="epbt", # Replace with your own username
    version="0.0.1",
    author="Fabrice Normandin",
    author_email="fabrice.normandin@gmail.com",
    description="Simple implementation of the [Population Based Training for Loss Function Optimization] paper.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lebrice/pop-based-training",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)