import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="turbigen",
    version="0.1.0",
    author="James Brind",
    author_email="turbigen@jamesbrind.uk",
    description="Axial turbine design system.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://jb753.user.srcf.net/compflow-docs/index.html",
    packages=["turbigen"],
    install_requires=[
        "numpy",
        "scipy",
        "compflow",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    keywords=["aerodynamics", "engineering", "turbomachinery", "design"],
)
