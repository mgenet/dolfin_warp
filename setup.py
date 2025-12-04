import datetime
import os
import setuptools

# version = os.environ['CI_COMMIT_TAG']
version = datetime.date.today().strftime("%Y.%m.%d")

setuptools.setup(
    name="dolfin_warp",
    version=version,
    author="Martin Genet",
    author_email="martin.genet@polytechnique.edu",
    description=open("README.md", "r").readlines()[1][:-1],
    long_description = open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mgenet/dolfin_warp",
    packages=["dolfin_warp"],
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=["gnuplot==5.4", "matplotlib==3.5", "meshio==5.3", "numpy==1.23.5", "pandas==1.3", "scipy==1.9", "vtk==9.2", "myPythonLibrary", "myVTKPythonLibrary", "dolfin_mech"],
)
