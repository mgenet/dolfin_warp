# dolfin_dic
A set of FEniCS- and VTK-based python tools for Finite Element Digital Image Correlation.
### Requirements
First you need to install [myPythonLibrary](https://github.com/mgenet/myPythonLibrary) as well as [myVTKPythonLibrary](https://github.com/mgenet/myVTKPythonLibrary). You also need a working installation of FEniCS (including DOLFIN python interface) & VTK (also including python interface).
### FEniCS installation
Install FEniCS from source using hashdist ([link](https://fenicsproject.org/download/installation_using_hashdist.html)):
```
git clone https://bitbucket.org/fenics-project/fenics-developer-tools.git
cd fenics-developer-tools
cp install/profiles/fenics.Linux.yaml fenics.yaml
```
In `fenics.yaml`, add `numpy` & `scipy` in the list of requested packages, and set `vtk_wrap_python` to true. Then run the installation again:
```
./install/fenics-install.sh fenics.yaml
```
### Installation
Get the code:
```
git clone git@bitbucket.org:mgenet/dolfin_dic.git
```
To be able to load the library within python, the simplest is to add the folder containing `dolfin_dic` to the `PYTHONPATH` environment variable:
```
export PYTHONPATH=$PYTHONPATH:/path/to/folder
```
(To make this permanent, add the line to `~/.bashrc`.) Then you should be able to load the library within python:
```
> import dolfin_dic as ddic
```