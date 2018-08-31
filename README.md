# dolfin_dic
A set of FEniCS- and VTK-based python tools for Finite Element Digital Image Correlation.
### Requirements
First you need to install [myPythonLibrary](https://gitlab.inria.fr/mgenet/myPythonLibrary) as well as [myVTKPythonLibrary](https://gitlab.inria.fr/mgenet/myVTKPythonLibrary). You also need a working installation of FEniCS (including DOLFIN python interface) & VTK (also including python interface).
### Installation
Get the code:
```
git clone https://gitlab.inria.fr/mgenet/dolfin_dic.git
```
To be able to load the library within python, the simplest is to add the folder containing `dolfin_dic` to the `PYTHONPATH` environment variable:
```
export PYTHONPATH=$PYTHONPATH:/path/to/folder
```
(To make this permanent, add the line to `~/.bashrc`.)
Then you should be able to load the library within python:
```
import dolfin_dic as ddic
```