# dolfin_warp
A set of FEniCS- and VTK-based python tools for Finite Element Digital Image Correlation, basically implementing the method described in [[Genet et al., 2018, Medical Image Analysis]](https://www.medicalimageanalysisjournal.com/article/S1361-8415(18)30534-6/fulltext).
### Requirements
First you need to install [myPythonLibrary](https://gitlab.inria.fr/mgenet/myPythonLibrary), [myVTKPythonLibrary](https://gitlab.inria.fr/mgenet/myVTKPythonLibrary) as well as [dolfin_cm](https://gitlab.inria.fr/mgenet/dolfin_cm). You also need a working installation of FEniCS (including DOLFIN python interface) & VTK (also including python interface).
### Installation
Get the code:
```
git clone https://gitlab.inria.fr/mgenet/dolfin_warp.git
```
To be able to load the library within python, the simplest is to add the folder containing `dolfin_warp` to the `PYTHONPATH` environment variable:
```
export PYTHONPATH=$PYTHONPATH:/path/to/folder
```
(To make this permanent, add the line to `~/.bashrc`.)
Then you should be able to load the library within python:
```
import dolfin_warp as dwarp
```