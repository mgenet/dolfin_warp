# FEniCS base
FROM quay.io/fenicsproject/stable:2019.1.0.r3

# basic stuff
RUN apt-get update; apt-get -y install rename; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# python3 stuff
RUN apt-get update; apt-get -y install python3-future python3-numpy python3-scipy python3-tk; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pip install numpy==1.15

# vtk stuff
RUN apt-get update; apt-get -y install libvtk7-dev python3-vtk7; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV CPATH="/usr/include/vtk-7.1":$CPATH

# gnuplot stuff
RUN apt-get update; apt-get -y install gnuplot git; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
WORKDIR /home/fenics
RUN git clone https://github.com/aschn/gnuplot-colorbrewer.git
RUN echo "set loadpath \"/home/fenics/gnuplot-colorbrewer/diverging:/home/fenics/gnuplot-colorbrewer/qualitative:/home/fenics/gnuplot-colorbrewer/sequential\"" > .gnuplot

# optimization stuff
RUN pip install git+https://github.com/dolfin-adjoint/pyadjoint.git@2019.1.0 cma nlopt sklearn

# environment setup
WORKDIR /home/fenics
RUN mkdir code
ENV PYTHONPATH=/home/fenics/code:$PYTHONPATH
ENV PYTHONPATH=/home/fenics/code/myPythonLibrary:$PYTHONPATH
ENV PYTHONPATH=/home/fenics/code/myVTKPythonLibrary:$PYTHONPATH
ENV PATH=/home/fenics/code/myPythonLibrary/myPythonLibrary:$PATH
ENV PATH=/home/fenics/code/myVTKPythonLibrary/myVTKPythonLibrary:$PATH
ENV PATH=/home/fenics/code/vtkpython_cbl:$PATH
