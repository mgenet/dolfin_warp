# FEniCS base

# FROM ubuntu:latest
# FROM quay.io/fenicsproject/stable:2016.1.0
# FROM quay.io/fenicsproject/stable:2016.2.0
# FROM quay.io/fenicsproject/stable:2017.1.0
FROM quay.io/fenicsproject/stable:2017.2.0
# FROM quay.io/fenicsproject/stable:current
# FROM quay.io/fenicsproject/stable:latest

# USER root

# RUN apt-get update; apt-get -y upgrade; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# RUN apt-get update; apt-get install -y software-properties-common; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# RUN add-apt-repository -y ppa:fenics-packages/fenics-2016.1.0
# RUN add-apt-repository -y ppa:fenics-packages/fenics-2016.2.0
# RUN add-apt-repository -y ppa:fenics-packages/fenics-2017.1.0
# RUN add-apt-repository -y ppa:fenics-packages/fenics-2017.2.0
# RUN add-apt-repository -y ppa:fenics-packages/fenics
# RUN apt-get update; apt-get install -y fenics; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# basic stuff
RUN apt-get update; apt-get -y install rename; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# python stuff
RUN apt-get update; apt-get -y install python-future python-numpy python-scipy python-tk; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pip install numpy==1.15

# vtk stuff
RUN apt-get update; apt-get -y install libvtk6-dev python-vtk6; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV CPATH="/usr/include/vtk-6.2":$CPATH

# gnuplot stuff
RUN apt-get update; apt-get -y install gnuplot git; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
WORKDIR /home/fenics
RUN git clone https://github.com/aschn/gnuplot-colorbrewer.git
RUN echo "set loadpath \"/home/fenics/gnuplot-colorbrewer/diverging:/home/fenics/gnuplot-colorbrewer/qualitative:/home/fenics/gnuplot-colorbrewer/sequential\"" > .gnuplot

# environment setup
WORKDIR /home/fenics
RUN mkdir code
ENV PYTHONPATH=/home/fenics/code:$PYTHONPATH
ENV PYTHONPATH=/home/fenics/code/myPythonLibrary:$PYTHONPATH
ENV PYTHONPATH=/home/fenics/code/myVTKPythonLibrary:$PYTHONPATH
ENV PATH=/home/fenics/code/myPythonLibrary/myPythonLibrary:$PATH
ENV PATH=/home/fenics/code/myVTKPythonLibrary/myVTKPythonLibrary:$PATH
ENV PATH=/home/fenics/code/vtkpython_cbl:$PATH

#USER root
