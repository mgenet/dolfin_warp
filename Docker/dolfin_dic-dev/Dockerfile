FROM quay.io/fenicsproject/stable:2019.1.0.r3

RUN apt-get update; apt-get -y install python-future python-numpy; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update; apt-get -y install libvtk7-dev python-vtk6 python3-vtk7; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV CPATH="/usr/include/vtk-7.1":$CPATH

RUN apt-get update; apt-get -y install gnuplot git; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
WORKDIR /home/fenics
RUN git clone https://github.com/aschn/gnuplot-colorbrewer.git
RUN echo "set loadpath \"/home/fenics/gnuplot-colorbrewer/diverging:/home/fenics/gnuplot-colorbrewer/qualitative:/home/fenics/gnuplot-colorbrewer/sequential\"" > .gnuplot

WORKDIR /home/fenics
RUN mkdir code
ENV PYTHONPATH=/home/fenics/code:$PYTHONPATH
ENV PATH=/home/fenics/code/myPythonLibrary:/home/fenics/code/myVTKPythonLibrary:/home/fenics/code/vtkpython_cbl:$PATH
