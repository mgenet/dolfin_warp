FROM registry.gitlab.inria.fr/mgenet/dolfin_warp/dolfin_warp-2019.1.0-dev

# RUN pip install myPythonLibrary
# RUN pip install myVTKPythonLibrary

RUN apt-get update; apt-get -y install git; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
WORKDIR /home/fenics/code
RUN git clone https://gitlab.inria.fr/mgenet/myPythonLibrary.git
RUN git clone https://gitlab.inria.fr/mgenet/myVTKPythonLibrary.git
RUN git clone https://gitlab.inria.fr/mgenet/vtkpython_cbl.git
RUN git clone https://gitlab.inria.fr/mgenet/dolfin_mech.git
RUN git clone https://gitlab.inria.fr/mgenet/dolfin_warp.git
