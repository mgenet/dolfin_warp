FROM dolfin_dic-dev

RUN apt-get update; apt-get -y install git; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
WORKDIR /home/fenics/code
RUN git clone https://gitlab.inria.fr/mgenet/myPythonLibrary.git
RUN git clone https://gitlab.inria.fr/mgenet/myVTKPythonLibrary.git
RUN git clone https://gitlab.inria.fr/mgenet/vtkpython_cbl.git
RUN git clone https://gitlab.inria.fr/mgenet/dolfin_cm.git
RUN git clone https://gitlab.inria.fr/mgenet/dolfin_dic.git
