FROM dolfin_dic-dev

RUN apt-get update; apt-get install -y git; apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
WORKDIR /home/fenics/code
RUN git clone https://github.com/mgenet/myPythonLibrary.git
RUN git clone https://github.com/mgenet/myVTKPythonLibrary.git
RUN git clone https://github.com/mgenet/vtkpython_cbl.git
RUN git clone https://bitbucket.org/mgenet/dolfin_dic.git
