FROM continuumio/anaconda3:latest
MAINTAINER Pierre-Elouan Rethore <pe@retho.re>

RUN apt-get update                                               \
 && apt-get install -y                                           \
    build-essential \
    gcc \
    gfortran \
    wget \
    binutils \
    gcc \
    libgtk2.0-0 \
    libgtk2.0-dev \
    psmisc



RUN mkdir /install
WORKDIR /install

#ENV REQUIREMENTS requirements1.txt
#ARG REQUIREMENTS

# Install python library requirements to run the notebooks
COPY docker/requirements2.txt /install
RUN pip install --upgrade pip \
 && pip install -r /install/requirements2.txt

## Install PyOptSparse & IpOpt
RUN apt-get update \
 && apt-get install -y \
    mercurial \
    meld
RUN apt-get update \
 && apt-get install -y \
    swig

ENV POSDIR /install/pyoptsparse
ENV IPV 3.11.7
ENV IPOPT_DIR $POSDIR/pyoptsparse/pyIPOPT/Ipopt


RUN hg clone https://bitbucket.org/mdolab/pyoptsparse $POSDIR

# Install Ipopt
COPY docker/install_ipopt2.sh /install
run pwd
run ls
COPY docker/ma27ad.f /install
RUN /install/install_ipopt2.sh
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$IPOPT_DIR/lib

# Install SNOPT
# COPY snopt/* $POSDIR/pyoptsparse/pySNOPT/source/

## Install PyOptSparse
#COPY docker/install_pyoptsparse.sh /install
WORKDIR $POSDIR


# Replace python 3 incomplatible files from pyoptsparse with files from pyipopt
RUN git clone https://github.com/xuy/pyipopt.git && \
    mv ./pyipopt/src/pyipoptcoremodule.c ./pyoptsparse/pyIPOPT/src/pyipoptcoremodule.c && \
    mv ./pyipopt/src/hook.h ./pyoptsparse/pyIPOPT/src/hook.h && \
    mv ./pyipopt/src/callback.c ./pyoptsparse/pyIPOPT/src/callback.c

RUN python setup.py install

#RUN mkdir /notebooks
#WORKDIR /notebooks

#COPY mycert.pem /install/
#COPY mykey.key /install/
#COPY jupyter_notebook_config.py /install/

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini


# Install the Colonel

#RUN mkdir /deb
#WORKDIR /deb
#COPY docker/*.deb /deb/
#RUN dpkg -i *.deb

RUN apt-get clean \
 && apt-get autoremove -y

RUN apt-get install lazarus -y 
 
#RUN mkdir /install
#RUN mkdir /install/source
#RUN mkdir /install/FugaLib
#WORKDIR   /install
#COPY topfarm/cost_models/fuga/Colonel/source/*.pas /install/source/
#COPY topfarm/cost_models/fuga/Colonel/FugaLib/FugaLib.lpr /install/FugaLib/
#COPY topfarm/cost_models/fuga/Colonel/FugaLib/FugaLib.lpi /install/FugaLib

## Build
RUN lazbuild /install/FugaLib/FugaLib.lpi

RUN curl -sL https://deb.nodesource.com/setup_8.x | bash -
RUN apt-get update -y && apt-get install -y nodejs


#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager \
# && jupyter labextension install jupyterlab_bokeh

#ENTRYPOINT ["/usr/bin/tini", "--"]


#EXPOSE 8898

CMD bash

     # CMD jupyter notebook \
     #      --notebook-dir=/notebooks \
     #      --config=/install/jupyter_notebook_config.py \
     #      --allow-root
