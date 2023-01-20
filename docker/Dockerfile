FROM continuumio/anaconda3:latest
MAINTAINER Mikkel Friis-Møller <mikf@dtu.dk>
ENV PATH="/opt/conda/bin:${PATH}"

# build: 
# docker build . -t dtuwindenergy/topfarm2
# tag:
# docker tag dtuwindenergy/topfarm2 dtuwindenergy/topfarm2:latest (if needed)
# docker tag dtuwindenergy/topfarm2 dtuwindenergy/topfarm2:<version number>
# push:
# docker login
# docker push --all-tags dtuwindenergy/topfarm2

RUN apt-get update
RUN apt-get -y install make pandoc graphviz

RUN conda install mongodb
RUN conda install -c conda-forge ffmpeg

RUN pip install matplotlib
RUN pip install numpy
RUN pip install numpy-financial
RUN pip install openmdao==3.16
RUN pip install pytest
RUN pip install pycodestyle
RUN pip install pytest-cov
RUN pip install pymongo[srv]
RUN pip install scipy
RUN pip install sphinx
RUN pip install sphinx_rtd_theme
RUN pip install scikit-learn
RUN pip install mock
RUN pip install tensorflow
RUN pip install shapely>=1.8.2
RUN pip install memory_profiler
