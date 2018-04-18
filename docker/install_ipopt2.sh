#!/bin/bash

curl https://www.coin-or.org/download/source/Ipopt/Ipopt-$IPV.tgz > /install/Ipopt-$IPV.tgz
cd /install
tar xvf Ipopt-$IPV.tgz
mv Ipopt-$IPV $IPOPT_DIR
cp /install/ma27ad.f $IPOPT_DIR/ThirdParty/HSLold/
cd $IPOPT_DIR/ThirdParty/Blas
sh ./get.Blas
cd $IPOPT_DIR/ThirdParty/Lapack
sh ./get.Lapack
cd $IPOPT_DIR
./configure --disable-linear-solver-loader
make install
