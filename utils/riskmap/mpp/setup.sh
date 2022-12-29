#!/bin/bash

ROOT_DIR=$(cd $(dirname "$0"); pwd)

cd ${ROOT_DIR}/lib/thirdparty/osqp

if ! [ -d "${ROOT_DIR}/lib/thirdparty/osqp/build" ]
then
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="../install"
    make -j4 && make install
fi

cd ${ROOT_DIR}/lib

if  [ -d "${ROOT_DIR}/lib/build" ] 
then
    rm -rf build
fi

mkdir build && cd build
cmake ../
make -j6
cp ./frenet*.so ../

# optional
export PYTHONPATH=$PYTHONPATH:${ROOT_DIR}/lib/build
pybind11-stubgen -o ./ frenet
cp ./frenet-stubs/__init__.pyi ../frenet.pyi