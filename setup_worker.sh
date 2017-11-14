#!/bin/bash

wget https://github.com/themcaffee/neural-network-genetic-algorithm/archive/distributed.zip
sudo apt install unzip
unzip distributed.zip
cd neural-network-genetic-algorithm
sudo pip3 install -r requirements.txt
