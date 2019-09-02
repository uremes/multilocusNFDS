#! /bin/sh
g++ -Werror -Wall -g -O2 -o $1/freqDepSelect  $1/main.cpp  $1/functions.cpp  -I /usr/local/include/ -L /usr/local/lib/ -lgsl -lgslcblas
