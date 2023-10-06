#!/bin/sh

rm log.txt
rm oneTime*
source compile.sh
./algorithm.exe r
python3 plotData.py
