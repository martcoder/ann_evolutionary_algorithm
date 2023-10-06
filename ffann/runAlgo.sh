#!/bin/sh

rm log.txt
rm oneTime*
python3 writeLinearData.py
source compile.sh
./algorithm.exe r
python3 plotData.py
