#!/bin/sh

rm log.txt
rm oneTimeOutput.data
python3 writeLinearData.py
source compile.sh
./algorithm.exe r
python3 plotData.py
