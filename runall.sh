#!/bin/bash

circuits='or sum adder semisum'

for circuit in $circuits
do
    python circuit.py $circuit
    python nn.py
    python regressors.py all
    python graph.py $circuit
done
