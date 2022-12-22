# NE 204 Lab 2
## Kalie Knecht
Lab Partners: Ian Kolaja and Trevor Arino

Data used for lab 2 can be downloaded [here](https://drive.google.com/drive/folders/1OlxsxKPx0cMWF8GUOX0B5gkdGfqoI0pU?usp=sharing). Please save lab data in 'data' folder.

This repo is forked from my lab 1 repository.

## Dependencies
* Code developed and tested in Python version 3.8.13
* Packages used:
    * numpy
    * matplotlib
    * h5py
    * scipy
    * sys
    * jupyter notebook/jupyter-lab

## Pulse processing framework
Tools developed for this laboratory are in `spectrum.py`, `filters.py`, and `tools.py`. The bulk of the calibration procedures are in `spectrum.py`, while the filtering and filter parameter optimization is done in `filters.py`. `tools.py` contains some low level tools. Many new tools for this laboratory are in `pulse_shape.py`. 

Follow the `Analysis.ipynb` notebook to see how the methods are used to analyze raw waveforms. The functions at least all have function headers to help understand what everything is doing, but improvements could still be made to the documentation.

## Data information
* all files used `ne204.json` configuration for data acquisition
* cs.h5
    * collection date: 10/28/22
    * sources used:
        * Cs-137
    * 10 minute data acquisiton
    * 25 cm source-detector distance
