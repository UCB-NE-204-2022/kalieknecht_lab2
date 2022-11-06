# NE 204 Lab 1
## Kalie Knecht
Lab Partners: Ian Kolaja, Trevor Arino, <s>Karishma Shah</s>

Data used for lab 1 can be downloaded [here](https://drive.google.com/drive/folders/17KCtdw0pnYLPe2_8L5wQfc0XqgJNOaZC?usp=sharing). Please save lab data in 'data' folder.

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
Tools developed for this laboratory are in `spectrum.py`, `filters.py`, and `tools.py`. The bulk of the calibration procedures are in `spectrum.py`, while the filtering and filter parameter optimization is done in `filters.py`. `tools.py` contains some low level tools.

Follow the `Analysis.ipynb` notebook to see how the methods are used to analyze raw waveforms. The functions at least all have function headers to help understand what everything is doing, but improvements could still be made to the documentation.

## Data information
* all files used `ne204.json` configuration for data acquisition
* cs.h5
    * collection date: 10/28/22
    * sources used:
        * Cs-137
    * 10 minute data acquisiton
    * 25 cm source-detector distance
* co.h5
    * collection date: 10/28/22
    * sources used:
        * Co-60
    * 10 minute data acquisition
    * 25 cm source-detector distance
* ba.h5
    * collection date: 10/28/22
    * sources used:
        * Ba-133
    * 10 minute data acquisition
    * 25 cm source-detector distance
* pulser.h5
    * collection date: 10/28/22
    * Pulse generator connected to pre-amp test input
    * Pulser settings:
        * 7 ms wide
        * 30 Hz frequency (33 ms period)
        * 300 mV height
        * Verified on oscilloscope
    * 5 minute data acquisition
