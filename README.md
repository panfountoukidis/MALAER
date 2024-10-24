# MALAER
This reposiratory contains info about the MALAER (MAchine Learning-based AERosol) retrieval algorithm.

# Description of the files:
1) The MALAER_ufs.py file, contains several utility functions (ufs) which are used in the MALAER algorithm. More detailed info are included in each function,
in the file itself.

2) The MALAER_main.py file, is the main file which contains the MALAER algorithm. More details are included in the file.


# Description of the MALAER retrieval algorithm:
The aim of the MALAER algorithm is the retrieval of AOD and PM2.5 surface concentrations, with the use of the L1 Radiance (RA) and Irradiance (IRR) S5P/TROPMI satllite data, based on Machine Learning (ML) techniques.
To do so, the algorithm takes as data, several other parameters. It utilizes several L2 data from S5P/TROPOMI satellite (O3, Cloud, TCWV, Wind), data from the CAMS model (relative humidity, specific humidity and boundary
layer height), data from AERONET (AOD) and data from the European Environmental Agency (PM2.5 surface concentrations). The last two data are the target parameters for the ML models.
