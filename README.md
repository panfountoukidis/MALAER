# MALAER
This reposiratory contains the Python files, which are used in the MALAER (MAchine Learning-based AERosol) retrieval algorithm.

# Description of the files:
1) The MALAER_main.py file, is the main file which contains the MALAER algorithm. More details are included in the file.

2) The MALAER_ufs.py file, contains several utility functions (ufs) which are used in the MALAER algorithm. More detailed info are included in each function,
in the file.

3) The download_data.py file, can be used to download several data, which are used in the MALAER algorithm as features/targets. More details are included in the file.

4) The make_s5p_overpass.py file, can be used to make overpass files of a specific area, from the S5P/TROPOMI file. More details are included in the file.


# Description of the MALAER retrieval algorithm:
The aim of the MALAER algorithm is the retrieval of AOD and PM2.5 surface concentrations, with the use of the L1 Radiance (RA) and Irradiance (IRR) S5P/TROPMI satllite data, based on Machine Learning (ML) techniques.
To do so, the algorithm takes as data, several other parameters. It utilizes several L2 data from S5P/TROPOMI satellite (O3, Cloud, TCWV, Wind), data from the CAMS model (relative humidity, specific humidity and boundary
layer height), data from AERONET (AOD) and data from the European Environmental Agency (PM2.5 surface concentrations). The last two data are the target parameters for the ML models.
