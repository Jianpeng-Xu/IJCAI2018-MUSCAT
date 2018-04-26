Description of data:
1. The data used in this work is in folder ./data. 
2. There are four files in data file, each for one response variable (including tmax, tmean, tmin, and prcp) 
Each file contains the following variables:
--- X1: S*T*d1 data tensor for resolution 1
--- X2: S*T*d2 data tensor for resolution 2
--- X: S*T*(d1+d2) data tensor for all resolutions
--- Y: S*T response variable
--- stationLat
--- stationLon

Description of codes:
--- main.m: This is the file that you can run MUSCAT with proper settings.
--- muscat_incremental_sparsa_space.m: This file solves the optimization for MUSCAT with incremental over space.
--- muscat_incremental_sparsa_time_preUpdate.m and muscat_incremental_sparsa_time_postUpdate.m: these two files are for MUSCAT with incremental over time. See main.m for the detail of how these two files are used. 

Other codes are to support the optimization for MUSCAT, including the codes under folder 'private'.
