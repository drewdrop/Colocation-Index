# Colocation Index or Coagglomeration Index
### Author： Lu Chen (Southeast University ,China) 

Industrial Colocation index is based on the researches of Billings and Johnson(2016) and Lu Chen et al.(2020). The index uses the longitude and latitude coordinates of enterprises to study the spatial group location rules among industries.

### Description
The program is based on the Wasserstein distance algorithm of machine learning, combined with Sinkhorn and entropy regularity optimization algorithm, to calculate the similarity of the two industries in the spatial distribution. Finally, we calculate the Colocation Index through Monte Carlo simulation and hypothesis test.


### Installation
The core call module is ot. Install the module by "pip install POT".
install CUDA

### Files
**example.csv**: We provide a demo file which includes all enterprises in a certain area of China. The core indicators include three digit industry code（SIC3） and geographic information (longitude and latitude).

**SIC3.csv**: This file inculdes all the three digit industry code of this certain area of China
