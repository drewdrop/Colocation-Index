# Colocation Index or Coagglomeration Index
### Author： Lu Chen (Southeast University ,China)  

#### Acknowledgment
**Zhen Tang (Loyola Marymount University, USA);**  
**Jun Yuan (New Jersey Institute of Technology, USA);**  
**Xinyue Ye(Texas A&M University, USA)**

### Description

Industrial Colocation index is based on the researches of Billings and Johnson(2016) and Lu Chen et al.(2020). The index uses the longitude and latitude coordinates of enterprises to study the spatial group location rules among industries.  
The program is based on the Wasserstein distance algorithm of machine learning, combined with Sinkhorn and entropy regularity optimization algorithm, to calculate the similarity of the two industries in the spatial distribution. Finally, we calculate the Colocation Index through Monte Carlo simulation and hypothesis test.


### Installation
The core call module is ot. Install the module by "pip install POT"  
install CUDA

### Files
**example.csv**: We provide a demo file which includes all enterprises in a certain area of China. The core indicators include three digit industry code（SIC3 or SIC2）, geographic information (longitude and latitude) and region code(city) .


### References
[1] Cuturi M. Sinkhorn distances: Lightspeed computation of optimal transport[C]. 2013.

[2] Martin A, Chintala S, and Bottou L. Wasserstein generative adversarial networks: International conference on machine learning[Z]. 2017.

[3] Billings S B, and Johnson E B. Agglomeration within an urban area[J]. Journal of Urban Economics. 2016, 91: 13-25.

[4] Lu Chen, Xiuyan Liu, Xinyue Ye and Hanhui Hu. Industrial Coagglomeration and Industrial Spatial Governance of Urban Cluster: Measurement Based on Machine Learning Algorithm and Analysis of Influence Factors[J]. China Industrial Economics,2020(05):99-117.
Chinese Journal version: 陈露，刘修岩，叶信岳，胡汉辉. 城市群视角下的产业共聚与产业空间治理：机器学习算法的测度[J]. 中国工业经济. 2020(05): 99-117.
