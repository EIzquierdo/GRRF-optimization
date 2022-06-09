# GRRF-optimization for remote sensing data
Demo of the paper Izquierdo-Verdiguier, E. and Zurita-Milla, R., 2020. An evaluation of Guided Regularized Random Forest for classification and regression tasks in remote sensing. International Journal of Applied Earth Observation and Geoinformation, 88, p.102051.

The authors would like to thank to Houtao Deng for developing the [RRF package](https://cran.r-project.org/web/packages/RRF/RRF.pdf) which GRRF-optimization for remotes sensing data was based. 

This code was also used in the following contributions:
+ Izquierdo-Verdiguier, E., Zurita-Milla, R., & Rolf, A. (2017). On the use of guided regularized random forests to identify crops in smallholder farm fields. In 2017 9th International Workshop on the Analysis of Multitemporal Remote Sensing Images (MultiTemp) (pp. 1-3). IEEE. Available [here](https://ieeexplore.ieee.org/document/8035248).
+ Izquierdo-Verdiguier, E. and Zurita-Milla, R. (2018). Use of Guided Regularized Random Forest for Biophysical Parameter Retrieval. IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium, pp. 5776-5779, IEEE. Available [here](https://ieeexplore.ieee.org/document/8517920)
+ Aguilar, R., Zurita-Milla, R., Izquierdo-Verdiguier, E., & A De By, R. (2018). A cloud-based multi-temporal ensemble classifier to map smallholder farming systems. Remote sensing, 10(5), 729. Available [here](https://www.mdpi.com/2072-4292/10/5/729).
+ Sanchez-Ruiz, S., Moreno-Martinez, A., Izquierdo-Verdiguier, E., Chiesi, M., Maselli, F., & Gilabert, M. A. (2019). Growing stock volume from multi-temporal landsat imagery through google earth engine. International Journal of Applied Earth Observation and Geoinformation, 83, 101913. Available [here](https://www.sciencedirect.com/science/article/pii/S0303243419301898?casa_token=YoLsQdtrtysAAAAA:MDCUBgYMZpyTxib2PffDOhxokS3kEXJFSQKA92qawAc-U31HwdNUTc6VXh55XoTGfJ-qKnY6).
+ Quesada-Ruiz, L., Rodriguez-Galiano, V. F., Zurita-Milla, R., & Izquierdo-Verdiguier, E. (2021). Area and Feature Guided Regularised Random Forest: a novel method for predictive modelling of binary phenomena. The case of illegal landfill in Canary Island. International Journal of Geographical Information Science. doi: 10.1080/13658816.2022.2075879. Available [here](https://www.tandfonline.com/doi/full/10.1080/13658816.2022.2075879?cookieSet=1).

The code uses the [Salinas hyperspectral image](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas) to show a classification example.

## Install:
[RRF package](https://cran.r-project.org/web/packages/RRF/RRF.pdf) must be installed in R.

To install the libraries:

    pip install -r requirements.txt

## Citation:
If you use this code or part of this code in your work, please cite it as follow:

**APA**:

Izquierdo-Verdiguier, E., & Zurita-Milla, R. (2021). GRRF-optimization: Guided Regularized Random Forest optimization for Resmote Sensing data (Version 0.3) [Computer software]. https://doi.org/10.5281/zenodo.5287370

**BibTeX**:

@misc{GRRF_optimization21, author = {Izquierdo-Verdiguier, Emma and Zurita-Milla, Raul}, doi = {10.5281/zenodo.5287370}, month = {8}, title = {{GRRF-optimization: Guided Regularized Random Forest optimization for Resmote Sensing data}}, year = {2021} }


