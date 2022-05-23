## Files
    data_preprocess/        Preprocess natural datasets
        employee_salaries.py
        misc_colleges.py
        mushroom.py
        url_spam.py
    dataset.py              Prepare all datasets in unified format
    exp.py                  Computational experiments. Conduct training and validation and record results
    proximity.py            Implemented a few method to estimate the proximity of an instance to the Optimal Decision Surface
                                computeClosenessToMarginByAlternatingChain: [ENNC][zhu2016extended]
                                computeClosenessToMarginByEntropy:          NeighborEntropy [NE][shin2002pattern]
                                computeClosenessToMarginByFreqeucyInOppKNN: NearCount [NC][zhu2020nearcount] 
    selectTopK.py           Select top K instances for each class based on weight of instance (negative proximity)
                            Consider diversity to ensure selected instances spreading-out
    train_util.py           Load dataset and prepare it for computational experiments
    voronoi_explore.py      PDOC-V and other Voronoi related methods
                                remove_noise_by_voronoi: PDOC-V algorithm
    lsh_methods.py          LSH based instance selection methods
                                LSH_IS_F: [LSH][AlvarLSH2016]


[zhu2016extended]: Fa Zhu, Jian Yang, Junbin Gao and Chunyan Xu, Extended nearest neighbor chain induced instance-weights for SVMs, Pattern Recognition, (2016) Vol. 60, pp. 863--874             
[shin2002pattern]: Hyunjung Shin and Sungzoon Cho, Pattern selection for support vector classifiers, International Conference on Intelligent Data Engineering and Automated Learning, (2002), pp. 469--474
[zhu2020nearcount]: Zonghai Zhu, Zhe Wang, Dongdong Li and Wenli Du, NearCount: Selecting critical instances based on the cited counts of nearest neighbors, Knowledge-Based Systems, (2020) Vol. 190, pp. 105196
[AlvarLSH2016]: Álvar Arnaiz-González, José-Francisco Díez-Pastor, Juan J. Rodríguez, César García-Osorio, Instance selection of linear complexity for big data, Knowledge-Based Systems, (2016) Vol. 107, pp. 83-95
[li2010selecting]: Yuhua Li and Liam Maguire, Selecting critical patterns based on local geometrical and statistical information, IEEE transactions on pattern analysis and machine intelligence, (2010) Vol. 33(6), pp. 1189--1201
