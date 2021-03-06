# PDOC-V
The python code for PDOC-V as described in our paper "Instance Selection via Voronoi Neighbors for Binary Classification Tasks"

## Files
	data/						collected datasets
	src/						python source code, see README.md for details
	result/						computational results
	    data_knn/                       Nearest neighbors for datasets generated by exp.py
	    exp-2022-05-20/                 Computational results generated by exp.py
	    summary_natural.csv             Complete results on 3 existing synthetic datasets and 6 natural datasets
	    summary_natural_baseline.csv    Results on fullset of 9 existing datasets
	    summary_synthetic.csv           Computational results on 12 new synthetic datasets
	    summary_synthetic_baseline.csv  Results on fullset of 12 new synthetic datasets
	    Table2-baseline.csv                 Table 2 in our paper
	    Table3-min-alpha-for-target.csv     Table 3 in our paper
	    Table4-ARF1-ART-at-alpha=0.2.csv    Table 4 in our paper

## Steps to repeat our experiments

1. Run experiments, typically on a few computers.

```
    python exp.py
```
results are stored in result/exp-2022-05-20/**/summary.csv

