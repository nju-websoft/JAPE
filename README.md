# JAPE
Source code and datasets of [ISWC2017](https://iswc2017.semanticweb.org/) paper "cross-lingual entity alignment via joint attribute-preserving embedding", a.k.a. JAPE.

## Code

#### Dependencies
* Python 3
* Tensorflow 1.2 
* Scipy
* Numpy

## Datasets

In our experiment, we do not use all the triples in datasets. For relationship triples, we select a portion whose head and tail entities are popular. For attribute triples, we discard their values due to diversity and cross-linguality.

The whole datasets can be find [here](http://ws.nju.edu.cn/jape/). 

The folder id_data contains

## Running and parameters

> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

> If you have any difficulty or question in running code and reproducing expriment results, please email to zqsun.nju@gmail.com and whu@nju.edu.cn.

## Cite
If you use this model or code, please cite it as follows:      
_Zequn Sun, Wei Hu, Chengkai Li. Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding. In: ISWC 2017._
