# JAPE
Source code and datasets of [ISWC2017](https://iswc2017.semanticweb.org/) research paper "Cross-lingual entity alignment via joint attribute-preserving embedding", a.k.a., JAPE.

## Code
The correspondence between python files and our JAPE variants are as follows:
* se_pos.py == SE w/o neg   
* se_pos_neg.py == SE   
* cse_pos_neg.py == SE+AE  

To run SE, use: <code> python3 se_pos.py ../data/dbp15k/zh_en/ 0.3 </code>

#### Dependencies
* Python 3
* Tensorflow 1.2 
* Scipy
* Numpy

## Datasets
In our experiment, we do not use all the triples in datasets. For relationship triples, we select a portion whose head and tail entities are popular. For attribute triples, we discard their values due to diversity and cross-linguality.

The whole datasets can be found [here](http://ws.nju.edu.cn/jape/). 

## Running and parameters
> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

> If you have any difficulty or question in running code and reproducing expriment results, please email to zqsun.nju@gmail.com and whu@nju.edu.cn.

## Citation
If you use this model or code, please cite it as follows:      
_Zequn Sun, Wei Hu, Chengkai Li. Cross-Lingual Entity Alignment via Joint Attribute-Preserving Embedding. In: ISWC 2017._
