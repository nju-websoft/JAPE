# JAPE
Source code and datasets for [ISWC2017](https://iswc2017.semanticweb.org/) research paper "Cross-lingual entity alignment via joint attribute-preserving embedding", a.k.a., JAPE.

## Code
The correspondence between python files and our JAPE variants is as follows:
* se_pos.py == SE w/o neg   
* se_pos_neg.py == SE   
* cse_pos_neg.py == SE + AE  

To run SE, please use:   
<code> python3 se_pos.py ../data/dbp15k/zh_en/ 0.3 </code>

To learn attribute embeddings, please use:   
<code> python3 attr2vec.py ../data/dbp15k/zh_en/ ../data/dbp15k/zh_en/0_3/ ../data/dbp15k/zh_en/all_attrs_range ../data/dbp15k/en_all_attrs_range  </code>

To calculate entity similarities, please use:   
<code> python3 ent2vec_sparse.py ../data/dbp15k/zh_en/ 0.3 0.95 0.95 0.9 </code>

#### Dependencies
* Python 3
* Tensorflow 1.2 
* Scipy
* Numpy

## Datasets
In our experiment, we do not use all the triples in datasets. For relationship triples, we select a portion whose head and tail entities are popular. For attribute triples, we discard their values due to diversity and cross-linguality.

The whole datasets can be found [here](http://ws.nju.edu.cn/jape/). 

### Directory structure
Take DBP15K (ZH-EN) as an example, the folder 'zh_en' contains:
* all_attrs_range: the range of attributes;
* ent_ILLs: all entity links;
* rel_ILLs: all relationship links;
* s_labels: cross-lingual entity labels of source KG (ZH);
* s_triples: relationship triples of source KG (ZH);
* sup_attr_pairs: all attribute links;
* t_labels: cross-lingual entity labels of target KG (EN);
* t_triples: relationship triples of target KG (EN);
* training_attrs_1: entity attributes in source KG (ZH);
* training_attrs_2: entity attributes in target KG (EN);

## Running and parameters
> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

> If you have any difficulty or question in running code and reproducing expriment results, please email to zqsun.nju@gmail.com and whu@nju.edu.cn.

## Citation
If you use this model or code, please cite it as follows:      
_Zequn Sun, Wei Hu, Chengkai Li. Cross-Lingual Entity Alignment via Joint Attribute-Preserving Embedding. In: ISWC 2017._

## Links
The following links point to some recent work that uses this dataset:
 
Sun, Zequn, et al. [Bootstrapping Entity Alignment with Knowledge Graph Embedding.](https://www.ijcai.org/proceedings/2018/0611.pdf) IJCAI. 2018.  
Wang, Zhichun, et al. [Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks.](http://aclweb.org/anthology/D18-1032) EMNLP, 2018.  

