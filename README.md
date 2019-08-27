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
Take DBP15K (ZH-EN) as an example, the folder "zh_en" contains:
* all_attrs_range: the range code of attributes in source KG (ZH);
* ent_ILLs: all entity links (15K);
* rel_ILLs: all relationship links (with the same URI or localname);
* s_labels: cross-lingual entity labels of source KG (ZH);
* s_triples: relationship triples of source KG (ZH);
* sup_attr_pairs: all attribute links (with the same URI or localname);
* t_labels: cross-lingual entity labels of target KG (EN);
* t_triples: relationship triples of target KG (EN);
* training_attrs_1: entity attributes in source KG (ZH);
* training_attrs_2: entity attributes in target KG (EN);

On top of this, we built 5 datasets (0_1, 0_2, 0_3, 0_4, 0_5) for embedding-based entity alignment models. "0_x" means that this dataset uses "x0%" entity links as training data and uses the rest for testing. The two entities of each entity link in training data have the same id. In our main experiments, we used the dataset in "0_3" which has 30% entity links as training data.

The folder "mtranse" contains the corresponding 5 datasets for MTransE. The difference lies in that the two entities of each entity link in training data have different ids.

### Dataset files
Take the dataset "0_3" of DBP15K (ZH-EN) as an example, the folder "0_3" contains:
* ent_ids_1: ids for entities in source KG (ZH);
* ent_ids_2: ids for entities in target KG (EN);
* ref_ent_ids: entity links encoded by ids for testing;
* ref_ents: URIs of entity links for testing;
* rel_ids_1: ids for relationships in source KG (ZH);
* rel_ids_2: ids for relationships in target KG (EN);
* sup_ent_ids: entity links encoded by ids for training;
* sup_rel_ids: relationship links encoded by ids for training;
* triples_1: relationship triples encoded by ids in source KG (ZH);
* triples_2: relationship triples encoded by ids in target KG (EN);

## Running and parameters
> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

> If you have any difficulty or question in running code and reproducing expriment results, please email to zqsun.nju@gmail.com and whu@nju.edu.cn.

## Citation
If you use this model or code, please cite it as follows:      
```
@inproceedings{JAPE,
  author    = {Zequn Sun and Wei Hu and Chengkai Li},
  title     = {Cross-Lingual Entity Alignment via Joint Attribute-Preserving Embedding},
  booktitle = {ISWC},
  pages     = {628--644},
  year      = {2017}
}
```

## Links
The following links point to some recent work that uses our datasets:
 
* Zequn Sun, et al. [Bootstrapping Entity Alignment with Knowledge Graph Embedding.](https://www.ijcai.org/proceedings/2018/0611.pdf) In: IJCAI 2018.  
* Zhichun Wang, et al. [Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks.](http://aclweb.org/anthology/D18-1032) In: EMNLP 2018.   
* Ning Pang, et al. [Iterative Entity Alignment with Improved Neural Attribute Embedding.](http://ceur-ws.org/Vol-2377/paper_5.pdf) In: DL4KG 2019. 
* Kun Xu, et al. [Cross-lingual Knowledge Graph Alignment via Graph Matching Neural Network.](https://www.aclweb.org/anthology/P19-1304) In: ACL 2019.  
* Yixin Cao, et al. [Multi-Channel Graph Neural Network for Entity Alignment.](https://www.aclweb.org/anthology/P19-1140) In: ACL 2019.  
* Yuting Wu, et al. [Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs.](https://www.ijcai.org/proceedings/2019/0733.pdf) In: IJCAI 2019.   
* Qiannan Zhu, et al. [Neighborhood-Aware Attentional Representation for Multilingual Knowledge Graphs.](https://www.ijcai.org/proceedings/2019/0269.pdf) In: IJCAI 2019.  
* Fan Xiong, et al. [Entity Alignment for Cross-lingual Knowledge Graph
with Graph Convolutional Networks.](https://www.ijcai.org/proceedings/2019/0929.pdf) In: IJCAI 2019 Doctoral Consortium.
* Tingting Jiang, et al. [Two-Stage Entity Alignment: Combining Hybrid Knowledge Graph Embedding with Similarity-Based Relation Alignment.](https://link.springer.com/chapter/10.1007/978-3-030-29908-8_13) In: PRICAI 2019.  

