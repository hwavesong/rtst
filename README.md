# Right Triangle Similarity Transformation Algorithm

This project hosts the official implementation for our ICANN 2020 paper:

Embedding Compression with Right Triangle Similarity Transformations[[Springer Link](https://link.springer.com/chapter/10.1007/978-3-030-61616-8_62)].

## Abstract
Word embedding technology has promoted the development of many NLP tasks. However, these embeddings often require a lot of storage, memory, and computation, resulting in low efficiency in NLP tasks. To address the problem, this paper proposes a new method for compressing word embeddings. We sample a set of orthogonal vector pairs from word embedding matrix in advance. Then these vector pairs are fed into a neural network sharing weights (i.e., Siamese network), and low-dimensional forms of the vector pairs are obtained. We get two vector triplets by adding the subtraction results of the vector pairs, respectively, which can be regarded as two triangles. The neural network is trained by minimizing the mean square error of the three internal angles between the two triangles. Finally, we extract its shared body as a compressor. The essence of this method is the right triangle similarity transformation (RTST), which is a combination of manifold learning and neural networks. It is distinguishable from other methods. The orthogonality in right triangles is beneficial to the compressed space construction. RTST also maintains the relative order of each edge (vector norm) in triangles. Experimental results on semantic similarity tasks reveal that the vector size is 64% of the original, while the performance is improved by 1.8%. When the compression rate reaches 2.7%, the performance drop is only 1.2%. Detailed analysis and ablation study further validate the rationality and the robustness of the RTST method.

## Algorithm
![Algorithm Overview](https://github.com/songs18/PictureSet/blob/main/RTST.svg)

## Usage
The word embedding compression based on the right triangle similarity transformations includes two steps: preparation step and compression step.

In the preparation step, we provide two methods to obtain orthogonal vector pairs, one is a heuristic method, which filters out orthogonal vector pairs according to part of speech, and the other is a constructive method, which randomly selects one vector, and then randomly generate an orthogonal partner. In this code repository, we provide a heuristic-based method.
One novelty is that the idea of RTST is based on manifold learning, and the implementation is based on Siamese neural networks.

1. Run bucketer.py under preprocessing to bucket word embeddings and filter out orthogonal vector pairs.
2. Run the main.py method to compress the word embedding.

You can modify the project through the configuration file named setup.ini to view the experimental results under different conditions.





## Citation
If you find our work or code useful in your research, please consider citing:

```
@inproceedings{DBLP:conf/icann/SongZHY20,
  author    = {Haohao Song and
               Dongsheng Zou and
               Lei Hu and
               Jieying Yuan},
  editor    = {Igor Farkas and
               Paolo Masulli and
               Stefan Wermter},
  title     = {Embedding Compression with Right Triangle Similarity Transformations},
  booktitle = {Artificial Neural Networks and Machine Learning - {ICANN} 2020 - 29th
               International Conference on Artificial Neural Networks, Bratislava,
               Slovakia, September 15-18, 2020, Proceedings, Part {II}},
  series    = {Lecture Notes in Computer Science},
  volume    = {12397},
  pages     = {773--785},
  publisher = {Springer},
  year      = {2020},
  url       = {https://doi.org/10.1007/978-3-030-61616-8\_62},
  doi       = {10.1007/978-3-030-61616-8\_62},
  timestamp = {Tue, 20 Oct 2020 18:10:19 +0200}
}
```

If you have any questions, please contact me via issue or [email](songhaohao2018@cqu.edu.cn).


