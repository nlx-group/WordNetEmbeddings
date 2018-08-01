# WordNet Embeddings

## wnet2vec

**Article**

Saedi, Chakaveh, António Branco, João António Rodrigues and João Ricardo Silva, 2018, ["WordNet Embeddings"](http://www.di.fc.ul.pt/~ahb/pubs/2018SaediBrancoRodriguesEtAL.pdf), In Proceedings, 3rd Workshop on Representation Learning for Natural Language Processing (RepL4NLP), 56th Annual Meeting of the Association for Computational Linguistics, 15-20 July 2018, Melbourne, Australia.

**WordNet used in the above paper**

[Princeton WordNet 3.0](http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz)

**Test sets used in above paper**

Please note that the semantic network to semantic space method presented in the above paper includes random-based subprocedures (e.g. selecting one word from a set of words with identical number of outgoing edges). The test scores may present slight fluctuations over different runs of the code.

[SimLex-999](https://www.cl.cam.ac.uk/~fh295/simlex.html)

[RG1965](http://delivery.acm.org/10.1145/370000/365657/p627-rubenstein.pdf?ip=194.117.40.49&id=365657&acc=ACTIVE%20SERVICE&key=2E5699D25B4FE09E%2E454625C777251F56%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1527501385_f2095c911da3627e99b9a6c8a9769558)

[WordSim-353-Similarity](http://alfonseca.org/eng/research/wordsim353.html)

[WordSim-353-Relatedness](http://alfonseca.org/eng/research/wordsim353.html)

[MEN](http://clic.cimec.unitn.it/~elia.bruni/MEN.html)

[MTurk-771](http://www2.mta.ac.il/~gideon/datasets/)

**Models**

The best wnet2vec model we have obtained that was ran with 60,000 words using [Princeton WordNet 3.0](http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz), referred in the article, is available for download [here](http://lxcenter.di.fc.ul.pt/wn2vec.zip).

**How to run wn2vec software**

To provide input files to the software the following structure must exist:

```
|-- main.py
|-- data
|   |-- input
|   |   |-- language_wnet
|   |   |   |-- *wnet_files
|   |   |-- language_testset
|   |   |   |-- *testset_files
|   |-- output
|-- modules
|   |-- input_output.py
|   |-- sort_rank_remove.py
|   |-- vector_accuracy_checker.py
|   |-- vector_distance.py
|   |-- vector_generator.py
```

Where *language* is the language that you are using that must be indicated in main.py in the variable **lang**.
If the language isn't supported by the current path routing in the code, which was mainly use for experiments, you may add the path to the directory in the files *input_output.py*, *vector_generator.py* and *vector_accuracy_checker.py*.

Various variables for the output of the model, such as embedding dimension, can be found in *main.py*. 

To run the software, you will need the following packages:

* Numpy
* progressbar
* keras
* sklearn
* scipy
* gensim

Python3.5 was used for the experimentation.
