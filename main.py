# coding=utf-8
#! /usr/bin/env python3.4

"""
MIT License

Copyright (c) 2018 NLX-Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This code creates a graph over the wordnet. Then creates word embedding based on PMI on the data

Chakaveh.saedi@di.fc.ul.pt
"""

from modules.vector_generator import *
from modules.vector_accuracy_checker import *
from modules.vector_distance import *

import time
from time import gmtime, strftime

# -------------------------------------------variables TO SET
lang = "English"                                    # TO be set: Dutch / English / Portuguese
only_one_word = False                             # TO be set: True if only one word is chosen from each synset
only_once = False                                 # TO be set: True if only one sense of ambiguous words are considered
equal_weight = True                               # TO be set: True if all relations receive same weight                         Not Done Yet
for_WSD = False                                   # TO be set: if True ambiguous words receive separate tags(wrd_synset offset)  Not Done Yet
accepted_rel = ["all", "syn", "self_loop"]        # TO be set: wordnet relation types to be considered
                                                  # if accepted_rel = ["all"], all relations included in wordnet settings will be used
                                                  # "syn": synonymy    "@":hypernymy    "~":hyponymy      "!": antonymy
                                                  #  ["~", "@", "!"]      "self_loop": to assign 1.1 for [i,i] position in the matrix
to_keep = "60000" #"12590" # "20154"   #13437                                # This number specifies how many of the extracted words are kept
                                                  #  if to_keep = all, all the words are kept
vec_dim = 850                                     # TO be set: Dimension of the final vectors

from_file = False                                  # TO be set: if True it uses the previously built np matrix saved in a file
                                                  #            otherwise the process begins from scratch
stage = "PMI"                             # TO be set: if from_file is True, it specifies which np array to use
                                                  #    result of    "random_walk"    or    "PMI"

normalization = True                              # TO be set: if True L1, or L2 or .... is calculated
norm = 2                                          # TO be set: an integer showing which norm (L1, L2, ...) should be calculated
                                                  # If norm = 0 and from_file = True the result of the previouse run is used

reduction_method = "PCA"                        # The methode for dimensionality reduction
                                                  # "PCA":classic pca    "IPCA":increamental PCA    "KPCA":kernel pca    "ISOMap":isomap
                                                  # "NN-1Hot":Neural Network          "NN-encoder": NN autoencoder
saved_model = False                               #  True if neural network is used for dimensinality reduction and a saved model is used

if lang == "English":
    ref_model = ["wordsim_rel.txt", "wordsim_sim.txt", "simlex999.txt", "MEN_dataset", "MTURK-771.csv", "RG1965.tsv"]
elif lang == "Portuguese":
    ref_model = ["LX-SimLex-999.txt", "LX-WordSim-353.txt"]
else:
    ref_model = ["simlex999.txt", "RG1965.tsv", "wordsim353.tsv"]
#ref_model = ["MEN_dataset"]
                                                  # Models used by Gensim for accuracy checking


all_pos = ["n","a","v","r"]                 # To be set: to identify which part of speeches in wordnet file should be used

extra_desc = ""                                   # A brief description over the test to be saved in the log file

approach = 1                                      # 1: random walk (article)          2: matrix & new edges (NLX)
iter = "infinite"                                 # If approach is 1 ---> "infinite" if  all arcs are needed
                                                  #    or        [a digit] if  a special iteration is considered
depth = 5                                        # if approach is 2 : [a digit] showing how deep to go down in the graph traverse

co_occurance_graph_based = False
just_test = False                                 # To be set: if true, only Gensim is called and previously created embedings are used for test
embedding_file_name = ("auto","abc")              # The input file to Gensim. "auto" to use the last created embeddig file for the test or the file name
#embedding_file_name = ("embeddings_infinite", "txt")

main_path = os.getcwd() + "/data/output/"
#-----------------------------------------------------------------------------------------------------------------------

if not just_test:
    path = main_path
    log_file = path + "en_1_log.txt"
    log = open(log_file, "w")

    file_names = {"n":"data.noun","v":"data.verb","a":"data.adj","r":"data.adv"}
    all_data = {}    # key: pos  , value: the summary over the coresponding data file
                     # see data_file_reader() for description over the fields
    emb_matrix = []
    word_list = []

    log_writer(log, extra_desc, only_one_word, only_once, equal_weight, for_WSD, accepted_rel, iter, vec_dim)

    start_time = time.time()
    log.write("Started at " + str(strftime("%Y-%m-%d %H:%M:%S", gmtime())) + "\n")

    if not from_file:
        if not co_occurance_graph_based:
            print("* Reading wordnet data files")
            log.write("* Reading wordnet data files\n")
            for pos in all_pos:
                all_data.update({pos: data_file_reader(file_names[pos], lang)})
                # Note: all_data["n"][0]["3"][2]  ---->  In data.noun file, the synset with offset = 3, fetch the second field
                # Note: all_data["n"][1]["3"]     ---->  In offset list related to the data.noun file, fetch the third entry

            # to extract all the requiered information from the data files
            word_set, synset_wrd = word_extractor(all_pos, all_data, only_one_word, only_once, log)

            # to create the relation matrix
            p_matrix, dim, word_list, non_zero, synonym_index = pMatrix_builder(all_data, all_pos, word_set, synset_wrd, equal_weight, approach, for_WSD, accepted_rel, to_keep, log, main_path, lang)
            array_writer(word_list, "word_list", "bin", main_path)
            array_writer(synonym_index, "synonym_index", "bin", main_path)
            array_writer(p_matrix, "p_matrix", "bin", main_path)
            if to_keep == "all":
                info_writer(dim,len(word_set),non_zero, for_WSD, main_path)
            else:
                info_writer(dim, int(to_keep), non_zero, for_WSD, main_path)
            wrd_cnt = len(word_set)
        else:
            p_matrix = array_loader("pMatrix", os.getcwd() + '/data/input/ngram/')
            word_list = array_loader("word_list", os.getcwd() + '/data/input/ngram/')
            wrd_cnt = len(word_list)
            dim = (wrd_cnt, wrd_cnt)
            non_zero = -10
    else:
        p_matrix = []
        word_list = array_loader("word_list", main_path)
        dim, for_WSD, wrd_cnt, non_zero = info_reader(main_path)
        dim = (int(dim),int(dim))
        wrd_cnt = int(wrd_cnt)
        non_zero = int(non_zero)
        synonym_index = array_loader("synonym_index", main_path)

    if approach == 1:
        #random walk -> PMI -> normalization
        emb_matrix = random_walk(p_matrix, dim, iter, log, from_file, stage, non_zero, main_path)

        # dimensionality reduction
        final_vec, feature_name, word_list = dimensionality_reduction(word_list, to_keep, reduction_method, emb_matrix, vec_dim, from_file, normalization, norm, log, saved_model, main_path)

        # writing the results into a file
        emb_writer(final_vec, word_list, vec_dim, iter, feature_name, for_WSD, main_path)

        finish_time = time.time()
        print("\nRequired time to process %d words: %.3f seconds ---" % (wrd_cnt, finish_time - start_time))
        log.write("\nFinished at %s <-----> total time: %.3f seconds" % (str(strftime("%Y-%m-%d %H:%M:%S", gmtime())),finish_time - start_time))
        log.close()

    elif approach == 2:
        # random walk
        emb_matrix = matrix_arc_update(p_matrix, synonym_index, accepted_rel, dim, depth, log, from_file, stage, main_path)

        # dimensionality reduction
        final_vec, feature_name, word_list = dimensionality_reduction(word_list, to_keep, reduction_method, emb_matrix, vec_dim, from_file, normalization, norm, log, saved_model, main_path)

        # writing the results into a file
        f_name = "depth_" + str(depth)
        emb_writer(final_vec, word_list, vec_dim, f_name, feature_name, for_WSD, main_path)

        finish_time = time.time()
        print("\nRequired time to process %d words: %.3f seconds ---" % (wrd_cnt, finish_time - start_time))
        log.write("\nFinished at %s <-----> total time: %.3f seconds" % (
        str(strftime("%Y-%m-%d %H:%M:%S", gmtime())), finish_time - start_time))

        log.close()


# Checking the accuracy using Gensim
vector_accuracy(ref_model, iter, approach, depth, for_WSD, embedding_file_name, main_path, lang)

