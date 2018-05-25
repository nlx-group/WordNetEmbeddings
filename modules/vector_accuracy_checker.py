# coding=utf-8
#! /usr/bin/env python3.4

"""
This code builds a word2vec model based on the test-set (the created embeddings)
Then model_accuracy is checked based on the comparison berween the existing accuracy_test_sets in Gensim

2017     Chakaveh.saedi@di.fc.ul.pt


help:
https://radimrehurek.com/gensim/models/keyedvectors.html
"""


import os
import logging
import gensim

from modules.input_output import *

def vector_accuracy(ref_model, iter, approach, depth, for_WSD, name, main_path, lang):
    if for_WSD:
        # there are more than one vector for each ambiguous words
        print("Code is not Complete YET")
    else:
        print("\n* Checking accuracy")

        log_file = main_path + "accuracy_results"

        # set logging definitions
        #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=log_file,
                            filemode='w')
        console = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        if name[0] == "auto":
            if approach == 1:
                testset = main_path + "embeddings_" + iter + ".txt"
            else:
                testset = main_path + "embeddings_depth_" + str(depth) + ".txt"
        else:
            if name[1] == "txt":
                testset = main_path + name[0] + ".txt"

            elif name[1] == "npy":              # This Condition DOES NOT WORK
                matrix = array_loader(name[0])
                array_writer(matrix, name[0], "txt")
                testset = main_path + name[0]
                f_name = open(testset)
                src = f_name.readlines()
                f_name.close()

                f_name = open(testset, "w")
                f_name.write("%d %d\n" % (matrix.shape[0],matrix.shape[1]))
                for line in src:
                    line = line.replace("0.000000000000000000e+00", "0.0").replace("1.000000000000000000e+00", "1.0").replace("2.000000000000000000e+00", "2.0")
                    f_name.write(line)
                f_name.close()

                testset = main_path + name[0]

            else:
                testset = main_path + name[0] + "." + name[1]



        # to build the model based on the created embeddings and then compare it to the reference
        # load and evaluate
        #model = model = gensim.models.KeyedVectors.load_word2vec_format(os.getcwd() + '/data/input/GoogleNews-vectors-negative300.bin' , binary=True)
        #model = gensim.models.Word2Vec.load_word2vec_format(testset, binary=False)
        model = gensim.models.KeyedVectors.load_word2vec_format(testset, binary=False)
        for ref in ref_model:
            print ("ref: %s-----------------------------"%ref)
            #ref = "/opt/gensim/gensim/test/test_data/" + ref
            if lang == "English":
                ref = os.getcwd() + '/data/input/English_testset/' + ref
            elif lang == "Portuguese":
                ref = os.getcwd() + '/data/input/Portuguese_testset/' + ref
            else:
                ref = os.getcwd() + '/data/input/Dutch_testset/' + ref
            if "questions-words" in ref:
                model.accuracy(ref, restrict_vocab=None)
            else:
                model.evaluate_word_pairs(ref)
