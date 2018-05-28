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

This code calculates the cosine similarity between two given vectors

Chakaveh.saedi@di.fc.ul.pt
"""

import math
from modules.input_output import *
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def word_similarity(wrd1, wrd2, for_WSD, from_emb_file):
    vec_extractor(wrd1, wrd2, for_WSD, from_emb_file)

def vec_extractor(wrd1, wrd2, for_WSD, from_emb_file):
    if from_emb_file == "auto":
        final_vec = array_loader("embeddings_matrix")
        word_list = array_loader("word_list")
        """
        all_words =[]
        for itm in word_list:
            all_words.append(itm.split("\t")[0])
        """
        if for_WSD:
            all_words = [itm.split("_offset")[0].replace("\n","") for itm in word_list]
        else:
            all_words = word_list

        all_words = np.array(all_words)
        indx1 = np.where(all_words == wrd1)[0]
        indx2 = np.where(all_words == wrd2)[0]
        com_wrd1 = [word_list[itm].split("\t")[0] for itm in indx1]
        com_wrd2 = [word_list[itm].split("\t")[0] for itm in indx2]
    else:
        indx1 = []
        indx2 = []
        com_wrd1 = []
        com_wrd2 = []
        final_vec = []
        indx = 0
        path = os.getcwd() + '/data/output/' + from_emb_file
        with open(path) as infile:
            for line in infile:
                if for_WSD:
                    if line[0:len(wrd1)] == wrd1 and line[len(wrd1):len(wrd1)+7] == "_offset":
                        temp = line[line.index(" ")+1:].replace(" \n","").replace("\n","").replace("'","").split(" ")
                        temp = [float(i) for i in temp]
                        final_vec.append(temp)
                        indx1.append(indx)
                        com_wrd1.append(line.split(" ")[0])
                        indx += 1
                    if line[0:len(wrd2)] == wrd2 and line[len(wrd2):len(wrd2)+7] == "_offset":
                        temp = line[line.index(" ")+1:].replace(" \n","").replace("\n","").replace("'","").split(" ")
                        temp = [float(i) for i in temp]
                        final_vec.append(temp)
                        indx2.append(indx)
                        com_wrd2.append(line.split(" ")[0])
                        indx += 1
                else:
                    if line[0:len(wrd1)] == wrd1 and line[len(wrd1):len(wrd1) + 1] == " ":
                        temp = line[line.index(" ") + 1:].replace(" \n", "").replace("\n", "").replace("'", "").split(" ")
                        temp = [float(i) for i in temp]
                        final_vec.append(temp)
                        indx1.append(indx)
                        com_wrd1.append(line.split(" ")[0])
                        indx += 1
                    if line[0:len(wrd2)] == wrd2 and line[len(wrd2):len(wrd2) + 1] == " ":
                        temp = line[line.index(" ") + 1:].replace(" \n", "").replace("\n", "").replace("'", "").split(" ")
                        temp = [float(i) for i in temp]
                        final_vec.append(temp)
                        indx2.append(indx)
                        com_wrd2.append(line.split(" ")[0])
                        indx += 1
        final_vec = np.array(final_vec)

    if len(indx1) > 1 :
        print('    "%s" is ambiguous with "%d" senses' % (wrd1, len(indx1)))

    if len(indx2) > 1:
        print('    "%s" is ambiguous with "%d" senses' % (wrd2, len(indx2)))

    if len(indx1) == 0 or len(indx2) == 0:
        print('    Cannot find both "%s" and "%s" in current word list' % (wrd1, wrd2))
    else:
        for i in range(len(indx1)):
            for j in range(len(indx2)):
                v1 = final_vec[indx1[i]]
                v2 = final_vec[indx2[j]]
                print('    Cosine similarity between "%s" and "%s": %f' % (com_wrd1[i],com_wrd2[j], cosine_sim(v1, v2, "auto")))


def cosine_sim(v1,v2,mode):
    if mode == "auto":
        #return(1 - distance.cosine(v1,v2))
        return(cosine_similarity(v1.reshape(1, -1),v2.reshape(1, -1)))
    else:
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        #synsDim = v2.split(" ")
        sumxx, sumxy, sumyy = 0, 0, 0
        j = 0
        for i in range(len(v1)):
            if v2[j] == "":
                j += 1
            y = float(v2[j])
            j += 1
            x = v1[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y

        if math.sqrt(sumxx*sumyy) == 0 :
            return (0.00000001)
        return (sumxy/math.sqrt(sumxx*sumyy))

def element_product(v1, v2):
    "compute elementwise product of v1 to v2: (v11 dot v21) (v12 dot v22) ..."

    if v2[0] == " ":
        v2 = v2.replace(" ","",1)

    synsVec = [float(a) for a in v2]

    vector1 = np.array(v1)
    vector2 = np.array(synsVec)

    return(vector1 * vector2)

