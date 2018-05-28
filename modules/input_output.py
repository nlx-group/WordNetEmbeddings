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

This code reads wordnet data and index files
data_file_reader(file_name):
    extract data from wordnet data files saved in "data/input" directory
    output is
    1- a dictionary with
        key = synsetoffsets
        data = (synsetWrds, synsetConnections, synsetRelationTypes, connectedSynsetPos, gloss)
    2- and offset_list

Chakaveh.saedi@di.fc.ul.pt
"""

import os, sys
import numpy as np
from progressbar import ProgressBar, Percentage, Bar

def data_file_reader(file_name, lang):
    print("    Working on " + file_name)
    if lang == "Dutch":
        path = os.getcwd() + '/data/input/Dutch_wnet/'
    elif lang == "Portuguese":
        path = os.getcwd() + '/data/input/Portuguese_wnet/'
    else:
        path = os.getcwd() + '/data/input/English_wnet/'

    fl = open(path + file_name)
    src = fl.readlines()
    fl.close()

    file_data = {}
    offset_list = []

    all_word = set()
    amb_word = set()

    for lineNum in range(len(src)):
        dataLine = src[lineNum]
        if dataLine[0:2] == "  ": #or " 000 " in dataLine:    # comments or synset with no relations
            continue
        else:
            synsetWrds = []
            synsetConnections = []
            synsetRelationTypes = []
            connectedSynsetPos = []

            dataLineParts = dataLine.split(" ")

            wrdCnt = int(dataLineParts[3], 16)

            indx = 4
            for i in range(wrdCnt):
                synsetWrds.append(dataLineParts[indx])
                """
                if dataLineParts[indx] not in all_word:
                    all_word.add(dataLineParts[indx])
                else:
                    amb_word.add(dataLineParts[indx])
                """
                indx += 2

            connCnt = int(dataLineParts[indx])
            indx += 1
            for i in range(connCnt):
                synsetRelationTypes.append(dataLineParts[indx])
                indx += 1
                synsetConnections.append(dataLineParts[indx])
                indx += 1
                connectedSynsetPos.append(dataLineParts[indx])
                indx += 1
                # the next field is 0000 or 000
                indx += 1

            gloss = dataLine.split("|")[1]
            gloss = gloss.replace("\n","")
            gloss = gloss.replace("'","''")

            data = (synsetWrds, synsetConnections, synsetRelationTypes, connectedSynsetPos, gloss)
            file_data.update({dataLineParts[0]:data})
            offset_list.append(dataLineParts[0])

            #if dataLineParts[0] in synsetConnections:
            #    print("    self loop", dataLineParts[0])

    #print("number of extracted words: ", len(all_word), ", ", len(amb_word), "of which are ambiguous")

    return file_data, offset_list

def emb_writer(emb_matrix, word_list, dim, iter, feature_name, for_WSD, main_path):
    try:
        if emb_matrix == []:
            print("no changes was made to the previously saved file")
        else:

            out_file = open(main_path + "embeddings_" + iter + ".txt", "w")

            out_file.write("%d %d\n" % (len(word_list), dim))

            if "pyspark" not in str(type(emb_matrix)):
                if dim > len(emb_matrix[0]):
                    dim = len(emb_matrix[0])
                pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(word_list))
                for i in pbar(range(len(word_list))):
                    if for_WSD:
                        wrd = word_list[i].split("\t")[0]
                    else:
                        wrd = word_list[i]
                    emb = ""
                    for j in range(dim):
                        emb += str(emb_matrix[i][j]) + " "
                    emb += "\n"
                    emb = emb.replace(" \n", "\n")
                    out_file.write(wrd + " " + emb)
            else:
                i = 0
                for row in emb_matrix.collect():
                    wrd = word_list[i].split("\t")[0]
                    i += 1
                    emb = row.asDict()
                    out_file.write(wrd + " " + str(emb[feature_name]).replace("[","").replace("]","").replace(","," ") + "\n")

            out_file.close()
            print("\n-------------------------------------------------------------")
            print("Vector Embeddings are created and saved in \data\output folder")
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("Unexpected error:", exc_value)

def array_writer(matrix, fname, type, main_path):
    try:
        print ("    Saving %s data into a file"%(fname))
        path = main_path + fname
        if type == "txt":
            np.savetxt(path, matrix)
        else:
            np.save(path, matrix)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("Unexpected error:", exc_value)
        print("        COULDN'T SAVE THE %s FILE"%(fname))

def array_loader(fname, main_path):
    path = main_path + fname + ".npy"
    mat_data = np.load(path)
    return(mat_data)

def info_writer(dim,wrd_cnt, non_zero, for_WSD, main_path):
    path = main_path + 'last_run_info'
    info = open(path,"w")
    info.write("dim: %d\n" % (dim[0]))
    info.write("for_WSD: %s\n" % (str(for_WSD)))
    info.write("wrd_cnt: %d\n" % (wrd_cnt))
    info.write("non_zero: %d\n" % (non_zero))

    info.close()

def info_reader(main_path):
    path = main_path+'last_run_info'
    info = open(path)
    data = info.readlines()
    info.close()

    dim = data[0].split(" ")[1].replace("\n","")
    for_WSD = data[1].split(" ")[1].replace("\n","")
    if for_WSD == "True":
        for_WSD = True
    else:
        for_WSD = False
    wrd_cnt = data[2].split(" ")[1].replace("\n","")
    non_zero = data[3].split(" ")[1].replace("\n","")

    return dim, for_WSD,  wrd_cnt,non_zero

def log_writer(log, description, only_one_word, only_once, equal_weight, for_WSD, accepted_rel, iter, vec_dim):
    try:
        log.write("Only one word from each synset: %s \n" %(only_one_word))
        log.write("Only one sense of each word: %s\n" %(only_once))
        log.write("Equal weight for different relation types: %s\n" %(str(equal_weight)))
        log.write("Different vectors for each sense of ambiguous words: %s \n" %(str(for_WSD)))
        log.write("Accepted relations: %s \n" %(str(accepted_rel)))
        log.write("Random walk method (infinite or itterative): %s \n" %(iter))
        log.write("Vector dimension: %d\n" % (vec_dim))

        if description != "":
            log.write("Description: %s\n" % (description))

        log.write("\n-----------------------------\n")
    except:
        print("        COULDN'T UPDATE THE LOG FILE")
