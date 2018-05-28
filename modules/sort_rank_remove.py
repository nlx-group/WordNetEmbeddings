import numpy as np
import string


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
"""


def sort_rem(emb_matrix, word_list, to_keep):

    if to_keep >= len(emb_matrix):
        print("    No row/column was eliminated")
        new_word_list= word_list
    else:
        words_to_keep = set(['a','b','c'])
        zero_index = [np.where(x == 0)[0] for x in emb_matrix]
        zero_cnt = [len(x) for x in zero_index]
        indx = np.array(zero_cnt).argsort()[::-1]

        indx = list(indx)
        to_del = len(emb_matrix) - to_keep
        i = 0
        stop = 0
        while i < to_del and stop <= to_del:
            itm = indx[i]
            if word_list[itm] in words_to_keep:
                indx.append(indx.pop(i))
                i -= 1
            i += 1
            stop += 1

        emb_matrix = np.delete(emb_matrix, indx[:to_del], axis=0)
        emb_matrix = np.delete(emb_matrix, indx[:to_del], axis=1)

        new_word_list = []
        for i in range(len(word_list)):
            if i not in indx[:to_del]:
                new_word_list.append(word_list[i])

        print(emb_matrix, new_word_list, "\n")

    return emb_matrix, new_word_list

emb_matrix = np.random.random_integers(-2,5,(8,8))
emb_matrix[emb_matrix < 0] = 0
word_list = list(string.ascii_lowercase)
word_list = np.array(word_list[:8])

print(emb_matrix, word_list, "\n")

emb_matrix, word_list = sort_rem(emb_matrix, word_list, 4)

print(emb_matrix, word_list, "\n")
