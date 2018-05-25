import numpy as np
import string

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
