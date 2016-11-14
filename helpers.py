from __future__ import division

__author__ = 'ShubhamTripathi'

from collections import Counter, OrderedDict
from nltk.util import ngrams
from iofunc import get_wordCorrect_data as wcd

def unpack(list):
    unpacked_list = []
    for i in list:
        unpacked_list += i
    return unpacked_list

def isAscii(utter):
    return all(ord(c) < 128 for c in utter)

def char_ngram(n, word):
    char_tokens = list(word)
    char_ngrams = ngrams(char_tokens, n)  # prefix-suffix is automatically generated here
    return map(lambda x: ''.join(x), char_ngrams)

def alter_point(lang):
    points = 0
    for i in xrange(len(lang)):
        try:
            l1,l2 = [lang[j] for j in (i,i+1)]
            if not l1 == l2:
                points += 1
        except:
            continue
    return points

def word_corrector(data , mode = 'both'):
    if mode == 'both':
        W,C = wcd('fOne')
        new_data = []
        print 'Correcting using file one.'
        for utter in data:
            new_utter = []
            for token in utter:
                if token in W:
                    index = W.index(token)
                    corr_word = C[index]
                    new_utter.append(corr_word)
                else:
                    new_utter.append(token)
            assert len(new_utter) == len(utter),'Error in correction'
            new_data.append(new_utter)
        assert len(new_data) == len(data), 'Error in correction.'

        W,C = wcd('fTwo')
        final_data = []
        print 'Correcting using file two.'
        for utter2 in new_data:
            new_utter2 = []
            for token in utter2:
                if token in W:
                    index = W.index(token)
                    corr_word = C[index]
                    new_utter2.append(corr_word)
                else:
                    new_utter2.append(token)
            assert len(new_utter2) == len(utter2), 'Error in correction. {0},{1}'.format(utter2,new_utter2)
            final_data.append(new_utter2)
        assert len(final_data) == len(data), 'Error in coorection.'
        return final_data
    elif mode == 'fOne':
        W, C = wcd('fOne')
        new_data = []
        print 'Correcting using file one.'
        for utter in data:
            new_utter = []
            for token in utter:
                if token in W:
                    index = W.index(token)
                    corr_word = C[index]
                    new_utter.append(corr_word)
                else:
                    new_utter.append(token)
            assert len(new_utter) == len(utter), 'Error in correction'
            new_data.append(new_utter)
        assert len(new_data) == len(data), 'Error in correction.'
        return new_data
    elif mode == 'fTwo':
        W, C = wcd('fTwo')
        final_data = []
        print 'Correcting using file two.'
        for utter2 in data:
            new_utter2 = []
            for token in utter2:
                if token in W:
                    index = W.index(token)
                    corr_word = C[index]
                    new_utter2.append(corr_word)
                else:
                    new_utter2.append(token)
            assert len(new_utter2) == len(utter2), 'Error in correction. {0},{1}'.format(utter2, new_utter2)
            final_data.append(new_utter2)
        assert len(final_data) == len(data), 'Error in coorection.'
        return final_data
    else:
        exit('Incorrect mode name. Modes can be fOne,fTwo or both')

def max_len(X, mode = 'pre'):
    maxi = ''
    for indx in xrange(len(X)):
        utterance = X[indx]
        for token in utterance:
            if len(token) > len(maxi):
                maxi = token

    if mode == 'pre':
        return len(char_ngram(2,maxi))
    elif mode == 'post':
        return len(char_ngram(2,maxi)),len(char_ngram(3,maxi)),len(char_ngram(4,maxi)),len(char_ngram(5,maxi))

def infreq(listOflist):
    counter = Counter()
    lists = unpack(listOflist)
    for word in lists:
        counter[word] += 1
    words = OrderedDict()
    for word in counter:
        words[word] = len(words)

    return counter, words

def word_normalisation(word):
    norm_word = []
    for letter in word:
        if letter.isupper():
            norm_word.append('A')
        elif letter.islower():
            norm_word.append('a')
        elif letter.isdigit():
            norm_word.append(0)
        else:
            norm_word.append(letter)
    try:
        rword = "".join(norm_word)
    except:
        rword = "0000"
    return rword

def check_labels(tf):
    counter = Counter()
    for utter in tf:
        labels = utter[2]
        for lbl in labels:
            counter[lbl] += 1
    return counter

def calc_prob(X, y, L):
    counter = Counter()
    counter['_UNK_'] = 0
    for idx in xrange(len(X)):
        utterance = X[idx]
        for words in utterance:
            counter[words] += 1
    data = zip(X, L, y)
    fdata = []
    for t in data:
        fdata.append(zip(t[0],t[1],t[2]))

    #Vector length
    ocount = Counter()
    for lbl in unpack(y):
        ocount[lbl] += 1
    odict = OrderedDict()
    for word in ocount:
        odict[word] = len(odict)
    fdata =  unpack(fdata)
    word_dict1 = OrderedDict()
    for word in counter:
        #Vector init
        vector = []
        for i in xrange(len(odict)):
            vector.append(0)

        mid_counter = Counter()
        for token in fdata:
            if word == token[0]:
                mid_counter[token[2]] += 1

        if not len(mid_counter) == 1:
            for (lbl,count) in mid_counter.most_common(1):
                vector[odict[lbl]] = 1
            vector = map(str, vector)
            word_dict1[word] = ''.join(vector)
        else:
            w = list(mid_counter.elements())
            vector[odict[w[0]]] = 1
            vector = map(str,vector)
            word_dict1[word] = ''.join(vector)

    word_dict2 = OrderedDict()
    for word in counter:

        #Vector init
        vector = []
        for i in xrange(len(odict)):
            vector.append(0)

        mid_counter = Counter()
        for token in fdata:
            if word == token[0]:
                mid_counter[token[2]] += 1

        if not len(mid_counter) == 1:
            for (lbl,count) in mid_counter.most_common(2):
                vector[odict[lbl]] = 1

            vector = map(str, vector)
            word_dict2[word] = ''.join(vector)
        else:
            w = list(mid_counter.elements())
            vector[odict[w[0]]] = 1
            vector = map(str,vector)
            word_dict2[word] = ''.join(vector)

    #reset vector
    rvector = []
    for i in xrange(len(odict)):
        rvector.append(0)
    rvector = map(str, rvector)
    rvector = ''.join(rvector)

    return word_dict1, word_dict2, rvector

def create_dict_full(X,L,Xt,Lt, ngram2_tr, ngram3_tr, ngram4_tr, ngram5_tr, ngram2_te, ngram3_te, ngram4_te, ngram5_te,
                         feat1_tr, feat2_tr, feat3_tr, feat4_tr, feat5_tr, feat6_tr, feat7_tr, feat8_tr, feat9_tr, feat10_tr, feat11_tr,
                         feat1_te, feat2_te, feat3_te, feat4_te, feat5_te, feat6_te, feat7_te, feat8_te, feat9_te, feat10_te, feat11_te):
    counter = Counter()
    for utter in X:
        for token in utter:
            counter[token] += 1
    # print L
    # exit()
    for utter in L:
        for token in utter:
            counter[token] += 1

    for utter in Xt:
        for token in utter:
            counter[token] += 1
    # print L
    # exit()
    for utter in Lt:
        for token in utter:
            counter[token] += 1

    for utter in ngram2_tr:
        for token in utter:
            for gram in token:
                counter[gram] += 1
    for utter in ngram3_tr:
        for token in utter:
            for gram in token:
                counter[gram] += 1
    for utter in ngram4_tr:
        for token in utter:
            for gram in token:
                counter[gram] += 1
    for utter in ngram5_tr:
        for token in utter:
            for gram in token:
                counter[gram] += 1

    for utter in ngram2_te:
        for token in utter:
            for gram in token:
                counter[gram] += 1
    for utter in ngram3_te:
        for token in utter:
            for gram in token:
                counter[gram] += 1
    for utter in ngram4_te:
        for token in utter:
            for gram in token:
                counter[gram] += 1
    for utter in ngram5_te:
        for token in utter:
            for gram in token:
                counter[gram] += 1

    for lst in feat1_tr:
        for token in lst:
            counter[token] += 1
    for lst in feat2_tr:
        for token in lst:
            counter[token] += 1
    for lst in feat3_tr:
        for token in lst:
            counter[token] += 1
    for lst in feat4_tr:
        for token in lst:
            counter[token] += 1
    for lst in feat5_tr:
        for token in lst:
            counter[token] += 1
    for lst in feat6_tr:
        for token in lst:
            counter[token] += 1
    for lst in feat7_tr:
        for token in lst:
            counter[token] += 1
    for lst in feat8_tr:
        for token in lst:
            counter[token] += 1
    for lst in feat9_tr:
        for token in lst:
            counter[token] += 1
    for lst in feat10_tr:
        for token in lst:
            counter[token] += 1
    for lst in feat11_tr:
        for token in lst:
            counter[token] += 1

    for lst in feat1_te:
        for token in lst:
            counter[token] += 1
    for lst in feat2_te:
        for token in lst:
            counter[token] += 1
    for lst in feat3_te:
        for token in lst:
            counter[token] += 1
    for lst in feat4_te:
        for token in lst:
            counter[token] += 1
    for lst in feat5_te:
        for token in lst:
            counter[token] += 1
    for lst in feat6_te:
        for token in lst:
            counter[token] += 1
    for lst in feat7_te:
        for token in lst:
            counter[token] += 1
    for lst in feat8_te:
        for token in lst:
            counter[token] += 1
    for lst in feat9_te:
        for token in lst:
            counter[token] += 1
    for lst in feat10_te:
        for token in lst:
            counter[token] += 1
    for lst in feat11_te:
        for token in lst:
            counter[token] += 1

    dictt = OrderedDict()
    for word in counter:
        dictt[word] = str(float(len(dictt)/len(counter)))
    return dictt
