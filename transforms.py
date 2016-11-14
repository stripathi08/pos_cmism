from __future__ import division

__author__= 'ShubhamTripathi'

import re
from iofunc import getAcro
from helpers import isAscii,char_ngram, alter_point, max_len, infreq, word_normalisation, calc_prob, create_dict_full, unpack
import os
import subprocess
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
# from sklearn.metrics import classification_report
from metaphone import dm as doubleMeta

class CleanTransformer:
    def __init__(self, trainData, testData, keep_label, kick_label, mode_data, mode = "train", apply = True):
        self.apply = apply
        self.trainData = trainData
        self.testData = testData
        self.mode = mode
        self.keep_label = keep_label
        self.kick_label = kick_label
        self.mode_data = mode_data

    def transform(self):
        tf = self.trainData
        tef = self.testData
        if self.mode == 'train':
            self.train_cleaned = self.train_cleaned(tf)
            self.mode = 'test'

        if self.mode == 'test':
            self.test_cleaned, self.test_bad_list = self.test_cleaned(tef)
        return self.train_cleaned, self.test_cleaned, self.test_bad_list

    def rm_residual(self, data, mode):
        gA = getAcro()
        if mode == 'train':
            clean_list_punc = []
            """This is for removing RD_PUNC/RD_SYM/RDF/UNK"""
            for token in data:
                if token[0] in gA[1] or re.findall(r'\.\.+', token[0]) or re.findall(r'--+', token[0]) or re.findall(r'\*\*+',token[0]) or re.findall(r',,+',token[0]) or re.findall(r'!!+',token[0]) or re.findall(r'\?\?+', token[0]) or re.findall(r'\'\'+', token[0]):
                    continue
                else:
                    clean_list_punc.append(token)
            clean_list_rdf = []
            for token in clean_list_punc:
                if token[0] in re.findall(r'~+', token[0]):
                    continue
                else:
                    clean_list_rdf.append(token)
            clean_list_unk = []
            for token in clean_list_rdf:
                if isAscii(token[0]):
                   clean_list_unk.append(token)
                else:
                    continue
            return clean_list_unk
        elif mode == 'test':
            clean_list_punc = []
            bad_list_punc = []
            bad_list_rdf = []
            bad_list_unk = []

            list_punc = []
            index = 0
            for token in data:
                token = list(token)
                token.append(index)
                list_punc.append(tuple(token))
                index += 1
            index = 0
            for token in list_punc:
                if token[0] in gA[1] or re.findall(r'\.\.+', token[0]) or re.findall(r'--+', token[0]) or re.findall(r'\*\*+', token[0]) or re.findall(r',,+', token[0]) or re.findall(r'!!+',token[0]) or re.findall( r'\?\?+', token[0]) or re.findall(r'\'\'+', token[0]):
                    token = list(token)
                    if self.mode_data == 'CR':
                        token.append('G_X')
                    elif self.mode_data == 'FN':
                        token.append('RD_PUNC')
                    bad_list_punc.append(tuple(token))
                    index += 1
                elif token[0] in re.findall(r'~+', token[0]):
                    token = list(token)
                    if self.mode_data == 'CR':
                        token.append('G_X')
                    elif self.mode_data == 'FN':
                        token.append('RD_SYM')
                    bad_list_rdf.append(tuple(token))
                    index += 1
                elif not isAscii(token[0]):
                    token = list(token)
                    if self.mode_data == 'CR':
                        token.append('G_X')
                    elif self.mode_data == 'FN':
                        token.append('RD_UNK')
                    bad_list_unk.append(tuple(token))
                    index += 1
                else:
                    clean_list_punc.append(token)
                    index += 1
            return clean_list_punc, bad_list_punc, bad_list_rdf, bad_list_unk

    def rm_num(self, data, mode = 'train'):
        gA = getAcro()
        if mode == 'test':
            clean_list_num = []
            bad_list_num = []
            for token in data:

                if token[0][0].isdigit():
                    token = list(token)
                    token.append('$')
                    bad_list_num.append(tuple(token))
                    continue

                if token[0][0].isdigit() and token[0][-1].isdigit():
                    token = list(token)
                    token.append('$')
                    bad_list_num.append(tuple(token))
                    continue

                if re.findall(r'^[0-9]+', token[0]):
                    if token[0].lower().endswith('th') or token[0].lower().endswith('st') or token[0].lower().endswith('nd') or token[0].lower().endswith('rd'):
                        token = list(token)
                        token.append('$')
                        bad_list_num.append(tuple(token))
                        continue

                if (token[0].isdigit() or token[0] in gA[3] or token[0].lower() in gA[3] or token[0] in gA[2] or token[0].startswith('+91')):
                    token = list(token)
                    token.append('$')
                    bad_list_num.append(tuple(token))
                    continue
                else:
                    clean_list_num.append(token)
            return clean_list_num, bad_list_num

        elif mode == 'train':
            clean_list_num = []
            for token in data:

                if token[0][0].isdigit() or token[0][-1].isdigit():
                    continue

                if re.findall(r'^[0-9]+', token[0]):
                    if token[0].lower().endswith('th') or token[0].lower().endswith('st') or token[0].lower().endswith('nd') or token[0].lower().endswith('rd'):
                        continue

                if not (token[0].isdigit() or token[0] in gA[3] or token[0].lower() in gA[3] or token[0] in gA[2] or token[0].startswith('+91')):
                    clean_list_num.append(token)
            return clean_list_num

    def rm_misc(self, data, mode = 'train'):
        sm_ex = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^) _/ O.o o.O \m/ _/\_ _|_ -_- _/\\_ [\\m/] ^_^ _/|\_ >.< <3 -.-
                     :v 8-D 8D B-) xD X-D XD =-D =D =-3 =3 B^D >_<""".split()
        pattern2 = "|".join(map(re.escape, sm_ex))

        if mode == 'train':
            clean_list_misc = []
            for token in data:
                if re.search(r'\.com', token[0]) or re.search(r'\.me', token[0]) or re.search(r'\.org', token[0]) or re.search(r'\.in', token[0]) or re.search(r'\.be', token[0]) or re.search(r'\.it', token[0]):
                    continue
                if token[0].startswith('http://') or token[0].startswith('https') or token[0].startswith('www') or token[0].endswith('.com'):
                    continue
                elif token[0].startswith('@') or token[0].startswith('#'):
                    continue
                elif re.match(r"(:|;|=)[-pPdDB)\\3(/'|\]}>oO]+", token[0]) or re.findall(pattern2, token[0]):
                    continue
                elif token[0] == 'RT':
                    continue
                else:
                    clean_list_misc.append(token)
            return clean_list_misc
        elif mode == 'test':
            clean_list_misc = []
            bad_list_u = []
            bad_list_at = []
            bad_list_hatch = []
            bad_list_emoji = []
            bad_list_rt = []
            for token in data:
                if re.search(r'\.com', token[0]) or re.search(r'\.me', token[0]) or re.search(r'\.org',token[0]) or re.search(r'\.in', token[0]) or re.search(r'\.be', token[0]) or re.search(r'\.it', token[0]):
                    token = list(token)
                    token.append('U')
                    bad_list_u.append(tuple(token))
                    continue
                elif token[0].startswith('http://') or token[0].startswith('https') or token[0].startswith('www') or token[0].endswith('.com'):
                    token = list(token)
                    token.append('U')
                    bad_list_u.append(tuple(token))
                    continue
                elif token[0].startswith('@'):
                    token = list(token)
                    token.append('@')
                    bad_list_at.append(tuple(token))
                    continue

                elif token[0].startswith('#'):
                    token = list(token)
                    token.append('#')
                    bad_list_hatch.append(tuple(token))
                    continue
                elif re.match(r"(:|;|=)[-pPdDB)\\3(/'|\]}>oO]+", token[0]) or re.findall(pattern2, token[0]):
                    token = list(token)
                    token.append('E')
                    bad_list_emoji.append(tuple(token))
                    continue
                elif token[0] == 'RT':
                    token = list(token)
                    token.append('~')
                    bad_list_rt.append(tuple(token))
                    continue
                else:
                    clean_list_misc.append(token)

            return clean_list_misc, bad_list_u, bad_list_emoji, bad_list_at, bad_list_hatch, bad_list_rt

    def empty_tag(self, data, mode):
        if mode == 'train':
            data_upd = []
            for token in data:
                if not token[0] == '' or token[1] == '' or token[2] == '':
                    data_upd.append(token)
                else:
                    continue
            return data_upd
        elif mode == 'test':
            data_upd = []
            for token in data:
                if not token[0] == '' or token[1] == '':
                    data_upd.append(token)
                else:
                    exit('Tag inconsistency in testing data.')
            return data_upd


    def check_labels_train(self, data):
        data_upd = []
        total_nos = 0
        mismatches = []
        for token in data:
            '''
            Check label
            '''
            if not token[2] in self.kick_label:
                if token[2] in self.keep_label:
                    data_upd.append(token)
            else:
                total_nos += 1
                mismatches.append(token)
                continue
        with open('./Resources/mismatches_train_label', 'a') as fp:
            for i in mismatches:
                for j in i:
                    fp.write("%s\t" % j)
                fp.write("\n")
            fp.close()
        return data_upd, total_nos


    # Returned data from this is a list of list. Use unpack.
    def train_cleaned(self,td):
       data_list = []
       for utterance in td:
           data = zip(utterance[0],utterance[1],utterance[2])
           data = self.empty_tag(data, mode = 'train')
           data_res = self.rm_residual(data, mode = 'train')
           data_num = self.rm_num(data_res, mode = 'train')
           data_misc = self.rm_misc(data_num)
           data_list.append(data_misc)
       data_list = [utter for utter in data_list if utter != []]

       #Removing data inconsistencies. While training.
       clean_final_data = []
       for utter in data_list:
           data = self.check_labels_train(utter)[0]
           clean_final_data.append(data)
       clean_final_data = [utter for utter in clean_final_data if utter != []]

       return clean_final_data

    def test_cleaned(self, ted):
        data_list = []
        bad_list = []
        for utterance in ted:
            data = zip(utterance[0], utterance[1])
            data = self.empty_tag(data, mode = 'test')
            data_res, b_punc, b_rdf, b_unk = self.rm_residual(data, mode = 'test')
            data_num, b_num = self.rm_num(data_res, mode = 'test')
            data_misc, b_u, b_emoji, b_at, b_hatch, b_rt = self.rm_misc(data_num, mode = 'test')
            data_list.append(data_misc)
            bb_list = [b_punc, b_rdf, b_unk, b_num, b_u, b_emoji, b_at, b_hatch, b_rt]
            bad_list.append(unpack(bb_list))
        return data_list, bad_list

class FeatureExtractor:

    def __init__(self, cleandata, contextWin = 4, text_to_feat = True):
        self.data = cleandata
        self.contextWin = contextWin
        self.text_to_feat = text_to_feat

    def data_transform_train(self, mode = 'train'):
        if mode == 'train':
            data = self.data
            utterance = []
            lang_tag = []
            pos_tag = []
            for utter in data:
                utterance.append(map(lambda x: x[0],utter))
                lang_tag.append(map(lambda x: x[1], utter))
                pos_tag.append(map(lambda x: x[2], utter))
            assert map(lambda x: len(x),utterance) == map(lambda x: len(x),lang_tag) and map(lambda x: len(x),lang_tag) == map(lambda x: len(x),pos_tag), 'Error in data transformation'
            self.X_train = utterance
            self.y_train = pos_tag
            self.Lang_train = lang_tag
            return utterance, lang_tag, pos_tag
        elif mode == 'test':
            data = self.data
            utterance = []
            lang_tag = []
            position_tag = []
            for utter in data:
                utterance.append(map(lambda x: x[0], utter))
                lang_tag.append(map(lambda x: x[1], utter))
                position_tag.append(map(lambda x: x[2], utter))
            assert map(lambda x: len(x), utterance) == map(lambda x: len(x), lang_tag) and map(lambda x: len(x), utterance) == map(lambda x: len(x), position_tag), 'Error in data transformation'
            self.X_train = utterance
            self.Lang_train = lang_tag
            return utterance, lang_tag, position_tag


    def transform(self, mode= 'train'):
        if mode == 'train':
            self.train_feature_extractor(self.X_train,self.Lang_train)

    def word2features(self,sentence, index, lang):
        Pstem = PorterStemmer()
        WNlem = WordNetLemmatizer()

        word = sentence[index]
        lang_tag = lang[index]

        feat = ['bias']
        #Current word
        feat += ['token:' + str(word.lower())]
        feat += ['token_lang:' + str(lang_tag)]
        feat += ['2gram%d:%s' %(i,x) for i, x in enumerate(char_ngram(2, word))]
        feat += ['3gram%d:%s' % (i, x) for i, x in enumerate(char_ngram(3, word))]
        feat += ['4gram%d:%s' % (i, x) for i, x in enumerate(char_ngram(4, word))]
        feat += ['5gram%d:%s' % (i, x) for i, x in enumerate(char_ngram(5, word))]
        feat += ['word_position:%f' % float(index/len(sentence))]
        feat += ['len_word:%d' % len(word)]
        # if len(word) >= 4:
        #     feat += ['len_word:%d' % 1]
        # else:
        #     feat += ['len_word:%d' % 0]

        feat += ['isupper:' + str(word.isupper())]
        feat += ['firstCharUpper:' + str(word[0].isupper())]

        count = 0
        for letter in word:
            if letter.isupper():
                count += 1
        feat += ['noCharUp:%d' % count]

        feat += ['word_count:%d' % self.word_count[word]]

        # if self.word_count[word] > 2:
        #     feat += ['word_count_bi:%d' % 1]
        # else:
        #     feat += ['word_count_bi:%d' % 0]

        if lang_tag == 'en':
            feat += ['wordStem:' + str(Pstem.stem(word))]
            feat += ['wordLemm:' + str(WNlem.lemmatize(word))]
        else:
            feat += ['wordStem:_NIL_']
            feat += ['wordLemm:_NIL_']

        feat += ['word_norm:' + str(word_normalisation(word))]
        # if word in self.word_dict:
        #     feat += ['word_present:%d' % 1]
        # else:
        #     feat += ['word_present:%d' % 0]
        #Context words
        for idx in xrange(self.contextWin):
            if index > idx:
                feat += ['prev%d:'% (index-idx) + str(sentence[index-idx-1])]
                feat += ['previsupper%d:'% (index-idx) + str(sentence[index-idx-1].isupper())]
                feat += ['prevleng%d:'% (index - idx) + str(len(sentence[index-idx-1]))]
                feat += ['prevlang%d:' % (index - idx) + str(lang[index - idx-1])]
                # feat += ["prevsuffix_%d_3:"% (index - idx) + str(sentence[index-idx - 1][-3:])]
                # feat += ["prevsuffix_%d_2:"% (index - idx) + str(sentence[index-idx -1][-2:])]
                # feat += ["prevprefix_%d_3:"% (index - idx) + str(sentence[index-idx-1][:3])]
                # feat += ["prevprefix_%d_2:"% (index - idx) + str(sentence[index-idx-1][:2])]
            else:
                feat += ['BOS_' + str(idx)]
        for idx in xrange(self.contextWin):
            if index < len(sentence) - idx - 1:
                feat += ['next%d:' % (index + idx) + str(sentence[index + idx + 1])]
                feat += ['nextisupper%d:' % (index + idx) + str(sentence[index + idx + 1].isupper())]
                feat += ['nextleng%d:' % (index + idx) + str(sentence[index + idx +1])]
                feat += ['nextlang%d:' % (index + idx) + str(lang[index + idx + 1])]
                # feat += ["nextsuffix_%d_3:"% (index + idx) + str(sentence[index + idx + 1][-3:])]
                # feat += ["nextsuffix_%d_2:"% (index + idx) + str(sentence[index + idx + 1][-2:])]
                # feat += ["nextprefix_%d_3:"% (index + idx) + str(sentence[index + idx + 1][:3])]
                # feat += ["nextprefix_%d_2:"% (index + idx) + str(sentence[index + idx + 1][:2])]
            else:
                feat += ["EOS_" + str(idx)]
        #Alternation points value for the entire sentence fed to each word as feature. Same value.
        feat += ['alternations:%d' % alter_point(lang)]

        return feat

    def sent2features(self,sentence, lang):
        return [self.word2features(sentence, index, lang) for index in xrange(len(sentence))]

    def train_feature_extractor(self,X,L):
        self.word_count, self.word_dict = infreq(X)
        num_samples = len(X)
        feats = []
        for ids in xrange(num_samples):
            feats.append(self.sent2features(X[ids], L[ids]))
        return feats

class crf_plus_format:

    def __init__(self, trainData, testData, bad_test_lbl):
        self.data = trainData
        self.testdata = testData
        self.bad_test_lbl = bad_test_lbl

    def transform(self, trainFile = True, runbash = True):
        X = self.data[0]
        y = self.data[2]
        L = self.data[1]
        X_e = self.testdata[0]
        L_e = self.testdata[1]
        self.position_tags_e = self.testdata[2]

        self.wd1, self.wd2, self.rvector = calc_prob(X, y, L)

        ngram2_tr, ngram3_tr, ngram4_tr, ngram5_tr, total_feat_length_train = self.ngram_extraction(X, X_e, Ngram=5, mode = 'train')
        feat1_tr, feat2_tr, feat3_tr, feat4_tr, feat5_tr, feat6_tr, feat7_tr, feat8_tr, feat9_tr, feat10_tr, feat11_tr, total_len = self.feature_extraction(X, L, X_e, L_e, mode='train')

        ngram2_te, ngram3_te, ngram4_te, ngram5_te = self.ngram_extraction(X, X_e, Ngram=5, mode = 'test')
        feat1_te, feat2_te, feat3_te, feat4_te, feat5_te, feat6_te, feat7_te, feat8_te, feat9_te, feat10_te, feat11_te = self.feature_extraction(X, L, X_e, L_e, mode='test')

        self.dictt = create_dict_full(X,L,X_e,L_e,ngram2_tr, ngram3_tr, ngram4_tr, ngram5_tr, ngram2_te, ngram3_te, ngram4_te, ngram5_te,
                         feat1_tr, feat2_tr, feat3_tr, feat4_tr, feat5_tr, feat6_tr, feat7_tr, feat8_tr, feat9_tr, feat10_tr, feat11_tr,
                         feat1_te, feat2_te, feat3_te, feat4_te, feat5_te, feat6_te, feat7_te, feat8_te, feat9_te,
                         feat10_te, feat11_te)

        if trainFile:
            self.write_2_file(X,y,L,X_e,L_e, write = True)

        if runbash:
            self.run_bash_cmd()
            self.check_accuracy('output.txt')
            self.print_final_test(X_e, L_e)

    def create_feat_file(self, total_feat_length, contextWin):
        try:
            os.remove('./Resources/crf_plus/feat_template.txt')
        except:
            print 'Creating template file.'

        main_idx = 0
        with open('./Resources/crf_plus/feat_template.txt', 'a') as fp:
            # Contextual information. Words and their lang tags
            for idx in range(-contextWin, contextWin+1):
                fp.write('U{0}:%x[{1},{2}]\n'.format(main_idx, idx, 0))
                main_idx += 1

            for idx in range(-contextWin, contextWin + 1):
                fp.write('U{0}:%x[{1},{2}]\n'.format(main_idx, idx, 1))
                main_idx += 1
            for idx in range(1,total_feat_length):
                fp.write('U{0}:%x[{1},{2}]\n'.format(main_idx, 0, idx))
                main_idx += 1
        print 'Template file created.'

    def output_train_file(self, X,L,y,ngram2, ngram3, ngram4, ngram5, feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10, feat11, mode = 'num_feat'):
        with open('./Resources/crf_plus/train.txt','a') as flp:
            if mode == 'num_feat':
                for tidx in xrange(len(X)):
                    ng2 = []
                    ng3 = []
                    ng4 = []
                    ng5 = []
                    for token in ngram2[tidx]:
                        ng2.append(self.dictt[token])
                    for token in ngram3[tidx]:
                        ng3.append(self.dictt[token])
                    for token in ngram4[tidx]:
                        ng4.append(self.dictt[token])
                    for token in ngram5[tidx]:
                        ng5.append(self.dictt[token])

                    if not tidx == len(X)-1:
                        flp.write("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17}".format(self.dictt[X[tidx]], self.dictt[L[tidx]], ' '.join(ng2), ' '.join(ng3), ' '.join(ng4), ' '.join(ng5), self.dictt[feat1[tidx]], self.dictt[feat2[tidx]], self.dictt[feat3[tidx]], self.dictt[feat4[tidx]], self.dictt[feat5[tidx]], self.dictt[feat6[tidx]], self.dictt[feat7[tidx]], self.dictt[feat8[tidx]], self.dictt[feat9[tidx]], self.dictt[feat10[tidx]], self.dictt[feat11[tidx]], y[tidx]))
                        flp.write('\n')
                    else:
                        flp.write("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17}".format(self.dictt[X[tidx]], self.dictt[L[tidx]], ' '.join(ng2), ' '.join(ng3), ' '.join(ng4), ' '.join(ng5), self.dictt[feat1[tidx]], self.dictt[feat2[tidx]], self.dictt[feat3[tidx]], self.dictt[feat4[tidx]], self.dictt[feat5[tidx]], self.dictt[feat6[tidx]], self.dictt[feat7[tidx]], self.dictt[feat8[tidx]], self.dictt[feat9[tidx]], self.dictt[feat10[tidx]], self.dictt[feat11[tidx]], y[tidx]))
                        flp.write('\n' + '\n')
                        break
            elif mode == 'str_feat':
                for tidx in xrange(len(X)):
                    if not tidx == len(X) - 1:
                        flp.write("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17}".format(X[tidx], L[tidx], ' '.join(ngram2[tidx]), ' '.join(ngram3[tidx]), ' '.join(ngram4[tidx]), ' '.join(ngram5[tidx]), feat1[tidx], feat2[tidx], feat3[tidx], feat4[tidx], feat5[tidx], feat6[tidx], feat7[tidx], feat8[tidx], feat9[tidx], feat10[tidx], feat11[tidx], y[tidx]))
                        flp.write('\n')
                    else:
                        flp.write("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17}".format(X[tidx], L[tidx], ' '.join(ngram2[tidx]), ' '.join(ngram3[tidx]),' '.join(ngram4[tidx]), ' '.join(ngram5[tidx]), feat1[tidx], feat2[tidx], feat3[tidx],feat4[tidx], feat5[tidx], feat6[tidx], feat7[tidx], feat8[tidx], feat9[tidx], feat10[tidx], feat11[tidx], y[tidx]))
                        flp.write('\n' + '\n')
                        break

    def output_test_file(self, X, L, ngram2, ngram3, ngram4, ngram5, feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10, feat11, mode = 'num_feat'):
        with open('./Resources/crf_plus/test.txt', 'a') as flp:
            if mode == 'num_feat':
                for tidx in xrange(len(X)):
                    ng2 = []
                    ng3 = []
                    ng4 = []
                    ng5 = []
                    for token in ngram2[tidx]:
                        ng2.append(self.dictt[token])
                    for token in ngram3[tidx]:
                        ng3.append(self.dictt[token])
                    for token in ngram4[tidx]:
                        ng4.append(self.dictt[token])
                    for token in ngram5[tidx]:
                        ng5.append(self.dictt[token])
                    if not tidx == len(X) - 1:
                        flp.write(
                            "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16}".format(
                                self.dictt[X[tidx]], self.dictt[L[tidx]], ' '.join(ng2), ' '.join(ng3), ' '.join(ng4),
                                ' '.join(ng5), self.dictt[feat1[tidx]], self.dictt[feat2[tidx]],
                                self.dictt[feat3[tidx]], self.dictt[feat4[tidx]], self.dictt[feat5[tidx]],
                                self.dictt[feat6[tidx]], self.dictt[feat7[tidx]], self.dictt[feat8[tidx]],
                                self.dictt[feat9[tidx]], self.dictt[feat10[tidx]], self.dictt[feat11[tidx]]))
                        flp.write('\n')
                    else:
                        flp.write(
                            "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16}".format(
                                self.dictt[X[tidx]], self.dictt[L[tidx]], ' '.join(ng2), ' '.join(ng3), ' '.join(ng4),
                                ' '.join(ng5), self.dictt[feat1[tidx]], self.dictt[feat2[tidx]],
                                self.dictt[feat3[tidx]], self.dictt[feat4[tidx]], self.dictt[feat5[tidx]],
                                self.dictt[feat6[tidx]], self.dictt[feat7[tidx]], self.dictt[feat8[tidx]],
                                self.dictt[feat9[tidx]], self.dictt[feat10[tidx]], self.dictt[feat11[tidx]]))
                        flp.write('\n' + '\n')
                        break
            elif mode == 'str_feat':
                for tidx in xrange(len(X)):
                    if not tidx == len(X) - 1:
                        flp.write(
                            "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16}".format(
                                X[tidx], L[tidx], ' '.join(ngram2[tidx]), ' '.join(ngram3[tidx]),
                                ' '.join(ngram4[tidx]), ' '.join(ngram5[tidx]), feat1[tidx], feat2[tidx], feat3[tidx],
                                feat4[tidx], feat5[tidx], feat6[tidx], feat7[tidx], feat8[tidx], feat9[tidx],
                                feat10[tidx], feat11[tidx]))
                        flp.write('\n')
                    else:
                        flp.write(
                            "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16}".format(
                                X[tidx], L[tidx], ' '.join(ngram2[tidx]), ' '.join(ngram3[tidx]),
                                ' '.join(ngram4[tidx]), ' '.join(ngram5[tidx]), feat1[tidx], feat2[tidx], feat3[tidx],
                                feat4[tidx], feat5[tidx], feat6[tidx], feat7[tidx], feat8[tidx], feat9[tidx],
                                feat10[tidx], feat11[tidx]))
                        flp.write('\n' + '\n')
                        break

    def ngram_extraction(self, X, X_e, Ngram = 5, mode = 'train'):
        mx2_tr = max_len(X, mode = 'pre')
        mx2_te = max_len(X_e, mode = 'pre')
        if mx2_tr > mx2_te:
            mx2, mx3, mx4, mx5 = max_len(X, mode = 'post')
        else:
            mx2, mx3, mx4, mx5 = max_len(X_e, mode = 'post')

        if mode == 'train':
            ngram2 = []
            ngram3 = []
            ngram4 = []
            ngram5 = []
            for idx in xrange(len(X)):
                utterance = X[idx]
                ngram2_mid = []
                ngram3_mid = []
                ngram4_mid = []
                ngram5_mid = []
                for tokens in utterance:
                    ng2 = char_ngram(2,tokens)
                    ng3 = char_ngram(3, tokens)
                    ng4 = char_ngram(4, tokens)
                    ng5 = char_ngram(5, tokens)
                    while len(ng2) < mx2:
                        ng2.append('_NIL_')
                    while len(ng3) < mx3:
                        ng3.append('_NIL_')
                    while len(ng4) < mx4:
                        ng4.append('_NIL_')
                    while len(ng5) < mx5:
                        ng5.append('_NIL_')
                    ngram2_mid.append(ng2)
                    ngram3_mid.append(ng3)
                    ngram4_mid.append(ng4)
                    ngram5_mid.append(ng5)
                ngram2.append(ngram2_mid)
                ngram3.append(ngram3_mid)
                ngram4.append(ngram4_mid)
                ngram5.append(ngram5_mid)

            total_len_feat = mx2 + mx3 + mx4 + mx5

            return ngram2, ngram3, ngram4, ngram5, total_len_feat

        elif mode == 'test':
            ngram2 = []
            ngram3 = []
            ngram4 = []
            ngram5 = []
            for idx in xrange(len(X_e)):
                utterance = X_e[idx]
                ngram2_mid = []
                ngram3_mid = []
                ngram4_mid = []
                ngram5_mid = []
                for tokens in utterance:
                    ng2 = char_ngram(2, tokens)
                    ng3 = char_ngram(3, tokens)
                    ng4 = char_ngram(4, tokens)
                    ng5 = char_ngram(5, tokens)
                    while len(ng2) < mx2:
                        ng2.append('_NIL_')
                    while len(ng3) < mx3:
                        ng3.append('_NIL_')
                    while len(ng4) < mx4:
                        ng4.append('_NIL_')
                    while len(ng5) < mx5:
                        ng5.append('_NIL_')
                    ngram2_mid.append(ng2)
                    ngram3_mid.append(ng3)
                    ngram4_mid.append(ng4)
                    ngram5_mid.append(ng5)
                ngram2.append(ngram2_mid)
                ngram3.append(ngram3_mid)
                ngram4.append(ngram4_mid)
                ngram5.append(ngram5_mid)

            return ngram2, ngram3, ngram4, ngram5

    def feature_extraction(self, X, L, X_e, L_e, mode = 'train'):
        Pstem = PorterStemmer()
        WNlem = WordNetLemmatizer()
        if mode == 'train':
            word_pos = []
            word_len = []
            is_upper = []
            fcharup = []
            nocharup = []
            stemm = []
            lemm = []
            wordnorm = []
            phn_norm = []
            prob1 = []
            prob2 = []
            for idx in xrange(len(X)):
                utterance = X[idx]
                language = L[idx]

                mid_word_pos = []
                mid_word_len = []
                mid_stemm = []
                mid_lemm = []
                mid_phn_norm = []
                for tokens in xrange(len(utterance)):
                    mid_word_pos.append(float(tokens/len(utterance)))
                    mid_word_len.append(len(utterance[tokens]))
                    if language[tokens] == 'E' or language[tokens] == 'en':
                        mid_stemm.append(Pstem.stem(utterance[tokens]))
                        mid_lemm.append(WNlem.lemmatize(utterance[tokens]))
                        if not doubleMeta(unicode(utterance[tokens], 'utf-8'))[0] == '':
                            mid_phn_norm.append(doubleMeta(unicode(utterance[tokens], 'utf-8'))[0])
                        else:
                            mid_phn_norm.append('_NIL_PHN_')
                    else:
                        mid_stemm.append(utterance[tokens])
                        mid_lemm.append(utterance[tokens])
                        mid_phn_norm.append('_NIL_PHN_')

                mid_isupper = []
                mid_fcharup = []
                mid_nocharup = []
                mid_wordnorm = []
                mid_prob1 = []
                mid_prob2 = []
                for tokens in utterance:
                    try:
                        mid_prob1.append(self.wd1[tokens])
                    except:
                        mid_prob1.append(self.rvector)
                    try:
                        mid_prob2.append(self.wd2[tokens])
                    except:
                        mid_prob2.append(self.rvector)

                    mid_isupper.append(tokens.isupper())
                    mid_fcharup.append(tokens[0].isupper())

                    count = 0
                    for letter in tokens:
                        if letter.isupper():
                            count += 1
                    mid_nocharup.append(float(count/len(tokens)))
                    mid_wordnorm.append(word_normalisation(tokens))


                word_pos.append(mid_word_pos)
                word_len.append(mid_word_len)
                is_upper.append(mid_isupper)
                fcharup.append(mid_fcharup)
                nocharup.append(mid_nocharup)
                stemm.append(mid_stemm)
                lemm.append(mid_lemm)
                wordnorm.append(mid_wordnorm)
                phn_norm.append(mid_phn_norm)
                prob1.append(mid_prob1)
                prob2.append(mid_prob2)

            # total_len = 9 + (2*len(self.rvector.split()))
            total_len = 9 + 2
            return word_pos, word_len, is_upper, fcharup, nocharup, stemm, lemm, wordnorm, phn_norm, prob1, prob2, total_len

        elif mode == 'test':
            word_pos = []
            word_len = []
            is_upper = []
            fcharup = []
            nocharup = []
            stemm = []
            lemm = []
            wordnorm = []
            phn_norm = []
            prob1 = []
            prob2 = []
            for idx in xrange(len(X_e)):
                utterance = X_e[idx]
                language = L_e[idx]

                mid_word_pos = []
                mid_word_len = []
                mid_stemm = []
                mid_lemm = []
                mid_phn_norm = []
                for tokens in xrange(len(utterance)):
                    mid_word_pos.append(float(tokens / len(utterance)))
                    mid_word_len.append(len(utterance[tokens]))
                    if language[tokens] == 'E' or language[tokens] == 'en':
                        mid_stemm.append(Pstem.stem(utterance[tokens]))
                        mid_lemm.append(WNlem.lemmatize(utterance[tokens]))
                        mid_phn_norm.append(doubleMeta(unicode(utterance[tokens], 'utf-8'))[0])
                    else:
                        mid_stemm.append(utterance[tokens])
                        mid_lemm.append(utterance[tokens])
                        mid_phn_norm.append('_NIL_PHN_')

                mid_isupper = []
                mid_fcharup = []
                mid_nocharup = []
                mid_wordnorm = []
                mid_prob1 = []
                mid_prob2 = []
                for tokens in utterance:
                    mid_isupper.append(tokens.isupper())
                    mid_fcharup.append(tokens[0].isupper())

                    try:
                        mid_prob1.append(self.wd1[tokens])
                    except:
                        mid_prob1.append(self.rvector)

                    try:
                        mid_prob2.append(self.wd2[tokens])
                    except:
                        mid_prob2.append(self.rvector)

                    count = 0
                    for letter in tokens:
                        if letter.isupper():
                            count += 1
                    mid_nocharup.append(float(count/len(tokens)))
                    mid_wordnorm.append(word_normalisation(tokens))

                word_pos.append(mid_word_pos)
                word_len.append(mid_word_len)
                is_upper.append(mid_isupper)
                fcharup.append(mid_fcharup)
                nocharup.append(mid_nocharup)
                stemm.append(mid_stemm)
                lemm.append(mid_lemm)
                wordnorm.append(mid_wordnorm)
                phn_norm.append(mid_phn_norm)
                prob1.append(mid_prob1)
                prob2.append(mid_prob2)

            return word_pos, word_len, is_upper, fcharup, nocharup, stemm, lemm, wordnorm, phn_norm, prob1, prob2

    def write_2_file(self,X,y,L,X_e,L_e, write = True):
        try:
            os.remove('./Resources/crf_plus/train.txt')
            os.remove('./Resources/crf_plus/test.txt')
            print 'Old files found. Creating new train/test files.'
        except:
            print 'Creating train/test file.'

        '''
        Training data preparation.
        '''

        ngram2, ngram3, ngram4, ngram5, total_feat_length_train = self.ngram_extraction(X, X_e, Ngram = 5, mode = 'train')
        feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10, feat11, total_len = self.feature_extraction(X, L, X_e, L_e, mode = 'train')
        self.total_leng = total_feat_length_train + total_len


        print total_feat_length_train + total_len
        ind = 0
        for idx in xrange(len(X)):
            utterance = X[idx]
            lang = L[idx]
            pos_tag = y[idx]
            self.output_train_file(utterance, lang, pos_tag, ngram2[ind], ngram3[ind], ngram4[ind], ngram5[ind], feat1[ind], feat2[ind], feat3[ind], feat4[ind], feat5[ind], feat6[ind], feat7[ind], feat8[ind], feat9[ind], feat10[ind], feat11[ind], mode = 'num_feat')
            ind += 1

        '''
        Testing data preparation
        '''

        ngram2, ngram3,ngram4,ngram5 = self.ngram_extraction(X, X_e, Ngram = 5, mode = 'test')
        feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10, feat11 = self.feature_extraction(X, L, X_e, L_e, mode = 'test')

        ind = 0
        for idt in xrange(len(X_e)):
            utterance = X_e[idt]
            lang = L_e[idt]
            self.output_test_file(utterance, lang, ngram2[ind], ngram3[ind], ngram4[ind], ngram5[ind], feat1[ind], feat2[ind], feat3[ind], feat4[ind], feat5[ind], feat6[ind], feat7[ind], feat8[ind], feat9[ind], feat10[ind], feat11[ind], mode = 'num_feat')
            ind += 1

    def run_bash_cmd(self):
        self.create_feat_file(self.total_leng, contextWin=4)
        self.owd = os.getcwd()
        os.chdir('./Resources')

        cmd = "crf_learn crf_plus/feat_template.txt crf_plus/train.txt crf_plus/modelOne"
        print 'Training...'
        subprocess.call(cmd, shell = True)

        cmd = "crf_test -m crf_plus/modelOne crf_plus/test.txt > crf_plus/output.txt"
        print 'Testing...'
        subprocess.call(cmd, shell = True)

    def check_accuracy(self, outFile):
        with open('./crf_plus/%s' %outFile,'r') as fp:
            oData = fp.readlines()
        y_true = []
        y_pred = []
        my_true = []
        my_pred = []
        for token in oData:
            splString = re.split(r'\t', token)
            if not splString[0] == '\n':
                my_true.append(splString[-2])
                my_pred.append(re.split(r'\n', splString[-1])[0])
            else:
                y_true.append(my_true)
                y_pred.append(my_pred)
                my_pred = []
                my_true = []
        self.y_pred = y_pred
        # print classification_report(y_true, y_pred)
        os.chdir(self.owd)

    def print_final_test(self, X_e, L_e):
        os.chdir('./Resources')
        try:
            os.remove('./crf_plus/FB_TE__UNCONS_EN_Test_FN.txt')
            print 'Old output found. Creating final output file.'
        except:
            print 'Creating output file.'
        pr_idx = 0
        for idx in xrange(len(X_e)):
            utter = X_e[idx]
            lang = L_e[idx]
            rule_lbl = self.bad_test_lbl[idx]
            position_lbl = self.position_tags_e[idx]
            if not len(utter) == 0:
                y_pr = self.y_pred[pr_idx]
                pr_idx +=1
            if len(rule_lbl) == 0:
                for token_idx in xrange(len(utter)):
                    with open('./crf_plus/FB_TE__UNCONS_EN_Test_FN.txt','a') as flp:
                        if not token_idx == len(utter)-1:
                            flp.write("{0}\t{1}\t{2}".format(utter[token_idx], lang[token_idx], y_pr[token_idx]))
                            flp.write("\n")
                        else:
                            flp.write("{0}\t{1}\t{2}".format(utter[token_idx], lang[token_idx], y_pr[token_idx]))
                            flp.write("\n"  + "\n")
            else:
                ids = [nos[2] for nos in rule_lbl]
                utter_r = [nos[0] for nos in rule_lbl]
                lang_r = [nos[1] for nos in rule_lbl]
                pred_r = [nos[3] for nos in rule_lbl]
                real_idx = 0
                for token_idx in range(0, (len(rule_lbl) + len(position_lbl))):
                    if token_idx in position_lbl:
                        with open('./crf_plus/FB_TE__UNCONS_EN_Test_FN.txt', 'a') as flp:
                            if not token_idx == len(rule_lbl) + len(position_lbl) - 1:
                                # assert flp.write("{0}\t{1}\t{2}".format(utter[real_idx], lang[real_idx], y_pr[real_idx])), 'ds'
                                flp.write("{0}\t{1}\t{2}".format(utter[real_idx], lang[real_idx], y_pr[real_idx]))
                                flp.write("\n")
                                real_idx += 1
                            else:
                                flp.write("{0}\t{1}\t{2}".format(utter[real_idx], lang[real_idx], y_pr[real_idx]))
                                flp.write("\n" + "\n")
                                real_idx += 1
                    elif token_idx in ids:
                        indt = ids.index(token_idx)
                        with open('./crf_plus/FB_TE__UNCONS_EN_Test_FN.txt', 'a') as flp:
                            if not token_idx == len(rule_lbl) + len(position_lbl) - 1:
                                flp.write("{0}\t{1}\t{2}".format(utter_r[indt], lang_r[indt], pred_r[indt]))
                                flp.write("\n")
                            else:
                                flp.write("{0}\t{1}\t{2}".format(utter_r[indt], lang_r[indt], pred_r[indt]))
                                flp.write("\n" + "\n")