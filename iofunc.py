__author__= 'ShubhamTripathi'

import re
import string
from collections import Counter

def get_training_data(fileName,tam_flag = 0, stats = 0):
    input_file = "./Resources/training_data/{0}".format(fileName)
    # input_file_xml = file(input_file)
    if tam_flag :
        with open(input_file,'r') as fpl:
            tam_file = fpl.readlines()
        sentence = [];lang_tag = [];pos_tag = []
        utters = []
        lang = []
        pos = []
        mismatch_token = 0
        for i in tam_file:
            if re.search(r'^.',i):
                splStr = re.split(r' ',i)
                try:
                    re.split(r'\n', splStr[2])[0]
                except IndexError:
                    with open("./Resources/mismatches_train_input.txt", 'a') as mm:
                        mm.write("Token : {0} \n".format(i))
                    mismatch_token += 1
                    continue
                sentence.append(splStr[0])
                lang_tag.append(splStr[1])
                pos_tag.append(re.split(r'\n', splStr[2])[0])
            else:
                assert len(sentence) == len(pos_tag) & len(pos_tag) == len(lang_tag), 'Error in utter extraction'
                if len(sentence) == 0 | len(lang_tag) == 0 | len(pos_tag) == 0:
                    continue
                utters.append(sentence);lang.append(lang_tag);pos.append(pos_tag)
                sentence = [];lang_tag = [];pos_tag = []
        td = zip(utters,lang,pos)
        with open("./Resources/mismatches_train_input.txt", 'a') as mm:
            mm.write("Total Mismatches : {0}".format(mismatch_token))
        return td
    else:
        with open(input_file,'r') as fl:
            fileString = fl.readlines()
        # xml = input_file_xml.read()
        sentence = [];lang_tag = []; pos_tag = []
        utters = []
        lang = []
        pos = []
        for i in fileString:
            if re.search(r'\t',i):
                splStr = re.split(r'\t',i)
                sentence.append(splStr[0])
                lang_tag.append(splStr[1])
                pos_tag.append(re.split(r'\n',splStr[2])[0])
            else:
                assert len(sentence) == len(pos_tag) & len(pos_tag) == len(lang_tag), 'Error in utter extraction'
                if len(sentence)== 0 | len(lang_tag) == 0 | len(pos_tag) == 0:
                    continue
                utters.append(sentence);lang.append(lang_tag); pos.append(pos_tag)
                sentence = [];lang_tag = [];pos_tag = []
        if stats:
            total_len = unpack(utters)
            tags = set(unpack(pos))

            with open('./Resources/statistics.txt', 'a') as flp:
                flp.write("{0}\n".format(fileName))
                flp.write("Total no of sentences = {0}\n".format(len(utters)))
                flp.write("Average length of sentences = {0}\n".format(float(len(total_len)/len(utters))))
                flp.write("No of POS tags = {0}\n".format(len(tags)))
                flp.write("Tag List = {0}\n".format(tags))
                # flp.write("Count of each tag = {0} + \n\n".format(countt))

        td = zip(utters,lang,pos)
        return td

def get_testing_data(fileName,tam_flag = 0, stats = 0):
    input_file = "./Resources/testing_data/{0}".format(fileName)
    with open(input_file,'r') as fl:
        fileString = fl.readlines()
    # xml = input_file_xml.read()
    sentence = []
    lang_tag = []
    utters = []
    lang = []

    for i in fileString:
        if re.search(r'\t',i):
            splStr = re.split(r'\t',i)
            spllang = re.split(r'\n', splStr[1])
            sentence.append(splStr[0])
            lang_tag.append(spllang[0])
        else:
            assert len(sentence) == len(lang_tag), 'Error in utter extraction'
            if len(sentence)== 0 | len(lang_tag) == 0:
                continue
            utters.append(sentence)
            lang.append(lang_tag)
            sentence = [];lang_tag = []
    assert len(sentence) == len(lang_tag), 'Error in utter extraction'
    utters.append(sentence)
    lang.append(lang_tag)

    if stats:
        total_len = unpack(utters)
        with open('./Resources/statistics.txt', 'a') as flp:
            flp.write("{0}\n".format(fileName))
            flp.write("Total no of sentences = {0}\n".format(len(utters)))
            flp.write("Average length of sentences = {0}\n".format(float(len(total_len)/len(utters))))
    td = zip(utters,lang)
    return td

def getAcro():
    # Creating list of acronyms
    with open('./Resources/AcroNew2.txt','r') as ip:
        acronyms = ip.read().split()
    # Other set of punctuations components
    punct = set(string.punctuation)
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    num_string = ['one','two','three','four','five','six','seven','eight','nine','ten','hundred','thousand','lakh','million','billion']
    return acronyms, punct, nums, num_string

def get_wordCorrect_data(fname):
    if fname == 'fOne':
        with open('./Resources/word_correct/emnlp_dict.txt') as flp:
            data = flp.readlines()
        Nspell = []
        spell = []
        for i in data:
            try:
                splString = re.split(r'\t',i)
                Nspell.append(splString[0])
                spell.append(re.split(r'\n',splString[1])[0])
            except:
                continue
        return Nspell, spell
    elif fname == 'fTwo':
        with open('./Resources/word_correct/Test_Set_3802_Pairs.txt') as flp:
            data = flp.readlines()
        Nspell = []
        spell = []
        for i in data:
            splString = re.split(r' | ', i)
            Nspell.append(re.split(r'\t',splString[0])[1])
            spell.append(re.split(r'\r\n',splString[2])[0])
        return Nspell, spell
    else:
        exit('Incorrect file name.')

def unpack(list):
    unpacked_list = []
    for i in list:
        unpacked_list += i
    return unpacked_list