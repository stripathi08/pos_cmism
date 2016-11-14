__author__ = 'ShubhamTripathi'

import re
from iofunc import get_training_data, get_testing_data
from transforms import CleanTransformer,FeatureExtractor, crf_plus_format
from helpers import word_corrector, check_labels, unpack
from sklearn.cross_validation import KFold
import pycrfsuite
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from itertools import chain

'''
Incase of switching back to sklearn's classifiers
'''
# from sklearn.svm import LinearSVC
# from sklearn.metrics import confusion_matrix,classification_report
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier


def bio_classification_report(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

def prepare_Xy_data(trId, teId,X,y):
    Xtrain = []
    Xtest = []
    ytrain = []
    ytest = []
    for idx in trId:
        Xtrain.append(X[idx])
        ytrain.append(y[idx])
    for idx in teId:
        Xtest.append(X[idx])
        ytest.append(y[idx])

    return Xtrain, Xtest, ytrain, ytest

def cross_validate_sklearn(X,y,L, n =4):
    kf = KFold(len(X), n_folds= n, shuffle= True)
    for train_idx, test_idx in kf:
        X_train,X_test, y_train, y_test = prepare_Xy_data(train_idx,test_idx,X,y)
        print len(X_train), len(y_train)
        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)
        print trainer.get_params()
        # trainer.set_params({
        #     'c1': 2.0,  # coefficient for L1 penalty
        #     'c2': 1e-2,  # coefficient for L2 penalty
        #     'max_iterations': 75,  # stop earlier
        #
        #     # include transitions that are possible, but not observed
        #     'feature.possible_transitions': True
        # })
        print 'Training...'
        trainer.train('model1.crfsuite')

        print 'Testing...'
        tagger = pycrfsuite.Tagger()
        tagger.open('model1.crfsuite')
        y_pred = [tagger.tag(xseq) for xseq in X_test]

        print(bio_classification_report(y_test, y_pred))


def cross_validate_crf_plus(data, n = 4):
    kf = KFold(len(data[0]), n_folds = n, shuffle=True)
    for trainIdx, testIdx in kf:
        crf_plus_format(data,trainIdx,testIdx).transform(trainFile = True, runbash = True)

## When not combining data
def cross_validate_crf_plus_non_comb(data0, data1, data2, n = 4):
    kf0 = KFold(len(data0[0]), n_folds = n, shuffle=True)
    kf1 = KFold(len(data1[0]), n_folds= n, shuffle=True)
    kf2 = KFold(len(data2[0]), n_folds= n, shuffle=True)
    kf = [kf0, kf1, kf2]
    trId = []
    teId = []
    for i in kf:
        for trainIdx, testIdx in i:
            trId.append(trainIdx)
            teId.append(testIdx)

    #No. of cross validations
    for i in range(0,4):
        new_data = []
        X = []
        L = []
        y = []
        X.append(data0[0])
        X.append(data1[0])
        X.append(data2[0])
        L.append(data0[1])
        L.append(data1[1])
        L.append(data2[1])
        y.append(data0[2])
        y.append(data1[2])
        y.append(data2[2])
        new_data.append(unpack(X))
        new_data.append(unpack(L))
        new_data.append(unpack(y))

        train_idx = []
        for idx in trId[i]:
            train_idx.append(idx)
        for idx in trId[i+4]:
            train_idx.append(idx + len(trId[i]))
        for idx in trId[i+8]:
            train_idx.append(idx + len(trId[i]) + len(trId[i + 4]))

        test_idx = []
        for idx in teId[i]:
            test_idx.append(idx)
        print 'Testing on Facebook data'
        crf_plus_format(new_data, train_idx, test_idx).transform(trainFile=True, runbash=True)

        test_idx = []
        for idx in teId[i + 4]:
            test_idx.append(idx + len(teId[i]))
        print 'Testing on Twitter data'
        crf_plus_format(new_data, train_idx, test_idx).transform(trainFile=True, runbash=True)

        test_idx = []
        for idx in teId[i + 8]:
            test_idx.append(idx + len(teId[i]) + len(teId[i + 4]))
        print 'Testing on WhatsApp data'
        crf_plus_format(new_data, train_idx, test_idx).transform(trainFile=True, runbash=True)
##

def run_training(tf,test_tf, lang_pair, mode = "crf++", mode_data = 'CR', tmode = 'train', word_correct = False, combined_result = 1):
    if mode_data == 'FN':
        keep_label = {
            "N_NN":0,
            "N_NNV":1,
            "N_NST":2,
            "N_NNP":3,
            "PR_PRP":4,
            "PR_PRL":5,
            "PR_PRF":6,
            "PR_PRC":7,
            "PR_PRQ":8,
            "V_VM":9,
            "V_VAUX":10,
            "JJ":11,
            "RB_ALC":12,
            "RB_AMN":13,
            "DM_DMD":14,
            "DM_DMI":15,
            "DM_DMQ":16,
            "DM_DMR":17,
            "QT_QTF":18,
            "QT_QTC":19,
            "QT_QTO":20,
            "RP_RPD":21,
            "RP_NEG":22,
            "RP_INTF":23,
            "RP_INJ":24,
            "CC":25,
            "PSP":26,
            "DT":27,
            "RD_RDF":28

        }

        kick_label = ["RD_PUNC","RD_SYM", "RD_UNK", "RD_ECH","@","#","~","$","E","U"]
    elif mode_data == 'CR':
        keep_label = {
            "G_N":0,
            "G_PRP":1,
            "G_V":2,
            "G_J":3,
            "G_R":4,
            "G_SYM":5,
            "G_PRT":6,
            "CC":7,
            "PSP":8,
            "DT":9,
            "G_X":10

        }
        kick_label = ["null", "@", "#", "~", "$", "E", "U"]

    else:
        exit('Incorrect data mode.')

    if combined_result:
        cleanXyL, test_cleanXyL, bad_test_lbl = CleanTransformer(tf, test_tf, keep_label, kick_label, mode_data, "train",True).transform()
        if tmode == 'train':
            form_cleanXyL = FeatureExtractor(cleanXyL, 4, True).data_transform_train(mode = 'train')
            if word_correct == True:
                correct_tokens = word_corrector(form_cleanXyL[0], mode='both')
                if mode == 'pycrf':
                    X = FeatureExtractor(cleanXyL, 2, True).train_feature_extractor(correct_tokens,form_cleanXyL[1])
                    cross_validate_sklearn(X, form_cleanXyL[2], form_cleanXyL[1], 4)
                elif mode == 'crf++':
                    data = []; data.append(correct_tokens)
                    data.append(form_cleanXyL[1])
                    data.append(form_cleanXyL[2])
                    cross_validate_crf_plus(data,4)
            else:
                if mode == 'pycrf':
                    X = FeatureExtractor(cleanXyL, 2, True).train_feature_extractor(form_cleanXyL[0], form_cleanXyL[1])
                    cross_validate_sklearn(X, form_cleanXyL[2], form_cleanXyL[1], 4)
                elif mode == 'crf++':
                    cross_validate_crf_plus(form_cleanXyL, 4)

        elif tmode == 'test':
            form_cleanXyL = FeatureExtractor(cleanXyL, 4, True).data_transform_train(mode='train')
            form_cleanXyL_test = FeatureExtractor(test_cleanXyL, 4, True).data_transform_train(mode = 'test')
            if word_correct == True:
                correct_tokens = word_corrector(form_cleanXyL[0], mode = 'both')
                correct_tokens_test = word_corrector(form_cleanXyL_test[0], mode = 'both')

                if mode == 'crf++':
                    data = []
                    data.append(correct_tokens)
                    data.append(form_cleanXyL[1])
                    data.append(form_cleanXyL[2])

                    testdata = []
                    testdata.append(correct_tokens_test)
                    testdata.append(form_cleanXyL[1])
                    crf_plus_format(data, testdata, bad_test_lbl).transform(True, True)
            else:
                if mode == 'crf++':
                    crf_plus_format(form_cleanXyL, form_cleanXyL_test, bad_test_lbl).transform(True, True)


if __name__ == '__main__':
        lang_pair = "BN_EN"
        # Coarse of fine mode
        mode_data = "FN"
        # train/test mode
        tmode = 'test'
        combined_result = 1
        # Telugu flag. Always zero
        tam_flag = 0

        if combined_result:
            domain = ["FB", "TWT", "WA"]
            main_train_frame = []
            main_test_frame = []
            for dom in domain:
                fileName = dom + "_" + lang_pair + "_" + mode_data + ".txt"
                training_frame = get_training_data(fileName, tam_flag, stats = 1)
                main_train_frame.append(training_frame)
            #for unconstrained, set true
            file_flag = 3
            if file_flag == 0:
                fileName_u = "BN_EN_TRAIN.txt"
            elif file_flag == 1:
                fileName_u = "HI_EN_TRAIN.txt"
            elif file_flag == 2:
                fileName_u = "TA_EN_TRAIN.txt"
            try:
                training_frame_u = get_training_data(fileName_u, tam_flag, stats = 0)
                main_train_frame.append(training_frame_u)
            except:
                print 'Constrained Run.'

            fileNameTest = "FB" + "_" + lang_pair + "_" + "Test_raw.txt"
            testing_frame = get_testing_data(fileNameTest, tam_flag, stats = 0)
            main_test_frame.append(testing_frame)

            run_training(unpack(main_train_frame),unpack(main_test_frame), lang_pair, mode = "crf++", mode_data = mode_data, tmode = tmode, word_correct = False, combined_result = combined_result)
