import time
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
RRF = importr("RRF")
from rpy2.robjects import numpy2ri

numpy2ri.activate()


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def ind_VfoldCross(data, selec):
    cls = np.unique(data)
    arr_train = []

    for i in cls:
        # get the indexes for each
        ind = np.where(data == i)

        if len(ind[0]) <= selec:
            arr_train.extend(ind[0])
        else:
            sel = random.sample(range(len(ind[0])), selec)
            arr_train.extend(ind[0][sel])

    return arr_train


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label', fontsize=13)
    plt.xlabel('Predicted label', fontsize=13)


def split_data(data, labels, Ntrain, Nval, method):
    
    if method.find('class')>=0:
        list_of_train = ind_VfoldCross(labels, Ntrain)
    else:
        list_of_train = random.sample(range(data.shape[0]),Ntrain)
    list_of_rest = diff(range(len(data)), list_of_train)
    X_sub = data[list_of_rest, :]
    Y_sub = labels[list_of_rest]
    if method.find('class')>=0:
        list_of_val = ind_VfoldCross(Y_sub, Nval)
    else:
        list_of_val = random.sample(range(X_sub.shape[0]),Nval)
    list_of_test = diff(range(len(X_sub)), list_of_val)

    data_tr = data[list_of_train, :]
    data_tr = np.array(data_tr, dtype="float64")
    labels_tr = labels[list_of_train]
    if method.find('class')>=0:
        labels_tr = [str(labels_tr[t][0]) for t in range(0, len(labels_tr))]
    else:
        labels_tr = [str(labels_tr[t]) for t in range(0, len(labels_tr))]

    data_vl = data[list_of_val, :]
    data_vl = np.array(data_vl, dtype="float64")
    labels_vl = labels[list_of_val]
    if method.find('class')>=0:
        labels_vl = [labels_vl[t][0] for t in range(0, len(labels_vl))]
    else:
        labels_vl = [labels_vl[t] for t in range(0, len(labels_vl))]

    data_ts = data[list_of_test, :]
    data_ts = np.array(data_ts, dtype="float64")
    labels_ts = labels[list_of_test]
    if method.find('class')>=0:
        labels_ts = [labels_ts[t][0] for t in range(0, len(labels_ts))]
    else:
        labels_ts = [labels_ts[t] for t in range(0, len(labels_ts))]

    return data_tr, labels_tr, data_vl, labels_vl, data_ts, labels_ts


def RandomForestFI(data_tr, labels_tr, method):

    t = time.time()
    if method.find('class')>=0:
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    else:
        rf = RandomForestRegressor(n_estimators=200, n_jobs=-1)
    rf = rf.fit(data_tr, labels_tr)
    print('RF done!')
    importances = rf.feature_importances_
    importRF_all = importances / max(importances)
    indicesRF = np.argsort(importRF_all, axis=None)[::-1]
    elapsed = time.time() - t
    print('time of RF using all features: ', elapsed)
    return rf, importRF_all, indicesRF


def GRRFoptimization(lambda0, gamma, importRF_all, indicesRF,
                     data_tr, labels_tr, data_vl, labels_vl, Niter, method):
    summary = []
    for lam in lambda0:
        for gam in gamma:
            if gam != 0 or lam != 0:
                if method.find('class')>=0:
                    OAgrrf = []
                    kgrrf = []
                else:
                    RMSEgrrf = []
                    Rgrrf = []

                BestcoefReg = (1 - gam) * lam + gam * importRF_all
                if method.find('class')>=0:
                    grrf = RRF.RRF(robjects.r.matrix(data_tr, nrow=data_tr.shape[0],
                                                 ncol=data_tr.shape[1]),
                               robjects.vectors.FactorVector(labels_tr),
                               flagReg=1, coefReg=BestcoefReg)
                else:
                    grrf = RRF.RRF(robjects.r.matrix(data_tr, nrow=data_tr.shape[0],
                                                     ncol=data_tr.shape[1]), 
                                   robjects.vectors.FloatVector(labels_tr),
                                   flagReg=1, coefReg=BestcoefReg)
                selected_features = grrf.rx2('feaSet')
                selected_features = np.array([x - 1 for x in selected_features])

                col = selected_features
                colRF = indicesRF[0:len(col)]

                for N2 in range(Niter):
                    X_train1 = data_tr[:, col]
                    X_val1 = data_vl[:, col]

                    if method.find('class')>=0:
                        grrf_F = RandomForestClassifier(n_estimators=200, n_jobs=-1)
                    else:
                        grrf_F = RandomForestRegressor(n_estimators=200, n_jobs=-1)
                    grrf_F = grrf_F.fit(X_train1, labels_tr)

                    Ypred_GRRF = grrf_F.predict(X_val1)

                    if method.find('class')>=0:
                        Ypred_GRRF = [int(i) for i in Ypred_GRRF]
                        OA_GRRF = accuracy_score(labels_vl, Ypred_GRRF)
                        kappa_GRRF = cohen_kappa_score(labels_vl, Ypred_GRRF)
                    else:
                        rmse_grrf =  np.sqrt(np.mean((labels_vl-Ypred_GRRF) ** 2))    
                        r_grrf = np.corrcoef(labels_vl,Ypred_GRRF)[1, 0]

                    X_train1 = data_tr[:, colRF]
                    X_val1 = data_vl[:, colRF]

                    if method.find('class')>=0:
                        rf_F = RandomForestClassifier(n_estimators=200, n_jobs=-1)
                    else:
                        rf_F = RandomForestRegressor(n_estimators=200, n_jobs=-1)

                    rf_F = rf_F.fit(X_train1, labels_tr)
                    Ypred_RF = rf_F.predict(X_val1)

                    if method.find('class')>=0:
                        Ypred_RF = [int(i) for i in Ypred_RF]
                        OA_RF = accuracy_score(labels_vl, Ypred_RF)
                        print(['gamma: ' + str(gam) + ' and lambda ' + str(lam) + ' select '
                               + str(len(colRF))+' features:' + ' GRRF OA= ' + str(OA_GRRF*100)
                               + ' and RFsel OA= ' + str(OA_RF*100)])

                        OAgrrf.append(OA_GRRF)
                        kgrrf.append(kappa_GRRF)
                    else:
                        rmse_rf = np.sqrt(np.mean((labels_vl-Ypred_RF) ** 2))

                        print(['gamma: ' + str(gam) + ' and lambda ' + str(lam) + ' select '
                               + str(len(colRF))+' features:' + ' GRRF RMSE= ' + str(rmse_grrf)
                               + ' and RFsel RMSE= ' + str(rmse_rf)])
                        RMSEgrrf.append(rmse_grrf)
                        Rgrrf.append(r_grrf)

                if method.find('class')>=0:
                    OA1 = np.mean(OAgrrf)
                    K1 = np.mean(kgrrf)
                    results = [gam, lam, OA1, K1]
                else:
                    RMSE1 = np.mean(RMSEgrrf)
                    R1 = np.mean(Rgrrf)
                    results = [gam, lam, RMSE1, R1] 
                summary.append(results)

    return summary


def SelectedGRRFfeatures(BestGamma, BestLambda, importRF_all, data_tr, labels_tr,
                         data_vl, labels_vl, method):
    BestcoefReg = (1 - BestGamma) * BestLambda + BestGamma * importRF_all
    X_train = np.concatenate((data_tr, data_vl), axis=0)
    Y_train = np.concatenate((labels_tr, labels_vl), axis=0)

    if method.find('class')>=0:
        final_grrf = RRF.RRF(robjects.r.matrix(X_train, nrow=X_train.shape[0], ncol=X_train.shape[1]),
                         robjects.vectors.FactorVector(Y_train),
                         flagReg=1, coefReg=BestcoefReg)
    else:
        final_grrf = RRF.RRF(robjects.r.matrix(X_train, nrow=X_train.shape[0],
                                                     ncol=X_train.shape[1]), 
                                   robjects.vectors.FloatVector(Y_train),
                                   flagReg=1, coefReg=BestcoefReg)

    selected_features = final_grrf.rx2('feaSet')
    importances = final_grrf.rx2('importance')
    # Indices python start 0 and in R in 1:
    selected_features = np.array([x - 1 for x in selected_features])

    importances = importances[selected_features]
    importances = importances / np.max(importances)
    indices = np.argsort(importances, axis=None)[::-1]

    return indices, importances, selected_features


def prediction(data_tr, labels_tr, data_ts, labels_ts, classes_names, method, selected_features=None):

    if selected_features is not None:
        X_train = data_tr[:, selected_features]
        X_ts = data_ts[:, selected_features]
    else:
        X_train = data_tr
        X_ts = data_ts

    if method.find('class')>=0:
        final = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    else:
        final = RandomForestRegressor(n_estimators=200, n_jobs=-1)
    final = final.fit(X_train, labels_tr)

    Ypred = final.predict(X_ts)
    if method.find('class')>=0:
        Ypred = [int(i) for i in Ypred]

        output1 = accuracy_score(labels_ts, Ypred)
        output3 = cohen_kappa_score(labels_ts, Ypred)
        output2 = confusion_matrix(labels_ts, Ypred)
        output4 = classification_report(labels_ts, Ypred, target_names=classes_names)
    else:
        output1 = np.sqrt(np.mean((labels_ts - Ypred) ** 2))
        output2 = np.mean(labels_ts-Ypred)
        output3 = np.mean(np.abs(labels_ts-Ypred))
        output4 = np.corrcoef(labels_ts,Ypred)[1, 0]
        
    return final, output1, output2, output3, output4


def mapping(final, data_total, labels_total, selected_features, nr, nc, method):

    if len(selected_features) > 0:
        X_total = data_total[:, selected_features]
    else:
        X_total = data_total
    Yimage = final.predict(X_total)
    if method.find('class')>=0:
        Yimage = np.array([int(i) for i in Yimage]).reshape(nr * nc, 1)
        Yimage[labels_total == 0] = 0
    Yimage = Yimage.reshape((nr, nc))
    return Yimage
