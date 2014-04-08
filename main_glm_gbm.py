import numpy as np
import time

from funcs import *
from sklearn.metrics import *
from smote import SMOTE, borderlineSMOTE
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation, preprocessing, ensemble, linear_model

WORKING_DIR = '/home/neik/Kaggle/LoanDefaultPrediction/'


def classifier(seed):
    K = 10
    bestFeatures=[520,521,268,1,767,330,331,48]
    model = linear_model.LogisticRegression(C=1.e20, penalty='l2', dual=False, class_weight='auto')
    scaler = preprocessing.StandardScaler()
    
    #read in  data, parse into training and target sets    
    train, test, loss = subData(bestFeatures)
    target = np.asarray([1 if x > 0 else 0 for x in loss])
    
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    X_one,X_two,y_one,y_two = cross_validation.train_test_split(train,
                                                                target,
                                                                test_size= 0.3,
                                                                random_state = seed)
    

    #Simple K-Fold cross validation. 10 folds.
    cv = cross_validation.KFold(len(X_one), n_folds=K, indices=False, shuffle=True)
    
    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    prob_vald = np.zeros(len(X_two))
    prob_test = np.zeros(len(test))
    for traincv, testcv in cv:
        model.fit(X_one[traincv], y_one[traincv])
        prob_01 = model.predict_proba(X_two)[:,1]
        prob_02 = model.predict_proba(test)[:,1]
        prob_vald += prob_01/K
        prob_test += prob_02/K
    
    auc = roc_auc_score(y_two, prob_vald)
    print "AUC (%d folds): %f" % (K, auc)

    best_thred, max_f1 = f1search(y_two, prob_vald)
    print "Best threshold: %f and max F1 score: %f" % (best_thred, max_f1)
    
    y_test = [1 if x>best_thred else 0 for x in prob_test]
    
    return y_test


def fitter(default_id, seed):
    K = 10
    bestFeatures=[745,585,250,584,368,655,111,59,615,316,37]
    scaler = preprocessing.StandardScaler()
    model = ensemble.GradientBoostingRegressor(loss='lad')
    
    # load the training and testing data with best features
    train, test, loss = subData(bestFeatures)
    idx = np.where(loss>0)[0]
    target = loss[idx]

    train = train[idx,:]
    train = scaler.fit_transform(train)
    
    new_loss = np.asarray(default_id)
    idy = np.where(new_loss==1)[0]
    test = test[idy,:]
    test = scaler.fit_transform(test)
    
    cv = cross_validation.KFold(len(train), n_folds=K, indices=False, shuffle=True, random_state=seed)
    
    mean_test_val = np.zeros(len(test))
    for traincv, testcv in cv:
        model.fit(train[traincv], fw_transform(target[traincv],0.01))
        prediction = model.predict(test)
        mean_test_val += prediction/K    
    
    test_val = bw_transform(mean_test_val,0.01)
    
    y_pred = boundLoss(test_val)    
    y_zero = np.zeros(len(default_id))
    new_loss[idy] = y_pred
    
    mae = mean_absolute_error(y_zero, new_loss)
    print "Loss MAE: = %f on test set with %d-fold CV" % (mae,K)
    
    return new_loss


if __name__ == '__main__':
    t0 = time.clock()  
    seed = 1
    
    id = np.genfromtxt(open(WORKING_DIR+'/CSVs/testId.csv','r'),skip_header=1).tolist()
    
    #default_id = classifier(seed)
    #print np.sum(default_id)
    
    sum_id = np.zeros(len(id))
    for i in np.random.randint(1,9999,5): 
        default_id = classifier(i)
        sum_id += default_id
        print i, np.sum(default_id)
        
    final_id = [1 if x==5 else 0 for x in sum_id]
    
    print 'Sum final ID: %d' % np.sum(final_id)
    #y_pred = fitter(final_id, seed)
    
    #print np.sum(y_pred)
    pred = [[id[i], x] for i, x in enumerate(final_id)]    
    
    np.savetxt(WORKING_DIR+'/CSVs/submission.csv', pred, delimiter=',', fmt='%d,%d', header='id,loss', comments = '')
    
    print 'Running time = ' + str(time.clock() - t0)

