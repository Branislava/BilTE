import sys
import pandas as pd

from feature_extraction.MWE import EnglishMWE, SerbianMWE
from feature_extraction.Features import SingleFeatures, JointFeatures

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, linear_model, naive_bayes, svm
from sklearn import ensemble
import xgboost

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 7

def train_model(alg, X, y, cv):

    accuracy_scores = model_selection.cross_val_score(alg, X, y, cv=cv, scoring='accuracy')
    f1_scores = model_selection.cross_val_score(alg, X, y, cv=cv, scoring='f1')
    p_scores = model_selection.cross_val_score(alg, X, y, cv=cv, scoring='precision')
    r_scores = model_selection.cross_val_score(alg, X, y, cv=cv, scoring='recall')

    return accuracy_scores, f1_scores, p_scores, r_scores

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Usage: python classify.py [-exp data/dataset.csv]|[-imp data/dataset-features.csv]')
        exit(1)
    
    for i in range(len(sys.argv)):
        arg = sys.argv[i]
        
        if arg == '-exp':
            filename =  sys.argv[i+1]
    
            df = pd.read_csv(filename)
                
            d = []
            for index, row in df.iterrows():

                label = row['class']
                MWEs = [SerbianMWE('GIZA_SR_ORIGINAL', row['GIZA_SR_ORIGINAL']),
                        SerbianMWE('GIZA_SR_LMTZ', row['GIZA_SR_LMTZ']),
                        SerbianMWE('EXTRACTED_SR', row['EXTRACTED_SR']),
                        EnglishMWE('DICTIONARY_EN', row['DICTIONARY_EN'])]
                
                features_dict = {'class': label}
                
                o = JointFeatures(MWEs)
                features = o.extract()
                features_dict.update(features)
                
                for MWE in MWEs:
                    o = SingleFeatures(MWE)
                    features = o.extract()
                    features_dict.update(features)
                
                d.append(features_dict)
                
            print('Number of extracted features', (len(d[0]) - 1))
            df = pd.DataFrame(d)
            df.to_csv('{0}-features.csv'.format(filename[:-4]), index=False)
            
            print('Encoding features...')
            le = LabelEncoder()
            for feature in df:
                if df[feature].dtype not in [int, float]:
                    df[feature] = le.fit_transform(df[feature])
            df.to_csv('{0}-features-encoded.csv'.format(filename[:-4]), index=False)
            
        elif arg == '-imp':
            filename =  sys.argv[i+1]
            df = pd.read_csv(filename)
            
            X = df.ix[:, df.columns != 'class']
            y = df['class']
            
            classifiers = [
                ('Naive Bayes', naive_bayes.MultinomialNB()),
                ('Logistic Regression', linear_model.LogisticRegression()),
                ('Linear SVM', svm.LinearSVC()),
                ('RBF SVM', svm.SVC()),
                ('Random Forest', ensemble.RandomForestClassifier()),
                ('Gradient Boosting', ensemble.GradientBoostingClassifier(n_estimators=80,learning_rate=0.1, min_samples_split=400,min_samples_leaf=50,max_depth=7,max_features='sqrt',subsample=0.8,random_state=10)),
                ('Extreme Gradient Boosting', xgboost.XGBClassifier())
            ]
            
            # output out_filename
            fout = open('results.txt', 'w')
            
            fout.write('Classifier,Acc,F1,P,R\n')
            for clf_name, clf in classifiers:
                
                if clf_name == 'Gradient Boosting':
                    clf.fit(X, y)
                    plt.figure(0)
                    dtrain_predictions = clf.predict(X)
                    dtrain_predprob = clf.predict_proba(X)[:,1]
                    feat_imp = pd.Series(clf.feature_importances_, list(X)).sort_values(ascending=False)[:15]
                    feat_imp.plot(kind='bar', title='Feature Importances')
                    plt.ylabel('Feature Importance Score')
                    plt.tight_layout()
                    plt.savefig('fig.png')
                    
                accuracy_scores, f1_scores, p_scores, r_scores = train_model(clf, X, y, cv=5)    
                fout.write('%s,%0.4f (+/- %0.4f),%0.4f (+/- %0.4f),%0.4f (+/- %0.4f),%0.4f (+/- %0.4f)\n' % 
                           (clf_name, 
                            accuracy_scores.mean(), accuracy_scores.std() * 4,
                            f1_scores.mean(), f1_scores.std() * 4,
                            p_scores.mean(), p_scores.std() * 4,
                            r_scores.mean(), r_scores.std() * 4))
                fout.flush()

            fout.close()
