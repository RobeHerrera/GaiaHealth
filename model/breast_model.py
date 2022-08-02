# Python libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
import warnings
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import itertools

warnings.filterwarnings('ignore') #ignore warning messages 
# Confusion matrix 
def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix"',
                          cmap = plt.cm.Blues) :
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Show metrics 
def show_metrics(cm):
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    print('Accuracy  =     {:.3f}'.format((tp+tn)/(tp+tn+fp+fn)))
    print('Precision =     {:.3f}'.format(tp/(tp+fp)))
    print('Recall    =     {:.3f}'.format(tp/(tp+fn)))
    print('F1_score  =     {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/
                                                 ((tp/(tp+fp))+(tp/(tp+fn))))))

# Precision-recall curve
def plot_precision_recall(recall, precision):
    plt.step(recall, precision, color = 'b', alpha = 0.2,
             where = 'post')
    plt.fill_between(recall, precision, step ='post', alpha = 0.2,
                 color = 'b')

    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0,1])
    plt.ylim([0.0,1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.show()

# ROC curve
def plot_roc(fpr, tpr):
    plt.plot(fpr, tpr, label = 'ROC curve', linewidth = 2)
    plt.plot([0,1],[0,1], 'k--', linewidth = 2)
   # plt.xlim([0.0,0.001])
   # plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

# Learning curve
def plot_learning_curve(estimator, title, X, y, ylim = None, cv = None,
                        n_jobs = 1, train_sizes = np.linspace(.1, 1.0, 5)):
    """
    Plots a learning curve. http://scikit-learn.org/stable/modules/learning_curve.html
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis = 1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha = 0.1, color = "g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color = "r",
             label = "Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color = "g",
             label = "Cross-validation score")
    plt.legend(loc = "best")
    return plt   

# Cross val metric
def cross_val_metrics(model, X, y) :
    scores = ['accuracy', 'precision', 'recall']
    for sc in scores:
        scores = cross_val_score(model, X, y, cv = 5, scoring = sc)
        print('[%s] : %0.5f (+/- %0.5f)'%(sc, scores.mean(), scores.std()))

def run():
    print("Hello form model")
    # Read data
    data = pd.read_csv('data/data.csv')
    null_feat = pd.DataFrame(len(data['id']) - data.isnull().sum(), columns = ['Count'])

    trace = go.Bar(x = null_feat.index, y = null_feat['Count'] ,opacity = 0.8, marker=dict(color = 'lightgrey',
            line=dict(color='#000000',width=1.5)))

    layout = dict(title =  "Missing Values")
                        
    fig = dict(data = [trace], layout=layout)
    # py.iplot(fig)

    # Drop useless variables
    data = data.drop(['Unnamed: 32','id'],axis = 1)

    # Reassign target
    data.diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = True)

    # ------------ Predicting the Breast Cancer Data Set ------------#
    # Def X and Y
    y = np.array(data.diagnosis.tolist())
    data = data.drop('diagnosis', 1)
    print(data.head())
    X = np.array(data.to_numpy())

    # Normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train_test split
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.12, random_state = random_state)

    # ------------ Predicting with model 1------------#
    if True:
        # Find best hyperparameters (accuracy)
        log_clf = LogisticRegression(random_state = random_state, max_iter=1_000_000)
        param_grid = {
                    'penalty' : ['l2','l1'],  
                    'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                    }

# Increase the number of iterations (max_iter) or scale the data as shown in:
#     https://scikit-learn.org/stable/modules/preprocessing.html
        CV_log_clf = GridSearchCV(estimator = log_clf, param_grid = param_grid , scoring = 'accuracy', verbose = 1, n_jobs = -1)
        CV_log_clf.fit(X_train, y_train)

        best_parameters = CV_log_clf.best_params_
        print('The best parameters for using this model is', best_parameters)

        #Log with best hyperparameters
        CV_log_clf = LogisticRegression(C = best_parameters['C'], 
                                        penalty = best_parameters['penalty'], 
                                        random_state = random_state)

        CV_log_clf.fit(X_train, y_train)
        y_pred = CV_log_clf.predict(X_test)
        y_score = CV_log_clf.decision_function(X_test)

        # Confusion maxtrix & metrics
        cm = confusion_matrix(y_test, y_pred)
        class_names = [0,1]
        plt.figure()
        plot_confusion_matrix(cm, 
                            classes=class_names, 
                            title='Logistic Confusion matrix')
        plt.savefig('6')
        # plt.show()

        show_metrics(cm)

        # ROC curve
        fpr, tpr, t = roc_curve(y_test, y_score)
        # plot_roc(fpr, tpr)

        # Logistic regression with RFE
        log_clf = LogisticRegression(C = best_parameters['C'], 
                                        penalty = best_parameters['penalty'], 
                                        random_state = random_state)

        selector = RFE(log_clf)
        selector = selector.fit(X_train, y_train)

        y_pred = selector.predict(X_test)
        y_score = selector.predict_proba(X_test)[:,1]


        # Confusion maxtrix & metrics
        cm = confusion_matrix(y_test, y_pred)
        class_names = [0,1]
        plt.figure()
        plot_confusion_matrix(cm, 
                            classes=class_names, 
                            title='Logistic Confusion matrix')
        # plt.show()

        show_metrics(cm)

        # ROC curve
        fpr, tpr, t = roc_curve(y_test, y_score)
        # plot_roc()

        # support and ranking RFE
        print(selector.support_)
        print(selector.ranking_)

        # Learning curve Log with best hyperpara
        # plot_learning_curve(CV_log_clf, 'Learning Curve For Logistic Model', X, y, (0.85,1.05), 10)
        plt.savefig('7')
        # plt.show()

        # Learning curve Log with RFE
        # plot_learning_curve(selector, 'Learning Curve For Logistic Model with RFE', X, y, (0.85,1.05), 10)
        # plt.show()

        # Cross val Log 
        cross_log = cross_val_metrics(CV_log_clf, X, y)

        # Cross val Log with RFE
        cross_selector = cross_val_metrics(selector, X, y)

        # Threshold
        thresholds_adj = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

        plt.figure(figsize = (15,15))

        j = 1
        for i in thresholds_adj:
            y_score = CV_log_clf.predict_proba(X_test)[:,1] > i
            
            
            plt.subplot(3,3,j)
            j += 1
            
            cm = confusion_matrix(y_test, y_score)
            
            tp = cm[1,1]
            fn = cm[1,0]
            fp = cm[0,1]
            tn = cm[0,0]

            print('Recall w/ threshold = %s :'%i, (tp/(tp+fn)))
            
            class_names = [0,1]
            plot_confusion_matrix(cm, 
                                classes=class_names, 
                                title='Threshold = %s'%i) 

        Recall = 1.
        y_score = CV_log_clf.predict_proba(X_test)[:,1] > 0.1
        cm = confusion_matrix(y_test, y_score)
        class_names = [0,1]
        show_metrics(cm)

    # ------------ Predicting with model 2------------#
    # else:
        # Find the best parameters (recall)
        log2_clf = LogisticRegression(random_state = random_state, max_iter=1_000_000)
        # logreg = LogisticRegression(solver='saga', max_iter=1)
        param_grid = {
                    'penalty' : ['l2','l1'],  
                    'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    }

        CV_log2_clf = GridSearchCV(estimator = log2_clf, param_grid = param_grid , scoring = 'recall', verbose = 1, n_jobs = -1)
        CV_log2_clf.fit(X_train, y_train)

        best_parameters = CV_log2_clf.best_params_
        print('The best parameters for using this model is', best_parameters)

        # 7.1. Logistic Regression and GridSearch CV to optimise hyperparameters (recall)
        # Log w best hyperparameters (recall)
        CV_log2_clf = LogisticRegression(C = best_parameters['C'], 
                                        penalty = best_parameters['penalty'], 
                                        random_state = random_state)


        CV_log2_clf.fit(X_train, y_train)

        y_pred = CV_log2_clf.predict(X_test)
        y_score = CV_log2_clf.decision_function(X_test)
        # Confusion maxtrix & metrics
        cm = confusion_matrix(y_test, y_pred)
        class_names = [0,1]

        # Cross val log2
        cross_val_metrics(CV_log2_clf, X, y)

        # 7.2. Voting classifier : log + log2
        #Voting Classifier
        voting_clf = VotingClassifier (
                estimators = [('log1', CV_log_clf), ('log_2', CV_log2_clf)],
                            voting='soft', weights = [1, 1])
            
        voting_clf.fit(X_train,y_train)

        y_pred = voting_clf.predict(X_test)
        y_score = voting_clf.predict_proba(X_test)[:,1]

        # Confusion maxtrix
        cm = confusion_matrix(y_test, y_pred)
        class_names = [0,1]
        show_metrics(cm)

        # # Cross val score voting
        cross_voting = cross_val_metrics(voting_clf, X, y)

        # #Learning curve Voting
        # plot_learning_curve(voting_clf, 'Learning Curve For Voting clf', X, y, (0.85,1.05), 10)
        # plt.savefig('9')
        # plt.show()

        # 7.3. Voting classifier : select threshold (recall = 100%)
        # Threshold
        thresholds_adj = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

        plt.figure(figsize = (15,15))

        j = 1
        for i in thresholds_adj:
            y_score = voting_clf.predict_proba(X_test)[:,1] > i
            
            
            plt.subplot(3,3,j)
            j += 1
            
            cm = confusion_matrix(y_test, y_score)
            
            tp = cm[1,1]
            fn = cm[1,0]
            fp = cm[0,1]
            tn = cm[0,0]

            print('Recall w/ threshold = %s :'%i, (tp/(tp+fn)))
            
            class_names = [0,1]
            # plot_confusion_matrix(cm, 
            #                     classes=class_names, 
            #                     title='Threshold = %s'%i) 

        # Ensemble, recall = 1.
        y_score = voting_clf.predict_proba(X_test)[:,1] > 0.23
        cm = confusion_matrix(y_test, y_score)
        class_names = [0,1]
        plt.figure()
        # plot_confusion_matrix(cm, 
        #                     classes = class_names, 
        #                     title = 'Ensemble Clf CM : recall = 100%')
        plt.savefig('8')
        # plt.show()

        show_metrics(cm)

        # ROC curve
        fpr, tpr, t = roc_curve(y_test, y_score)
        # plot_roc()

        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        # plot_precision_recall()

        models_metrics = {'log_clf': [0.982, 0.990, 0.962], 
                        'selector': [0.974, 0.981, 0.948],
                        'log2_clf' : [0.974,0.976,0.953],
                        'voting_clf' : [0.979,0.985,0.958]
                        }
        df = pd.DataFrame(data = models_metrics)
        df.rename(index={0:'Accuracy',1:'Precision', 2: 'Recall'}, 
                        inplace=True)
        ax = df.plot(kind='bar', figsize = (15,10), ylim = (0.94, 1), 
                color = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue'],
                rot = 0, title ='Models performance (cross val mean)',
                edgecolor = 'grey', alpha = 0.5)
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.0005))
        # plt.show()

    print("End of model")

