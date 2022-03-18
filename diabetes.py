import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.manifold import TSNE 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import constants

numerical_features = constants.NUMERICAL_FEATURES

#plot graph for each feature
def visualize_data(diabetes):
    print("\nPlot graph for each feature\n")
    plt.figure(figsize = (constants.FIGSIZE_X, constants.FIGSIZE_Y))
    for x in range(len(diabetes.columns)):
        plt.subplot(constants.N_PLOT_X, constants.N_PLOT_Y, x + 1)
        if diabetes.columns[x] in numerical_features:
            plt.hist(diabetes[diabetes.columns[x]], facecolor = "g") 
        else:
            diabetes[diabetes.columns[x]].value_counts().sort_index().plot(kind = "bar")
        
        plt.ylabel("N* of occurrences")
        plt.xlabel(diabetes.columns[x])
        plt.xticks(rotation = constants.ROTATION, horizontalalignment = "center")

    plt.subplots_adjust(left = constants.LEFT, bottom = constants.BOTTOM, right = constants.RIGHT, top = constants.TOP, wspace = constants.W_SPACE, hspace = constants.H_SPACE)
    plt.show()

#computing measures for each feature
def compute_measures(diabetes):
    print("\nCompute measures for each feature\n")
    for x in range(len(diabetes.columns)):
        if diabetes.columns[x] in numerical_features:
            print("AVG of " + diabetes.columns[x] + " is: " + str(np.average(diabetes[diabetes.columns[x]])))
            print("STD of " + diabetes.columns[x] + " is: " + str(np.std(diabetes[diabetes.columns[x]])))
        else:
            print("Median of " + diabetes.columns[x] + " is: " + str(np.median(diabetes[diabetes.columns[x]])))
            print("Mode of " + diabetes.columns[x] + " is: " + str(stats.mode(diabetes[diabetes.columns[x]])))

#split dataset in train and test
def split_dataset(y, X):
    print("\nSplitting dataset\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = constants.TEST_SIZE, stratify = y)
    print("Dimension of training-test is: " + str(X_train.shape) + " and " + str(y_train.shape))
    print("Dimension of test-test is: " + str(X_test.shape) + " and " + str(y_test.shape))

    return X_train, X_test, y_train, y_test

#perform feature selection with MI and CHI2
def features_selection_MI_CHI2(X, y, mode):
    print("\nFeature selection\n")
    best_features = SelectKBest(score_func = mode, k = "all")
    fit = best_features.fit(X, y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)
    feature_scores = pd.concat([df_columns, df_scores], axis = 1)
    feature_scores.columns = ['Specs', 'Score']

    plt.figure(figsize = (constants.FIGSIZE_X_FS, constants.FIGSIZE_Y_FS))
    plt.bar([X.columns[i] for i in range(len(best_features.scores_))], best_features.scores_)
    plt.xticks(rotation = constants.ROTATION_FS, horizontalalignment = "center")
    plt.subplots_adjust(left = constants.LEFT_FS, bottom = constants.BOTTOM_FS, right = constants.RIGHT_FS, top = constants.TOP_FS, wspace = constants.W_SPACE_FS, hspace = constants.H_SPACE_FS)
    plt.show()
    return feature_scores
    
#build a new dataset only with important features
def optimal_dataset(score_MI, X):
    print("\nMost important features\n")
    score_ordered = score_MI[score_MI.Score > constants.DIVIDER]
    print(score_ordered)
    for i in X.columns:
        if i not in list(score_ordered.Specs):
            X.pop(i)
    return X

#ML models comparison
def models_comparison(X_train, y_train):
    print("\nSearching best algorithm\n")
    kf = StratifiedKFold(n_splits = constants.N_SPLITS, random_state = None, shuffle = False)
    logistic_regression = LogisticRegression(solver = "liblinear")
    random_forest = RandomForestClassifier()
    ada_boost = AdaBoostClassifier()
    gradient_boosting = GradientBoostingClassifier()
    decision_tree = DecisionTreeClassifier()

    acc_regr, acc_rf, acc_ab, acc_gb, acc_dt = 0, 0, 0, 0, 0

    for train_index, validation_index in kf.split(X_train, y_train):
        xt = X_train.iloc[train_index]
        xv = X_train.iloc[validation_index]
        yt = y_train.iloc[train_index]
        yv = y_train.iloc[validation_index]

        logistic_regression.fit(xt, yt)
        random_forest.fit(xt, yt)
        logistic_regression.fit(xt, yt)
        ada_boost.fit(xt, yt)
        gradient_boosting.fit(xt, yt)
        decision_tree.fit(xt, yt)

        y_pred_vt_regr = logistic_regression.predict(xv)
        y_pred_vt_rf = random_forest.predict(xv)
        y_pred_vt_ab = ada_boost.predict(xv)
        y_pred_vt_gb = gradient_boosting.predict(xv)
        y_pred_vt_dt = decision_tree.predict(xv)

        acc_regr += accuracy_score(yv,  y_pred_vt_regr)
        acc_rf += accuracy_score(yv,  y_pred_vt_rf)
        acc_ab += accuracy_score(yv,  y_pred_vt_ab)
        acc_gb += accuracy_score(yv,  y_pred_vt_gb)
        acc_dt += accuracy_score(yv,  y_pred_vt_dt)
    
    print("Logistic Regression's accuracy: " + str((acc_regr / constants.N_SPLITS) * 100))
    print("Random Forest's accuracy: " + str((acc_rf / constants.N_SPLITS) * 100))
    print("AdaBoost's accuracy: " + str((acc_ab / constants.N_SPLITS) * 100))
    print("GradientBoosting's accuracy: " + str((acc_gb / constants.N_SPLITS) * 100))
    print("Decision Tree's accuracy: " + str((acc_dt / constants.N_SPLITS) * 100))

#perform fine tuning with random search to take the best n_estimators 
def fine_tuning_gb_rs(X_train, y_train):
    kf = StratifiedKFold(n_splits = constants.N_SPLITS, random_state = None, shuffle = False)
    print("\nRandom Search\n")
    train_results = []
    validation_results = []       
    acc_val, acc_train = 0, 0
    for i in constants.N_TREES:
        acc_val, acc_train = 0, 0
        optimal_gradient_boosting = GradientBoostingClassifier(n_estimators = i)
        for train_index, validation_index in kf.split(X_train, y_train):
            xt = X_train.iloc[train_index]
            xv = X_train.iloc[validation_index]
            yt = y_train.iloc[train_index]
            yv = y_train.iloc[validation_index] #ho utilizzato iloc anche qui in quanto passando come parametro y_train e non y, non ho una sola colonna ma gli indici e il valore corrispondente
            optimal_gradient_boosting.fit(xt, yt)
            y_pred_vt_gb = optimal_gradient_boosting.predict(xv)
            y_pred_xt_gb = optimal_gradient_boosting.predict(xt)
            acc_val += accuracy_score(yv, y_pred_vt_gb)
            acc_train += accuracy_score(yt, y_pred_xt_gb)

        train_results.append(acc_train / constants.N_SPLITS)
        validation_results.append(acc_val / constants.N_SPLITS)
        print("Computing " +  str(i) + " with validation accuracy: " + str((acc_val / constants.N_SPLITS) * 100))

    plt.plot(constants.N_TREES, train_results, color = "blue", label = "Training")
    plt.plot(constants.N_TREES, validation_results, color = "red", label = "Validation")
    plt.scatter(constants.N_TREES, validation_results, s = 40, facecolors = "none", edgecolors = "r")
    plt.scatter(constants.N_TREES, train_results, s = 40, facecolors = "none", edgecolors = "b")
    plt.legend()
    plt.show()

#perform fine tuning with grid search to take the best n_estimators, learning_rate and max_depth
def fine_tuning_gb_gs(X_train, y_train):
    print("\nGrid Search\n")
    kf = StratifiedKFold(n_splits = constants.N_SPLITS, random_state = None, shuffle = False)
    for i in range(constants.N_DEFAULT_TREES - 25, constants.N_DEFAULT_TREES + 100, 25):
        for j in range(constants.MAX_DEFAULT_DEPTH, constants.MAX_DEFAULT_DEPTH + 5, 1):
            for h in constants.N_LEARNING_RATE:
                final_gradient_boosting = GradientBoostingClassifier(n_estimators = i, learning_rate = j, max_depth = h)
                for train_index, validation_index in kf.split(X_train, y_train):
                    xt = X_train.iloc[train_index]
                    xv = X_train.iloc[validation_index]
                    yt = y_train.iloc[train_index]
                    yv = y_train.iloc[validation_index]
                    final_gradient_boosting.fit(xt, yt)
                    y_pred_vt_gb = final_gradient_boosting.predict(xv)
                print(str(accuracy_score(yv, y_pred_vt_gb) * 100) + " with trees " + str(i) + ", learning rate " + str(j) + ", max depth " + str(h))

#perform a final evaluation on the test-set
def final_evaluation_ts(X_train, y_train, X_test, y_test):
    print("\nEvaluation on test-set\n")
    gb = GradientBoostingClassifier(n_estimators = constants.N_ESTIMATORS, learning_rate = constants.LEARNING_RATE, max_depth = constants.MAX_DEPTH)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    print("Final accuracy test set: " + str(accuracy_score(y_test, y_pred) * 100))
    conf_matrix_view = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels = ["negative", "positive"])
    conf_matrix_view.plot()
    conf_matrix_view.ax_.set(title = "Confusion matrix for diabetes", xlabel = "Predicted", ylabel = "Real class")
    plt.show()

#view a t-SNE 3D plot
def tsne_3d(X_train_tsne, y_train_tsne):
    ax = plt.axes(projection = "3d")
    labels = ["Negative", "Positive"]
    tsne = TSNE(n_components = 3, perplexity = 53, init = "pca", learning_rate = "auto") 
    dim_result = tsne.fit_transform(X_train_tsne) 
    scatter = ax.scatter(dim_result[:, 0], dim_result[:, 1], dim_result[:, 2], c = y_train_tsne)
    handles, _ = scatter.legend_elements(prop = "colors")
    ax.legend(handles, labels)
    plt.show()

#view a t-SNE 2D plot
def tsne_2d(X_train_tsne, y_train_tsne):
    labels = ["Negative", "Positive"]
    tsne = TSNE(n_components = 2, perplexity = 53, init = "pca", learning_rate = "auto") 
    dim_result = tsne.fit_transform(X_train_tsne) 
    scatter = plt.scatter(dim_result[:, 0], dim_result[:, 1], c = y_train_tsne)
    handles, _ = scatter.legend_elements(prop = "colors")
    plt.legend(handles, labels)
    plt.show()

#get a subset of dataset for calculating t-SNE
def subset_for_tsne(X_train, y_train):
    _, X_test_tsne, _, y_test_tsne = train_test_split(X_train, y_train, test_size = constants.TEST_SIZE_TSNE, stratify = y_train)
    return X_test_tsne, y_test_tsne

#retrieve data from mongoDB
def get_data():
    print("I'm loading data...")
    diabetes = pd.read_csv(constants.PATH)

    return diabetes

#start of program
def main():
    diabetes = get_data()
    visualize_data(diabetes)
    compute_measures(diabetes)
    y, X = diabetes.pop("Diabetes_binary"), diabetes
    score_MI = features_selection_MI_CHI2(X, y, mutual_info_classif) 
    #score_CHI2 = features_selection_MI_CHI2(X, y, chi2) 
    X = optimal_dataset(score_MI, X)
    X_train, X_test, y_train, y_test = split_dataset(y, X)
    models_comparison(X_train, y_train)
    fine_tuning_gb_rs(X_train, y_train)
    #fine_tuning_gb_gs(X_train, y_train)
    final_evaluation_ts(X_train, y_train, X_test, y_test)
    X_test_tsne, y_test_tsne = subset_for_tsne(X_train, y_train)
    tsne_2d(X_test_tsne, y_test_tsne)
    tsne_3d(X_test_tsne, y_test_tsne)
    
main()
