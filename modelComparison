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
