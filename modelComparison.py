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
