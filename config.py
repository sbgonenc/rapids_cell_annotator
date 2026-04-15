#### Default values ###

classifier_default_values = {
    "logistic_regression" : {'max_iter': 1000, 'tol': 1e-4, 'C': 1.0, 'fit_intercept': True, 'penalty': 'l2'},
    "random_forest" : {'n_estimators': 100, 'max_depth': 18, 'max_features': 'sqrt', 'n_streams': 4, 'random_state': 42},
    "svm" : {'C': 1.0, 'probability': True, 'fit_intercept': True, 'tol': 1e-4, 'max_iter': 1000, 'loss': 'hinge', 'class_weight': None},
    #"linear_regression" : {'fit_intercept': True, 'normalize': True}

}