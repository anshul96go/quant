class Model:

    def prepare_features(self, df):
        """
        :param df: this is the data you want to use to prepare the features for your model
        :return: X, a matrix of features (can be a numpy array or a pandas dataframe, your choice!)
        """
        # todo: implement this function - you can use some of the features given to you or you can build a batch of
        #  your own based on the data that you are given.
        # *** PLEASE ENSURE THAT DO NOT INTRODUCE A LOOKAHEAD IN THIS MATRIX ***
        # *** Bonus points for coding a function that tests against lookahead in X ***
        pass

    def fit(self, path_to_train_csv, *args, **kwargs):
        # todo: read train csv
        # todo: do any operation you would like on it

        # todo: prepare features for the model fit
        X = self.prepare_features(some_dataframe)

        # todo: fit your model here - use X (features matrix), y (the target variable) and any other information you
        #  want to use

        # this follows the scikit-learn pattern by returning self
        return self

    def predict(self, path_to_test_csv, *args, **kwargs):
        # todo: read test csv
        # todo: do any operation you would like on it

        # todo: prepare features for the model predict
        X = self.prepare_features(some_dataframe)

        # todo: calculate your model prediction (call it ypred) using X and any other information you want to use

        # this follows the scikit-learn pattern by returning ypred
        return ypred


if __name__ == '__main__':
    train_csv_path = 'put your local path here'
    test_csv_path = 'put your local path here'

    fit_args = []  # todo: populate this as you see fit
    fit_kwargs = {}  # todo: populate this as you see fit
    clf = Model()
    clf.fit(train_csv_path, *fit_args, **fit_kwargs)

    predict_args = []  # todo: populate this as you see fit
    predict_kwargs = {}  # todo: populate this as you see fit
    ypred = clf.predict(test_csv_path, *predict_args, **predict_kwargs)