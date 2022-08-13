import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials

class FareMLPrediction:

    def __init__(self):
        self.debug_flag = True
        dataset_csv_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'flight_offers.csv')
        self.dataset = pd.read_csv(dataset_csv_file_path)

    def filter_dataset_by_airline_code(self, dataset, airline_code):
        return dataset[dataset['validatingAirlineCodes'] == airline_code]

    def cleaning(self, dataset):
        return dataset[dataset['validatingAirlineCodes'] != 'SVO']

    def dataset_extract_target(self, dataset):
        target_col_name = 'total'
        y = dataset.pop(target_col_name)
        return dataset, y

    def feature_selection(self, X, selected_features):
        return X[selected_features]

    def feature_engineering(self, X):
        X['departure_date'] = pd.to_datetime(X['departure_date'], format='%d/%m/%Y')
        X['year'] = X['departure_date'].dt.year
        X['month'] = X['departure_date'].dt.month
        X['day'] = X['departure_date'].dt.day
        X.pop('departure_date')
        return X

    def outlier_detection(self, X, excluded_column):
        isolation_forest_model = IsolationForest(max_samples=100, random_state=42)
        X_without_excluded_columns = X.loc[:, X.columns != excluded_column]
        X['outlier'] = isolation_forest_model.fit_predict(X_without_excluded_columns)
        return X

    def encoding(self, X, categorical_columns_names):
        X_categorical_columns = X.loc[:, categorical_columns_names]
        X[categorical_columns_names] = X_categorical_columns.apply(LabelEncoder().fit_transform)
        return X

    def preprocessing(self, dataset):
        predict_for_airline_code = 'AF'
        dataset = self.filter_dataset_by_airline_code(dataset, predict_for_airline_code)
        dataset = self.cleaning(dataset)
        X, y = self.dataset_extract_target(dataset)
        selected_features = ['origin_location_code', 'destination_location_code', 'departure_date', 'distance'] # 'validatingAirlineCodes',
        X = self.feature_selection(X, selected_features)
        categorical_columns_names = ['origin_location_code', 'destination_location_code',
                                     #'validatingAirlineCodes'
                                     ]
        X = self.feature_engineering(X)
        X = self.encoding(X, categorical_columns_names)
        # excluded_column = 'departure_date'
        # X = self.outlier_detection(X, excluded_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_linear_regression(self, X_train, y_train):
        classifier = LinearRegression()
        classifier.fit(X_train, y_train)
        return classifier

    def train_xgboost_regressor(self, X_train, y_train):
        classifier = XGBRegressor(colsample_bytree=4, gamma=7.549847309471019, learning_rate=4, max_depth=7, min_child_weight=6, n_estimators=4,
                                  reg_alpha=111, reg_lambda=0.6663906133223992, subsample=0.9818632459626231)
        classifier.fit(X_train, y_train)
        return classifier

    def train_random_forest_regressor(self, X_train, y_train):
        classifier = RandomForestRegressor()
        classifier.fit(X_train, y_train)
        return classifier

    def evaluate(self, classifier, X, y):
        y_pred = classifier.predict(X)
        root_mean_square_error = mean_squared_error(y, y_pred, squared=False)
        X['y_true'] = y
        X['y_pred'] = y_pred
        return X, root_mean_square_error


    def ml_flow(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self.dataset)
        classifier = self.train_random_forest_regressor(X_train, y_train)
        X, train_root_mean_square_error = self.evaluate(classifier, X_train, y_train)
        print(f"train_root_mean_square_error: {train_root_mean_square_error}")
        X, test_root_mean_square_error = self.evaluate(classifier, X_test, y_test)
        print(f"test_root_mean_square_error: {test_root_mean_square_error}")
        pass

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def xgb_reg(self, para):
        reg = XGBRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        reg.fit(self.X_train, self.y_train,
                eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict(self.X_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}

    def hyperparameters_optimization(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self.dataset)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # XGB parameters
        xgb_reg_params = {
            'learning_rate': hp.choice('learning_rate', np.arange(0.05, 0.31, 0.05)),
            'max_depth': hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
            'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
            'subsample': hp.uniform('subsample', 0.8, 1),
            'n_estimators': hp.choice('n_estimators', (1, 3, 5, 7, 10)),
            'gamma': hp.uniform('gamma', 1, 9),
            'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        }
        xgb_fit_params = {
            'eval_metric': 'rmse',
            'early_stopping_rounds': 10,
            'verbose': False
        }
        xgb_para = dict()
        xgb_para['reg_params'] = xgb_reg_params
        xgb_para['fit_params'] = xgb_fit_params
        xgb_para['loss_func'] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))

        xgb_opt = self.process(fn_name='xgb_reg', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=100)
        print(xgb_opt)

if __name__ == "__main__":
    fareMLPrediction = FareMLPrediction()
    fareMLPrediction.hyperparameters_optimization()
    # fareMLPrediction.ml_flow()