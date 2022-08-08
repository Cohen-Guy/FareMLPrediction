import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

class FareMLPrediction:

    def __init__(self):
        self.debug_flag = True
        dataset_csv_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'flight_offers.csv')
        self.dataset = pd.read_csv(dataset_csv_file_path)

    def filter_dataset_by_airline_code(self, dataset, airline_code):
        return dataset[dataset['validatingAirlineCodes'] == airline_code]

    def dataset_extract_target(self, dataset):
        target_col_name = 'total'
        y = dataset.pop(target_col_name)
        return dataset, y

    def feature_selection(self, X, selected_features):
        return X[selected_features]

    def feature_engineering(self, X):
        X['departure_date'] = pd.to_datetime(X['departure_date'])
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
        predict_for_airline_code = 'LH'
        dataset = self.filter_dataset_by_airline_code(dataset, predict_for_airline_code)
        X, y = self.dataset_extract_target(dataset)
        selected_features = ['origin_location_code', 'destination_location_code', 'departure_date',# 'validatingAirlineCodes',
                             'distance', 'duration']
        X = self.feature_selection(X, selected_features)
        categorical_columns_names = ['origin_location_code', 'destination_location_code',
                                     #'validatingAirlineCodes'
                                     ]
        X = self.feature_engineering(X)
        X = self.encoding(X, categorical_columns_names)
        excluded_column = 'departure_date'
        # X = self.outlier_detection(X, excluded_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        return X_train, X_test, y_train, y_test


    def train(self, X_train, y_train):
        classifier = LinearRegression()
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
        classifier = self.train(X_train, y_train)
        X, train_root_mean_square_error = self.evaluate(classifier, X_train, y_train)
        print(f"train_root_mean_square_error: {train_root_mean_square_error}")
        X, test_root_mean_square_error = self.evaluate(classifier, X_test, y_test)
        print(f"test_accuracy_score: {test_root_mean_square_error}")
        pass


if __name__ == "__main__":
    fareMLPrediction = FareMLPrediction()
    fareMLPrediction.ml_flow()