import os
import pandas as pd
from sklearn.ensemble import IsolationForest

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

    def outlier_detection(self, X):
        isolation_forest_model = IsolationForest(max_samples=100, random_state=42)
        X['outlier'] = isolation_forest_model.fit_predict(X)
        return X

    def encoding(self, X):
        pass

    def preprocessing(self, dataset):
        predict_for_airline_code = 'LH'
        dataset = self.filter_dataset_by_airline_code(dataset, predict_for_airline_code)
        X, y = self.dataset_extract_target(dataset)
        selected_features = ['origin_location_code', 'destination_location_code', 'departure_date', 'distance', 'duration']
        X = self.feature_selection(X, selected_features)

        X = self.outlier_detection(X)
        pass



    def ml_flow(self):
        self.preprocessing(self.dataset)

if __name__ == "__main__":
    fareMLPrediction = FareMLPrediction()
    fareMLPrediction.ml_flow()