import os
import datetime
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
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

class FareMLPrediction:

    def __init__(self):
        self.time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.debug_flag = True
        dataset_csv_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'flight_offers.csv')
        self.dataset = pd.read_csv(dataset_csv_file_path)
        self.selected_features = ['origin_location_code', 'destination_location_code', 'departure_date', 'validatingAirlineCodes', 'distance'] # 'validatingAirlineCodes',

    def filter_dataset_by_airline_code(self, dataset, airline_code):
        return dataset[dataset['validatingAirlineCodes'] == airline_code]

    # def cleaning(self, dataset):
    #     return dataset[dataset['validatingAirlineCodes'] != 'SVO']

    def dataset_extract_target(self, dataset):
        target_col_name = 'total'
        y = dataset.pop(target_col_name)
        return dataset, y

    def feature_selection(self, X):
        return X[self.selected_features]

    def feature_engineering(self, X):
        X['departure_date'] = pd.to_datetime(X['departure_date'], format='%d/%m/%Y')
        X['year'] = X['departure_date'].dt.year
        X['month'] = X['departure_date'].dt.month
        X['day'] = X['departure_date'].dt.day
        X['weekday'] = X['departure_date'].dt.weekday
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
        # predict_for_airline_code = 'AF'
        # dataset = self.filter_dataset_by_airline_code(dataset, predict_for_airline_code)
        # dataset = self.cleaning(dataset)
        X, y = self.dataset_extract_target(dataset)
        X = self.feature_selection(X)
        categorical_columns_names = ['origin_location_code', 'destination_location_code', 'validatingAirlineCodes']
        X = self.feature_engineering(X)
        X = self.encoding(X, categorical_columns_names)
        # excluded_column = 'departure_date'
        # X = self.outlier_detection(X, excluded_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_linear_regression(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def train_xgboost_regressor(self, X_train, y_train):
        model = XGBRegressor(colsample_bytree=0.7, learning_rate=0.07, max_depth=7, min_child_weight=4, n_estimators=100)
        model.fit(X_train, y_train)
        return model

    def explainability(self, model, X, y):
        shap_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'explainability', 'shap')
        sample_ind = 20
        sampled_x = X.sample(frac=0.1)
        explainer = shap.Explainer(model, sampled_x)
        shap_values = explainer(sampled_x)
        waterfall_diagram = shap.plots.waterfall(shap_values[0], show=False)
        waterfall_diagram_name = f"waterfall_diagram_{self.time_str}.png"
        waterfall_diagram_path = os.path.join(shap_folder_path, waterfall_diagram_name)
        waterfall_diagram.savefig(waterfall_diagram_path, format='png', dpi=600, bbox_inches='tight')
        plt.clf()
        shap.plots.beeswarm(shap_values, show=False)
        beeswarm_diagram_name = f"beeswarm_diagram_{self.time_str}.png"
        beeswarm_diagram_path = os.path.join(shap_folder_path, beeswarm_diagram_name)
        beeswarm_diagram = plt.gcf()
        beeswarm_diagram.savefig(beeswarm_diagram_path, format='png', dpi=600, bbox_inches='tight')
        plt.clf()
        # summary_plot_fig = shap.summary_plot(shap_values[0], X_test, show=False)
        # summary_plot_diagram_name = f"summary_plot_diagram_{self.time_str}.png"
        # summary_plot_path = os.path.join(summary_plot_fig, summary_plot_diagram_name)
        # summary_plot_fig.savefig(summary_plot_path, format='png', dpi=600, bbox_inches='tight')

        # for feature in self.selected_features:
        feature = 'distance'
        self.shap_plots_for_feature(model, feature, shap_values, shap_folder_path, sampled_x, sample_ind)
        # feature = 'departure_date'
        # self.shap_plots_for_feature(feature, shap_values, shap_folder_path, sampled_x, sample_ind)

    def shap_plots_for_feature(self, model, feature, shap_values, shap_folder_path, sampled_x, sample_ind):
        feature_shap_values = shap_values[:, feature]
        shap.plots.scatter(feature_shap_values, color=shap_values, show=False)
        scatter_diagram_name = f"{feature}_scatter_diagram_{self.time_str}.png"
        scatter_diagram_path = os.path.join(shap_folder_path, scatter_diagram_name)
        scatter_diagram = plt.gcf()
        scatter_diagram.savefig(scatter_diagram_path, format='png', dpi=600, bbox_inches='tight')
        plt.clf()
        partial_dependence_diagram, ax = shap.partial_dependence_plot(
            feature, model.predict, sampled_x, model_expected_value=True,
            feature_expected_value=True, ice=False,
            shap_values=shap_values[sample_ind:sample_ind + 1, :],
            show=False
        )
        partial_dependence_diagram_name = f"{feature}_partial_dependence_diagram_{self.time_str}.png"
        partial_dependence_diagram_path = os.path.join(shap_folder_path, partial_dependence_diagram_name)
        partial_dependence_diagram.savefig(partial_dependence_diagram_path, format='png', dpi=600, bbox_inches='tight')
        plt.clf()

    def train_random_forest_regressor(self, X_train, y_train):
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        return model

    def evaluate_train(self, model, X, y):
        y_pred = model.predict(X)
        root_mean_square_error = mean_squared_error(y, y_pred, squared=False)
        X['y_true'] = y
        X['y_pred'] = y_pred
        return X, root_mean_square_error

    def evaluate_test(self, model, X, y):
        y_pred = model.predict(X)
        root_mean_square_error = mean_squared_error(y, y_pred, squared=False)
        return X, root_mean_square_error


    def ml_flow(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self.dataset)
        model = self.train_xgboost_regressor(X_train, y_train)
        X_train, train_root_mean_square_error = self.evaluate_train(model, X_train, y_train)
        print(f"train_root_mean_square_error: {train_root_mean_square_error}")
        X_test, test_root_mean_square_error = self.evaluate_test(model, X_test, y_test)
        print(f"test_root_mean_square_error: {test_root_mean_square_error}")
        return model, X_test, y_test

    def hyperparameters_optimization(self):
        X_train, X_test, y_train, y_test = self.preprocessing(self.dataset)
        parameters = {'objective': ['reg:squarederror'],
                      'learning_rate': [.03, 0.05, .07],  # so called `eta` value
                      'max_depth': [5, 6, 7],
                      'min_child_weight': [4],
                      'subsample': [0.7],
                      'colsample_bytree': [0.7],
                      'n_estimators': [1, 4, 10, 50, 100, 500]}

        xgb = XGBRegressor()
        random_search = GridSearchCV(xgb,
                        parameters,
                        cv=5,
                        n_jobs=5,
                        verbose=True)
        random_search.fit(X_train, y_train)
        print(random_search.best_params_)


if __name__ == "__main__":
    fareMLPrediction = FareMLPrediction()
    # fareMLPrediction.hyperparameters_optimization()
    model, X_test, y_test = fareMLPrediction.ml_flow()
    # fareMLPrediction.explainability(model, X_test, y_test)