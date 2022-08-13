import os
import pandas as pd
# import plotly.io as pio
import plotly.express as px
# import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff

class FareMLEDA:

    def __init__(self):
        pd.set_option('display.max_columns', None)
        self.debug_flag = True
        dataset_csv_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'flight_offers.csv')
        self.dataset = pd.read_csv(dataset_csv_file_path)

    def basic_info(self):
        print(f"shape:\n {self.dataset.shape}")
        print(f"head:\n {self.dataset.head(10)}")
        print(f"columns:\n {self.dataset.columns}")
        print(f"unique:\n {self.dataset.nunique(axis=0)}")
        print(f"describe:\n {self.dataset.describe(include='all')}")

    def correlation(self):
        corr = self.dataset.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        df_mask = corr.mask(mask)

        fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(),
                                          x=df_mask.columns.tolist(),
                                          y=df_mask.columns.tolist(),
                                          colorscale=px.colors.diverging.RdBu,
                                          hoverinfo="none",  # Shows hoverinfo for null values
                                          showscale=True, ygap=1, xgap=1
                                          )

        fig.update_xaxes(side="bottom")

        fig.update_layout(
            title_text='Heatmap',
            title_x=0.5,
            width=1000,
            height=1000,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_zeroline=False,
            yaxis_zeroline=False,
            yaxis_autorange='reversed',
            template='plotly_white'
        )

        # NaN values are not handled automatically and are displayed in the figure
        # So we need to get rid of the text manually
        for i in range(len(fig.layout.annotations)):
            if fig.layout.annotations[i].text == 'nan':
                fig.layout.annotations[i].text = ""

        fig.show()

        # fig = go.Figure()
        # fig.add_trace(
        #     go.Heatmap(
        #         x=correlation.columns,
        #         y=correlation.index,
        #         z=np.array(correlation)
        #     )
        # )
        # fig.show()

    def plots_for_columns(self, x_column_name, y_column_name):
        self.scatter_plot(x_column_name, y_column_name)
        self.histogram(x_column_name, y_column_name)
        self.plotbox(x_column_name, y_column_name)

    def scatter_plot(self, x_column_name, y_column_name):
        fig = px.scatter(x=self.dataset[x_column_name], y=self.dataset[y_column_name])
        fig.show()

    def histogram(self, x_column_name, y_column_name):
        fig = px.histogram(self.dataset, x=x_column_name, y=y_column_name)
        fig.show()

    def plotbox(self, x_column_name, y_column_name):
        fig = px.box(self.dataset, x=x_column_name, y=y_column_name)
        fig.show()

    def scatter_matrix(self):
        fig = px.scatter_matrix(self.dataset)
        fig.show()

    def eda(self):
        self.basic_info()
        self.correlation()
        self.plots_for_columns('validatingAirlineCodes', 'total')
        self.plots_for_columns('origin_location_code', 'total')
        self.plots_for_columns('destination_location_code', 'total')
        self.plots_for_columns('distance', 'total')
        self.plots_for_columns('departure_date', 'total')
        self.plots_for_columns('base', 'total')


if __name__ == "__main__":
    fareMLPrediction = FareMLEDA()
    fareMLPrediction.eda()