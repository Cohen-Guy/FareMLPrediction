import pandas as pd
import datetime
import os
from amadeus import Client, ResponseError
from geodata.distance_matrix_api import DistanceMatrixAPI

class PrepareDataset:

    def __init__(self):
        self.amadeus_client = Client(client_id='rPUxmofxNT5y0SVlynGKGVxYYwpdv81r', client_secret='fGyqtjXAceJKYq4d')
        self.data_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'prepare')
        self.first_csv_write = True
        airports_top_50_and_tel_aviv_iata_location_codes_list = ['TLV', 'IST', 'AMS', 'FRA', 'DME', 'SVO', 'SAW', 'CDG', 'MAD', 'LED', 'LHR', 'ATH',
                                    'ORY', 'BCN', 'VKO', 'PMI', 'MUC', 'FCO', 'LIS', 'OSL', 'AER', 'VIE', 'ZRH', 'LPA', 'CPH', 'MXP', 'KBP', 'BRU',
                                    'AYT', 'ARN', 'NCE', 'BER', 'OTP', 'WAW', 'AGP', 'ESB', 'TFN', 'SIP', 'LYS', 'ADB', 'BGO', 'GVA', 'DUB', 'HEL',
                                    'CTA', 'KRR', 'MRS', 'IBZ', 'HAM', 'LIN', 'OPO']
        airports_top_10_and_tel_aviv_iata_location_codes_list = ['LHR', 'CDG', 'AMS', 'FRA', 'IST', 'MAD', 'BCN', 'MUC', 'LGW', 'SVO', 'TLV']
        self.iata_location_codes_list = airports_top_10_and_tel_aviv_iata_location_codes_list
        self.distanceMatrixAPI = DistanceMatrixAPI(self.iata_location_codes_list)
        self.time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def get_price_offers_for_date(self, departure_date, adults):
        for location_code_src in self.iata_location_codes_list:
            for location_code_dst in self.iata_location_codes_list:
                if (location_code_dst != location_code_src):
                    df, has_price_offers = self.get_price_offers_for_src_dst_date_adult(location_code_src, location_code_dst, departure_date, adults)
                    if has_price_offers:
                        self.store_df_to_csv(df)

    def get_price_offers_for_date_range(self, start_date, end_date, adults):
        start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        date_list = [start_datetime + day_index * datetime.timedelta(days=1) for day_index in range((end_datetime-start_datetime).days + 1)]
        for departure_date in date_list:
            self.get_price_offers_for_date(departure_date.strftime('%Y-%m-%d'), adults)

    def get_price_offers_for_src_dst_date_adult(self, origin_location_code, destination_location_code, departure_date, adults):
        try:
            response = self.amadeus_client.shopping.flight_offers_search.get(
                originLocationCode=origin_location_code,
                destinationLocationCode=destination_location_code,
                departureDate=departure_date,
                adults=adults
                )
            df, has_price_offers = self.get_price_offer_df(response.data, origin_location_code, destination_location_code, departure_date, adults)
            return df, has_price_offers
        except ResponseError as error:
            print(error)
            return None, False

    def prepare_price_offer_list(self, data, origin_location_code, destination_location_code, departure_date, adults):
        price_offer_list = []
        distance_matrix_json_result = self.distanceMatrixAPI.get_travel_durations(origin_location_code, destination_location_code, departure_date)
        if (distance_matrix_json_result['status'] != 'MAX_ROUTE_LENGTH_EXCEEDED'):
            for price_offer in data:
                is_price_offer_valid = self.is_price_offer_valid(price_offer)
                if is_price_offer_valid:
                    dict = {}
                    dict['origin_location_code'] = origin_location_code
                    dict['destination_location_code'] = destination_location_code
                    dict['departure_date'] = departure_date
                    dict['adults'] = adults
                    dict['currency'] = price_offer['price']['currency']
                    dict['total'] = price_offer['price']['total']
                    dict['base'] = price_offer['price']['base']
                    dict['validatingAirlineCodes'] = price_offer['validatingAirlineCodes'][0]
                    dict['distance'] = distance_matrix_json_result['rows'][0]['elements'][0]['distance']['value']
                    dict['duration'] = distance_matrix_json_result['rows'][0]['elements'][0]['duration']['value']
                    if (self.is_price_offer_unique(price_offer_list, dict['currency'], dict['total'], dict['base'], dict['validatingAirlineCodes'])):
                        price_offer_list.append(dict)
        return price_offer_list

    def is_price_offer_valid(self, price_offer):
        if ((price_offer['price']['currency'] is not None) and
                (price_offer['price']['total'] is not None) and
                (price_offer['price']['base'] is not None) and
                (price_offer['validatingAirlineCodes'] is not None)):
            return True
        else:
            return False

    def is_price_offer_unique(self, price_offer_list, currency, total, base, validatingAirlineCodes):
        for price_offer in price_offer_list:
            if(price_offer['currency'] == currency and
                    price_offer['total'] == total and
                    price_offer['base'] == base and
                    price_offer['validatingAirlineCodes'] == validatingAirlineCodes):
                return False
        return True

    def get_price_offer_df(self, data, origin_location_code, destination_location_code, departure_date, adults):
        price_offer_list = self.prepare_price_offer_list(data, origin_location_code, destination_location_code, departure_date, adults)
        df = pd.DataFrame()
        if len(price_offer_list) > 0:
            has_price_offers = True
            for price_offer in price_offer_list:
                df_dictionary = pd.DataFrame([price_offer])
                df = pd.concat([df, df_dictionary], ignore_index=True)
            return df, has_price_offers
        else:
            has_price_offers = False
            return df, has_price_offers

    def store_df_to_csv(self, df):
        data_file_name = f"flight_offers_{self.time_str}.csv"
        # data_file_name = f"flight_offers.csv"
        data_file_path = os.path.join(self.data_folder_path, data_file_name)
        if (self.first_csv_write):
            df.to_csv(data_file_path, index=False)
            self.first_csv_write = False
        else:
            df.to_csv(data_file_path, mode='a', header=False, index=False)

if __name__ == "__main__":
    prepare_dataset = PrepareDataset()
    prepare_dataset.get_price_offers_for_date_range('2022-11-01', '2022-11-30', 1)