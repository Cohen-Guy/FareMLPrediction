# -*- coding: utf-8 -*-

import os
import urllib
from dotenv import load_dotenv
from geodata.geo_data import GetIATACodeGeodata
import datetime
import json
import traceback as tb

class DistanceMatrixAPI(object):

    def __init__(self, iata_location_codes_list):
        secrets_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'secrets')
        env_file_path = os.path.abspath(os.path.join(secrets_folder_path, '.env'))
        load_dotenv(dotenv_path=env_file_path)
        self.api_key = os.getenv("API_KEY")
        getIATACodeGeodata = GetIATACodeGeodata()
        self.geodata_dict_list = getIATACodeGeodata.get_iata_codes_geodata(iata_location_codes_list)

    def build_query_url(self, origin_iata_code, destination_iata_code, departure_time):
        base_url = 'https://maps.googleapis.com/maps/api/distancematrix/json?'
        origin_geodata_dict = next((geodata_dict for geodata_dict in self.geodata_dict_list if geodata_dict['iata_code'] == origin_iata_code), None)
        destination_geodata_dict = next((geodata_dict for geodata_dict in self.geodata_dict_list if geodata_dict['iata_code'] == destination_iata_code), None)
        build_origin_url_str = self.build_location_url_str(origin_geodata_dict)
        build_destination_url_str = self.build_location_url_str(destination_geodata_dict)
        origins_url_str = 'origins=' + build_origin_url_str
        destinations_url_str = '&destinations=' + build_destination_url_str
        departure_time_str = f"&departure_time={round(datetime.datetime.strptime(departure_time, '%Y-%m-%d').timestamp())}"
        query_url = f"{base_url}{origins_url_str}{destinations_url_str}{departure_time_str}&key={self.api_key}"
        return query_url

    def build_location_url_str(self, geodata_dict):
        iata_geodata_str = f"{geodata_dict['Latitude']}%2C{geodata_dict['Longitude']}"
        return iata_geodata_str

    def make_query(self, query_url):
        number_of_tries = 10
        try:
            for current_try in range(number_of_tries):
                try:
                    query_result = urllib.request.urlopen(query_url).read()
                    return query_result
                except:
                    print(f"query:\n {query_url}\n resulted error, try number: {current_try + 1}")
        except:
            print(f"query:\n {query_url}\n resulted error, 10 retries has been made with no success")

    def get_travel_durations(self, origin_iata_code, destination_iata_code, departure_time):
        query_url = self.build_query_url(origin_iata_code, destination_iata_code, departure_time)
        query_result = self.make_query(query_url)
        distance_matrix_json_result = json.loads(query_result)
        return distance_matrix_json_result

if __name__ == "__main__":
    iata_location_codes_list = ['TLV', 'IST', 'AMS', 'FRA', 'DME', 'SVO', 'SAW', 'CDG', 'MAD', 'LED', 'LHR', 'ATH',
                                    'ORY', 'BCN', 'VKO', 'PMI', 'MUC',
                                    'FCO', 'LIS', 'OSL', 'AER', 'VIE', 'ZRH', 'LPA', 'CPH', 'MXP', 'KBP', 'BRU', 'AYT',
                                    'ARN', 'NCE', 'BER', 'OTP',
                                    'WAW', 'AGP', 'ESB', 'TFN', 'SIP', 'LYS', 'ADB', 'BGO', 'GVA', 'DUB', 'HEL', 'CTA',
                                    'KRR', 'MRS', 'IBZ', 'HAM',
                                    'LIN', 'OPO']
    distanceMatrixAPI = DistanceMatrixAPI(iata_location_codes_list)
    departure_time = '2022-11-11'
    distanceMatrixAPI.get_travel_durations('TLV', 'BER', departure_time)