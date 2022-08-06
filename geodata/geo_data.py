import pandas as pd
import os


class GetIATACodeGeodata:
    def get_iata_code_geodata(self, df, iata_code):
        geodata_dict = {}
        geodata_dict['iata_code'] = iata_code
        geodata_dict['Longitude'] = df[df['IATA'] == iata_code]['Longitude'].values[0]
        geodata_dict['Latitude'] = df[df['IATA'] == iata_code]['Latitude'].values[0]
        return geodata_dict

    def get_iata_codes_geodata(self, iata_location_codes_list):
        airports_geodata_csv_file_path = os.path.join(os.path.dirname(__file__), 'data', 'Airports-Only.csv')
        df = pd.read_csv(airports_geodata_csv_file_path, encoding='utf8')
        geodata_dict_list = []
        for iata_code in iata_location_codes_list:
            geodata_dict = self.get_iata_code_geodata(df, iata_code)
            geodata_dict_list.append(geodata_dict)
        return geodata_dict_list

if __name__ == "__main__":
    iata_location_codes_list = ['TLV', 'IST', 'AMS', 'FRA', 'DME', 'SVO', 'SAW', 'CDG', 'MAD', 'LED', 'LHR', 'ATH',
                                'ORY', 'BCN', 'VKO', 'PMI', 'MUC',
                                'FCO', 'LIS', 'OSL', 'AER', 'VIE', 'ZRH', 'LPA', 'CPH', 'MXP', 'KBP', 'BRU', 'AYT',
                                'ARN', 'NCE', 'OTP',
                                'WAW', 'AGP', 'ESB', 'TFN', 'SIP', 'LYS', 'ADB', 'BGO', 'GVA', 'DUB', 'HEL', 'CTA',
                                'KRR', 'MRS', 'IBZ', 'HAM',
                                'LIN', 'OPO']
    getIATACodeGeodata = GetIATACodeGeodata()
    geodata_dict_list = getIATACodeGeodata.get_iata_codes_geodata(iata_location_codes_list)
    print(geodata_dict_list)