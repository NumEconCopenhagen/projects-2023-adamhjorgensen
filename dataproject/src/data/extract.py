import numpy as np
import pandas as pd
import PyDST as Dst # install with `pip install git+https://github.com/Kristianuruplarsen/pydst.git`
import os
from datetime import date
from pathlib import Path

def to_raw(data: pd.DataFrame or pd.Series, filename: str):
    """ Exports data to csv file in <root>/data/raw folder

    Args:
        data (pd.DataFrame or pd.Series): Data in pandas format
        filename (str): Name of csv file
    """
    
    # a. Create folder in <root>/data/raw with today's date
    today = str(date.today())
    path = Path(fr'data/raw/downloaded_at={today}')
    os.makedirs(path, exist_ok=True)
    
    # b. Export file to csv
    data.to_csv(f"{path}/{filename}.csv")
    
    
    
def get_capital():
    """ Extracts data on capital stocks from DST and exports it to csv
    """
    
    #a. Get data from DST
    data = Dst.utils.to_dataframe(Dst.get_data(
        table_id = 'NABK69', 
        variables={
            'BEHOLD':['LEN'], # Faste aktiver, nettobeholdning ultimo året
            'AKTIV': ['N111', # Boliger
                    'N1121', # Andre_bygninger
                    'N1122_3', # Anlæg
                    'N1131', # Transport
                    'N11P', # ICT udstyr, andre maskiner og inventar samt våbensystemer
                    'N115', # Stambesætninger mv.
                    'N117'], # Intellektuelle rettigheder
            'BRANCHE':['*'],
            'PRISENHED':['V', # Løbende priser
                         'LAN'], # 2010-priser, kædede værdier
            'TID': ['*']
            }
        ))
    
    #b. Export to csv
    filename = 'NABK69_K'
    to_raw(data, filename)
    
    
def get_investment():
    """ Extracts data on capital investment from DST and exports it to csv
    """
    
    #a. Get data from DST
    data = Dst.utils.to_dataframe(Dst.get_data(
        table_id = 'NABK69', 
        variables={
            'BEHOLD':['P51G'], # Faste aktiver, nettobeholdning ultimo året
            'AKTIV': ['N111', # Boliger
                    'N1121', # Andre_bygninger
                    'N1122_3', # Anlæg
                    'N1131', # Transport
                    'N11P', # ICT udstyr, andre maskiner og inventar samt våbensystemer
                    'N115', # Stambesætninger mv.
                    'N117'], # Intellektuelle rettigheder
            'BRANCHE':['*'],
            'PRISENHED':['V', # Løbende priser
                         'LAN'], # 2010-priser, kædede værdier
            'TID': ['*']
            }
        ))
    
    #b. Export to csv
    filename = 'NABK69_I'
    to_raw(data, filename)
   
def get_production():
    """ Extracts data on production from DST and exports it to csv
    """
    
    #a. Get data from DST
    data = Dst.utils.to_dataframe(Dst.get_data(
        table_id = 'NIO4F', 
        variables={
            'TILGANG1':['P1'], #Production
            'ANVENDELSE':['*'],
            'PRISENHED':['V', # Løbende priser
                         'Y'], # Sidste års priser
            'TID': ['*']
            }
        ))
    
    #b. Export to csv
    filename = 'NIO4F_Y'
    to_raw(data, filename) 
    
def get_materials():
    """ Extracts data on materials from DST and exports it to csv
    """
    
    #a. Get data from DST
    data = Dst.utils.to_dataframe(Dst.get_data(
        table_id = 'NIO4F', 
        variables={
            'TILGANG1':['TP2'], #Forbrug i produktionen
            'ANVENDELSE':['*'],
            'PRISENHED':['V', # Løbende priser
                         'Y'], # Sidste års priser
            'TID': ['*']
            }
        ))
    
    #b. Export to csv
    filename = 'NIO4F_R'
    to_raw(data, filename)
    


def get_employment():
    """ Extracts data on employment from DST and exports it to csv
    """
    
    #a. Get data from DST
    data = Dst.utils.to_dataframe(Dst.get_data(
        table_id = 'NIO3F', 
        variables={
            'TILGANG1':['D1'], #Wage income
            'ANVENDELSE':['*'],
            'PRISENHED':['V', # Løbende priser
                         'Y'], # Sidste års priser
            'TID': ['*']
            }
        ))
    
    #b. Export to csv
    filename = 'NIO3F_L'
    to_raw(data, filename)
    
def get_taxes():
    """ Extracts data on production taxes from DST and exports it to csv
    """
    
    #a. Get data from DST
    data = Dst.utils.to_dataframe(Dst.get_data(
        table_id = 'NIO3F', 
        variables={
            'TILGANG1':['D29X39'], #production taxes excl. VAT
            'ANVENDELSE':['*'],
            'PRISENHED':['V', # Løbende priser
                         'Y'], # Sidste års priser
            'TID': ['*']
            }
        ))
    
    #b. Export to csv
    filename = 'NIO3F_T'
    to_raw(data, filename)
    
def get_interestrate():
    """ Extracts data on interest rate from DST and exports it to csv
    """
    
    # a. Get data from DST
    data = Dst.utils.to_dataframe(Dst.get_data(
        table_id = 'MPK100',
        variables={
            'LAND':['420'],
            'TID':['*']
        }
    ))
    
    #b. Export to csv
    filename = 'MPK100'
    to_raw(data, filename)
            
    
def extract():
    get_capital()
    get_investment()
    get_production()
    get_materials()
    get_employment()
    get_taxes()
    get_interestrate()
    















