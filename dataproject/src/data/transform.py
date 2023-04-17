import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

def read_csv(filename: str):
    today = str(date.today())
    path = Path(fr'data/raw/downloaded_at={today}')
    data = pd.read_csv(f"{path}/{filename}.csv")
    return data

def to_processed(data: pd.DataFrame or pd.Series, filename: str):
    """ Exports data to csv file in <root>/data/processed folder

    Args:
        data (pd.DataFrame or pd.Series): Data in pandas format
        filename (str): Name of csv file
    """

    path = Path(fr'data/processed')
    data.to_csv(f"{path}/{filename}.csv")

def set_years(df: pd.DataFrame):
    # a. Set time
    col = 'TID'
    start_year = 1993
    end_year  = 2018
    mask = (df[col] >= start_year) & (df[col] <= end_year)
    
    # b. Slice on time
    df = df.loc[mask]
    return df

def set_sectors(df: pd.DataFrame):
    
    # a. Set sectors
    col = 'BRANCHE'
    mask = (df[col] != 'Total') & (df[col] != 'Of which: General government')
    
    # b. Slice on sectors
    df = df.loc[mask].copy()
    
    # c. Keep sector code only (5 first characters)
    df[col] = df[col].transform(lambda x: x[0:5])
    
    return df

def aggregate(data: pd.DataFrame, mapping: pd.Series, idx: str):
    """Aggregate data based on mapping

    Args:
        data (pd.DataFrame): Main data to be aggregated
        mapping (pd.Series): Mapping 1-to-1 or many-to-1, where many must be in index
        idx (str): Name of index variable in data to be aggregated over

    Returns:
        df: aggregated data
    """
    # a. apply mapping
    df = data.join(mapping, on=idx)

    # b. Create group to aggregate over
    new_idx = mapping.columns[0]
    group = [new_idx if i == idx else i for i in df.index.names]

    # c. Aggregate
    df = df.groupby(group).sum()
    
    # d. Set column name back to original
    df.index.set_names(idx, level=new_idx, inplace=True)
    
    return df

def load_sectormap():
    # a. Load
    sectormap = pd.read_excel('data/external/sectormapping.xlsx', 
                                  usecols=['IO69','MAKRO'], 
                                  dtype='str')
    
    # b. clean
    sectormap = sectormap.drop_duplicates().dropna().set_index('IO69')
    
    return sectormap

def make_aktivmap():
    aktiv = ['Buildings other than dwellings',
    'Cultivated biological resources',
    'Dwellings',
    'ICT equipment, other machinery and equipment and weapon systems',
    'Intellectual property products',
    'Other structures and land improvements',
    'Transport equipment']

    aggr_aktiv =['B', 'M', 'B', 'M', 'M', 'B', 'M']

    aktivmap = pd.DataFrame({'AKTIV':aktiv, 'aggr':aggr_aktiv}).set_index('AKTIV')
    
    return aktivmap

def transform_capital():

    #a. Load
    filename = 'NABK69_K'
    data = read_csv(filename)

    #b. Clean data
    data = set_years(data)
    data = set_sectors(data)

    # c. Reshape
    data = data.pivot_table(values ='INDHOLD', 
                            index  = ['BRANCHE','TID','AKTIV'],
                            columns= 'PRISENHED')
    
    # d. Aggregate
    # i. Aggregate over sectors
    sectormap = load_sectormap()
    data = aggregate(data, sectormap, 'BRANCHE')
    
    # ii. Aggregate over aktiver
    # o. Load and set up aktivmapping
    aktivmap = make_aktivmap()
    data = aggregate(data, aktivmap, 'AKTIV')
    
    # e. Export
    to_processed(data, filename)

def transform_investment():

    #a. Load
    filename = 'NABK69_I'
    data = read_csv(filename)

    #b. Clean data
    data = set_years(data)
    data = set_sectors(data)

    # c. Reshape
    data = data.pivot_table(values ='INDHOLD', 
                            index  = ['BRANCHE','TID','AKTIV'],
                            columns= 'PRISENHED')
    
    # d. Aggregate
    # i. Aggregate over sectors
    sectormap = load_sectormap()
    data = aggregate(data, sectormap, 'BRANCHE')
    
    # ii. Aggregate over aktiver
    # o. Load and set up aktivmapping
    aktivmap = make_aktivmap()
    data = aggregate(data, aktivmap, 'AKTIV')
    
    # e. Export
    to_processed(data, filename)
    
def transform():
    transform_capital()
    transform_investment()

