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
    col = 'year'
    start_year = 1993
    end_year  = 2018
    mask = (df[col] >= start_year) & (df[col] <= end_year)
    
    # b. Slice on time
    df = df.loc[mask]
    return df

def set_sectors(df: pd.DataFrame, col: str = 'sector'):
    
    # a. Keep sector code only (5 first characters)
    df[col] = df[col].transform(lambda x: x[0:5])
    
    # b. Keep only numeric values which are sectors)
    mask = df[col].str.isnumeric()
    df = df.loc[mask]
    
    return df

def rename_colums(df: pd.DataFrame):
    
    column_names_map = {
        'Unnamed: 0': 'index',
        'BEHOLD': 'type',
        'AKTIV': 'asset',
        'BRANCHE': 'sector',
        'PRISENHED': 'unit',
        'TID': 'year',
        'INDHOLD': 'value',
        'TILGANG1': 'type',
        'ANVENDELSE': 'sector',
    }
    
    df = df.copy()
    for col in df.columns:
            df.rename(columns={col: column_names_map[col]}, inplace=True)
    
    return df

def rename_prices(df: pd.Series):


    price_name_map = {
        'Current prices': 'V',
        'Constant prices of the previous year': 'D',
        '2010-prices, chained values': 'Q',
    }
    
    df = df.copy()
    for name in price_name_map:
        df.loc[df==name] = price_name_map[name]
        
    return df

def VQ_to_VD(V: pd.Series, Q: pd.Series):
    """ Converts data from values and quantities to values and values in previous years prices"""

    # a. Calculate prices
    # Calculate current year's prices
    P = V / Q
    #Calculate index exclusive years
    group = [i for i in P.index.names if i != 'year']
    # Lag prices
    P_lag = P.groupby(group).shift(1)
    
    # b. Calculate values in previous year's prices
    D = pd.Series(np.nan, index=V.index, name='D')
    D[:] = Q * P_lag
    
    return V, D

def chain_index(V: pd.Series, D: str, base_year: int = 2010):
    
    # Allocate
    P = pd.Series(np.nan, index=V.index, name='P')
    Q = pd.Series(np.nan, index=V.index, name='Q')
    
    #Year settings
    year_index = V.index.get_level_values('year')
    years = year_index.unique().sort_values()
    start_year = year_index==years[0]
    group = [i for i in P.index.names if i != 'year']
    
    # Initial values
    P.loc[start_year] = 1
    Q.loc[start_year] = V.loc[start_year]
    
    #Chain index
    for year in years[1:]: 
        t = year_index==year
        # t_lag = year_index==(year-1)
        P.loc[t] = (V.loc[t] / D.loc[t]) * P.groupby(group).shift(1).loc[t]
        Q.loc[t] = V.loc[t] / P.loc[t]
    
    #Reindex base year 
    baseprice = P.loc[year_index==base_year].droplevel('year')
    P = P.groupby('year').transform(lambda x: x / baseprice)
    Q = Q.groupby('year').transform(lambda x: x * baseprice)
    
    return P, Q
    

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
    # Set names
    aktiv = ['Buildings other than dwellings',
    'Cultivated biological resources',
    'Dwellings',
    'ICT equipment, other machinery and equipment and weapon systems',
    'Intellectual property products',
    'Other structures and land improvements',
    'Transport equipment']

    # Set aggregate type
    aggr_aktiv =['B', 'M', 'B', 'M', 'M', 'B', 'M']

    # Make mapping from disaggregate to aggregate assets
    aktivmap = pd.DataFrame({'AKTIV':aktiv, 'aggr':aggr_aktiv}).set_index('AKTIV')
    
    return aktivmap

def transform_capital():

    #a. Load
    filename = 'NABK69_K'
    data = read_csv(filename)

    #b. Clean data
    data = rename_colums(data)
    data['unit'] = rename_prices(data['unit'])
    data = set_years(data)
    data = set_sectors(data)

    # c. Reshape
    data = data.pivot_table(values ='value', 
                            index  = ['sector','year','asset'],
                            columns= 'unit')
    
    # d. Aggregate
    # i. Aggregate over sectors
    sectormap = load_sectormap()
    data = aggregate(data, sectormap, 'sector')
    
    # ii. Aggregate over aktiver
    # o. Load and set up aktivmapping
    aktivmap = make_aktivmap()
    data = aggregate(data, aktivmap, 'asset')
    
    # iii. Apply chain index
    V, D = VQ_to_VD(data['V'], data['Q'])
    P, Q = chain_index(V, D, base_year=2010)
    data = pd.concat([V, D, P, Q], axis=1)
    
    # e. Export
    to_processed(data, filename)

def transform_investment():

    #a. Load
    filename = 'NABK69_I'
    data = read_csv(filename)

    #b. Clean data
    data = rename_colums(data)
    data['unit'] = rename_prices(data['unit'])
    data = set_years(data)
    data = set_sectors(data)

    # c. Reshape
    data = data.pivot_table(values ='value', 
                            index  = ['sector','year','asset'],
                            columns= 'unit')
    
    # d. Aggregate
    # i. Aggregate over sectors
    sectormap = load_sectormap()
    data = aggregate(data, sectormap, 'sector')
    
    # ii. Aggregate over aktiver
    # Load and set up aktivmapping
    aktivmap = make_aktivmap()
    data = aggregate(data, aktivmap, 'asset')
    
    # iii. Apply chain index
    V, D = VQ_to_VD(data['V'], data['Q'])
    P, Q = chain_index(V, D, base_year=2010)
    data = pd.concat([V, D, P, Q], axis=1)
    
    # e. Export
    to_processed(data, filename)
    
def transform_production():

    #a. Load
    filename = 'NIO4F_Y'
    data = read_csv(filename)

    #b. Clean data
    data = rename_colums(data)
    data['unit'] = rename_prices(data['unit'])
    data = set_years(data)
    data = set_sectors(data, col='sector')

    # c. Reshape
    data = data.pivot_table(values ='value', 
                            index  = ['sector','year'],
                            columns= 'unit')
    
    # d. Aggregate
    # Aggregate over sectors
    sectormap = load_sectormap()
    data = aggregate(data, sectormap, 'sector')
    
    # e. Export
    to_processed(data, filename)
    
def transform_materials():

    #a. Load
    filename = 'NIO4F_R'
    data = read_csv(filename)

    #b. Clean data
    data = rename_colums(data)
    data['unit'] = rename_prices(data['unit'])
    data = set_years(data)
    data = set_sectors(data, col='sector')

    # c. Reshape
    data = data.pivot_table(values ='value', 
                            index  = ['sector','year'],
                            columns= 'unit')
    
    # d. Aggregate
    # Aggregate over sectors
    sectormap = load_sectormap()
    data = aggregate(data, sectormap, 'sector')
    
    # e. Export
    to_processed(data, filename)
    
def transform_employment():

    #a. Load
    filename = 'NIO3F_L'
    data = read_csv(filename)

    #b. Clean data
    data = rename_colums(data)
    data['unit'] = rename_prices(data['unit'])
    data = set_years(data)
    data = set_sectors(data, col='sector')

    # c. Reshape
    data = data.pivot_table(values ='value', 
                            index  = ['sector','year'],
                            columns= 'unit')
    
    # d. Aggregate
    # Aggregate over sectors
    sectormap = load_sectormap()
    data = aggregate(data, sectormap, 'sector')
    
    # e. Export
    to_processed(data, filename)
    
def transform_taxes():

    #a. Load
    filename = 'NIO3F_T'
    data = read_csv(filename)

    #b. Clean data
    data = rename_colums(data)
    data['unit'] = rename_prices(data['unit'])
    data = set_years(data)
    data = set_sectors(data, col='sector')

    # c. Reshape
    data = data.pivot_table(values ='value', 
                            index  = ['sector','year'],
                            columns= 'unit')
    
    # d. Aggregate
    # Aggregate over sectors
    sectormap = load_sectormap()
    data = aggregate(data, sectormap, 'sector')
    
    # e. Export
    to_processed(data, filename)

    
def transform():
    transform_capital()
    transform_investment()
    transform_production()
    transform_materials()
    transform_employment()
    transform_taxes()
    
    # #Create chain index (using Values in current and previous year's prices)
    # for year in years:
    #     #Initial year
    #     if year==years[0]:
    #         P=[1]
    #         #V where time is yr
    #         V_agg=V_input.loc[V_input.index.get_level_values('TID')==year]
    #         Q=[V_agg]
    #     else:
    #         #Price and quantity index
    #         V=V_input.loc[V_input.index.get_level_values('TID')==year]
    #         D=D_input.loc[D_input.index.get_level_values('TID')==year]
    #         P.append((V/D)*P[-1])
    #         Q.append(V/D[-1])
            
    # P = pd.Series(np.concatenate(P), index=V_input.index)

