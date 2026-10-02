import pandas as pd
from pathlib import Path

df = pd.read_csv(Path("data/raw/felixstowe-waverider.csv"), sep=',')
print(df.columns)
#print(df[['MEETPUNT_IDENTIFICATIE','GROOTHEID_OMSCHRIJVING', 'EENHEID_CODE', 'WAARNEMINGDATUM', 'WAARNEMINGTIJD', 'NUMERIEKEWAARDE', 'LAT', 'LON', 'EPSG']])
#print(df['GROOTHEID_OMSCHRIJVING'].unique())
print(df)