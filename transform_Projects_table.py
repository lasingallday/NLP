import pandas as pd

proj = pd.read_csv('/Users/jif/Donors_choose/Projects.csv', encoding='utf-8', iterator=True, chunksize=10000)

df_empty = pd.DataFrame()
for chunk in proj:
    chunk_edit = chunk.fillna('')
    df_empty = pd.concat([df_empty,chunk_edit])

df_empty.to_csv(r'/Users/jif/Desktop/Projects_nans_replaced.csv', encoding='utf-8')
