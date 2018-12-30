import pandas as pd

proj = pd.read_csv('/Users/jif/Donors_choose/Projects.csv', encoding='utf-8', iterator=True, chunksize=10000)

df_empty = pd.DataFrame()
for chunk in proj:
    chunk_edit = chunk.fillna('')
    df_empty = pd.concat([df_empty,chunk_edit])

# df_empty = df_empty.dropna(subset=['Project ID','School ID','Teacher ID','Teacher Project Posted Sequence','Project Type','Project Title','Project Essay','Project Short Description','Project Need Statement','Project Subject Category Tree','Project Subject Subcategory Tree','Project Grade Level Category','Project Resource Category','Project Cost','Project Posted Date','Project Expiration Date','Project Current Status','Project Fully Funded Date'])
df_empty.to_csv(r'/Users/jif/Desktop/Projects_nans_replaced_v2.csv', encoding='utf-8')
