import pandas as pd
import requests

master = pd.DataFrame()

for idx in range(0, 10001, 50):
    print(idx, end=' ')
    
    params = (
        ('order_by', 'sale_price'),
        ('order_direction', 'desc'),
        ('offset', idx),
        ('limit', '50'),
        ('collection', 'cryptopunks'),
    )

    original_response = requests.get('https://api.opensea.io/api/v1/assets', params=params)
    response = original_response.json()
    
    if "assets" not in response:
        break
    #NB. Original query string below. It seems impossible to parse and
    #reproduce query strings 100% accurately so the one below is given
    #in case the reproduced version is not "correct".
    # response = requests.get('https://api.opensea.io/api/v1/assets?order_by=sale_price&order_direction=desc&offset=0&limit=50&collection=cryptopunks')

    df_nested_list = pd.json_normalize(response, record_path=['assets'])
    if master.empty:
        master = df_nested_list
    else:
        master = master.append(df_nested_list, ignore_index=True)

print(master.info())
master.to_json("set_"+str(idx)+".json", orient="records")