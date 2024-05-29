# Recall feature

```python
sample = get_user('recall')
df = get_hdf(dtype='all', if_filter_label=True, if_drop_duplicates=True)
```

两个主要的数据集



```python
def add_prefix(df, exclude_columns, prefix):
    if isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]
        
    column_names = [col for col in df.columns if col not in exclude_columns]
    df.rename(columns = dict(zip(column_names, [prefix + name for name in column_names])), inplace=True)
    return df

def group_func(df, group_func_dic, group_key):
    if isinstance(group_func_dic, str):
        group_func_dic = [group_func_dic]
        
    features = df.groupby(group_key).agg(group_func_dic)
    features.columns = [e[0] + "_" + e[1].upper() for e in features.columns.tolist()]
    features.reset_index(inplace=True)
    return features

def filter_sample(df, key=None):
    if key is None:
        df = df.merge(sample[['buyer_admin_id']].drop_duplicates(), on=['buyer_admin_id'], how='inner')
    else:
        df = df.merge(sample[['buyer_admin_id', key]].drop_duplicates(), on=['buyer_admin_id', key], how='inner')
    return df
```

三个常用函数

add_prefix：除了**exclude_columns**， 其他列都更名为**prefix+name**

group_func：对数据集进行分组聚合操作， 聚合函数group_func_dic， 分组key :**group_key**

filter_sample: 按不同用户和key为关键字进行去重



##  get_user_store_dedup_feature函数

以**'buyer_admin_id', 'store_id'**为关键字分组，获取聚合特征。



# get_user_cate_dedup_feature函数

以**'buyer_admin_id', 'cate_id'**为关键字分组，获取聚合特征。



## get_item_feature函数

以**'intem_id'**为关键字分组，获取聚合特征。





## get_store_feature函数

以**'store_id'**为关键字分组，获取聚合特征。



## get_user_second_diff_feature函数