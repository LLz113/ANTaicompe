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

```python
def get_user_second_diff_feature(df):
    """
    用户时间间隔统计特征：
    聚合层级：cate_id, store_id, item_id
    
    1. 商品与下个商品间隔
    2. 商品与下个同样商品间隔
    3. 商品与下个同品类商品间隔
    4. 商品与下个同店铺商品间隔
    
    备注：线下：0.8843→0.8852  提升：0.009
    ---------------------------------------------
    
    """
    #
    df = df[['buyer_admin_id', 'store_id', 'cate_id', 'item_id', 'second']].drop_duplicates()  # 去除相同时间的同样商品
    df['second_diff'] = df['second'] - df.groupby(['buyer_admin_id'])['second'].shift(1)  # 商品与下个商品间隔
    df['cate_id_second_diff'] = df['second'] - df.groupby(['buyer_admin_id', 'cate_id'])['second'].shift(1)  # 与下个同品类商品间隔
    df['store_id_second_diff'] = df['second'] - df.groupby(['buyer_admin_id', 'store_id'])['second'].shift(1)  #与下个同店铺商品间隔
    #上述三个操作是统计用户购买时间间隔、不同用户相同品类的购买间隔、用户相同店铺的购买间隔
     
    feature_type = {
        'second_diff' : ['max', 'min', 'mean', 'std', np.ptp],
        'cate_id_second_diff':['max', 'min', 'mean', 'std', np.ptp],
        'store_id_second_diff':['max', 'min', 'mean', 'std', np.ptp],
    }
    
    df = filter_sample(df)
    feature = group_func(df, feature_type, group_key=['buyer_admin_id'])
    feature = add_prefix(feature, ['buyer_admin_id'], 'user_second_diff_')
    feature.to_hdf('../feature/recall/user_second_diff_feature', 'user')
    
    for level in ['cate_id', 'store_id']:
        feature = group_func(df, feature_type, group_key=['buyer_admin_id', level])
        feature = add_prefix(feature, ['buyer_admin_id', level], 'user_' + level + '_second_diff_')
        feature.to_hdf('../feature/recall/user_second_diff_feature', level)
    print('>>> user_second_diff_feature success')
    return feature
```





## get_item_conv_feature函数

```python
def get_item_conv_feature(df):
    """
    商品转化率特征
    
    """
    
    item_pv = df.drop_duplicates(subset=['buyer_admin_id', 'item_id', 'second']).groupby(['item_id']).size().to_frame('pv').reset_index()   #现对用户、商品、时间去重， 最后得到商品购买次数
    item_uv = df.groupby(['item_id'])['buyer_admin_id'].nunique().to_frame('uv').reset_index()  #购买商品的用户人数
    item_buy_uv = df[df['buy_flag']==1].groupby(['item_id'])['buyer_admin_id'].nunique().to_frame('buy_uv').reset_index()    #已购买的商品的用户个数

    dup = df[df['buy_flag']==1][df.duplicated(subset=['buyer_admin_id', 'item_id', 'second'], keep=False)]  #已购且重复购买的商品
    multi_buy_uv = dup.groupby(['item_id'])['buyer_admin_id'].nunique().to_frame('multi_buy_uv').reset_index()  #某商品被重复购买的次数

    view_time = df.groupby(['buyer_admin_id', 'item_id']).size().to_frame('user_view_time').reset_index()
    view_one_time = view_time.groupby(['item_id'])['user_view_time'].value_counts(normalize=True).to_frame('view_onetime_prop').reset_index()  #分组内value_counts，且以占比的形式展现
    view_one_time = view_one_time[view_one_time['user_view_time']==1].drop(['user_view_time'],1 )    
    
    last = df.drop_duplicates(subset=['buyer_admin_id'], keep='first')    #删除买家重复值， 保留第一个
    last_cnt = last.groupby(['item_id']).size().to_frame('last_buy').reset_index()
    
    last_via_day = df.drop_duplicates(subset=['buyer_admin_id', 'day'], keep='first')\
        .drop_duplicates(subset=['buyer_admin_id', 'item_id'], keep='first')   #每天每样商品
    last_via_day_cnt = last_via_day.groupby(['item_id']).size().to_frame('last_buy_day').reset_index()   #每天购买不同商品
    
    
    feature = item_pv.merge(item_uv, on=['item_id'], how='left')\
            .merge(item_buy_uv, on=['item_id'], how='left')\
            .merge(multi_buy_uv, on=['item_id'], how='left')\
            .merge(view_one_time, on=['item_id'], how='left')\
            .merge(last_cnt, on=['item_id'], how='left')\
            .merge(last_via_day_cnt, on=['item_id'], how='left').fillna(0)

    feature['pv/uv'] = feature['pv'] / feature['uv']
    feature['buy_uv/pv'] = feature['buy_uv'] / feature['uv']
    feature['multi_buy_uv/buy_uv'] = feature['multi_buy_uv'] / feature['buy_uv']
    feature['multi_buy_uv/uv'] = feature['multi_buy_uv'] / feature['uv']
    feature['last_buy/uv'] = feature['last_buy'] / feature['uv']
    feature['last_buy/buy_uv'] = feature['last_buy'] / feature['buy_uv']
    feature = feature.fillna(0)
    
    feature = add_prefix(feature, ['item_id'], 'item_conv_')
    feature.to_hdf('../feature/recall/item_conv_feature', 'all')
    print('>>> item_conv_feature success')
    return feature
```





```python
def get_user_rank_feature(feature, feature_name, key, group_key=[], ascending=False):
    if ascending:
        name = 'asc'
    else:
        name = 'desc'
    columns = []
    for col in feature.columns:
        if col not in ['buyer_admin_id', 'item_id', 'cate_id', 'store_id']:
            column_name = col + '_rank_' + name
            feature[column_name] = feature.groupby(['buyer_admin_id'] + group_key)[col].rank(ascending=ascending, method='dense')   #分用户来进行组内Rank，dense 类似与min方法，以Rank最小值来作为Rank值
            columns.append(column_name)
            
    if len(group_key)>0:
        feature = feature[['buyer_admin_id', 'item_id'] + group_key + columns]
        name = group_key[0] + '_' + name
    else:
        feature = feature[['buyer_admin_id']+ key + columns]
        
    feature.to_hdf('../feature/recall/' + feature_name, name)
    print('>>> user_rank_feature feature success')
    return feature
```



