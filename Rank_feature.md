# Rank_feature

本文件中存在大量与recall_feature 中功能相似的函数，因此不再解读

user = df[df[['buyer_admin_id', 'item_id', 'log_time']].duplicated(keep=False)][['buyer_admin_id','item_id']].drop_duplicates()
    dup = df.merge(user, how='inner', on=['buyer_admin_id', 'item_id'])
    

## get_user_item_dupli_feature 函数



```python
def get_user_item_dupli_feature(df):
    """
    df = get_hdf('buy', if_filter_label=True)
    import modin.pandas
    get_user_item_dupli_feature(df.copy())
    """
    user = df[df[['buyer_admin_id', 'item_id', 'log_time']].duplicated(keep=False)][['buyer_admin_id','item_id']].drop_duplicates()  # 首先找出存在[用户、商品、时间]的重复记录，再删除重复[用户、商品]的记录 -- 目的找出有重复时间的用户记录
    dup = df.merge(user, how='inner', on=['buyer_admin_id', 'item_id'])  #内连接，找出表中所有重复数据
    
    feature_type = {
        'dense_rank' : ['max', 'min', np.ptp],
        'first_second_diff':['max', 'min', 'mean'],    #first_second_diff 购物时间与第一次购物时间差
        'last_second_diff':['max', 'min', 'mean']		#last_second_diff 购物时间与最后一次购物时间差
    }    
    dup_feature = group_func(dup, feature_type, group_key=['buyer_admin_id', 'item_id'])
    dup_feature = add_prefix(dup_feature, ['buyer_admin_id', 'item_id'], 'user_item_dup_')
    
    dup_cnt = dup.groupby(['buyer_admin_id', 'item_id', 'log_time']).size().to_frame('dup_cnt').reset_index()

    feature_type = {
        'dup_cnt':['first', 'max', 'min', 'last', 'nunique'],
    }
    
    feature = group_func(dup_cnt, feature_type, group_key=['buyer_admin_id', 'item_id'])
    feature = add_prefix(feature, ['buyer_admin_id', 'item_id'], 'user-item_')
    feature['user-item_dup_cnt_FIRST=MAX'] = feature['user-item_dup_cnt_MAX'] - feature['user-item_dup_cnt_FIRST']
    
    irank2 = df[df['irank']==2][['buyer_admin_id', 'item_id']]
    irank2_flag = irank2.merge(dup_cnt[['buyer_admin_id', 'item_id']].drop_duplicates(), how='inner', on=['buyer_admin_id', 'item_id'])\
        .merge(feature, how='left', on=['buyer_admin_id', 'item_id'])  # 将feature特征与重复购买数据进行左连接
    irank2_flag['irank2_is_dup'] = 1  #重复购买标记
    irank2_flag['irank2_is_dup_scope'] = irank2_flag['irank2_is_dup'] * (irank2_flag['user-item_dup_cnt_FIRST'] < irank2_flag['user-item_dup_cnt_MAX'])  #这里特征不是很理解
    irank2_flag = irank2_flag.drop([col for col in irank2_flag.columns if 'user-item' in col], 1)
	
    #同上述操作
    irank3 = df[df['irank']==3][['buyer_admin_id', 'item_id']]
    irank3_flag = irank3.merge(dup_cnt[['buyer_admin_id', 'item_id']].drop_duplicates(), how='inner', on=['buyer_admin_id', 'item_id'])\
        .merge(feature, how='left', on=['buyer_admin_id', 'item_id'])
    irank3_flag['irank3_is_dup'] = 1
    irank3_flag['irank3_is_dup_scope'] = irank3_flag['irank3_is_dup'] * (irank3_flag['user-item_dup_cnt_FIRST'] < irank3_flag['user-item_dup_cnt_MAX'])
    irank3_flag = irank3_flag.drop([col for col in irank3_flag.columns if 'user-item' in col], 1)
    feature = feature.merge(irank2_flag, how='left', on=['buyer_admin_id', 'item_id'])
    feature = feature.merge(irank3_flag, how='left', on=['buyer_admin_id', 'item_id'])
    feature = feature.merge(dup_feature, how='left', on=['buyer_admin_id', 'item_id'])
    feature.to_hdf('../feature/rank/user_item_dupli_feature', 'all')
    print('>>> get_user_item_dupli_feature success')
    return feature
```