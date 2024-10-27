1. calculate daily index averaged weighted barra and barra std

```python

# barra mean
df_index.loc[date_, barra_list] = np.average(df_[barra_list], weights=df_['weight'], axis=0) 

# barraa std
df_index.loc[date_, barra_std_list] = np.average(
            (df_[barra_list] - df_index.loc[date_, barra_list]) ** 2, weights=df_['weight'],
            axis=0) ** 0.5
```

2. set portfolio daily barra limit as:

```python

df_index[barra_ + '_lower'] = df_index[barra_] - mc.barra_std_num * df_index[barra_ + '_std']
df_index[barra_ + '_upper'] = df_index[barra_] + mc.barra_std_num * df_index[barra_ + '_std']

```

3. constraints for industry

```python

constraints.append(weights[:, 0] @ stock_today_industry[industry_list_name] - index_today[industry_list].iloc[0,
                                                                                      :] <= mc.industry_diff)

constraints.append(weights[:, 0] @ stock_today_industry[industry_list_name] - index_today[industry_list].iloc[0,
                                                                                        :] >= -mc.industry_diff)
    
``` 

4. constraints for turnover

```python
# calculate trade weighted difference and cost
    try:
        trade_cost_sum = (cp.sum(cp.abs(weights - past_weight[['past_weight']].iloc[:len(stock_today)]))
                          + cp.sum(past_weight[['past_weight']].iloc[len(stock_today):]))
    except:
        trade_cost_sum = cp.sum(cp.abs(weights - past_weight[['past_weight']].iloc[:len(stock_today)]))
    
    # limit for turnover
    if past_weight['past_weight'].sum() > 0.5:  
        constraints.append(trade_cost_sum <= turnover * 2)  
```


