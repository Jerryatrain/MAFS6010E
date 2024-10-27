T_list = [1, 3, 5, 10, 22]  # Model prediction frequency
# T_list = [1]
T_pre_list = [5, 10, 22]  # Model prediction frequency
T_pre_list_300 = [10, 22]  # 300 model prediction frequency
# T_pre_list = [5]  # Model prediction frequency
adjustment_frequency = 'monthly'  # Model update frequency, can be monthly, weekly, or quarterly

# model_list = ['Xgb']
model_list = ['MLP', 'Xgb']
loss_cate = 'WCCC'
train_window_dict = {'3': 365*2,
                     '5': 365*2,
                     '10': 365*2,
                     '22': 365*2}  # Training window dates

# Portfolio optimization parameters
trade_cost = 0.001  # Slippage of 0.1%
buy_fee = 0.0089 / 100  # Buy fee
sell_fee = 0.0589 / 100  # Sell fee
industry_diff = 0.02  # Industry deviation
barra_std_num = 0.5
size_thresh = 0.1
turnover_punish = 500  # Turnover penalty
risk_aversion = 100  # Risk aversion coefficient

# industry_diff = 100  # Industry deviation
# barra_std_num = 100
# turnover_punish = 2  # Turnover penalty