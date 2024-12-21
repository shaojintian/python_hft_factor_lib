import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

# 计算COFI指标，衡量买卖力量的不平衡
def cofi(depth, trade):
    # 计算买入和卖出的不平衡，正值表示买入力量强，负值表示卖出力量强
    a = depth['bv1'] * np.where(depth['bp1'].diff() >= 0, 1, 0)  # 买入价差增加
    b = depth['bv1'].shift() * np.where(depth['bp1'].diff() <= 0, 1, 0)  # 买入价差减少
    c = depth['av1'] * np.where(depth['ap1'].diff() <= 0, 1, 0)  # 卖出价差减少
    d = depth['av1'].shift() * np.where(depth['ap1'].diff() >= 0, 1, 0)  # 卖出价差增加
    return (a - b - c + d).fillna(0)

# 计算最佳卖价（BP）的排名，用于衡量价格变动的相对位置
def bp_rank(depth, trade, n=100):
    return ((depth.bp1.rolling(n).rank()) / n * 2 - 1).fillna(0)

# 计算最佳买价（AP）的排名，用于衡量价格变动的相对位置
def ap_rank(depth, trade, n=100):
    return ((depth.ap1.rolling(n).rank()) / n * 2 - 1).fillna(0)

# 计算价格影响，衡量订单簿中价格变动对市场的影响
def price_impact(depth, trade, n=10):
    # 计算ask和bid的加权平均价格
    ask, bid, ask_v, bid_v = 0, 0, 0, 0
    for i in range(1, n+1):
        ask += depth[f'api{i}'] * depth[f'avi{i}']
        bid += depth[f'bp{i}'] * depth[f'bv{i}']
        ask_v += depth[f'av{i}']
        bid_v += depth[f'bv{i}']
    ask /= ask_v
    bid /= bid_v
    # 计算价格影响
    return pd.Series(-(depth['ap1'] - ask) / depth['ap1'] - (depth['bp1'] - bid) / depth['bp1'], name="price_impact")

# 计算信息比率，衡量策略的超额收益与其跟踪误差的比率
def inf_ratio(depth=None, trade=None, n=100):
    quasi = trade.p.diff().abs().rolling(n).sum().fillna(10)
    dif = trade.p.diff(n).abs().fillna(10)
    return quasi / (dif + quasi)

# 计算深度价格范围，衡量订单簿中价格的波动范围
def depth_price_range(depth=None, trade=None, n=100):
    return (depth.ap1.rolling(n).max() / depth.ap1.rolling(n).min() - 1).fillna(0)

# 计算价格变化的持续时间
def get_age(prices):
    last_value = prices[-1]
    age = 0
    for i in range(2, len(prices)):
        if prices[i-1] != last_value:
            return age
        age += 1
    return age

# 计算卖价（BP）变化的持续时间
def ask_bid_age(depth, trade, n=100):
    bp1 = depth['bp1']
    bp1changes = bp1.rolling(n).apply(get_age, engine='numba', raw=True).fillna(0)
    return bp1changes

# 计算到达率，衡量交易频率
def arrive_rate(depth, trade, n=300):
    res = trade['ts'].diff(n).fillna(0) / n
    return res

# 计算订单簿价格的偏度，衡量价格分布的不对称性
def depth_price_skew(depth, trade):
    prices = ["bp5", "bp4", "bp3", "bp2", "bp1", "ap1", "ap2", "ap3", "ap4", "ap5"]
    return depth[prices].skew(axis=1)

# 计算订单簿价格的峰度，衡量价格分布的尖峭程度
def depth_price_kurt(depth, trade):
    prices = ["bp5", "bp4", "bp3", "bp2", "bp1", "ap1", "ap2", "ap3", "ap4", "ap5"]
    return depth[prices].kurt(axis=1)

# 计算滚动回报率，衡量价格的变动
def rolling_return(depth, trade, n=100):
    mp = ((depth.ap1 + depth.bp1) / 2)
    return (mp.diff(n) / mp).fillna(0)

# 计算买入量增加的对数，衡量买入压力
def buy_increasing(depth, trade, n=100):
    v = trade.v.copy()
    v[v < 0] = 0
    return np.log1p(((v.rolling(2 * n).sum() + 1) / (v.rolling(n).sum() + 1)).fillna(1))

# 计算卖出量增加的对数，衡量卖出压力
def sell_increasing(depth, trade, n=100):
    v = trade.v.copy()
    v[v > 0] = 0
    return np.log1p(((v.rolling(2 * n).sum() - 1) / (v.rolling(n).sum() - 1)).fillna(1))

# 找到最大值的第一个位置
def first_location_of_maximum(x):
    max_value = max(x)
    for loc in range(len(x)):
        if x[loc] == max_value:
            return loc + 1

# 计算价格最大值的第一个位置
def price_idxmax(depth, trade, n=20):
    return depth['ap1'].rolling(n).apply(
        first_location_of_maximum, engine='numba', raw=True).fillna(0)

# 计算中心二阶导数
def mean_second_derivative_centra(x):
    sum_value = 0
    for i in range(len(x) - 5):
        sum_value += (x[i+5] - 2*x[i+3] + x[i]) / 2
    return sum_value / (2 * (len(x) - 5))

# 计算中心二阶导数的滚动应用
def center_deri_two(depth, trade, n=20):
    return depth['ap1'].rolling(n).apply(
        mean_second_derivative_centra, engine='numba', raw=True).fillna(0)

# 计算准价格，衡量价格的绝对变化
def quasi(depth, trade, n=100):
    return depth.ap1.diff(1).abs().rolling(n).sum().fillna(0)

# 计算最后的价格范围
def last_range(depth, trade, n=100):
    return trade.p.diff(1).abs().rolling(n).sum().fillna(0)

# 计算平均交易量
def avg_trade_volume(depth, trade, n=100):
    return (trade.v[::-1].abs().rolling(n).sum().shift(-n+1).fillna(0))[:n-1]

# 计算平均价差
def avg_spread(depth, trade, n=200):
    return (depth.ap1 - depth.bp1).rolling(n).mean().fillna(0)

# 计算平均交易周转量
def avg_turnover(depth, trade, n=500):
    return depth[['av1', 'av2', 'av3', 'av4', 'av5',
                  'bv1', 'bv2', 'bv3', 'bv4', 'bv5']].sum(axis=1)

# 计算绝对交易量的峰度
def abs_volume_kurt(depth, trade, n=500):
    return trade.v.abs().rolling(n).kurt().fillna(0)

# 计算绝对交易量的偏度
def abs_volume_skew(depth, trade, n=500):
    return trade.v.abs().rolling(n).skew().fillna(0)

# 计算交易量的峰度
def volume_kurt(depth, trade, n=500):
    return trade.v.rolling(n).kurt().fillna(0)

# 计算交易量的偏度
def volume_skew(depth, trade, n=500):
    return trade.v.rolling(n).skew().fillna(0)

# 计算价格的峰度
def price_kurt(depth, trade, n=500):
    return trade.p.rolling(n).kurt().fillna(0)

# 计算价格的偏度
def price_skew(depth, trade, n=500):
    return trade.p.rolling(n).skew().abs().fillna(0)

def rolling_return(depth, trade, n=100):
    mp = ((depth.ap1 + depth.bp1) / 2)
    return (mp.diff(n) / mp).fillna(0)

def first_location_of_maximum(x):
    max_value = max(x)  # 一个for循环
    for loc in range(len(x)):
        if x[loc] == max_value:
            return loc + 1

def price_idxmax(depth, trade, n=20):
    return depth['ap1'].rolling(n).apply(
        first_location_of_maximum, engine='numba', raw=True).fillna(0)

def mean_second_derivative_centra(x):
    sum_value = 0
    for i in range(len(x) - 5):
        sum_value += (x[i+5] - 2*x[i+3] + x[i]) / 2
    return sum_value / (2 * (len(x) - 5))


def weighted_price_to_mid(depth, trade, levels=10, alpha=1):
    def get_columns(name, levels):
        return [name + str(i) for i in range(1, levels + 1)]
    avs = depth[get_columns("av", levels)]
    bvs = depth[get_columns("bv", levels)]
    aps = depth[get_columns("ap", levels)]
    bps = depth[get_columns("bp", levels)]
    mp = (depth['ap1'] + depth['bp1']) / 2
    return (avs.values * aps.values + bvs.values * bps.values).sum(axis=1) / (avs.values + bvs.values).sum(axis=1) - mp

def ask_withdraws(depth, trade):
    ob_values = depth.values
    flows = np.zeros(len(ob_values))
    for i in range(1, len(ob_values)):
        flows[i] = _ask_withdraws_volume(ob_values[i-1], ob_values[i])
    return pd.Series(flows)

def bid_withdraws(depth, trade):
    ob_values = depth.values
    flows = np.zeros(len(ob_values))
    for i in range(1, len(ob_values)):
        flows[i] = _bid_withdraws_volume(ob_values[i-1], ob_values[i])
    return pd.Series(flows)


def mean_second_derivative_centra(x):
    sum_value = 0
    for i in range(len(x) - 5):
        sum_value += (x[i + 5] - 2 * x[i + 3] + x[i]) / 2
    return sum_value / (2 * (len(x) - 5))

def center_deri_two(depth, trade, n=20):
    return depth['ap1'].rolling(n).apply(
        mean_second_derivative_centra, engine='numba', raw=True).fillna(0)

def bv_divide_tn(depth, trade, n=10):
    bvs = depth.bv1 + depth.bv2 + ... + depth.bv10
    def volume(depth, trade, n):
        return trade.v
    v = volume(depth=depth, trade=trade, n=n)
    v[v > 0] = 0
    return (v.rolling(n).sum() / bvs).fillna(0)

def av_divide_tn(depth, trade, n=10):
    avs = depth.av1 + depth.av2 + ... + depth.av10
    def volume(depth, trade, n):
        return trade.v
    v = volume(depth=depth, trade=trade, n=n)
    v[v < 0] = 0
    return (v.rolling(n).sum() / avs).fillna(0)