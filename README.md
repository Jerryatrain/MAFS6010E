# 组合优化模型理论&代码

## 一、等权
所有资产等权配置

## 二、等波动率
通过计算过去一段时间资产的历史波动率，来作为该资产的收益风险特征。为了让不同的资产的最终风险暴露度一致，可以通过调整资产的相对权重来完成。  
w 为权重，σ 为波动率，满足如下关系式：

\[
w_1 \sigma_1 = w_2 \sigma_2 = \ldots = w_n \sigma_n
\]


即权重满足如下关系式：

\[
\frac{1}{\sigma_1} = \frac{1}{\sigma_2} = ... = \frac{1}{\sigma_n}
\]

```python
wts = 1 / pct_temp.std()
weights2 = wts / wts.sum()
```

## 三、均值方差
最小方差资产组合单独聚焦于风险并且忽略了资产组合的期望收益。大多数投资者希望在二者之间取得平衡，如果他们对股票有期望收益的话。均值方差优化起到了在给定一定程度风险的情况下，作为最大化期望收益下寻找最优资产组合的作用。我们首先考虑包括现金的资产组合用 Rf 表示收益，且用 w 表示权重，r 为股票的期望收益。

**参考资源：**
- 二、资产组合理论 - 最小方差资产组合 - 知乎 ([zhihu.com](https://zhihu.com))
- 均值方差模型在投资组合中的简单应用 - 拉姆达投资 - JoinQuant

### 1. 最小化方差/主动最小化方差
参考马科维茨最优投资组合理论  
投资组合的方差计算公式:

\[
std(F_c) = \sqrt{\sum_{n=1}^{i} \sum_{n=1}^{j} w_i w_j \sigma_i \sigma_j \rho_{ij}}
\]

最小化方差即的核心是投资组合的整体方差计算，并实现最小化的过程。w 为投资权重向量，Σ 为各标的的收益率协方差矩阵。投资组合整体的波动率满足：**预期投资组合方差 = SQRT(W^T * (协方差矩阵) * W)**

\[
\sigma = \sqrt{w^T \Sigma w}
\]

```python
def min_variance(returns,w):
    returns = np.log(returns)
    r_cov = returns.cov()
    p_var = np.dot(w.T,np.dot(r_cov,w))
    p_std = np.sqrt(p_var)

    return p_std
    

def min_active_variance(returns,w,benchmark):
    returns = np.log(returns)
    active_returns = returns - benchmark
    r_cov = active_returns.cov()
    p_var = np.dot(w.T,np.dot(r_cov,w))
    p_std = np.sqrt(p_var)

    return p_std
```

### 2. 均值方差效用方程

\[
U_{MV} = \mu_R - \frac{1}{2} \gamma \sigma_R^2
\]

其中 \(\mu_R\), \(\sigma_R^2\) 分布代表投资组合的收益率均值和方差， \(\gamma\) 代表投资者的风险厌恶系数。

```python
def utility_theory(weights, temp_ret):
    p_r = (1 + np.dot(weights, temp_ret.mean())) ** 252 - 1
    temp_cov = temp_ret.cov()
    lam = 10  # 超参数可以调整
    p_sigma = np.dot(weights.T, np.dot(temp_cov, weights)) * 252
    p_utility = p_r - (lam / 2 * p_sigma)
    return -1 * p_utility
```

### 3. 最大化夏普
逻辑与最小化方差一致，只是将目标函数变成夏普比率。

\[
SharpeRatio = \frac{R_p - R_f}{\sigma_p}
\]

这里也需要计算组合方差，参上。最终线性规划化目标函数为:

\[
MAX \frac{W_i * f_i - R_f}{\sqrt{W^T \sum w * \sqrt{252}}}
\]

```python
def max_sharpe(returns, w):
    r_b = 0.03
    returns = np.log(returns)
    r_mean = returns.mean() * 252
    p_final_mean = np.sum(r_mean * w)
    r_cov = returns.cov()
    p_var = np.dot(w.T, np.dot(r_cov, w))
    p_std = np.sqrt(p_var)
    p_sharpe = (p_final_mean - r_b) / p_std
    
    return p_sharpe
```

### 4. 最大化信息比率
逻辑与最小化方差一致，只是将目标函数变成信息比率。

\[
IR = \frac{W_i * f_i - R_f}{\sigma_{tr}}
\]

这里也需要计算组合方差，参上。最终线性规划化目标函数为:

\[
MAX \frac{W_i * f_i - R_f}{\sigma_{W_i*f_i - rm} * \sqrt{252}}
\]

```python
def max_ir(returns, w, benchmark):
    r_b = 0.03
    returns = np.log(returns)
    r_mean = returns.mean() * 252
    p_mean = np.sum(returns * w)
    p_final_mean = np.sum(r_mean * w)
    tr = (p_mean - benchmark)
    std_tr = tr.std() * np.sqrt(252)
    p_ir = (p_final_mean - r_b) / std_tr
    
    return p_ir
```

### 四、风险平价
风险平价模型是通过衡量组合各标的对于组合的风险贡献程度，来制定投资模型的各类资产权重，目标是使得组合标的的波动等比例风险。假设投资组合中共有 n 个资产，第一 i 个资产的收益率为 f，假定 w 表示资产的投资权重向量，则投资组合的波动率可以定义为:

\[
\sigma = \sqrt{w^T \Sigma w}
\]

每个资产 i 的对投资组合的边际风险贡献为 MRC 为:

\[
MRC_i = \frac{\partial \sigma_p}{\partial w} = \frac{(\sum w)_i}{\sqrt{w^T \Sigma w}}
\]

我们可以得到每个资产对投资组合的风险贡献 RC 为:

\[
RC_i = w_i \frac{(\sum w)_i}{\sqrt{w^T \Sigma w}}
\]

而所有资产的总风险贡献 TRC 为:

\[
TRC = \sum_{i=1}^{n} RC_i = \sum_{i=1}^{n} w_i \frac{\partial \sigma_p}{\partial w} = \sqrt{w^T \Sigma w} = \sigma_p
\]


故可得，组合的波动率的分解式为各项资产的边际风险之和:

\[
Risk(r) = \sigma_p = RC_1 + RC_2 + ... + RC_n
\]

为了能够消除不同资产对组合风险贡献的不平衡，得到风险更加分散化的组合，研究者提出等风险的组合，以保证各资产的风险贡献相等。例如： \( RC_i = RC_j, \ i \neq j \) 则可以得到整体的表达式:

\[
\sum_{i=1}^{n} \sum_{j=1}^{n} (RC_i - RC_j)^2 = 0 \iff \sum_{i=1}^{n} \sum_{j=1}^{n} \left( w_i \left( \sum w \right)_i - w_j \left( \sum w \right)_j \right)^2 = 0
\]

因为 RC 的分母项一致，所以可以简化为:

\[
w_i \frac{\sum w_i}{\sqrt{w^T \Sigma w}} = w_j \frac{\sum w_j}{\sqrt{w^T \Sigma w}} \iff w_i \left( \sum w \right)_i - w_j \left( \sum w \right)_j
\]

在线性规划问题中该模型的最优权重式为:

\[
\min \sum_{i=1}^{n} \sum_{j=1}^{n} \left( w_i \left( \sum w \right)_i - w_j \left( \sum w \right)_j \right)^2 = 0
\]

**约束条件:**

\[
s.t. \sum_{i=1}^{n} w_i = 1, \ 0 \leq w_i \leq 1, \ i = 1, 2, ..., n
\]

```python
def risk_budget_objective(weights,cov):
    weights = np.array(weights) #weights为一维数组
    sigma = np.sqrt(np.dot(weights, np.dot(cov, weights))) #获取组合标准差   
    MRC = np.dot(cov,weights)/sigma
    TRC = weights * MRC
    delta_TRC = [sum((i - TRC)**2) for i in TRC]
    return sum(delta_TRC)
```

### 五、最小化跟踪误差
个股组合权重及其当期收益的构建的组合收益与基准收益之间的关系的波动率最小化，目标函数:

\[
\sigma_{W_i * f_i - rm}
\]

```python
def min_tracking_error(returns, w, benchmark):
    returns = np.log(returns)
    p_mean = np.sum(returns * w)
    tr = (p_mean - benchmark)
    std_tr = tr.std() * np.sqrt(252)
    
    return std_tr
```
