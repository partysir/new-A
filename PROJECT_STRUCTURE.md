# 项目结构说明

## 📁 目录结构

```
multi-factor-ml-strategy/
│
├── config.py                 # 配置文件 - 所有参数设置
├── data_manager.py           # 数据管理 - 数据获取、清洗、缓存
├── factor_engine.py          # 因子计算 - 技术、资金流、基本面因子
├── ml_model.py              # 机器学习 - 模型训练、预测
├── strategy.py              # 策略逻辑 - 选股、风控、择时
├── backtest.py              # 回测引擎 - 模拟交易、性能分析
├── main.py                  # 主程序 - 整合所有模块
├── examples.py              # 示例代码 - 快速入门
├── requirements.txt         # 依赖列表
├── README.md               # 使用文档
│
├── data_cache/             # 数据缓存目录
│   ├── daily_*.pkl         # 日线数据缓存
│   └── ...
│
├── output/                 # 输出目录
│   ├── equity_curve.png    # 权益曲线图
│   ├── drawdown.png        # 回撤图
│   ├── metrics_table.png   # 指标表
│   ├── trades.csv          # 交易记录
│   ├── positions.csv       # 持仓记录
│   ├── feature_importance.csv  # 特征重要性
│   └── live/               # 实盘结果
│       ├── selected_*.csv
│       └── trades_*.csv
│
└── logs/                   # 日志目录
    └── strategy.log        # 运行日志
```

## 📄 文件说明

### 核心模块

#### 1. config.py
**配置管理**

包含所有配置类:
- `DataConfig` - 数据相关配置
- `FactorConfig` - 因子计算配置
- `ModelConfig` - 模型训练配置
- `StrategyConfig` - 策略参数配置
- `RiskConfig` - 风控参数配置
- `BacktestConfig` - 回测设置
- `SystemConfig` - 系统设置

主要类:
```python
Config() - 全局配置类
  .validate() - 验证配置
  .save() - 保存配置
  .load() - 加载配置
```

#### 2. data_manager.py
**数据管理模块**

主要类和功能:

```python
DataManager(config)
  .get_stock_list() - 获取股票列表
  .get_daily_data() - 获取日线数据
  .get_fundamental_data() - 获取基本面数据
  .get_money_flow_data() - 获取资金流数据
  .get_index_data() - 获取指数数据
  .incremental_update() - 增量更新
  .clear_cache() - 清除缓存

DataQuality
  .check_missing_values() - 检查缺失值
  .check_duplicates() - 检查重复
  .check_outliers() - 检查异常值
  .generate_report() - 生成质量报告
```

特点:
- 自动数据清洗
- 智能缓存机制
- 增量更新支持
- 质量检查

#### 3. factor_engine.py
**因子计算模块**

主要类:

```python
FactorEngine(config)
  .calculate_all_factors() - 计算所有因子
  ._calculate_technical_factors() - 技术因子
  ._calculate_money_flow_factors() - 资金流因子
  ._calculate_alpha_factors() - Alpha因子
  ._winsorize_factors() - 极值处理
  ._standardize_factors() - 标准化
  ._neutralize_factors() - 行业中性化
  ._orthogonalize_factors() - 正交化
  .calculate_ic() - 计算IC值
  .select_factors() - 因子筛选

FeatureImportance
  .get_model_importance() - 获取特征重要性
  .plot_importance() - 绘制重要性图
```

支持的因子:
- 技术因子: 11个
- 资金流因子: 6个
- 基本面因子: 10个
- Alpha因子: 3个(可扩展)

#### 4. ml_model.py
**机器学习模块**

主要类:

```python
MLModel(config, model_type)
  .build_model() - 构建模型
  .train() - 训练模型
  .predict() - 预测
  .get_feature_importance() - 特征重要性
  .save() - 保存模型
  .load() - 加载模型

EnsembleModel(config)
  .build_models() - 构建多个模型
  .train() - 训练所有模型
  .predict() - 集成预测

WalkForwardTrainer(config)
  .walk_forward_train() - 滚动训练
  .get_model_performance() - 模型性能

LabelGenerator
  .generate_forward_return() - 生成标签
  .add_labels() - 添加标签
```

支持的模型:
- XGBoost
- LightGBM
- CatBoost
- 集成学习

#### 5. strategy.py
**策略模块**

主要类:

```python
Strategy(config)
  .select_stocks() - 选股
  .calculate_weights() - 权重分配
  .should_rebalance() - 判断调仓

RiskManager(config)
  .check_stop_loss() - 止损检查
  .check_take_profit() - 止盈检查
  .check_holding_period() - 持仓期限检查
  .check_portfolio_risk() - 组合风险检查

MarketTiming(config)
  .get_market_signal() - 择时信号
  ._calculate_rsrs() - RSRS指标
  ._calculate_ma_signal() - 均线信号

SentimentAnalyzer(config)
  .get_sentiment_score() - 舆情评分
  .apply_sentiment_filter() - 舆情过滤

PortfolioManager(config)
  .update_positions() - 更新持仓
  .get_portfolio_value() - 组合市值
  ._buy_stock() - 买入
  ._sell_stock() - 卖出
```

#### 6. backtest.py
**回测模块**

主要类:

```python
BacktestEngine(config)
  .run() - 运行回测
  .calculate_metrics() - 计算指标
  .generate_report() - 生成报告
  .plot_results() - 绘制图表

RealTimeTrader(config)
  .execute_trades() - 执行交易
  .get_positions() - 获取持仓
  .get_balance() - 获取余额
```

输出指标:
- 收益类: 总收益、年化收益、超额收益
- 风险类: 波动率、最大回撤
- 风险调整: 夏普比率、索提诺比率、信息比率
- 其他: 胜率、换手率

#### 7. main.py
**主程序**

主要类:

```python
StrategySystem(config)
  .run_backtest() - 运行回测
  .run_live() - 运行实盘
  .run_train_only() - 仅训练

main() - 入口函数
```

运行模式:
- backtest: 回测模式
- live: 实盘模式
- train: 训练模式

## 🔄 数据流程

```
1. 数据获取
   DataManager.get_daily_data()
   └─> 返回原始日线数据

2. 因子计算
   FactorEngine.calculate_all_factors()
   └─> 计算技术/资金流/基本面/Alpha因子
   └─> 因子处理(极值/标准化/中性化)
   └─> 返回带因子的数据

3. 标签生成
   LabelGenerator.add_labels()
   └─> 计算未来收益率
   └─> 返回带标签的数据

4. 模型训练
   WalkForwardTrainer.walk_forward_train()
   └─> 滚动窗口训练
   └─> 生成预测分数
   └─> 返回带ML评分的数据

5. 策略执行
   Strategy.select_stocks()
   └─> 根据评分选股
   └─> 应用风控规则
   └─> 计算权重
   └─> 返回目标持仓

6. 回测/实盘
   BacktestEngine.run() / RealTimeTrader.execute_trades()
   └─> 模拟/执行交易
   └─> 计算收益
   └─> 生成报告
```

## 🎯 关键设计

### 1. 防止未来函数

**问题**: 使用未来数据训练模型会导致过拟合

**解决方案**:
- Walk-Forward训练: 严格按时间顺序训练
- 标签生成: 只使用当前及之前的数据
- 预测隔离: ml_score和持仓严格分离

```python
# Walk-Forward示意
训练集: [day 1 ... day 252]  -> 预测 day 253
训练集:     [day 21 ... day 272]  -> 预测 day 273
训练集:         [day 41 ... day 292]  -> 预测 day 293
```

### 2. 因子正交化

**目的**: 去除因子间的相关性

**方法**: QR分解
```python
def orthogonalize(factors):
    Q, R = qr(factors)
    return Q  # 正交化后的因子
```

### 3. 行业中性化

**目的**: 消除行业偏差

**方法**: 减去行业均值
```python
factor_neutral = factor - industry_mean
```

### 4. 智能缓存

**策略**:
- 日线数据缓存到本地
- 因子结果缓存
- 模型缓存
- 增量更新机制

### 5. 内存优化

**方法**:
- 分块处理大数据
- 及时释放内存
- 使用生成器
- 数据类型优化(float32)

## 📊 性能指标

### 收益指标
- **总收益**: 整个回测期的累计收益
- **年化收益**: 折算为年化的收益率
- **超额收益**: 相对基准的超额部分

### 风险指标
- **波动率**: 收益率的标准差
- **最大回撤**: 从最高点到最低点的跌幅
- **下行风险**: 只计算负收益的波动

### 风险调整收益
- **夏普比率**: (收益 - 无风险利率) / 波动率
- **索提诺比率**: 收益 / 下行风险
- **信息比率**: 超额收益 / 跟踪误差
- **卡玛比率**: 年化收益 / 最大回撤

### 其他指标
- **胜率**: 盈利交易日占比
- **盈亏比**: 平均盈利 / 平均亏损
- **换手率**: 年化交易金额 / 平均资产

## 🔧 扩展指南

### 添加新因子

1. 在 `factor_engine.py` 中添加计算函数:
```python
def _calculate_my_factor(self, df):
    df['my_factor'] = ...  # 你的计算逻辑
    return df
```

2. 在配置中启用:
```python
config.factor.technical_factors.append('my_factor')
```

### 添加新模型

1. 在 `ml_model.py` 的 `build_model()` 中添加:
```python
elif self.model_type == 'my_model':
    from my_model_lib import MyModel
    self.model = MyModel(...)
```

### 添加新的风控规则

1. 在 `strategy.py` 的 `RiskManager` 中添加:
```python
def check_my_rule(self, ...):
    # 你的规则逻辑
    return to_sell_list
```

2. 在回测循环中调用

### 自定义择时指标

1. 在 `strategy.py` 的 `MarketTiming` 中添加:
```python
def _calculate_my_timing(self, df, date):
    # 你的择时逻辑
    return signal  # 0-1之间
```

## ⚠️ 注意事项

1. **数据质量**: 确保数据完整、准确
2. **未来函数**: 严格遵循时间顺序
3. **过拟合**: 不要过度优化参数
4. **交易成本**: 考虑佣金、滑点、冲击成本
5. **市场变化**: 定期重新训练模型
6. **风险控制**: 设置合理的止损止盈
7. **资金管理**: 控制单只股票和行业权重

## 📚 参考资料

- Tushare文档: https://tushare.pro/document/2
- XGBoost文档: https://xgboost.readthedocs.io/
- LightGBM文档: https://lightgbm.readthedocs.io/
- 量化投资书籍: 《量化投资策略》等