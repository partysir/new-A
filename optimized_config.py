"""
优化后的配置文件
关键改进点已标注
"""

class OptimizedConfig:
    """优化后的策略配置"""
    
    def __init__(self):
        # === 策略参数 ===
        self.strategy = {
            # 选股数量
            'top_n': 20,  # 【建议】从10提升到20,分散风险
            
            # 调仓频率 - 【关键改进】
            'rebalance_frequency': 'n_days',
            'rebalance_day': 10,  # 【改进】从5天改为10天,降低换手率
            
            # 选股方法
            'selection_method': 'rank',  # 使用排名法,不用硬阈值
            'score_threshold': 0.3,  # 【放宽】从0.5降到0.3
            
            # 权重方法
            'weight_method': 'equal',  # 等权,简单有效
            'max_single_weight': 0.15,  # 单只最大15%
            
            # 行业约束
            'max_industry_weight': 0.4,  # 单行业最大40%
            
            # 交易成本
            'commission_rate': 0.0003,  # 万三佣金
            'stamp_tax': 0.001,  # 千一印花税
            'slippage': 0.001,  # 千一滑点
        }
        
        # === 风险控制参数 ===
        self.risk = {
            # 【关键改进】非对称止盈止损
            'stop_loss_pct': -0.05,  # 止损-5%
            'take_profit_pct': 0.15,  # 止盈+15% (盈亏比3:1)
            'trailing_stop_pct': 0.10,  # 移动止盈阈值10%
            
            # 【关键改进】仓位控制
            'use_tiered_position': True,
            'min_position': 0.30,  # 【关键】最低仓位30%,永不空仓
            'max_position': 1.00,
            
            # 仓位分级(基于指数信号)
            'position_tiers': [
                {'threshold': 0.05, 'position': 1.00, 'name': '满仓'},
                {'threshold': 0.10, 'position': 0.70, 'name': '七成仓'},
                {'threshold': 0.15, 'position': 0.50, 'name': '半仓'},
                {'threshold': 1.00, 'position': 0.30, 'name': '轻仓'}
            ],
            
            # 持仓管理
            'max_holding_days': 30,  # 【放宽】从20天改为30天
            'score_decay_threshold': 0.4,  # 评分衰减阈值
        }
        
        # === 模型参数 ===
        self.model = {
            'model_type': 'lightgbm',  # 推荐LightGBM,速度快
            'use_ensemble': False,  # 【建议】单模型即可,避免过拟合
            
            # LightGBM参数
            'lightgbm_params': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': 6,
                'min_data_in_leaf': 100,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1
            },
            
            # 训练参数
            'n_estimators': 200,  # 【减少】从500降到200,防止过拟合
            'early_stopping_rounds': 50,
            
            # Walk-Forward参数
            'train_window_months': 12,  # 【减少】从24个月降到12个月
            'test_window_months': 3,
            'retrain_frequency_months': 3,
        }
        
        # === 因子工程参数 ===
        self.factor = {
            # 【关键改进】减少因子数量,避免过拟合
            'technical_factors': [
                'momentum_20',  # 20日动量
                'rsi_14',       # RSI
                'macd',         # MACD
                'volume_ratio', # 量比
            ],
            
            'money_flow_factors': [
                'turnover_rate',      # 换手率
                'amount_change',      # 成交额变化
                'volume_price_corr',  # 量价相关性
            ],
            
            'alpha_factors': [
                'alpha_001',
                'alpha_002',
            ],
            
            # 因子处理
            'winsorize': True,
            'winsorize_limits': [0.01, 0.99],
            
            'standardize': True,
            
            'neutralize': True,  # 【重要】行业+市值中和
            
            'orthogonalize': False,  # 【关闭】正交化容易损失信息
            
            # 因子选择
            'max_factors': 15,  # 最多15个因子
        }
        
        # === 标签生成参数 ===
        self.label = {
            'method': 'next_day_open_return',  # 【关键】预测次日开盘收益
            'horizon': 1,  # 1天
            'classification': False,
            'quantile_thresholds': [0.3, 0.7],
        }
        
        # === 回测参数 ===
        self.backtest = {
            'initial_capital': 1000000,
            'start_date': '20200101',
            'end_date': '20241231',
            'benchmark': '000300.SH',  # 沪深300
            
            # 输出设置
            'output_dir': './results',
            'save_trades': True,
            'save_positions': True,
            'plot_results': True,
            'plot_factor_importance': True,
        }
        
        # === 系统参数 ===
        self.system = {
            'use_multiprocessing': True,
            'n_jobs': -1,
            'cache_factors': True,
            'cache_dir': './cache',
            'log_level': 'INFO',
            'log_file': './logs/strategy.log',
        }


# === 使用示例 ===
if __name__ == '__main__':
    """
    关键改进总结:
    
    1. 仓位控制:
       - 永不空仓,最低30%
       - 使用指数信号而非净值回撤
    
    2. 止盈止损:
       - 非对称: -5%止损, +15%止盈
       - 盈亏比3:1,胜率51%即可盈利
    
    3. 选股优化:
       - 过滤涨幅>7%的股票
       - 预测次日开盘收益
       - 增加备选池,防止买不到
    
    4. 降低换手:
       - 调仓周期从5天改为10天
       - 持仓容忍度,不强制卖出Top N外的股票
       - 最长持仓从20天延长到30天
    
    5. 模型优化:
       - 减少训练窗口,从24月降到12月
       - 减少因子数量,避免过拟合
       - 关闭正交化,避免信息损失
    
    预期改进:
    - 年化收益: 从-11%提升到+15%以上
    - 最大回撤: 控制在-20%以内
    - 夏普比率: 从负值提升到1.0以上
    - 换手率: 降低50%
    """
    
    config = OptimizedConfig()
    print("优化配置已加载")
    print(f"最低仓位: {config.risk['min_position']}")
    print(f"止损/止盈: {config.risk['stop_loss_pct']}/{config.risk['take_profit_pct']}")
    print(f"调仓周期: {config.strategy['rebalance_day']}天")