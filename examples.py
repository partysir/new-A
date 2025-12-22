"""
快速入门示例
演示基本的使用方法
"""


def example_1_basic_backtest():
    """示例1: 基础回测"""
    print("=" * 60)
    print("示例1: 基础回测")
    print("=" * 60)

    from config import Config
    from main import StrategySystem

    # 创建配置
    config = Config()

    # 设置参数
    config.data.tushare_token = "2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211"  # 请替换为你的token
    config.backtest.start_date = "20200101"
    config.strategy.top_n = 10
    config.strategy.rebalance_frequency = "weekly"

    # 创建系统
    system = StrategySystem(config)

    # 运行回测
    print("运行回测中...")
    results = system.run_backtest()

    # 打印结果
    print("\n回测结果:")
    print(f"年化收益: {results['metrics']['annual_return']:.2%}")
    print(f"夏普比率: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['metrics']['max_drawdown']:.2%}")


def example_2_custom_config():
    """示例2: 自定义配置"""
    print("\n" + "=" * 60)
    print("示例2: 自定义配置")
    print("=" * 60)

    from config import Config

    # 创建配置
    config = Config()

    # 数据配置
    config.data.start_date = "20210101"
    config.data.exclude_st = True
    config.data.min_market_cap = 20  # 20亿市值

    # 因子配置
    config.factor.technical_factors = [
        'momentum_20', 'momentum_60', 'rsi_14', 'macd'
    ]
    config.factor.neutralize = True

    # 模型配置
    config.model.model_type = "lightgbm"
    config.model.train_window = 180

    # 策略配置
    config.strategy.top_n = 15
    config.strategy.weight_method = "score_weighted"
    config.strategy.rebalance_frequency = "monthly"

    # 风控配置
    config.risk.use_stop_loss = True
    config.risk.stop_loss_pct = -0.08
    config.risk.use_timing = True

    # 保存配置
    config.save("my_config.json")
    print("配置已保存到 my_config.json")

    # 加载配置
    loaded_config = Config.load("my_config.json")
    print("配置已加载")


def example_3_factor_analysis():
    """示例3: 因子分析"""
    print("\n" + "=" * 60)
    print("示例3: 因子分析")
    print("=" * 60)

    from config import Config
    from data_manager import DataManager
    from factor_engine import FactorEngine
    from ml_model import LabelGenerator
    import pandas as pd

    # 创建配置
    config = Config()
    config.data.tushare_token = "YOUR_TOKEN"

    # 加载数据(使用少量数据做演示)
    print("加载数据...")
    dm = DataManager(config)

    # 这里使用缓存的数据或少量数据
    # df = dm.get_daily_data(start_date="20230101", end_date="20230131")

    print("计算因子...")
    fe = FactorEngine(config)
    # df_with_factors = fe.calculate_all_factors(df)

    print("计算IC值...")
    # df_with_labels = LabelGenerator.add_labels(df_with_factors, config)
    # ic_df = fe.calculate_ic(df_with_labels)

    # 分析因子
    # ic_stats = ic_df.groupby('factor')['ic'].agg(['mean', 'std', 'count'])
    # ic_stats['ic_ir'] = ic_stats['mean'] / ic_stats['std']
    # ic_stats = ic_stats.sort_values('ic_ir', ascending=False)

    # print("\n因子IC统计:")
    # print(ic_stats.head(10))


def example_4_model_comparison():
    """示例4: 模型对比"""
    print("\n" + "=" * 60)
    print("示例4: 模型对比")
    print("=" * 60)

    from config import Config
    from main import StrategySystem

    models = ['xgboost', 'lightgbm', 'catboost']
    results_dict = {}

    for model_type in models:
        print(f"\n测试 {model_type}...")

        config = Config()
        config.data.tushare_token = "YOUR_TOKEN"
        config.model.model_type = model_type
        config.backtest.start_date = "20220101"

        system = StrategySystem(config)
        # results = system.run_backtest()

        # results_dict[model_type] = results['metrics']

    # 打印对比
    print("\n模型对比:")
    print("-" * 60)
    # for model, metrics in results_dict.items():
    #     print(f"{model:12s}: 年化收益={metrics['annual_return']:.2%}, "
    #           f"夏普比率={metrics['sharpe_ratio']:.2f}")


def example_5_parameter_optimization():
    """示例5: 参数优化"""
    print("\n" + "=" * 60)
    print("示例5: 参数优化")
    print("=" * 60)

    from config import Config
    from main import StrategySystem

    # 测试不同的持仓数量
    top_n_list = [5, 10, 15, 20]
    results = []

    for n in top_n_list:
        print(f"\n测试 top_n={n}...")

        config = Config()
        config.data.tushare_token = "YOUR_TOKEN"
        config.strategy.top_n = n
        config.backtest.start_date = "20220101"

        system = StrategySystem(config)
        # result = system.run_backtest()

        # results.append({
        #     'top_n': n,
        #     'annual_return': result['metrics']['annual_return'],
        #     'sharpe_ratio': result['metrics']['sharpe_ratio'],
        #     'max_drawdown': result['metrics']['max_drawdown']
        # })

    # 打印结果
    # import pandas as pd
    # df_results = pd.DataFrame(results)
    # print("\n参数优化结果:")
    # print(df_results.to_string(index=False))


def example_6_live_trading_simulation():
    """示例6: 模拟实盘"""
    print("\n" + "=" * 60)
    print("示例6: 模拟实盘")
    print("=" * 60)

    from config import Config
    from main import StrategySystem

    # 创建配置
    config = Config()
    config.data.tushare_token = "YOUR_TOKEN"

    # 启用模拟模式
    config.system.live_trading = True
    config.system.dry_run = True  # 模拟运行,不实际交易

    # 创建系统
    system = StrategySystem(config)

    print("运行模拟实盘...")
    # results = system.run_live()

    print("模拟完成,查看 ./output/live/ 目录获取结果")


def main():
    """主函数"""
    print("=" * 60)
    print("多因子综合评分+机器学习选股策略 - 快速入门示例")
    print("=" * 60)

    print("\n请选择要运行的示例:")
    print("1. 基础回测")
    print("2. 自定义配置")
    print("3. 因子分析")
    print("4. 模型对比")
    print("5. 参数优化")
    print("6. 模拟实盘")
    print("0. 退出")

    choice = input("\n请输入选择 (0-6): ").strip()

    examples = {
        '1': example_1_basic_backtest,
        '2': example_2_custom_config,
        '3': example_3_factor_analysis,
        '4': example_4_model_comparison,
        '5': example_5_parameter_optimization,
        '6': example_6_live_trading_simulation,
    }

    if choice == '0':
        print("退出程序")
        return

    if choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"\n执行出错: {e}")
            print("请确保:")
            print("1. 已安装所有依赖: pip install -r requirements.txt")
            print("2. 已设置Tushare Token")
            print("3. 有足够的数据和计算资源")
    else:
        print("无效的选择")


if __name__ == '__main__':
    main()