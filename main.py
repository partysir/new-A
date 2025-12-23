"""
多因子综合评分+机器学习选股策略 - 主程序
Version: 4.0

使用方法:
python main.py --mode backtest  # 回测模式
python main.py --mode live      # 实盘模式
python main.py --mode train     # 仅训练模式
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

# 导入模块
from config import Config
from data_manager import DataManager, DataQuality
from factor_engine import FactorEngine, FeatureImportance
from ml_model import MLModel, EnsembleModel, WalkForwardTrainer, LabelGenerator
from strategy import Strategy, RiskManager, MarketTiming, SentimentAnalyzer, PortfolioManager
from backtest import BacktestEngine, RealTimeTrader


def setup_logging(config: Config):
    """设置日志"""
    log_dir = Path(config.system.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.system.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.system.log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"Multi-Factor ML Stock Selection Strategy v{config.version}")
    logger.info(f"Started at {datetime.now()}")
    logger.info("=" * 80)

    return logger


class StrategySystem:
    """策略系统主类"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 初始化组件
        self.data_manager = DataManager(config)
        self.factor_engine = FactorEngine(config)
        self.trainer = WalkForwardTrainer(config)

    def run_backtest(self):
        """运行回测"""
        self.logger.info("Starting backtest mode...")

        # 1. 数据获取
        self.logger.info("Step 1: Loading data...")
        df_daily = self._load_data()

        # 2. 数据质量检查
        self.logger.info("Step 2: Checking data quality...")
        quality_report = DataQuality.generate_report(df_daily)
        self.logger.info(f"Data shape: {quality_report['shape']}")
        self.logger.info(f"Memory usage: {quality_report['memory_usage_mb']:.2f} MB")

        # 3. 因子计算
        self.logger.info("Step 3: Calculating factors...")
        df_with_factors = self.factor_engine.calculate_all_factors(df_daily)

        # 4. 添加标签
        self.logger.info("Step 4: Generating labels...")
        df_with_labels = LabelGenerator.add_labels(df_with_factors, self.config)

        # 5. 模型训练和预测(Walk-Forward)
        self.logger.info("Step 5: Training models (Walk-Forward)...")
        df_predictions = self.trainer.walk_forward_train(df_with_labels)

        # 合并预测结果
        df_final = df_with_factors.merge(
            df_predictions[['ts_code', 'trade_date', 'ml_score']],
            on=['ts_code', 'trade_date'],
            how='left'
        )

        # 6. 运行回测
        self.logger.info("Step 6: Running backtest...")
        backtest_engine = BacktestEngine(self.config)

        # 获取指数数据
        index_data = self.data_manager.get_index_data(
            self.config.backtest.benchmark,
            self.config.backtest.start_date
        )

        results = backtest_engine.run(
            df_with_scores=df_final,
            price_data=df_daily,
            index_data=index_data
        )

        # 7. 生成报告
        self.logger.info("Step 7: Generating report...")
        self._print_metrics(results['metrics'])

        backtest_engine.generate_report()
        backtest_engine.plot_results(results['metrics'])

        # 生成增强版报告(中文)
        self.logger.info("Generating enhanced Chinese reports...")
        from enhanced_report import EnhancedReportGenerator
        enhanced_report = EnhancedReportGenerator(self.config)
        enhanced_report.generate_complete_report(
            results['results'],
            results['metrics'],
            self.config.backtest.output_dir
        )

        # 8. 特征重要性分析
        if self.config.backtest.plot_factor_importance:
            self.logger.info("Step 8: Analyzing feature importance...")
            self._analyze_feature_importance()

        self.logger.info("Backtest completed successfully!")
        return results

    def run_live(self):
        """运行实盘"""
        self.logger.info("Starting live trading mode...")

        if not self.config.system.live_trading:
            self.logger.warning("Live trading is disabled in config")
            return

        # 1. 加载最新数据
        self.logger.info("Loading latest data...")
        df_daily = self._load_data(incremental=True)

        # 2. 计算因子
        self.logger.info("Calculating factors...")
        df_with_factors = self.factor_engine.calculate_all_factors(df_daily)

        # 3. 加载或训练模型
        self.logger.info("Loading model...")
        model = self._load_or_train_model(df_with_factors)

        # 4. 生成预测
        self.logger.info("Generating predictions...")
        latest_date = df_with_factors['trade_date'].max()
        df_latest = df_with_factors[df_with_factors['trade_date'] == latest_date]

        X, _ = self.trainer.prepare_data(df_latest)
        predictions = model.predict(X)

        df_latest['ml_score'] = predictions

        # 5. 选股
        self.logger.info("Selecting stocks...")
        strategy = Strategy(self.config)
        
        # 根据配置决定使用哪种选股方法
        if self.config.strategy.use_composite_score:
            selected = strategy.select_stocks_live(df_latest, latest_date)
        else:
            selected = strategy.select_stocks(df_latest, latest_date)
        
        selected = strategy.calculate_weights(selected)

        # 6. 生成交易指令
        self.logger.info("Generating trade orders...")
        trader = RealTimeTrader(self.config)

        # 获取当前持仓
        current_positions = trader.get_positions()

        # 计算目标持仓与当前持仓的差异
        trades = self._generate_trade_orders(selected, current_positions)

        # 7. 执行交易
        self.logger.info(f"Executing {len(trades)} trades...")
        results = trader.execute_trades(trades)

        # 8. 记录结果
        self._save_live_results(selected, trades, results)

        # 9. 生成实盘推荐看板
        self.logger.info("Generating live recommendation report...")
        from enhanced_report import EnhancedReportGenerator
        enhanced_report = EnhancedReportGenerator(self.config)
        enhanced_report.generate_live_recommendation_report(
            selected, 
            self.config.backtest.output_dir
        )

        self.logger.info("Live trading completed!")
        return results

    def run_train_only(self):
        """仅训练模式"""
        self.logger.info("Starting train-only mode...")

        # 1. 加载数据
        df_daily = self._load_data()

        # 2. 计算因子
        df_with_factors = self.factor_engine.calculate_all_factors(df_daily)

        # 3. 添加标签
        df_with_labels = LabelGenerator.add_labels(df_with_factors, self.config)

        # 4. 训练模型
        self.logger.info("Training model...")

        if self.config.model.use_ensemble:
            model = EnsembleModel(self.config)
            model.build_models()
        else:
            model = MLModel(self.config)
            model.build_model()

        # 准备训练数据
        X, y = self.trainer.prepare_data(df_with_labels)

        # 去除NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        # 训练
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = model.train(X_train, y_train, X_val, y_val)

        # 5. 保存模型
        model_path = Path(self.config.system.cache_factors) / 'latest_model.pkl'
        model.save(str(model_path))

        # 6. 特征重要性
        importance = model.get_feature_importance()
        self.logger.info(f"Top 10 features:\n{importance.head(10)}")

        # 保存特征重要性
        importance.to_csv(Path(self.config.backtest.output_dir) / 'feature_importance.csv', index=False)

        self.logger.info("Training completed!")
        return results

    def _load_data(self, incremental: bool = False) -> pd.DataFrame:
        """加载数据"""
        if incremental and self.config.data.incremental_update:
            df = self.data_manager.incremental_update()
        else:
            df = self.data_manager.get_daily_data(
                start_date=self.config.data.start_date,
                end_date=self.config.data.end_date
            )

        # 获取行业信息
        industry_df = self.data_manager.get_industry_data()
        df = df.merge(industry_df[['ts_code', 'industry']], on='ts_code', how='left')

        return df

    def _load_or_train_model(self, df: pd.DataFrame):
        """加载或训练模型"""
        model_path = Path(self.config.system.cache_factors) / 'latest_model.pkl'

        if model_path.exists():
            self.logger.info("Loading existing model...")
            model = MLModel(self.config)
            model.load(str(model_path))
        else:
            self.logger.info("Training new model...")
            # 训练新模型
            df_with_labels = LabelGenerator.add_labels(df, self.config)

            if self.config.model.use_ensemble:
                model = EnsembleModel(self.config)
            else:
                model = MLModel(self.config)

            # 简化训练(使用最近的数据)
            df_recent = df_with_labels.tail(50000)
            X, y = self.trainer.prepare_data(df_recent)

            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]

            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model.train(X_train, y_train, X_val, y_val)
            model.save(str(model_path))

        return model

    def _generate_trade_orders(self, target: pd.DataFrame, current: pd.DataFrame) -> list:
        """生成交易指令"""
        trades = []

        # 这里简化处理,实际需要根据当前持仓计算差异
        for _, row in target.iterrows():
            trades.append({
                'code': row['ts_code'],
                'action': 'buy',
                'weight': row['weight'],
                'price': 0,  # 市价
                'shares': 0  # 根据权重计算
            })

        return trades

    def _save_live_results(self, selected: pd.DataFrame, trades: list, results: list):
        """保存实盘结果"""
        output_dir = Path(self.config.backtest.output_dir) / 'live'
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存选股结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        selected.to_csv(output_dir / f'selected_{timestamp}.csv', index=False)

        # 保存交易记录
        if trades:
            import pandas as pd
            pd.DataFrame(trades).to_csv(output_dir / f'trades_{timestamp}.csv', index=False)

        if results:
            import pandas as pd
            pd.DataFrame(results).to_csv(output_dir / f'results_{timestamp}.csv', index=False)

    def _print_metrics(self, metrics: dict):
        """打印性能指标"""
        self.logger.info("=" * 60)
        self.logger.info("Performance Metrics:")
        self.logger.info("-" * 60)
        self.logger.info(f"Total Return:        {metrics['total_return']:>10.2%}")
        self.logger.info(f"Annual Return:       {metrics['annual_return']:>10.2%}")
        self.logger.info(f"Volatility:          {metrics['volatility']:>10.2%}")
        self.logger.info(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        self.logger.info(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        self.logger.info(f"Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
        self.logger.info(f"Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")
        self.logger.info(f"Win Rate:            {metrics['win_rate']:>10.2%}")
        self.logger.info("-" * 60)
        self.logger.info(f"Benchmark Return:    {metrics['benchmark_annual_return']:>10.2%}")
        self.logger.info(f"Excess Return:       {metrics['excess_return']:>10.2%}")
        self.logger.info(f"Information Ratio:   {metrics['information_ratio']:>10.2f}")
        self.logger.info("=" * 60)

    def _analyze_feature_importance(self):
        """分析特征重要性"""
        # 获取最新模型
        model_dates = sorted(self.trainer.models.keys())
        if not model_dates:
            return

        latest_model = self.trainer.models[model_dates[-1]]
        importance = latest_model.get_feature_importance()

        # 绘图
        output_dir = Path(self.config.backtest.output_dir)
        FeatureImportance.plot_importance(
            importance,
            top_n=20,
            save_path=str(output_dir / 'feature_importance.png')
        )

        # 保存
        importance.to_csv(output_dir / 'feature_importance.csv', index=False)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Multi-Factor ML Stock Selection Strategy')
    parser.add_argument('--mode', type=str, default='backtest',
                       choices=['backtest', 'live', 'train'],
                       help='Running mode')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')

    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()

    # 验证配置
    config.validate()

    # 设置日志
    logger = setup_logging(config)

    try:
        # 创建策略系统
        system = StrategySystem(config)

        # 运行
        if args.mode == 'backtest':
            results = system.run_backtest()
        elif args.mode == 'live':
            results = system.run_live()
        elif args.mode == 'train':
            results = system.run_train_only()
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        logger.info("Program completed successfully!")

    except Exception as e:
        logger.error(f"Program failed with error: {e}", exc_info=True)
        raise

    finally:
        logger.info(f"Program ended at {datetime.now()}")


if __name__ == '__main__':
    main()