"""
回测引擎
用于策略回测和性能评估
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)


class BacktestEngine:
    """回测引擎"""

    def __init__(self, config):
        self.config = config
        self.results = {
            'dates': [],
            'portfolio_values': [],
            'positions': [],
            'trades': [],
            'cash': []
        }

    def run(
        self,
        df_with_scores: pd.DataFrame,
        price_data: pd.DataFrame,
        index_data: pd.DataFrame = None
    ) -> Dict:
        """
        运行回测

        Args:
            df_with_scores: 包含ML评分的数据
            price_data: 价格数据
            index_data: 指数数据(用于择时)

        Returns:
            回测结果字典
        """
        from strategy import Strategy, RiskManager, MarketTiming, SentimentAnalyzer, PortfolioManager

        logger.info("Starting backtest...")

        # 初始化组件
        strategy = Strategy(self.config)
        risk_manager = RiskManager(self.config)
        timing = MarketTiming(self.config)
        sentiment = SentimentAnalyzer(self.config)
        portfolio = PortfolioManager(self.config)

        # 获取交易日期
        dates = sorted(df_with_scores['trade_date'].unique())
        start_date = self.config.backtest.start_date
        if start_date:
            dates = [d for d in dates if d >= start_date]

        # 逐日回测
        trading_day_count = 0
        for i, current_date in enumerate(dates):
            # 获取当前价格
            price_date = price_data[price_data['trade_date'] == current_date]
            current_prices = dict(zip(price_date['ts_code'], price_date['close']))

            if not current_prices:
                continue

            # 择时信号
            if index_data is not None:
                market_signal = timing.get_market_signal(index_data, current_date)
            else:
                market_signal = 1.0

            # 风控检查
            portfolio_value = portfolio.get_portfolio_value(current_prices)

            # 组合止损
            if risk_manager.check_portfolio_risk(portfolio_value, self.config.backtest.initial_capital, current_date):
                logger.warning(f"Portfolio stop loss triggered on {current_date}")
                # 清仓
                for code in list(portfolio.positions.keys()):
                    shares = portfolio.positions[code]['shares']
                    price = current_prices.get(code, portfolio.positions[code]['cost'])
                    portfolio._sell_stock(code, shares, price, current_date, reason='组合止损')
                trading_day_count += 1
                continue

            # 个股止损止盈
            to_sell = []
            stop_loss_list = risk_manager.check_stop_loss(portfolio.positions, current_prices)
            take_profit_list = risk_manager.check_take_profit(portfolio.positions, current_prices)
            holding_period_list = risk_manager.check_holding_period(current_date)

            to_sell.extend(stop_loss_list)
            to_sell.extend(take_profit_list)
            to_sell.extend(holding_period_list)

            for code in set(to_sell):
                if code in portfolio.positions:
                    shares = portfolio.positions[code]['shares']
                    price = current_prices.get(code, portfolio.positions[code]['cost'])

                    # 确定卖出原因
                    if code in stop_loss_list:
                        reason = '止损'
                    elif code in take_profit_list:
                        reason = '止盈'
                    elif code in holding_period_list:
                        reason = '持仓期满'
                    else:
                        reason = '风控卖出'

                    portfolio._sell_stock(code, shares, price, current_date, reason=reason)

                    # 更新持仓日期
                    if code in risk_manager.position_entry_dates:
                        del risk_manager.position_entry_dates[code]

            # 判断是否调仓
            if strategy.should_rebalance(current_date, trading_day_count):
                # 选股
                selected_stocks = strategy.select_stocks(df_with_scores, current_date)

                if len(selected_stocks) > 0:
                    # 舆情过滤
                    selected_stocks = sentiment.apply_sentiment_filter(selected_stocks, current_date)

                    # 检查并过滤涨停板股票
                    selected_stocks = self._filter_limit_up_stocks(
                        selected_stocks, price_data, current_date
                    )

                    # 计算权重
                    selected_stocks = strategy.calculate_weights(selected_stocks)

                    # 应用择时信号
                    selected_stocks['weight'] = selected_stocks['weight'] * market_signal

                    # 更新持仓
                    trades = portfolio.update_positions(selected_stocks, current_prices, current_date)

                    # 记录新持仓的开始日期
                    for _, row in selected_stocks.iterrows():
                        code = row['ts_code']
                        if code not in risk_manager.position_entry_dates:
                            risk_manager.position_entry_dates[code] = current_date

            # 记录结果
            self.results['dates'].append(current_date)
            self.results['portfolio_values'].append(portfolio_value)
            self.results['cash'].append(portfolio.cash)
            self.results['positions'].append(
                portfolio.get_positions_df(current_prices, current_date).copy()
            )

            # 进度
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i+1}/{len(dates)}, Portfolio Value: {portfolio_value:,.2f}")

            trading_day_count += 1

        # 计算性能指标
        metrics = self.calculate_metrics(price_data, index_data)

        logger.info("Backtest completed")
        return {
            'results': self.results,
            'metrics': metrics,
            'trades': portfolio.trades
        }

    def _filter_limit_up_stocks(
        self,
        selected_stocks: pd.DataFrame,
        price_data: pd.DataFrame,
        current_date: str
    ) -> pd.DataFrame:
        """
        过滤涨停板股票(无法买入)

        Args:
            selected_stocks: 选中的股票
            price_data: 价格数据
            current_date: 当前日期

        Returns:
            过滤后的股票列表
        """
        if len(selected_stocks) == 0:
            return selected_stocks

        # 获取前一交易日数据
        dates_list = sorted(price_data['trade_date'].unique())
        try:
            current_idx = dates_list.index(current_date)
            if current_idx == 0:
                return selected_stocks

            prev_date = dates_list[current_idx - 1]
        except (ValueError, IndexError):
            return selected_stocks

        # 获取当日和前一日价格
        current_prices = price_data[price_data['trade_date'] == current_date].set_index('ts_code')
        prev_prices = price_data[price_data['trade_date'] == prev_date].set_index('ts_code')

        filtered_stocks = []
        limit_up_count = 0

        for _, stock in selected_stocks.iterrows():
            code = stock['ts_code']

            # 检查是否涨停
            if code in current_prices.index and code in prev_prices.index:
                current_close = current_prices.loc[code, 'close']
                prev_close = prev_prices.loc[code, 'close']

                # 计算涨幅
                pct_change = (current_close - prev_close) / prev_close

                # ST股票涨停5%, 普通股票涨停10%
                # 考虑浮点数精度,使用9.8%和4.8%作为阈值
                is_st = code.startswith('ST') or 'ST' in stock.get('name', '')
                limit_threshold = 0.048 if is_st else 0.098

                if pct_change >= limit_threshold:
                    limit_up_count += 1
                    logger.warning(f"过滤涨停股: {code} {stock.get('name', '')} 涨幅{pct_change:.2%}")
                    continue

            filtered_stocks.append(stock)

        if limit_up_count > 0:
            logger.info(f"共过滤涨停股票: {limit_up_count} 只, 剩余: {len(filtered_stocks)} 只")

        return pd.DataFrame(filtered_stocks) if filtered_stocks else pd.DataFrame()

    def calculate_metrics(
        self,
        price_data: pd.DataFrame,
        index_data: pd.DataFrame = None
    ) -> Dict:
        """计算性能指标"""
        # 转换为DataFrame
        df = pd.DataFrame({
            'date': self.results['dates'],
            'portfolio_value': self.results['portfolio_values']
        })

        # 计算收益率
        df['return'] = df['portfolio_value'].pct_change()
        df['cumulative_return'] = (1 + df['return']).cumprod() - 1

        # 基准收益
        if index_data is not None:
            benchmark_dates = df['date'].values
            index_data = index_data[index_data['trade_date'].isin(benchmark_dates)]
            index_data = index_data.sort_values('trade_date')

            benchmark_return = index_data['close'].pct_change()
            benchmark_cumulative = (1 + benchmark_return).cumprod() - 1

            df['benchmark_return'] = benchmark_return.values
            df['benchmark_cumulative'] = benchmark_cumulative.values
            df['excess_return'] = df['return'] - df['benchmark_return']
        else:
            df['benchmark_return'] = 0
            df['benchmark_cumulative'] = 0
            df['excess_return'] = df['return']

        # 回撤
        df['cumulative_max'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['portfolio_value'] - df['cumulative_max']) / df['cumulative_max']

        # 统计指标
        total_return = df['cumulative_return'].iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(df)) - 1

        volatility = df['return'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        max_drawdown = df['drawdown'].min()

        win_rate = (df['return'] > 0).sum() / len(df)

        # 基准比较
        if index_data is not None:
            benchmark_total_return = df['benchmark_cumulative'].iloc[-1]
            benchmark_annual_return = (1 + benchmark_total_return) ** (252 / len(df)) - 1
            excess_return = annual_return - benchmark_annual_return

            # 信息比率
            excess_std = df['excess_return'].std() * np.sqrt(252)
            information_ratio = excess_return / excess_std if excess_std > 0 else 0
        else:
            benchmark_annual_return = 0
            excess_return = annual_return
            information_ratio = 0

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'benchmark_annual_return': benchmark_annual_return,
            'excess_return': excess_return,
            'information_ratio': information_ratio,
            'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown < 0 else 0,
            'sortino_ratio': self._calculate_sortino(df['return'], annual_return),
        }

        return metrics

    def _calculate_sortino(self, returns: pd.Series, target_return: float = 0) -> float:
        """计算Sortino比率"""
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return 0

        downside_std = downside_returns.std() * np.sqrt(252)
        annual_return = returns.mean() * 252

        return annual_return / downside_std if downside_std > 0 else 0

    def generate_report(self, output_dir: str = None):
        """生成回测报告"""
        output_dir = output_dir or self.config.backtest.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存结果
        if self.config.backtest.save_trades:
            trades_df = pd.DataFrame(self.results.get('trades', []))
            trades_df.to_csv(output_path / 'trades.csv', index=False)

        if self.config.backtest.save_positions:
            # 保存每日持仓
            all_positions = []
            for date, positions in zip(self.results['dates'], self.results['positions']):
                if not positions.empty:
                    positions['date'] = date
                    all_positions.append(positions)

            if all_positions:
                positions_df = pd.concat(all_positions, ignore_index=True)
                positions_df.to_csv(output_path / 'positions.csv', index=False)

        if self.config.backtest.save_metrics:
            # 保存每日净值
            equity_df = pd.DataFrame({
                'date': self.results['dates'],
                'portfolio_value': self.results['portfolio_values'],
                'cash': self.results['cash']
            })
            equity_df.to_csv(output_path / 'equity_curve.csv', index=False)

        logger.info(f"Report saved to {output_path}")

    def plot_results(self, metrics: Dict, output_dir: str = None):
        """绘制回测结果图"""
        output_dir = output_dir or self.config.backtest.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            'date': pd.to_datetime(self.results['dates']),
            'portfolio_value': self.results['portfolio_values']
        })

        df['return'] = df['portfolio_value'].pct_change()
        df['cumulative_return'] = (1 + df['return']).cumprod() - 1

        # 回撤
        df['cumulative_max'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['portfolio_value'] - df['cumulative_max']) / df['cumulative_max']

        # 设置样式
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # 1. 权益曲线
        if self.config.backtest.plot_equity_curve:
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(df['date'], df['portfolio_value'], linewidth=2, label='Portfolio Value')
            ax.axhline(y=self.config.backtest.initial_capital, color='r',
                      linestyle='--', alpha=0.5, label='Initial Capital')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Portfolio Value (¥)', fontsize=12)
            ax.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'equity_curve.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 2. 回撤曲线
        if self.config.backtest.plot_drawdown:
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.fill_between(df['date'], df['drawdown'] * 100, 0, alpha=0.3, color='red')
            ax.plot(df['date'], df['drawdown'] * 100, linewidth=1, color='red')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Drawdown (%)', fontsize=12)
            ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'drawdown.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 3. 收益率分布
        if self.config.backtest.plot_returns_dist:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # 直方图
            ax1.hist(df['return'].dropna() * 100, bins=50, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Daily Return (%)', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
            ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax1.grid(True, alpha=0.3)

            # QQ图
            from scipy import stats
            stats.probplot(df['return'].dropna(), dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path / 'returns_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 4. 性能指标表
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('tight')
        ax.axis('off')

        metrics_data = [
            ['Total Return', f"{metrics['total_return']:.2%}"],
            ['Annual Return', f"{metrics['annual_return']:.2%}"],
            ['Volatility', f"{metrics['volatility']:.2%}"],
            ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"],
            ['Sortino Ratio', f"{metrics['sortino_ratio']:.2f}"],
            ['Max Drawdown', f"{metrics['max_drawdown']:.2%}"],
            ['Calmar Ratio', f"{metrics['calmar_ratio']:.2f}"],
            ['Win Rate', f"{metrics['win_rate']:.2%}"],
            ['Benchmark Return', f"{metrics['benchmark_annual_return']:.2%}"],
            ['Excess Return', f"{metrics['excess_return']:.2%}"],
            ['Information Ratio', f"{metrics['information_ratio']:.2f}"],
        ]

        table = ax.table(cellText=metrics_data, colLabels=['Metric', 'Value'],
                        cellLoc='left', loc='center',
                        colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        # 设置表格样式
        for i in range(len(metrics_data) + 1):
            if i == 0:
                table[(i, 0)].set_facecolor('#4CAF50')
                table[(i, 1)].set_facecolor('#4CAF50')
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            else:
                if i % 2 == 0:
                    table[(i, 0)].set_facecolor('#f1f1f2')
                    table[(i, 1)].set_facecolor('#f1f1f2')

        plt.title('Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(output_path / 'metrics_table.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Plots saved to {output_path}")


class RealTimeTrader:
    """实盘交易器"""

    def __init__(self, config):
        self.config = config
        self.api = None

        if config.system.live_trading:
            self._init_api()

    def _init_api(self):
        """初始化交易API"""
        if self.config.system.trade_api == 'easytrader':
            try:
                import easytrader
                self.api = easytrader.use('ths')  # 同花顺
                # self.api.connect()  # 连接客户端
                logger.info("EasyTrader API initialized")
            except Exception as e:
                logger.error(f"Failed to initialize EasyTrader: {e}")
        else:
            logger.warning(f"Unknown trade API: {self.config.system.trade_api}")

    def execute_trades(self, trades: List[Dict]) -> List[Dict]:
        """
        执行交易

        Args:
            trades: 交易列表

        Returns:
            执行结果列表
        """
        if self.config.system.dry_run:
            logger.info("Dry run mode - trades not executed")
            return trades

        if self.api is None:
            logger.error("Trading API not initialized")
            return []

        results = []

        for trade in trades:
            try:
                if trade['action'] == 'buy':
                    result = self.api.buy(
                        trade['code'],
                        price=trade['price'],
                        amount=trade['shares']
                    )
                elif trade['action'] == 'sell':
                    result = self.api.sell(
                        trade['code'],
                        price=trade['price'],
                        amount=trade['shares']
                    )
                else:
                    continue

                results.append({**trade, 'result': result})
                logger.info(f"Executed: {trade['action']} {trade['code']}")

            except Exception as e:
                logger.error(f"Failed to execute trade {trade}: {e}")
                results.append({**trade, 'result': 'failed', 'error': str(e)})

        return results

    def get_positions(self) -> pd.DataFrame:
        """获取当前持仓"""
        if self.api is None:
            return pd.DataFrame()

        try:
            positions = self.api.position
            return pd.DataFrame(positions)
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return pd.DataFrame()

    def get_balance(self) -> Dict:
        """获取账户余额"""
        if self.api is None:
            return {}

        try:
            balance = self.api.balance
            return balance
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return {}