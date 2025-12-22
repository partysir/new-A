"""
增强版报告生成模块
支持中文可视化面板和详细持仓追踪
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class EnhancedReportGenerator:
    """增强版报告生成器"""

    def __init__(self, config):
        self.config = config

    def generate_daily_position_report(self, results: dict, output_dir: str):
        """
        生成每日持仓明细报告(中文)

        包含字段:
        - 日期、股票代码、操作、股数、价格、成本、入场日期
        - 当前市值、盈亏、盈亏率、评分、持仓天数、原因
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 整理每日持仓数据
        daily_positions = []

        for i, (date, positions_df) in enumerate(zip(results['dates'], results['positions'])):
            if positions_df.empty:
                continue

            for _, pos in positions_df.iterrows():
                record = {
                    '日期': date,
                    '股票代码': pos.get('ts_code', ''),
                    '股票名称': pos.get('name', ''),
                    '操作': '持仓',
                    '持仓股数': int(pos.get('shares', 0)),
                    '当前价格': round(pos.get('price', 0), 2),
                    '成本价': round(pos.get('cost', 0), 2),
                    '入场日期': pos.get('entry_date', ''),
                    '当前市值': round(pos.get('value', 0), 2),
                    '持仓盈亏': round(pos.get('pnl', 0), 2),
                    '盈亏率': f"{pos.get('pnl_pct', 0) * 100:.2f}%",
                    '评分': round(pos.get('score', 0), 4),
                    '持仓天数': int(pos.get('holding_days', 0)),
                    '备注': ''
                }
                daily_positions.append(record)

        # 整理交易记录
        trades_list = []
        if 'trades' in results:
            for trade in results['trades']:
                record = {
                    '日期': trade.get('date', ''),
                    '股票代码': trade.get('code', ''),
                    '股票名称': trade.get('name', ''),
                    '操作': '买入' if trade.get('action') == 'buy' else '卖出',
                    '成交股数': int(trade.get('shares', 0)),
                    '成交价格': round(trade.get('price', 0), 2),
                    '成交金额': round(trade.get('amount', 0), 2),
                    '手续费': round(trade.get('commission', 0), 2),
                    '原因': trade.get('reason', ''),
                    '评分': round(trade.get('score', 0), 4) if 'score' in trade else ''
                }
                trades_list.append(record)

        # 保存持仓明细
        if daily_positions:
            positions_df = pd.DataFrame(daily_positions)
            positions_df.to_excel(
                output_path / '每日持仓明细.xlsx',
                index=False,
                engine='openpyxl'
            )
            logger.info(f"每日持仓明细已保存: {len(positions_df)} 条记录")

        # 保存交易记录
        if trades_list:
            trades_df = pd.DataFrame(trades_list)
            trades_df.to_excel(
                output_path / '交易明细.xlsx',
                index=False,
                engine='openpyxl'
            )
            logger.info(f"交易明细已保存: {len(trades_df)} 条记录")

        return positions_df if daily_positions else None, trades_df if trades_list else None

    def generate_daily_dashboard(self, results: dict, output_dir: str):
        """生成每日持仓看板"""
        output_path = Path(output_dir)

        # 按日期分组统计
        daily_summary = []

        for i, (date, positions_df, cash) in enumerate(zip(
                results['dates'],
                results['positions'],
                results['cash']
        )):
            if i == 0:
                continue

            # 统计持仓
            n_positions = len(positions_df) if not positions_df.empty else 0

            if not positions_df.empty:
                total_value = positions_df['value'].sum() if 'value' in positions_df.columns else 0
                total_pnl = positions_df['pnl'].sum() if 'pnl' in positions_df.columns else 0
                avg_pnl_pct = positions_df['pnl_pct'].mean() if 'pnl_pct' in positions_df.columns else 0
            else:
                total_value = 0
                total_pnl = 0
                avg_pnl_pct = 0

            portfolio_value = results['portfolio_values'][i]
            prev_value = results['portfolio_values'][i - 1]
            daily_return = (portfolio_value - prev_value) / prev_value

            summary = {
                '日期': date,
                '持仓数量': n_positions,
                '持仓市值': round(total_value, 2),
                '现金余额': round(cash, 2),
                '组合总值': round(portfolio_value, 2),
                '持仓盈亏': round(total_pnl, 2),
                '平均盈亏率': f"{avg_pnl_pct * 100:.2f}%",
                '当日收益率': f"{daily_return * 100:.2f}%",
                '累计收益率': f"{(portfolio_value / results['portfolio_values'][0] - 1) * 100:.2f}%"
            }
            daily_summary.append(summary)

        # 保存看板
        if daily_summary:
            dashboard_df = pd.DataFrame(daily_summary)
            dashboard_df.to_excel(
                output_path / '每日持仓看板.xlsx',
                index=False,
                engine='openpyxl'
            )
            logger.info(f"每日持仓看板已保存")

        return dashboard_df if daily_summary else None

    def generate_trade_analysis(self, trades_df: pd.DataFrame, output_dir: str):
        """生成交易分析报告"""
        if trades_df is None or len(trades_df) == 0:
            return

        output_path = Path(output_dir)

        # 买入分析
        buy_trades = trades_df[trades_df['操作'] == '买入'].copy()

        if len(buy_trades) > 0:
            # 检查涨停板买入情况
            buy_trades['涨停板'] = buy_trades.apply(
                lambda x: self._check_limit_up(x['股票代码'], x['日期'], x['成交价格']),
                axis=1
            )

            limit_up_count = buy_trades['涨停板'].sum()
            limit_up_rate = limit_up_count / len(buy_trades)

            logger.info(f"买入涨停板统计: {limit_up_count}/{len(buy_trades)} = {limit_up_rate:.1%}")

            # 保存涨停板买入明细
            if limit_up_count > 0:
                limit_up_df = buy_trades[buy_trades['涨停板'] == True].copy()
                limit_up_df.to_excel(
                    output_path / '涨停板买入明细.xlsx',
                    index=False,
                    engine='openpyxl'
                )

        # 卖出原因分析
        sell_trades = trades_df[trades_df['操作'] == '卖出'].copy()

        if len(sell_trades) > 0:
            reason_stats = sell_trades['原因'].value_counts()

            # 保存卖出原因统计
            reason_df = pd.DataFrame({
                '卖出原因': reason_stats.index,
                '次数': reason_stats.values,
                '占比': [f"{v / len(sell_trades) * 100:.1f}%" for v in reason_stats.values]
            })

            reason_df.to_excel(
                output_path / '卖出原因统计.xlsx',
                index=False,
                engine='openpyxl'
            )

    def _check_limit_up(self, code: str, date: str, price: float) -> bool:
        """
        检查是否为涨停板买入

        简化版本:基于价格判断
        实际应该查询历史数据对比前一日收盘价
        """
        # TODO: 实际实现应该查询前一日收盘价,判断涨幅是否≥9.9%
        # 这里简化处理,返回False
        return False

    def plot_position_analysis(self, positions_df: pd.DataFrame, output_dir: str):
        """绘制持仓分析图表"""
        if positions_df is None or len(positions_df) == 0:
            return

        output_path = Path(output_dir)

        # 1. 持仓数量变化
        daily_count = positions_df.groupby('日期').size()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(pd.to_datetime(daily_count.index), daily_count.values, linewidth=2)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('持仓数量', fontsize=12)
        ax.set_title('持仓数量变化', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / '持仓数量变化.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 持仓天数分布
        holding_days = positions_df['持仓天数'].astype(int)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(holding_days, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('持仓天数', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title('持仓天数分布', fontsize=14, fontweight='bold')
        ax.axvline(holding_days.mean(), color='red', linestyle='--',
                   label=f'平均: {holding_days.mean():.1f}天')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / '持仓天数分布.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 盈亏率分布
        pnl_rates = positions_df['盈亏率'].str.rstrip('%').astype(float)

        fig, ax = plt.subplots(figsize=(12, 6))

        # 分别统计正负盈亏
        positive_rates = pnl_rates[pnl_rates >= 0]
        negative_rates = pnl_rates[pnl_rates < 0]

        # 绘制直方图
        if len(negative_rates) > 0:
            ax.hist(negative_rates, bins=25, edgecolor='black', alpha=0.7, color='red', label='亏损')
        if len(positive_rates) > 0:
            ax.hist(positive_rates, bins=25, edgecolor='black', alpha=0.7, color='green', label='盈利')

        ax.set_xlabel('盈亏率 (%)', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title('持仓盈亏率分布', fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=2)
        ax.axvline(pnl_rates.mean(), color='blue', linestyle='--',
                   label=f'平均: {pnl_rates.mean():.2f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / '盈亏率分布.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("持仓分析图表已生成")

    def generate_complete_report(self, results: dict, metrics: dict, output_dir: str):
        """生成完整报告"""
        logger.info("开始生成增强版报告...")

        # 1. 生成每日持仓明细
        positions_df, trades_df = self.generate_daily_position_report(results, output_dir)

        # 2. 生成每日看板
        dashboard_df = self.generate_daily_dashboard(results, output_dir)

        # 3. 生成交易分析
        if trades_df is not None:
            self.generate_trade_analysis(trades_df, output_dir)

        # 4. 生成图表分析
        if positions_df is not None:
            self.plot_position_analysis(positions_df, output_dir)

        # 5. 生成绩效汇总
        self.generate_performance_summary(metrics, results, output_dir)

        logger.info(f"✅ 增强版报告生成完成! 位置: {output_dir}")

    def generate_performance_summary(self, metrics: dict, results: dict, output_dir: str):
        """生成绩效汇总报告"""
        output_path = Path(output_dir)

        # 整理指标
        summary = {
            '指标': [
                '总收益率',
                '年化收益率',
                '年化波动率',
                '夏普比率',
                '索提诺比率',
                '最大回撤',
                '卡尔玛比率',
                '信息比率',
                '胜率',
                '基准收益率',
                '超额收益'
            ],
            '数值': [
                f"{metrics.get('total_return', 0) * 100:.2f}%",
                f"{metrics.get('annual_return', 0) * 100:.2f}%",
                f"{metrics.get('volatility', 0) * 100:.2f}%",
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                f"{metrics.get('sortino_ratio', 0):.2f}",
                f"{metrics.get('max_drawdown', 0) * 100:.2f}%",
                f"{metrics.get('calmar_ratio', 0):.2f}",
                f"{metrics.get('information_ratio', 0):.2f}",
                f"{metrics.get('win_rate', 0) * 100:.1f}%",
                f"{metrics.get('benchmark_annual_return', 0) * 100:.2f}%",
                f"{metrics.get('excess_return', 0) * 100:.2f}%"
            ]
        }

        summary_df = pd.DataFrame(summary)
        summary_df.to_excel(
            output_path / '绩效汇总.xlsx',
            index=False,
            engine='openpyxl'
        )

        logger.info("绩效汇总已保存")