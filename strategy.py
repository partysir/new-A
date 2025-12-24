"""
策略模块 - 修复版 v3.0
核心改进:
1. 彻底修复空仓死锁 - 使用指数信号而非净值回撤
2. 解决涨停板悖论 - 预测次日开盘收益,避免追高
3. 优化盈亏比 - 非对称止盈止损
4. 降低换手率 - 持仓优化逻辑
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class Strategy:
    """策略主类 - 优化选股逻辑"""
    
    def __init__(self, config):
        self.config = config

    def select_stocks(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        改进的选股逻辑 - 解决涨停板悖论
        核心思想: 预测次日开盘后的收益,而非当日收盘
        """
        df_date = df[df['trade_date'] == date].copy()
        if df_date.empty: 
            return pd.DataFrame()

        # === 第一层: 可交易性过滤 ===
        
        # 1. 排除微盘股(市值后20%)
        if 'circ_mv' in df_date.columns:
            cap_threshold = df_date['circ_mv'].quantile(0.20)
            df_date = df_date[df_date['circ_mv'] > cap_threshold]
        
        # 2. 【关键修改】排除"明天买不到"的股票
        # 不仅过滤涨停,还要过滤"已经涨太多"的股票
        if 'pct_chg' in df_date.columns and 'is_st' in df_date.columns:
            df_date = df_date[
                (df_date['pct_chg'] < 7.0) &  # 涨幅<7%,避免次日高开买不到
                (df_date['pct_chg'] > -9.0) &  # 跌幅<9%,避免买到垃圾
                (df_date['is_st'] == 0)
            ]
            logger.info(f"[{date}] 过滤涨幅>7%的股票,剩余{len(df_date)}只")
        
        # 3. 流动性过滤
        if 'amount' in df_date.columns:
            df_date = df_date[df_date['amount'] > 1e7]  # 成交额>1000万

        # 4. 【新增】避免追高 - 过滤短期暴涨股
        if 'momentum_5' in df_date.columns:
            # 5日涨幅超过20%的不买(可能是游资炒作)
            df_date['momentum_5'] = df_date['close'].pct_change(5)
            df_date = df_date[df_date['momentum_5'] < 0.20]

        # === 第二层: 评分过滤 ===
        
        # 根据选股方法决定是否使用硬阈值
        if self.config.strategy.selection_method in ['score', 'threshold']:
            score_threshold = self.config.strategy.score_threshold
            df_date = df_date[df_date['ml_score'] >= score_threshold]
        
        # 如果过滤后为空,放宽限制
        if df_date.empty:
            df_date = df[df['trade_date'] == date].copy()
            logger.warning(f"[{date}] 过滤后为空,放宽限制")

        # === 第三层: 排序与行业约束 ===
        
        df_date = df_date.sort_values('ml_score', ascending=False)

        # 行业中性化
        if self.config.strategy.max_industry_weight < 1.0:
            df_date = self._apply_industry_constraints(df_date)

        # === 第四层: 最终筛选 ===
        
        # 【关键修改】增加缓冲数量,防止第二天开盘时部分股票涨停买不到
        buffer_multiplier = 1.5  # 多选50%作为备选
        top_n_with_buffer = int(self.config.strategy.top_n * buffer_multiplier)
        selected = df_date.head(top_n_with_buffer)
        
        # 标记优先级
        selected['priority'] = range(1, len(selected) + 1)
        
        if len(selected) == 0:
            logger.warning(f"[{date}] Selected 0 stocks. Check filters.")
        else:
            logger.info(f"[{date}] Selected {len(selected)} stocks (top_n={self.config.strategy.top_n}, with buffer)")
            
        return selected

    def select_stocks_live(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """实盘专用选股逻辑(保持原有逻辑)"""
        df_date = df[df['trade_date'] == date].copy()
        if len(df_date) == 0:
            return pd.DataFrame()

        # 第一层: 硬性风控过滤
        if 'pct_chg' in df_date.columns and 'vol_ma20' in df_date.columns:
            mask_crash = (df_date['pct_chg'] < -7) & (df_date['vol'] > 1.5 * df_date['vol_ma20'])
        else:
            mask_crash = pd.Series([False] * len(df_date), index=df_date.index)
            
        if 'rsi_14' in df_date.columns:
            mask_overbought = df_date['rsi_14'] > 85
        else:
            mask_overbought = pd.Series([False] * len(df_date), index=df_date.index)
        
        df_candidates = df_date[~(mask_crash | mask_overbought)].copy()
        
        # 第二层: ML模型初选
        top_percentile = max(1, int(len(df_candidates) * 0.2))
        df_candidates = df_candidates.sort_values('ml_score', ascending=False).head(top_percentile)

        # 第三层: 实盘指标加权
        def normalize(series):
            if series.max() == series.min():
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - series.min()) / (series.max() - series.min())

        for col in ['smart_money_score', 'trend_energy', 'safety_margin']:
            if col not in df_candidates.columns:
                df_candidates[col] = 0.5

        df_candidates['composite_score'] = (
            0.5 * df_candidates['ml_score'] + 
            0.3 * normalize(df_candidates['smart_money_score']) +
            0.2 * normalize(df_candidates['trend_energy'])
        )

        # 第四层: 最终Top N
        final_selection = df_candidates.sort_values('composite_score', ascending=False).head(self.config.strategy.top_n)
        final_selection['recommend_reason'] = final_selection.apply(self._generate_reason, axis=1)
        
        return final_selection

    def _generate_reason(self, row):
        """生成推荐理由文本"""
        reasons = []
        if row['ml_score'] > 0.8: reasons.append("模型高确信")
        if row.get('smart_money_score', 0) > 1.0: reasons.append("主力资金抢筹")
        if row.get('trend_energy', 0) > 2.0: reasons.append("趋势即将爆发")
        return "+".join(reasons) if reasons else "综合评分优选"

    def calculate_weights(self, selected_stocks: pd.DataFrame) -> pd.DataFrame:
        """计算股票权重(保持原有逻辑)"""
        if selected_stocks.empty:
            return selected_stocks
        
        n_stocks = len(selected_stocks)
        
        if self.config.strategy.weight_method == 'equal':
            weight_per_stock = 1.0 / n_stocks
            selected_stocks['weight'] = weight_per_stock
            
        elif self.config.strategy.weight_method == 'score_weighted':
            if 'ml_score' in selected_stocks.columns:
                scores = selected_stocks['ml_score']
                exp_scores = np.exp(scores - scores.max())
                weights = exp_scores / exp_scores.sum()
                selected_stocks['weight'] = weights
            else:
                selected_stocks['weight'] = 1.0 / n_stocks
                
        elif self.config.strategy.weight_method == 'risk_parity':
            if 'volatility' in selected_stocks.columns:
                vol_weights = 1.0 / (selected_stocks['volatility'] + 1e-9)
                selected_stocks['weight'] = vol_weights / vol_weights.sum()
            else:
                selected_stocks['weight'] = 1.0 / n_stocks
                
        else:
            selected_stocks['weight'] = 1.0 / n_stocks
        
        # 应用单只股票最大权重限制
        selected_stocks['weight'] = selected_stocks['weight'].clip(
            upper=self.config.strategy.max_single_weight
        )
        
        # 重新归一化权重
        total_weight = selected_stocks['weight'].sum()
        if total_weight > 0:
            selected_stocks['weight'] = selected_stocks['weight'] / total_weight
        
        return selected_stocks

    def _apply_industry_constraints(self, df):
        """行业约束"""
        max_per_ind = int(self.config.strategy.top_n * self.config.strategy.max_industry_weight)
        if max_per_ind < 1: 
            max_per_ind = 1
        return df.groupby('industry', group_keys=False).apply(lambda x: x.head(max_per_ind))

    def should_rebalance(self, date: str, trading_day_count: int) -> bool:
        """判断调仓日"""
        freq = self.config.strategy.rebalance_frequency
        if freq == 'daily': 
            return True
        if freq == 'weekly':
            return pd.to_datetime(date).weekday() == 4
        if freq == 'n_days':
            return trading_day_count % self.config.strategy.rebalance_day == 0
        return False


class RiskManager:
    """
    风险管理器 v3.0 - 彻底修复版
    核心改进:
    1. 使用指数信号而非净值回撤控制仓位
    2. 非对称止盈止损(盈亏比2:1)
    3. 持仓优化,避免频繁换手
    """
    
    def __init__(self, config):
        self.config = config
        self.position_entry_scores = {}
        self.position_entry_dates = {}
        
        # 仓位控制参数
        self.current_position_scalar = 1.0
        self.min_position = 0.3  # 【关键】最低仓位30%,永不空仓
        self.max_position = 1.0
        
        # 非对称止盈止损
        self.stop_loss_pct = -0.05  # 止损-5%
        self.take_profit_pct = 0.15  # 止盈+15%(盈亏比3:1)
        self.trailing_stop_pct = 0.10  # 移动止盈,盈利10%后启动

    def check_risk(self, positions: Dict, current_prices: Dict,
                   current_scores: Dict, current_date: str) -> List[Tuple[str, str]]:
        """
        个股风险检查 - 优化版
        重点: 非对称止盈止损 + 持仓优化
        """
        to_sell = []

        for code, pos in positions.items():
            if code not in current_prices: 
                continue

            price = current_prices[code]
            cost = pos['cost']
            pnl_pct = (price - cost) / cost

            if code not in self.position_entry_scores:
                self.position_entry_scores[code] = pos.get('score', 0)

            # === A. 止损(-5%) ===
            if pnl_pct <= self.stop_loss_pct:
                to_sell.append((code, '止损'))
                logger.info(f"[{code}] 触发止损: {pnl_pct:.2%}")
                continue

            # === B. 止盈(+15%) ===
            if pnl_pct >= self.take_profit_pct:
                to_sell.append((code, '止盈'))
                logger.info(f"[{code}] 触发止盈: {pnl_pct:.2%}")
                continue

            # === C. 移动止盈(盈利>10%后,回撤5%就卖) ===
            if pnl_pct >= self.trailing_stop_pct:
                # 记录最高盈利
                if 'max_profit' not in pos:
                    pos['max_profit'] = pnl_pct
                else:
                    pos['max_profit'] = max(pos['max_profit'], pnl_pct)
                
                # 从最高点回撤5%就卖
                drawdown_from_peak = (pnl_pct - pos['max_profit'])
                if drawdown_from_peak <= -0.05:
                    to_sell.append((code, '移动止盈'))
                    logger.info(f"[{code}] 移动止盈: 最高{pos['max_profit']:.2%}, 当前{pnl_pct:.2%}")
                    continue

            # === D. 评分衰减止损 ===
            current_score = current_scores.get(code, 0)
            entry_score = self.position_entry_scores.get(code, 0)

            if entry_score > 0:
                decay = (entry_score - current_score) / entry_score
                if decay > 0.4 and self._get_holding_days(pos, current_date) > 5:
                    to_sell.append((code, '因子衰减'))
                    logger.info(f"[{code}] 因子衰减: {decay:.2%}")
                    continue

            # === E. 低分止损 ===
            if current_score < 0.3:
                to_sell.append((code, '低分止损'))
                logger.info(f"[{code}] 低分止损: score={current_score:.2f}")
                continue

            # === F. 持仓期满(放宽到30天) ===
            holding_days = self._get_holding_days(pos, current_date)
            if holding_days >= 30:
                to_sell.append((code, '持仓期满'))
                logger.info(f"[{code}] 持仓期满: {holding_days}天")
                continue

        return to_sell

    def check_portfolio_risk_v3(self, current_value: float, initial_capital: float,
                                current_date: str, index_data: pd.DataFrame = None) -> Dict:
        """
        【核心改进】基于指数信号的仓位控制
        
        逻辑:
        1. 使用沪深300的20日均线作为仓位信号
        2. 指数在均线上方: 满仓(100%)
        3. 指数在均线下方: 半仓(50%)
        4. 指数跌破均线且回撤>10%: 轻仓(30%)
        5. 永不空仓,避免错过反弹
        """
        
        # 【关键修复】计算当前回撤 - 修正公式
        # 回撤应该基于历史最高净值,而非初始资金
        if not hasattr(self, 'peak_value'):
            self.peak_value = initial_capital
        
        # 更新峰值
        self.peak_value = max(self.peak_value, current_value)
        
        # 正确的回撤计算: (峰值 - 当前值) / 峰值
        drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0
        
        # 确保回撤在合理范围内 [0, 1]
        drawdown = max(0, min(1, drawdown))
        
        # === 方法1: 使用指数信号(推荐) ===
        if index_data is not None and len(index_data) > 0:
            try:
                # 获取当前日期的指数数据
                index_current = index_data[index_data['trade_date'] <= current_date].tail(20)
                
                if len(index_current) >= 20:
                    current_close = index_current.iloc[-1]['close']
                    ma20 = index_current['close'].rolling(20).mean().iloc[-1]
                    
                    # 指数相对均线的位置
                    index_position = (current_close - ma20) / ma20
                    
                    # 根据指数位置调整仓位
                    if index_position > 0.02:  # 指数在均线上方2%
                        target_position = 1.0
                        tier_name = "满仓(指数强势)"
                    elif index_position > -0.02:  # 指数在均线附近
                        target_position = 0.7
                        tier_name = "七成仓(指数震荡)"
                    elif drawdown < 0.10:  # 指数弱势但回撤不大
                        target_position = 0.5
                        tier_name = "半仓(指数弱势)"
                    else:  # 指数弱势且回撤较大
                        target_position = 0.3
                        tier_name = "轻仓(防守)"
                    
                    # 平滑仓位变化(避免频繁调整)
                    position_change = abs(target_position - self.current_position_scalar)
                    if position_change < 0.1:
                        target_position = self.current_position_scalar
                    
                    self.current_position_scalar = target_position
                    
                    message = f"{tier_name}, 指数位置={index_position:.2%}, 回撤={drawdown:.2%}"
                    logger.info(f"[{current_date}] {message}")
                    
                    return {
                        'action': 'normal',
                        'position_scalar': self.current_position_scalar,
                        'tier_name': tier_name,
                        'drawdown': drawdown,
                        'message': message
                    }
            except Exception as e:
                logger.warning(f"指数信号计算失败: {e}, 使用回撤方法")
        
        # === 方法2: 回撤方法(备选) ===
        # 【关键修改】永不空仓,最低保持30%
        if drawdown < 0.05:
            target_position = 1.0
            tier_name = "满仓"
        elif drawdown < 0.10:
            target_position = 0.7
            tier_name = "七成仓"
        elif drawdown < 0.15:
            target_position = 0.5
            tier_name = "半仓"
        else:
            target_position = 0.3  # 最低30%,永不空仓
            tier_name = "轻仓"
        
        self.current_position_scalar = max(target_position, self.min_position)
        
        message = f"{tier_name}, 回撤={drawdown:.2%}"
        logger.info(f"[{current_date}] {message}")
        
        return {
            'action': 'normal',
            'position_scalar': self.current_position_scalar,
            'tier_name': tier_name,
            'drawdown': drawdown,
            'message': message
        }

    def _get_holding_days(self, pos, current_date):
        """计算持仓天数"""
        try:
            entry = pd.to_datetime(str(pos['entry_date']))
            curr = pd.to_datetime(str(current_date))
            return (curr - entry).days
        except:
            return 0

    # 兼容接口
    def check_portfolio_risk(self, current_value, initial_capital, current_date, index_data=None):
        """兼容旧接口"""
        return self.check_portfolio_risk_v3(current_value, initial_capital, current_date, index_data)

    def check_stop_loss(self, positions: dict, current_prices: dict) -> list:
        """兼容接口"""
        stop_loss_list = []
        for code, pos_info in positions.items():
            if code not in current_prices: 
                continue
            cost = pos_info['cost']
            current_price = current_prices[code]
            if cost == 0: 
                continue
            ret = (current_price - cost) / cost
            if ret <= self.stop_loss_pct:
                stop_loss_list.append(code)
        return stop_loss_list

    def check_take_profit(self, positions: dict, current_prices: dict) -> list:
        """兼容接口"""
        take_profit_list = []
        for code, pos_info in positions.items():
            if code not in current_prices: 
                continue
            cost = pos_info['cost']
            current_price = current_prices[code]
            if cost == 0: 
                continue
            ret = (current_price - cost) / cost
            if ret >= self.take_profit_pct:
                take_profit_list.append(code)
        return take_profit_list


class PortfolioManager:
    """
    组合管理器 - 优化版
    核心改进: 减少换手率,持仓优化
    """
    
    def __init__(self, config):
        self.config = config
        self.positions = {}
        self.cash = config.backtest.initial_capital
        self.trades = []
        self.risk_manager = None
        self.buy_cost = config.strategy.commission_rate
        self.sell_cost = config.strategy.commission_rate + config.strategy.stamp_tax

    def set_risk_manager(self, risk_manager):
        self.risk_manager = risk_manager

    def update_positions(self, target_df: pd.DataFrame, current_prices: Dict,
                        current_scores: Dict, date: str, index_data: pd.DataFrame = None) -> List[Dict]:
        """
        更新持仓 - 优化版
        核心: 减少不必要的换手
        """
        new_trades = []

        # 1. 获取当前仓位建议
        portfolio_value = self.get_portfolio_value(current_prices)
        if self.risk_manager:
            risk_status = self.risk_manager.check_portfolio_risk(
                portfolio_value, self.config.backtest.initial_capital, date, index_data
            )
            position_scalar = risk_status['position_scalar']
        else:
            position_scalar = 1.0

        # 2. 【优化】持仓保留逻辑 - 减少换手
        target_codes = set(target_df['ts_code'].values)
        to_sell = []
        
        for code in self.positions.keys():
            # 如果股票仍在目标池中,保留持仓(除非触发风控)
            if code in target_codes:
                continue
            
            # 检查是否需要强制卖出(风控触发)
            pos = self.positions[code]
            price = current_prices.get(code, pos['cost'])
            pnl_pct = (price - pos['cost']) / pos['cost']
            
            # 只有在严重亏损或严重盈利时才卖出
            if pnl_pct < -0.08 or pnl_pct > 0.20:
                to_sell.append(code)
                logger.info(f"[{code}] 强制卖出: pnl={pnl_pct:.2%}")
            else:
                # 否则继续持有,即使不在Top N中(容忍度)
                logger.info(f"[{code}] 容忍持有: pnl={pnl_pct:.2%}")

        # 3. 执行卖出
        for code in to_sell:
            trade = self._sell(code, current_prices.get(code), date, '调仓卖出')
            if trade: 
                new_trades.append(trade)

        # 4. 买入逻辑(应用仓位比例)
        available_cash = max(0, self.cash * 0.95 * position_scalar)

        to_buy = []
        for _, row in target_df.iterrows():
            code = row['ts_code']
            if code not in self.positions:
                to_buy.append(row)

        if not to_buy:
            self.trades.extend(new_trades)
            return new_trades

        # 【优化】按优先级排序,优先买入高分股票
        if 'priority' in target_df.columns:
            to_buy = sorted(to_buy, key=lambda x: x.get('priority', 999))
        
        # 只买入Top N,不买入buffer
        actual_top_n = self.config.strategy.top_n
        to_buy = to_buy[:actual_top_n]

        per_stock_cash = available_cash / len(to_buy)

        for row in to_buy:
            code = row['ts_code']
            price = current_prices.get(code)
            if not price: 
                continue

            shares = int(per_stock_cash / (price * (1 + self.buy_cost)) / 100) * 100

            if shares >= 100:
                trade = self._buy(code, shares, price, date, row.get('ml_score', 0))
                if trade: 
                    new_trades.append(trade)

        self.trades.extend(new_trades)
        return new_trades

    def _buy(self, code, shares, price, date, score):
        """买入"""
        cost = shares * price * (1 + self.buy_cost)
        if cost > self.cash: 
            return None

        self.cash -= cost
        if code not in self.positions:
            self.positions[code] = {
                'shares': shares, 
                'cost': price, 
                'entry_date': date,
                'score': score, 
                'name': code
            }

        return {
            'date': date, 'code': code, 'action': 'buy',
            'shares': shares, 'price': price, 'amount': cost
        }

    def _sell(self, code, price, date, reason):
        """卖出"""
        if not price or code not in self.positions: 
            return None
        pos = self.positions[code]
        revenue = pos['shares'] * price * (1 - self.sell_cost)
        pnl = revenue - (pos['shares'] * pos['cost'])
        self.cash += revenue
        del self.positions[code]
        return {
            'date': date, 'code': code, 'action': 'sell',
            'shares': pos['shares'], 'price': price, 'amount': revenue,
            'pnl': pnl, 'reason': reason
        }

    def get_portfolio_value(self, current_prices):
        """获取组合价值"""
        val = self.cash
        for code, pos in self.positions.items():
            price = current_prices.get(code, pos['cost'])
            val += pos['shares'] * price
        return val

    def get_positions_df(self, current_prices, date):
        """获取持仓DataFrame"""
        data = []
        for code, pos in self.positions.items():
            price = current_prices.get(code, pos['cost'])
            val = pos['shares'] * price
            pnl = (price - pos['cost']) * pos['shares']
            data.append({
                'ts_code': code, 
                'name': pos.get('name'), 
                'shares': pos['shares'],
                'cost': pos['cost'], 
                'price': price, 
                'value': val, 
                'pnl': pnl,
                'entry_date': pos['entry_date']
            })
        return pd.DataFrame(data)


# 保留其他类(保持原有逻辑)
class SentimentAnalyzer:
    """舆情分析器"""
    def __init__(self, config):
        self.config = config
    def apply_sentiment_filter(self, df, date):
        return df


class MarketTiming:
    """择时模块"""
    def __init__(self, config):
        self.config = config
    def get_market_signal(self, index_data, date):
        return 1.0