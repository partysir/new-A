"""
策略模块
包含选股、择时、风控和组合管理
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Strategy:
    """策略主类"""

    def __init__(self, config):
        self.config = config
        self.positions = {}  # 当前持仓
        self.cash = config.backtest.initial_capital
        self.portfolio_value = self.cash

    def select_stocks(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        选股

        Args:
            df: 包含ml_score的数据
            date: 当前日期

        Returns:
            选中的股票DataFrame
        """
        df_date = df[df['trade_date'] == date].copy()

        if len(df_date) == 0:
            return pd.DataFrame()

        # 按评分排序
        df_date = df_date.sort_values('ml_score', ascending=False)

        # 选择方法
        if self.config.strategy.selection_method == 'score':
            # Top N
            selected = df_date.head(self.config.strategy.top_n)

        elif self.config.strategy.selection_method == 'threshold':
            # 分数阈值
            selected = df_date[df_date['ml_score'] >= self.config.strategy.score_threshold]
            selected = selected.head(self.config.strategy.top_n)

        elif self.config.strategy.selection_method == 'rank':
            # 排名前N%
            n_stocks = int(len(df_date) * 0.1)  # 前10%
            selected = df_date.head(n_stocks)
        else:
            selected = df_date.head(self.config.strategy.top_n)

        # 行业限制
        if self.config.strategy.max_industry_weight < 1.0 and 'industry' in selected.columns:
            selected = self._apply_industry_limits(selected)

        logger.info(f"Selected {len(selected)} stocks for {date}")
        return selected

    def _apply_industry_limits(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用行业限制"""
        max_per_industry = int(len(df) * self.config.strategy.max_industry_weight)

        # 每个行业最多选max_per_industry只
        df_limited = df.groupby('industry', group_keys=False).apply(
            lambda x: x.head(max_per_industry)
        )

        # 确保至少min_industry_count个行业
        industries = df_limited['industry'].nunique()
        if industries < self.config.strategy.min_industry_count:
            logger.warning(f"Only {industries} industries, less than min requirement")

        return df_limited

    def calculate_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算持仓权重

        Args:
            df: 选中的股票

        Returns:
            包含权重的DataFrame
        """
        method = self.config.strategy.weight_method

        if method == 'equal':
            # 等权重
            df['weight'] = 1.0 / len(df)

        elif method == 'score_weighted':
            # 按分数加权
            df['weight'] = df['ml_score'] / df['ml_score'].sum()

        elif method == 'risk_parity':
            # 风险平价
            if 'volatility' in df.columns:
                inv_vol = 1.0 / df['volatility']
                df['weight'] = inv_vol / inv_vol.sum()
            else:
                df['weight'] = 1.0 / len(df)

        elif method == 'optimize':
            # 优化权重(最大夏普)
            df['weight'] = self._optimize_weights(df)
        else:
            df['weight'] = 1.0 / len(df)

        # 权重限制
        max_weight = self.config.strategy.max_single_weight
        df['weight'] = df['weight'].clip(upper=max_weight)

        # 归一化
        df['weight'] = df['weight'] / df['weight'].sum()

        return df

    def _optimize_weights(self, df: pd.DataFrame) -> np.ndarray:
        """优化权重(简化版)"""
        n = len(df)

        # 这里使用简化版本,实际可以使用cvxpy等优化库
        # 假设等权重
        weights = np.ones(n) / n

        return weights

    def should_rebalance(self, date: str, trading_day_count: int = 0) -> bool:
        """
        判断是否应该调仓

        Args:
            date: 当前日期
            trading_day_count: 当前是第几个交易日(从0开始)

        Returns:
            是否调仓
        """
        freq = self.config.strategy.rebalance_frequency

        if freq == 'daily':
            return True

        elif freq == 'weekly':
            # 检查是否是周五(或指定日期)
            date_dt = pd.to_datetime(date)
            weekday = date_dt.weekday()  # 0=周一, 4=周五
            target_day = self.config.strategy.rebalance_day - 1  # 配置中5代表周五,转为0-6
            return weekday == target_day

        elif freq == 'monthly':
            # 检查是否是每月指定日期
            date_dt = pd.to_datetime(date)
            return date_dt.day == self.config.strategy.rebalance_day

        elif freq == 'n_days':
            # 每N个交易日调仓
            n_days = self.config.strategy.rebalance_day
            return trading_day_count % n_days == 0

        return False


class RiskManager:
    """风险管理器"""

    def __init__(self, config):
        self.config = config
        self.position_entry_dates = {}  # 持仓开始日期

    def check_stop_loss(self, positions: Dict, current_prices: Dict) -> List[str]:
        """
        检查止损

        Args:
            positions: 持仓字典 {ts_code: {'shares': n, 'cost': p}}
            current_prices: 当前价格字典 {ts_code: price}

        Returns:
            需要止损的股票列表
        """
        if not self.config.risk.use_stop_loss:
            return []

        to_sell = []

        for code, pos in positions.items():
            if code not in current_prices:
                continue

            cost = pos['cost']
            price = current_prices[code]
            pnl_pct = (price - cost) / cost

            # 个股止损
            if pnl_pct <= self.config.risk.stop_loss_pct:
                to_sell.append(code)
                logger.info(f"Stop loss triggered for {code}: {pnl_pct:.2%}")

        return to_sell

    def check_take_profit(self, positions: Dict, current_prices: Dict) -> List[str]:
        """检查止盈"""
        if not self.config.risk.use_take_profit:
            return []

        to_sell = []

        for code, pos in positions.items():
            if code not in current_prices:
                continue

            cost = pos['cost']
            price = current_prices[code]
            pnl_pct = (price - cost) / cost

            # 个股止盈
            if pnl_pct >= self.config.risk.take_profit_pct:
                to_sell.append(code)
                logger.info(f"Take profit triggered for {code}: {pnl_pct:.2%}")

        return to_sell

    def check_holding_period(self, current_date: str) -> List[str]:
        """检查持仓期限"""
        to_sell = []
        current_dt = pd.to_datetime(current_date)

        for code, entry_date in self.position_entry_dates.items():
            entry_dt = pd.to_datetime(entry_date)
            holding_days = (current_dt - entry_dt).days

            # 最大持仓期
            if holding_days >= self.config.risk.max_holding_days:
                to_sell.append(code)
                logger.info(f"Max holding period reached for {code}: {holding_days} days")

        return to_sell

    def check_portfolio_risk(self, portfolio_value: float, initial_capital: float,
                            current_date: str = None) -> bool:
        """
        检查组合风险

        Args:
            portfolio_value: 当前组合价值
            initial_capital: 初始资金
            current_date: 当前日期(用于止损恢复判断)

        Returns:
            是否触发组合止损
        """
        pnl_pct = (portfolio_value - initial_capital) / initial_capital

        # 检查是否触发新的止损
        if pnl_pct <= self.config.risk.portfolio_stop_loss_pct:
            # 如果之前没有记录止损,记录本次止损时间
            if not hasattr(self, '_stop_loss_triggered_date'):
                self._stop_loss_triggered_date = current_date
                self._stop_loss_count = getattr(self, '_stop_loss_count', 0) + 1
                logger.warning(f"Portfolio stop loss triggered #{self._stop_loss_count}: {pnl_pct:.2%} on {current_date}")
                return True

            # 如果已经触发过止损,检查是否应该恢复
            if current_date and hasattr(self, '_stop_loss_triggered_date'):
                # 计算距离上次止损的天数
                from datetime import datetime
                current_dt = datetime.strptime(current_date, '%Y%m%d')
                stop_dt = datetime.strptime(self._stop_loss_triggered_date, '%Y%m%d')
                days_since_stop = (current_dt - stop_dt).days

                # 如果超过30个交易日(约1.5个月),且回撤不再恶化,允许恢复交易
                if days_since_stop >= 30:
                    # 计算当前回撤相对于触发时的改善
                    if pnl_pct > self.config.risk.portfolio_stop_loss_pct * 0.9:  # 回撤改善10%以上
                        logger.info(f"Portfolio stop loss recovery: current return {pnl_pct:.2%}, {days_since_stop} days since stop")
                        # 清除止损标记,允许重新开始
                        delattr(self, '_stop_loss_triggered_date')
                        return False

            return True

        # 如果收益率恢复到正常水平,清除止损标记
        if hasattr(self, '_stop_loss_triggered_date'):
            if pnl_pct > self.config.risk.portfolio_stop_loss_pct * 0.5:  # 回撤恢复50%以上
                logger.info(f"Portfolio recovered from stop loss: {pnl_pct:.2%}")
                delattr(self, '_stop_loss_triggered_date')

        return False


class MarketTiming:
    """择时模块"""

    def __init__(self, config):
        self.config = config

    def get_market_signal(self, index_data: pd.DataFrame, current_date: str) -> float:
        """
        获取市场择时信号

        Args:
            index_data: 指数数据
            current_date: 当前日期

        Returns:
            仓位比例(0-1)
        """
        if not self.config.risk.use_timing:
            return 1.0  # 满仓

        indicator = self.config.risk.timing_indicator

        if indicator == 'rsrs':
            signal = self._calculate_rsrs(index_data, current_date)
        elif indicator == 'ma':
            signal = self._calculate_ma_signal(index_data, current_date)
        elif indicator == 'macd':
            signal = self._calculate_macd_signal(index_data, current_date)
        else:
            signal = 1.0

        # 根据信号决定仓位
        if signal < self.config.risk.rsrs_threshold:
            position_ratio = self.config.risk.cash_ratio_when_bear
            logger.info(f"Bear market signal: {signal:.2f}, position ratio: {position_ratio:.2f}")
        else:
            position_ratio = 1.0

        return position_ratio

    def _calculate_rsrs(self, df: pd.DataFrame, current_date: str) -> float:
        """
        计算RSRS指标(阻力支撑相对强度)

        原理:
        1. 用最高价和最低价做线性回归,得到斜率β
        2. 计算β的标准分
        3. 修正后的RSRS = 标准分 * β * R²
        """
        df = df[df['trade_date'] <= current_date].tail(100).copy()

        if len(df) < 18:
            return 1.0

        # 计算最近18日的回归斜率
        rsrs_scores = []

        for i in range(18, len(df)):
            window = df.iloc[i-18:i]

            # 线性回归: low = a + b * high
            X = window['high'].values
            y = window['low'].values

            # 计算斜率
            coef = np.polyfit(X, y, 1)
            beta = coef[0]

            # 计算R²
            y_pred = np.polyval(coef, X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            rsrs_scores.append(beta * r2)

        if not rsrs_scores:
            return 1.0

        # 标准化
        current_rsrs = rsrs_scores[-1]
        mean_rsrs = np.mean(rsrs_scores)
        std_rsrs = np.std(rsrs_scores)

        if std_rsrs > 0:
            zscore = (current_rsrs - mean_rsrs) / std_rsrs
        else:
            zscore = 0

        # 转换为0-1区间
        signal = 1 / (1 + np.exp(-zscore))

        return signal

    def _calculate_ma_signal(self, df: pd.DataFrame, current_date: str) -> float:
        """均线择时信号"""
        df = df[df['trade_date'] <= current_date].copy()

        if len(df) < 60:
            return 1.0

        # 计算均线
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

        latest = df.iloc[-1]

        # 短期均线在长期均线之上为多头信号
        if latest['ma20'] > latest['ma60']:
            return 1.0
        else:
            return 0.5

    def _calculate_macd_signal(self, df: pd.DataFrame, current_date: str) -> float:
        """MACD择时信号"""
        df = df[df['trade_date'] <= current_date].copy()

        if len(df) < 26:
            return 1.0

        # 计算MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd = dif - dea

        latest_macd = macd.iloc[-1]

        # MACD为正为多头信号
        if latest_macd > 0:
            return 1.0
        else:
            return 0.5


class SentimentAnalyzer:
    """舆情分析器"""

    def __init__(self, config):
        self.config = config
        self.sentiment_cache = {}

    def get_sentiment_score(self, ts_code: str, date: str) -> float:
        """
        获取舆情评分

        Args:
            ts_code: 股票代码
            date: 日期

        Returns:
            舆情分数(-1到1)
        """
        if not self.config.risk.use_sentiment:
            return 0.0

        # 从缓存获取
        key = f"{ts_code}_{date}"
        if key in self.sentiment_cache:
            return self.sentiment_cache[key]

        # 实际实现中,这里应该调用新闻API或爬虫
        # 这里返回随机值作为示例
        score = np.random.randn() * 0.3
        score = np.clip(score, -1, 1)

        self.sentiment_cache[key] = score
        return score

    def apply_sentiment_filter(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        应用舆情过滤

        Args:
            df: 候选股票
            date: 日期

        Returns:
            过滤后的股票
        """
        if not self.config.risk.use_sentiment:
            return df

        # 获取舆情分数
        df['sentiment'] = df['ts_code'].apply(
            lambda x: self.get_sentiment_score(x, date)
        )

        # 一票否决
        veto_threshold = self.config.risk.sentiment_veto_threshold
        df = df[df['sentiment'] > veto_threshold]

        # 利好加分
        bonus = self.config.risk.sentiment_bonus
        df['ml_score'] = df['ml_score'] + df['sentiment'] * bonus

        logger.info(f"Sentiment filter: {len(df)} stocks remaining")

        return df


class PortfolioManager:
    """组合管理器"""

    def __init__(self, config):
        self.config = config
        self.positions = {}
        self.cash = config.backtest.initial_capital
        self.trades = []

    def update_positions(
        self,
        target_stocks: pd.DataFrame,
        current_prices: Dict,
        current_date: str
    ) -> List[Dict]:
        """
        更新持仓

        Args:
            target_stocks: 目标持仓(包含weight, ml_score, name等)
            current_prices: 当前价格
            current_date: 当前日期

        Returns:
            交易列表
        """
        trades = []

        # 计算目标持仓市值
        target_positions = {}
        target_info = {}  # 保存score和name
        total_value = self.get_portfolio_value(current_prices)

        for _, row in target_stocks.iterrows():
            code = row['ts_code']
            weight = row['weight']
            price = current_prices.get(code, 0)

            if price > 0:
                target_value = total_value * weight
                target_shares = int(target_value / price / 100) * 100  # 取整手
                target_positions[code] = target_shares
                target_info[code] = {
                    'score': row.get('ml_score', 0.0),
                    'name': row.get('name', '')
                }

        # 卖出不在目标中的股票
        for code in list(self.positions.keys()):
            if code not in target_positions:
                shares = self.positions[code]['shares']
                price = current_prices.get(code, self.positions[code]['cost'])

                trade = self._sell_stock(code, shares, price, current_date, reason='调仓卖出')
                trades.append(trade)

        # 调整持仓
        for code, target_shares in target_positions.items():
            current_shares = self.positions.get(code, {}).get('shares', 0)
            price = current_prices[code]
            info = target_info[code]

            if target_shares > current_shares:
                # 买入
                shares_to_buy = target_shares - current_shares
                trade = self._buy_stock(
                    code, shares_to_buy, price, current_date,
                    score=info['score'], name=info['name']
                )
                trades.append(trade)

            elif target_shares < current_shares:
                # 卖出
                shares_to_sell = current_shares - target_shares
                trade = self._sell_stock(code, shares_to_sell, price, current_date, reason='调仓减仓')
                trades.append(trade)

        self.trades.extend(trades)
        return trades

        # 调整持仓
        for code, target_shares in target_positions.items():
            current_shares = self.positions.get(code, {}).get('shares', 0)
            price = current_prices[code]

            if target_shares > current_shares:
                # 买入
                shares_to_buy = target_shares - current_shares
                trade = self._buy_stock(code, shares_to_buy, price, current_date)
                trades.append(trade)

            elif target_shares < current_shares:
                # 卖出
                shares_to_sell = current_shares - target_shares
                trade = self._sell_stock(code, shares_to_sell, price, current_date)
                trades.append(trade)

        self.trades.extend(trades)
        return trades

    def _buy_stock(self, code: str, shares: int, price: float, date: str,
                   score: float = 0.0, name: str = '') -> Dict:
        """买入股票"""
        amount = shares * price
        commission = amount * self.config.strategy.commission_rate
        slippage = amount * self.config.strategy.slippage
        total_cost = amount + commission + slippage

        if total_cost > self.cash:
            logger.warning(f"Insufficient cash to buy {code}")
            return {}

        # 更新持仓
        if code in self.positions:
            old_shares = self.positions[code]['shares']
            old_cost = self.positions[code]['cost']
            new_shares = old_shares + shares
            new_cost = (old_shares * old_cost + shares * price) / new_shares
            self.positions[code]['shares'] = new_shares
            self.positions[code]['cost'] = new_cost
            # 保持原始entry_date和score
        else:
            self.positions[code] = {
                'shares': shares,
                'cost': price,
                'entry_date': date,
                'score': score,
                'name': name
            }

        self.cash -= total_cost

        trade = {
            'date': date,
            'code': code,
            'name': name,
            'action': 'buy',
            'shares': shares,
            'price': price,
            'amount': amount,
            'commission': commission,
            'slippage': slippage,
            'score': score,
            'reason': '调仓买入'
        }

        logger.info(f"Buy {code}: {shares} shares @ {price:.2f}")
        return trade

    def _sell_stock(self, code: str, shares: int, price: float, date: str,
                    reason: str = '调仓卖出') -> Dict:
        """卖出股票"""
        if code not in self.positions:
            return {}

        current_shares = self.positions[code]['shares']
        shares = min(shares, current_shares)

        amount = shares * price
        commission = amount * self.config.strategy.commission_rate
        stamp_tax = amount * self.config.strategy.stamp_tax
        slippage = amount * self.config.strategy.slippage
        total_proceeds = amount - commission - stamp_tax - slippage

        # 计算盈亏
        cost = self.positions[code]['cost']
        pnl = (price - cost) * shares
        pnl_rate = (price - cost) / cost

        # 更新持仓
        if shares >= current_shares:
            del self.positions[code]
        else:
            self.positions[code]['shares'] -= shares

        self.cash += total_proceeds

        trade = {
            'date': date,
            'code': code,
            'name': self.positions.get(code, {}).get('name', '') if code in self.positions else '',
            'action': 'sell',
            'shares': shares,
            'price': price,
            'cost': cost,
            'amount': amount,
            'commission': commission,
            'stamp_tax': stamp_tax,
            'slippage': slippage,
            'pnl': pnl,
            'pnl_rate': pnl_rate,
            'reason': reason
        }

        logger.info(f"Sell {code}: {shares} shares @ {price:.2f}, reason: {reason}")
        return trade

    def get_portfolio_value(self, current_prices: Dict) -> float:
        """计算组合总市值"""
        position_value = sum(
            pos['shares'] * current_prices.get(code, pos['cost'])
            for code, pos in self.positions.items()
        )
        return self.cash + position_value

    def get_positions_df(self, current_prices: Dict = None, current_date: str = None) -> pd.DataFrame:
        """
        获取持仓DataFrame（完整信息）

        Args:
            current_prices: 当前价格字典
            current_date: 当前日期

        Returns:
            包含完整信息的持仓DataFrame
        """
        if not self.positions:
            return pd.DataFrame()

        positions_list = []
        for code, pos in self.positions.items():
            shares = pos['shares']
            cost = pos['cost']
            entry_date = pos.get('entry_date', '')
            score = pos.get('score', 0.0)
            name = pos.get('name', '')

            # 计算当前市值和盈亏
            if current_prices and code in current_prices:
                price = current_prices[code]
            else:
                price = cost

            value = shares * price
            pnl = (price - cost) * shares
            pnl_pct = (price - cost) / cost if cost > 0 else 0

            # 计算持仓天数
            holding_days = 0
            if current_date and entry_date:
                try:
                    from datetime import datetime
                    current_dt = datetime.strptime(str(current_date), '%Y%m%d')
                    entry_dt = datetime.strptime(str(entry_date), '%Y%m%d')
                    holding_days = (current_dt - entry_dt).days
                except:
                    holding_days = 0

            positions_list.append({
                'ts_code': code,
                'name': name,
                'shares': shares,
                'cost': cost,
                'price': price,
                'value': value,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'entry_date': entry_date,
                'score': score,
                'holding_days': holding_days
            })

        return pd.DataFrame(positions_list)