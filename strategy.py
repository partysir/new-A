"""
策略模块 - 改进版 v2.0
主要改进：
1. 分级仓位管理（替代二元止损）
2. 增强的风险控制逻辑
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class Strategy:
    """策略主类（保持不变）"""
    def __init__(self, config):
        self.config = config

    def select_stocks(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """根据ML评分选股"""
        df_date = df[df['trade_date'] == date].copy()
        if df_date.empty: return pd.DataFrame()

        # CRITICAL: Filter trading universe BEFORE scoring
        # 1. Exclude micro-caps (bottom 20% by market cap)
        if 'circ_mv' in df_date.columns:
            cap_threshold = df_date['circ_mv'].quantile(0.20)
            df_date = df_date[df_date['circ_mv'] > cap_threshold]
        
        # 2. Exclude limit-up/down stocks (execution impossible)
        if 'pct_chg' in df_date.columns and 'is_st' in df_date.columns:
            df_date = df_date[
                (df_date['pct_chg'].abs() < 9.8) &
                (df_date['is_st'] == 0)
            ]
        
        # 3. Liquidity filter
        if 'amount' in df_date.columns:
            df_date = df_date[df_date['amount'] > 1e7]

        # 4. 过滤评分过低的
        score_threshold = self.config.strategy.score_threshold
        df_date = df_date[df_date['ml_score'] >= score_threshold]

        # 5. 排序
        df_date = df_date.sort_values('ml_score', ascending=False)

        # 6. 行业中性化
        if self.config.strategy.max_industry_weight < 1.0:
            df_date = self._apply_industry_constraints(df_date)

        # 7. Top N
        selected = df_date.head(self.config.strategy.top_n)
        logger.info(f"Selected {len(selected)} stocks for {date}")
        return selected

    def select_stocks_live(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """实盘专用选股逻辑（漏斗模型）"""
        df_date = df[df['trade_date'] == date].copy()
        if len(df_date) == 0:
            return pd.DataFrame()

        # 第一层：硬性风控过滤
        if 'pct_chg' in df_date.columns and 'vol_ma20' in df_date.columns:
            mask_crash = (df_date['pct_chg'] < -7) & (df_date['vol'] > 1.5 * df_date['vol_ma20'])
        else:
            mask_crash = pd.Series([False] * len(df_date), index=df_date.index)
            
        if 'rsi_14' in df_date.columns:
            mask_overbought = df_date['rsi_14'] > 85
        else:
            mask_overbought = pd.Series([False] * len(df_date), index=df_date.index)
        
        df_candidates = df_date[~(mask_crash | mask_overbought)].copy()
        
        # 第二层：ML模型初选
        top_percentile = max(1, int(len(df_candidates) * 0.2))
        df_candidates = df_candidates.sort_values('ml_score', ascending=False).head(top_percentile)

        # 第三层：实盘指标加权
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

        # 第四层：最终Top N
        final_selection = df_candidates.sort_values('composite_score', ascending=False).head(self.config.strategy.top_n)
        final_selection['recommend_reason'] = final_selection.apply(self._generate_reason, axis=1)
        
        return final_selection

    def _generate_reason(self, row):
        """生成推荐理由文本"""
        reasons = []
        if row['ml_score'] > 0.8: reasons.append("模型高确信")
        if row['smart_money_score'] > 1.0: reasons.append("主力资金抢筹")
        if row['trend_energy'] > 2.0: reasons.append("趋势即将爆发")
        return "+".join(reasons) if reasons else "综合评分优选"

    def calculate_weights(self, selected_stocks: pd.DataFrame) -> pd.DataFrame:
        """计算股票权重"""
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
        if max_per_ind < 1: max_per_ind = 1
        return df.groupby('industry', group_keys=False).apply(lambda x: x.head(max_per_ind))

    def should_rebalance(self, date: str, trading_day_count: int) -> bool:
        """判断调仓日"""
        freq = self.config.strategy.rebalance_frequency
        if freq == 'daily': return True
        if freq == 'weekly':
            return pd.to_datetime(date).weekday() == 4
        if freq == 'n_days':
            return trading_day_count % self.config.strategy.rebalance_day == 0
        return False


class RiskManager:
    """
    风险管理器 - v2.0 改进版
    核心改进：分级仓位管理系统
    """
    def __init__(self, config):
        self.config = config
        self.position_entry_scores = {}
        self.position_entry_dates = {}
        
        # 新增：仓位状态追踪
        self.current_position_scalar = 1.0  # 当前仓位比例
        self.drawdown_confirmation_days = 0  # 回撤确认天数
        self.last_drawdown_level = 0  # 上次回撤等级
        
        # 分级仓位配置（从config读取）
        if hasattr(self.config.risk, 'position_tiers'):
            self.position_tiers = self.config.risk.position_tiers
        else:
            self.position_tiers = [
                {'threshold': 0.10, 'position': 1.00, 'name': '正常'},
                {'threshold': 0.15, 'position': 0.50, 'name': '半仓'},
                {'threshold': 0.20, 'position': 0.25, 'name': '轻仓'},
                {'threshold': 1.00, 'position': 0.00, 'name': '空仓'}
            ]
        
        # 初始化当前仓位比例
        self.current_position_scalar = 1.0
        if hasattr(self.config.risk, 'use_tiered_position') and not self.config.risk.use_tiered_position:
            self.current_position_scalar = 1.0  # 如果未启用分级仓位，则始终使用100%仓位

    def check_risk(self, positions: Dict, current_prices: Dict,
                   current_scores: Dict, current_date: str) -> List[Tuple[str, str]]:
        """个股风险检查"""
        to_sell = []

        for code, pos in positions.items():
            if code not in current_prices: continue

            price = current_prices[code]
            cost = pos['cost']
            pnl_pct = (price - cost) / cost

            if code not in self.position_entry_scores:
                self.position_entry_scores[code] = pos.get('score', 0)

            # A. 止损
            if pnl_pct <= self.config.risk.stop_loss_pct:
                to_sell.append((code, '止损'))
                continue

            # B. 止盈
            if pnl_pct >= self.config.risk.take_profit_pct:
                to_sell.append((code, '止盈'))
                continue

            # C. 评分衰减止损
            current_score = current_scores.get(code, 0)
            entry_score = self.position_entry_scores.get(code, 0)

            if entry_score > 0:
                decay = (entry_score - current_score) / entry_score
                if decay > 0.3 and self._get_holding_days(pos, current_date) > 3:
                    to_sell.append((code, '因子衰减'))
                    continue

            # D. 排名止损
            if current_score < 0.4:
                to_sell.append((code, '低分止损'))
                continue

            # E. 持仓期满
            if self._get_holding_days(pos, current_date) >= self.config.risk.max_holding_days:
                to_sell.append((code, '持仓期满'))
                continue

        return to_sell

    def check_portfolio_risk_v2(self, current_value: float, initial_capital: float, 
                                current_date: str) -> Dict:
        """
        【核心改进】分级仓位风险管理
        
        返回格式:
        {
            'action': 'normal'|'reduce'|'stop',
            'position_scalar': 0.0-1.0,
            'tier_name': '正常'|'半仓'|'轻仓'|'空仓',
            'drawdown': 当前回撤率,
            'message': 操作说明
        }
        """
        # 计算当前回撤
        drawdown = (initial_capital - current_value) / initial_capital
        
        # 【关键修复】空仓恢复逻辑
        if self.current_position_scalar == 0.0:
            # 记录空仓开始时间（如果尚未记录）
            if not hasattr(self, 'empty_position_start_date'):
                self.empty_position_start_date = current_date
            
            # 计算空仓持续时间
            try:
                from datetime import datetime
                empty_start = datetime.strptime(str(self.empty_position_start_date), '%Y%m%d')
                current_dt = datetime.strptime(str(current_date), '%Y%m%d')
                empty_days = (current_dt - empty_start).days
            except:
                empty_days = 0
            
            # 恢复条件1：回撤降至15%以下（轻仓阈值）
            if drawdown < 0.15:
                self.current_position_scalar = 0.25
                self.drawdown_confirmation_days = 0
                self.last_drawdown_level = 2  # 轻仓档位
                
                # 重置空仓开始时间
                self.empty_position_start_date = None
                
                message = f"空仓恢复：回撤降至{drawdown:.2%}，升至【轻仓】"
                logger.warning(message)
                
                return {
                    'action': 'normal',
                    'position_scalar': 0.25,
                    'tier_name': '轻仓',
                    'drawdown': drawdown,
                    'message': message
                }
            # 恢复条件2：空仓超过90天，强制尝试小仓位复活（避免死锁）
            elif empty_days >= 90:
                self.current_position_scalar = 0.1  # 小仓位尝试
                self.drawdown_confirmation_days = 0
                self.last_drawdown_level = 3  # 从轻仓档位开始
                
                # 重置空仓开始时间
                self.empty_position_start_date = None
                
                message = f"空仓超90天：强制尝试小仓位(10%)复活，当前回撤{drawdown:.2%}"
                logger.warning(message)
                
                return {
                    'action': 'normal',
                    'position_scalar': 0.1,
                    'tier_name': '试探',
                    'drawdown': drawdown,
                    'message': message
                }
            # 继续空仓等待
            else:
                return {
                    'action': 'stop',
                    'position_scalar': 0.0,
                    'tier_name': '空仓',
                    'drawdown': drawdown,
                    'message': f"空仓中，已持续{empty_days}天，等待回撤降至15%以下（当前{drawdown:.2%}）"
                }
        
        # 确定当前应处于哪个档位
        target_tier = None
        for tier in self.position_tiers:
            if drawdown < tier['threshold']:
                target_tier = tier
                break
        
        if target_tier is None:
            target_tier = self.position_tiers[-1]  # 默认最后一档
        
        # 计算当前档位等级
        current_tier_index = self._get_tier_index(self.current_position_scalar)
        target_tier_index = self.position_tiers.index(target_tier)
        
        # 档位变化确认机制（避免频繁切换）
        if target_tier_index != current_tier_index:
            # 需要降档（回撤加深）
            if target_tier_index > current_tier_index:
                self.drawdown_confirmation_days += 1
                
                # 连续2天确认才触发降档
                if self.drawdown_confirmation_days >= 2:
                    self.current_position_scalar = target_tier['position']
                    self.last_drawdown_level = target_tier_index
                    self.drawdown_confirmation_days = 0
                    
                    message = f"回撤{drawdown:.2%}，连续2日确认，降至【{target_tier['name']}】"
                    logger.warning(message)
                    
                    return {
                        'action': 'stop' if target_tier['position'] == 0 else 'reduce',
                        'position_scalar': target_tier['position'],
                        'tier_name': target_tier['name'],
                        'drawdown': drawdown,
                        'message': message
                    }
                else:
                    # 等待确认
                    return {
                        'action': 'normal',
                        'position_scalar': self.current_position_scalar,
                        'tier_name': self.position_tiers[current_tier_index]['name'],
                        'drawdown': drawdown,
                        'message': f"回撤{drawdown:.2%}，等待确认中({self.drawdown_confirmation_days}/2天)"
                    }
            
            # 需要升档（回撤恢复）- 立即生效
            else:
                self.current_position_scalar = target_tier['position']
                self.last_drawdown_level = target_tier_index
                self.drawdown_confirmation_days = 0
                
                message = f"回撤恢复至{drawdown:.2%}，升至【{target_tier['name']}】"
                logger.info(message)
                
                return {
                    'action': 'normal',
                    'position_scalar': target_tier['position'],
                    'tier_name': target_tier['name'],
                    'drawdown': drawdown,
                    'message': message
                }
        
        # 档位未变化，重置确认计数
        self.drawdown_confirmation_days = 0
        
        return {
            'action': 'normal',
            'position_scalar': self.current_position_scalar,
            'tier_name': target_tier['name'],
            'drawdown': drawdown,
            'message': f"当前【{target_tier['name']}】，回撤{drawdown:.2%}"
        }

    def _get_tier_index(self, position_scalar: float) -> int:
        """根据仓位比例确定档位索引"""
        for i, tier in enumerate(self.position_tiers):
            if abs(tier['position'] - position_scalar) < 0.01:
                return i
        return 0

    def _get_holding_days(self, pos, current_date):
        try:
            entry = pd.to_datetime(str(pos['entry_date']))
            curr = pd.to_datetime(str(current_date))
            return (curr - entry).days
        except:
            return 0

    # 保留旧接口兼容性
    def check_portfolio_stop(self, portfolio_value, initial_capital, current_date):
        """
        【废弃】旧版二元止损接口
        保留用于向后兼容，实际调用新版check_portfolio_risk_v2
        """
        result = self.check_portfolio_risk_v2(portfolio_value, initial_capital, current_date)
        return result['action'] == 'stop'

    def check_portfolio_risk(self, current_value, initial_capital, current_date):
        """
        【推荐】新版接口别名
        """
        return self.check_portfolio_risk_v2(current_value, initial_capital, current_date)

    def check_stop_loss(self, positions: dict, current_prices: dict) -> list:
        """检查个股止损"""
        stop_loss_list = []
        stop_loss_pct = self.config.risk.stop_loss_pct if hasattr(self.config.risk, 'stop_loss_pct') else -0.10
        
        for code, pos_info in positions.items():
            if code not in current_prices:
                continue
                
            cost = pos_info['cost']
            current_price = current_prices[code]
            
            if cost == 0: continue
            
            ret = (current_price - cost) / cost
            
            if ret <= stop_loss_pct:
                stop_loss_list.append(code)
                
        return stop_loss_list

    def check_take_profit(self, positions: dict, current_prices: dict) -> list:
        """检查个股止盈"""
        take_profit_list = []
        take_profit_pct = self.config.risk.take_profit_pct if hasattr(self.config.risk, 'take_profit_pct') else 0.20
        
        for code, pos_info in positions.items():
            if code not in current_prices:
                continue
                
            cost = pos_info['cost']
            current_price = current_prices[code]
            
            if cost == 0: continue
            
            ret = (current_price - cost) / cost
            
            if ret >= take_profit_pct:
                take_profit_list.append(code)
                
        return take_profit_list

    def check_holding_period(self, current_date: str) -> list:
        """检查持仓天数"""
        sell_list = []
        max_days = self.config.strategy.max_holding_days if hasattr(self.config.strategy, 'max_holding_days') else 20
        
        try:
            current_dt = datetime.strptime(str(current_date), "%Y%m%d")
        except:
            return []

        for code, entry_date in self.position_entry_dates.items():
            try:
                entry_dt = datetime.strptime(str(entry_date), "%Y%m%d")
                days_held = (current_dt - entry_dt).days
                
                if days_held >= max_days:
                    sell_list.append(code)
            except ValueError:
                continue
                
        return sell_list


class PortfolioManager:
    """
    组合管理器 - 改进版
    适配分级仓位系统
    """
    def __init__(self, config):
        self.config = config
        self.positions = {}
        self.cash = config.backtest.initial_capital
        self.trades = []
        self.risk_manager = None  # 外部注入

        # 交易成本
        self.buy_cost = config.strategy.commission_rate
        self.sell_cost = config.strategy.commission_rate + config.strategy.stamp_tax

    def set_risk_manager(self, risk_manager):
        """注入风险管理器"""
        self.risk_manager = risk_manager

    def update_positions(self, target_df: pd.DataFrame, current_prices: Dict,
                        current_scores: Dict, date: str) -> List[Dict]:
        """
        执行调仓：适配动态仓位
        """
        new_trades = []

        # 1. 获取当前仓位建议
        portfolio_value = self.get_portfolio_value(current_prices)
        if self.risk_manager:
            risk_status = self.risk_manager.check_portfolio_risk(
                portfolio_value, self.config.backtest.initial_capital, date
            )
            position_scalar = risk_status['position_scalar']
            
            # 记录仓位变化日志
            if risk_status['action'] in ['reduce', 'stop']:
                logger.warning(f"[{date}] {risk_status['message']}")
        else:
            position_scalar = 1.0

        # 2. 卖出逻辑
        target_codes = set(target_df['ts_code'].values)
        to_sell = [code for code in self.positions.keys() if code not in target_codes]

        for code in to_sell:
            trade = self._sell(code, current_prices.get(code), date, '调仓卖出')
            if trade: new_trades.append(trade)

        # 3. 买入逻辑（应用仓位比例）
        available_cash = self.cash * 0.95 * position_scalar  # 关键改动：乘以仓位比例

        to_buy = []
        for _, row in target_df.iterrows():
            code = row['ts_code']
            if code not in self.positions:
                to_buy.append(row)

        if not to_buy:
            self.trades.extend(new_trades)
            return new_trades

        per_stock_cash = available_cash / len(to_buy)

        for row in to_buy:
            code = row['ts_code']
            price = current_prices.get(code)
            if not price: continue

            shares = int(per_stock_cash / (price * (1 + self.buy_cost)) / 100) * 100

            if shares >= 100:
                trade = self._buy(code, shares, price, date, row.get('ml_score', 0))
                if trade: new_trades.append(trade)

        self.trades.extend(new_trades)
        return new_trades

    def _buy(self, code, shares, price, date, score):
        cost = shares * price * (1 + self.buy_cost)
        if cost > self.cash: return None

        self.cash -= cost
        if code not in self.positions:
            self.positions[code] = {
                'shares': shares, 'cost': price, 'entry_date': date,
                'score': score, 'name': code
            }

        return {
            'date': date, 'code': code, 'action': 'buy',
            'shares': shares, 'price': price, 'amount': cost
        }

    def _sell(self, code, price, date, reason):
        if not price or code not in self.positions: return None

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
        val = self.cash
        for code, pos in self.positions.items():
            price = current_prices.get(code, pos['cost'])
            val += pos['shares'] * price
        return val

    def get_positions_df(self, current_prices, date):
        data = []
        for code, pos in self.positions.items():
            price = current_prices.get(code, pos['cost'])
            val = pos['shares'] * price
            pnl = (price - pos['cost']) * pos['shares']
            data.append({
                'ts_code': code, 'name': pos.get('name'), 'shares': pos['shares'],
                'cost': pos['cost'], 'price': price, 'value': val, 'pnl': pnl,
                'entry_date': pos['entry_date']
            })
        return pd.DataFrame(data)


class SentimentAnalyzer:
    """舆情分析器（保持不变）"""
    def __init__(self, config):
        self.config = config
    def apply_sentiment_filter(self, df, date):
        return df

class MarketTiming:
    """择时模块（保持不变）"""
    def __init__(self, config):
        self.config = config
    def get_market_signal(self, index_data, date):
        return 1.0