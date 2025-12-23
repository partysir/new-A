"""
机器学习模型模块 - 增强版
整合了 Triple Barrier 标签生成和集成投票机制
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, roc_auc_score
import joblib
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TripleBarrierLabeler:
    """
    三屏障标签生成器
    基于 止盈(上界)、止损(下界) 和 持仓期(垂直界) 生成标签
    """
    def __init__(self,
                 profit_threshold=0.05,    # 止盈 5%
                 stop_loss_threshold=-0.03, # 止损 -3%
                 holding_period=10):
        self.profit_threshold = profit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.holding_period = holding_period

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成标签: 1(Buy), 0(Hold/Sell)"""
        # 确保数据按股票和时间排序
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

        # 预计算未来价格序列
        grouped = df.groupby('ts_code')

        labels = []
        # 注意：这里使用循环处理每只股票，向量化实现较复杂
        # 为了效率，实际生产中可使用numba加速
        for name, group in grouped:
            close_prices = group['close'].values
            n = len(close_prices)
            group_labels = np.zeros(n)

            for i in range(n - self.holding_period):
                current_price = close_prices[i]
                future_prices = close_prices[i+1 : i+1+self.holding_period]

                # 计算收益率
                returns = (future_prices - current_price) / current_price

                # 触碰上界 (止盈)
                if np.any(returns >= self.profit_threshold):
                    group_labels[i] = 1
                # 触碰下界 (止损) - 这里简化为二分类，触碰止损或未达标都为0
                elif np.any(returns <= self.stop_loss_threshold):
                    group_labels[i] = 0
                # 持仓期满
                else:
                    # 如果期末收益为正，可视情况定，这里保守设为0
                    group_labels[i] = 1 if returns[-1] > 0 else 0

            labels.extend(group_labels)

        df['label'] = labels
        return df


class MLModel:
    """机器学习模型基类"""

    def __init__(self, config, model_type: str = None):
        self.config = config
        self.model_type = model_type or config.model.model_type
        self.model = None
        self.feature_names = None

    def build_model(self):
        """构建模型"""
        # 参数自适应调整
        xgb_params = self.config.model.xgb_params.copy()
        lgb_params = self.config.model.lgb_params.copy()

        if self.model_type == 'xgboost':
            import xgboost as xgb
            self.model = xgb.XGBRegressor(**xgb_params)

        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(**lgb_params)

        elif self.model_type == 'catboost':
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor(verbose=False, random_seed=42)

        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if self.model is None:
            self.build_model()

        self.feature_names = list(X_train.columns)

        # 训练
        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        # 评估
        train_pred = self.model.predict(X_train)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'r2': r2_score(y_train, train_pred)
        }
        return {'train_metrics': metrics}

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        return pd.DataFrame()

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class EnsembleModel:
    """
    集成模型 (Voting Mechanism)
    整合了老项目中的多模型投票思想
    """
    def __init__(self, config):
        self.config = config
        self.models = []
        self.model_types = ['xgboost', 'lightgbm', 'random_forest'] # 默认集成这三种

    def build_models(self):
        for m_type in self.model_types:
            try:
                model = MLModel(self.config, model_type=m_type)
                model.build_model()
                self.models.append(model)
            except Exception as e:
                logger.warning(f"Failed to build {m_type}: {e}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        for model in self.models:
            logger.info(f"Training ensemble component: {model.model_type}")
            model.train(X_train, y_train, X_val, y_val)

    def predict(self, X) -> np.ndarray:
        """
        加权平均预测
        也可以改为投票制：只有 > 阈值的才算1
        """
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            # 归一化预测值到 0-1 (如果是回归模型输出收益率，这里可以跳过或做Rank)
            predictions.append(pred)

        # 简单平均
        return np.mean(predictions, axis=0)


class WalkForwardTrainer:
    """滚动训练器"""
    def __init__(self, config):
        self.config = config
        self.models = {}

    def prepare_data(self, df):
        # 排除非特征列
        exclude_cols = ['ts_code', 'trade_date', 'label', 'forward_return',
                       'open', 'high', 'low', 'close', 'vol', 'amount', 'name', 'industry']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].fillna(0)
        # 优先使用 Triple Barrier 生成的 label，否则使用 forward_return
        y = df['label'] if 'label' in df.columns else df['forward_return']

        # 替换无限值
        X = X.replace([np.inf, -np.inf], 0)
        y = y.fillna(0)

        return X, y

    def walk_forward_train(self, df):
        """执行滚动训练"""
        df = df.sort_values('trade_date')
        dates = sorted(df['trade_date'].unique())

        train_window = self.config.model.train_window
        predictions_list = []

        # 检查是否使用集成模型
        use_ensemble = self.config.model.use_ensemble

        # 滚动窗口
        for i in range(train_window, len(dates), self.config.model.retrain_frequency):
            train_dates = dates[i-train_window : i]
            test_dates = dates[i : i+self.config.model.retrain_frequency]

            if not test_dates: break

            logger.info(f"Walk-forward: Training {train_dates[0]}-{train_dates[-1]}, Predicting {test_dates[0]}-{test_dates[-1]}")

            # 训练数据
            train_data = df[df['trade_date'].isin(train_dates)]
            X_train, y_train = self.prepare_data(train_data)

            if len(X_train) < 100: continue

            # 训练
            if use_ensemble:
                model = EnsembleModel(self.config)
                model.build_models()
            else:
                model = MLModel(self.config)
                model.build_model()

            model.train(X_train, y_train)
            self.models[test_dates[0]] = model # 保存模型引用

            # 预测
            test_data = df[df['trade_date'].isin(test_dates)]
            if len(test_data) == 0: continue

            X_test, _ = self.prepare_data(test_data)
            preds = model.predict(X_test)

            result_df = test_data[['ts_code', 'trade_date']].copy()
            result_df['ml_score'] = preds
            predictions_list.append(result_df)

        if not predictions_list:
            return pd.DataFrame()

        return pd.concat(predictions_list)


class LabelGenerator:
    """标签生成器工厂"""
    @staticmethod
    def add_labels(df, config):
        """根据配置添加标签"""
        # 1. 基础收益率标签
        periods = config.model.forward_return_days
        df['forward_return'] = df.groupby('ts_code')['close'].shift(-periods) / df['close'] - 1

        # 2. 如果启用 Triple Barrier (高级标签)
        # 这里我们默认启用以增强效果
        labeler = TripleBarrierLabeler(
            profit_threshold=0.05,
            stop_loss_threshold=-0.03,
            holding_period=periods
        )
        df = labeler.generate_labels(df)

        return df