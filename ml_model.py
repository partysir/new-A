"""
机器学习模型模块
包含模型训练、预测和Walk-Forward验证
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MLModel:
    """机器学习模型基类"""

    def __init__(self, config, model_type: str = None):
        self.config = config
        self.model_type = model_type or config.model.model_type
        self.model = None
        self.feature_names = None
        self.trained_dates = []

    def build_model(self):
        """构建模型"""
        if self.model_type == 'xgboost':
            import xgboost as xgb
            self.model = xgb.XGBRegressor(**self.config.model.xgb_params)

        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(**self.config.model.lgb_params)

        elif self.model_type == 'catboost':
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                loss_function='RMSE',
                random_seed=42,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        logger.info(f"Built {self.model_type} model")

    def train(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame = None,
            y_val: pd.Series = None
    ) -> Dict:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签

        Returns:
            训练结果字典
        """
        if self.model is None:
            self.build_model()

        self.feature_names = list(X_train.columns)

        # 训练
        if self.config.model.use_early_stopping and X_val is not None:
            if self.model_type in ['xgboost', 'lightgbm']:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

        # 评估
        train_pred = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred)

        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred)

        logger.info(f"Training completed - Train RMSE: {train_metrics['rmse']:.4f}")
        if val_metrics:
            logger.info(f"Validation RMSE: {val_metrics['rmse']:.4f}")

        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(X)

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """计算评估指标"""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'ic': np.corrcoef(y_true, y_pred)[0, 1]
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if self.model is None:
            return pd.DataFrame()

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            return pd.DataFrame()

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })

        return df.sort_values('importance', ascending=False)

    def save(self, filepath: str):
        """保存模型"""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        logger.info(f"Model loaded from {filepath}")


class EnsembleModel:
    """集成模型"""

    def __init__(self, config):
        self.config = config
        self.models = []
        self.weights = config.model.ensemble_weights

    def build_models(self):
        """构建多个模型"""
        for model_type in self.config.model.ensemble_models:
            model = MLModel(self.config, model_type)
            model.build_model()
            self.models.append(model)

        logger.info(f"Built {len(self.models)} ensemble models")

    def train(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame = None,
            y_val: pd.Series = None
    ) -> Dict:
        """训练所有模型"""
        results = []

        for i, model in enumerate(self.models):
            logger.info(f"Training model {i + 1}/{len(self.models)}")
            result = model.train(X_train, y_train, X_val, y_val)
            results.append(result)

        return {'individual_results': results}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """加权预测"""
        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # 加权平均
        predictions = np.array(predictions)
        weights = np.array(self.weights)

        return np.average(predictions, axis=0, weights=weights)

    def get_feature_importance(self) -> pd.DataFrame:
        """获取平均特征重要性"""
        importance_dfs = []

        for model in self.models:
            df = model.get_feature_importance()
            if not df.empty:
                importance_dfs.append(df)

        if not importance_dfs:
            return pd.DataFrame()

        # 合并并平均
        combined = pd.concat(importance_dfs)
        avg_importance = combined.groupby('feature')['importance'].mean().reset_index()

        return avg_importance.sort_values('importance', ascending=False)


class WalkForwardTrainer:
    """Walk-Forward训练器"""

    def __init__(self, config):
        self.config = config
        self.models = {}  # {date: model}
        self.predictions = []

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练数据

        Args:
            df: 包含特征和标签的数据

        Returns:
            (X, y) 特征和标签
        """
        # 获取特征列
        feature_cols = [col for col in df.columns if col not in [
            'ts_code', 'trade_date', 'forward_return', 'label',
            'industry', 'name', 'symbol'
        ]]

        X = df[feature_cols]
        y = df.get('forward_return', df.get('label', None))

        if y is None:
            raise ValueError("No label column found")

        return X, y

    def walk_forward_train(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Walk-Forward滚动训练

        Args:
            df: 完整数据集

        Returns:
            包含预测分数的DataFrame
        """
        logger.info("Starting walk-forward training...")

        # 按日期排序
        df = df.sort_values('trade_date')
        dates = sorted(df['trade_date'].unique())

        train_window = self.config.model.train_window
        retrain_freq = self.config.model.retrain_frequency

        predictions_list = []

        for i, current_date in enumerate(dates):
            # 确定训练窗口
            if i < train_window:
                continue  # 跳过前面不足训练窗口的日期

            # 判断是否需要重训练
            should_train = (i == train_window or
                            (i - train_window) % retrain_freq == 0)

            if should_train:
                # 准备训练数据
                train_dates = dates[i - train_window:i]
                df_train = df[df['trade_date'].isin(train_dates)]

                # 分离特征和标签
                X_train, y_train = self.prepare_data(df_train)

                # 去除NaN
                mask = ~(X_train.isna().any(axis=1) | y_train.isna())
                X_train = X_train[mask]
                y_train = y_train[mask]

                if len(X_train) < 100:
                    logger.warning(f"Insufficient training data for {current_date}")
                    continue

                # 训练验证分割
                if self.config.model.validation_split > 0:
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_train, y_train,
                        test_size=self.config.model.validation_split,
                        random_state=42
                    )
                else:
                    X_tr, y_tr = X_train, y_train
                    X_val, y_val = None, None

                # 训练模型
                if self.config.model.use_ensemble:
                    model = EnsembleModel(self.config)
                    model.build_models()
                else:
                    model = MLModel(self.config)
                    model.build_model()

                model.train(X_tr, y_tr, X_val, y_val)

                # 保存模型
                self.models[current_date] = model

                logger.info(f"Trained model for date {current_date} ({i}/{len(dates)})")

            # 预测当前日期
            df_current = df[df['trade_date'] == current_date]

            if len(df_current) == 0:
                continue

            # 使用最近的模型
            model_date = max([d for d in self.models.keys() if d <= current_date])
            model = self.models[model_date]

            # 准备预测数据
            X_pred, _ = self.prepare_data(df_current)

            # 预测
            try:
                predictions = model.predict(X_pred)

                # 保存预测结果
                df_pred = df_current[['ts_code', 'trade_date']].copy()
                df_pred['ml_score'] = predictions
                predictions_list.append(df_pred)

            except Exception as e:
                logger.error(f"Prediction failed for {current_date}: {e}")
                continue

        # 合并所有预测
        if predictions_list:
            df_predictions = pd.concat(predictions_list, ignore_index=True)
            logger.info(f"Walk-forward training completed: {len(df_predictions)} predictions")
            return df_predictions
        else:
            logger.warning("No predictions generated")
            return pd.DataFrame()

    def get_model_performance(self) -> pd.DataFrame:
        """获取模型性能统计"""
        performance = []

        for date, model in self.models.items():
            importance = model.get_feature_importance()

            performance.append({
                'date': date,
                'model_type': model.model_type,
                'n_features': len(model.feature_names),
                'top_feature': importance.iloc[0]['feature'] if not importance.empty else None,
                'top_importance': importance.iloc[0]['importance'] if not importance.empty else None
            })

        return pd.DataFrame(performance)


class LabelGenerator:
    """标签生成器"""

    @staticmethod
    def generate_forward_return(
            df: pd.DataFrame,
            periods: int = 5,
            method: str = 'return'
    ) -> pd.Series:
        """
        生成未来收益标签

        Args:
            df: 数据(需包含ts_code, trade_date, close)
            periods: 预测期数
            method: 标签类型(return, excess_return, rank)

        Returns:
            标签Series
        """
        df = df.sort_values(['ts_code', 'trade_date'])

        if method == 'return':
            # 简单收益率
            label = df.groupby('ts_code')['close'].shift(-periods) / df['close'] - 1

        elif method == 'excess_return':
            # 超额收益率(相对市场平均)
            label = df.groupby('ts_code')['close'].shift(-periods) / df['close'] - 1
            market_return = label.groupby(df['trade_date']).transform('mean')
            label = label - market_return

        elif method == 'rank':
            # 排名标签
            forward_close = df.groupby('ts_code')['close'].shift(-periods)
            label = forward_close / df['close'] - 1
            label = label.groupby(df['trade_date']).rank(pct=True)

        else:
            raise ValueError(f"Unknown method: {method}")

        return label

    @staticmethod
    def add_labels(df: pd.DataFrame, config) -> pd.DataFrame:
        """添加标签到数据集"""
        df = df.copy()

        label = LabelGenerator.generate_forward_return(
            df,
            periods=config.model.forward_return_days,
            method=config.model.label_type
        )

        df['forward_return'] = label
        df['label'] = label

        return df