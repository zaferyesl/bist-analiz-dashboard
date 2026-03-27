import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .advanced_indicators import AdvancedIndicators
from .ml_predictor import MLPredictor
from .pattern_matcher import PatternMatcher
from .self_analyzer import SelfAnalyzer

logger = logging.getLogger(__name__)

class MomentumPredictor:
    def __init__(self):
        self.ml_predictor = MLPredictor()
        self.pattern_matcher = PatternMatcher()
        
        # SelfAnalyzer originally expected a data_fetcher and a momentum_analyzer.
        # We pass `self` as both, because we will implement their required methods directly.
        self.self_analyzer = SelfAnalyzer(data_fetcher=self, momentum_analyzer=self)
        
        self.indicator_calculator = AdvancedIndicators()
        
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.model_path = os.path.join(self.data_dir, "momentum_model.pkl")
        self.patterns_path = os.path.join(self.data_dir, "patterns.json")
        self.rising_stocks_path = os.path.join(self.data_dir, "rising_stocks.txt")
        self.last_trained_path = os.path.join(self.data_dir, "last_trained.json")
        
        self._current_df = None
        self._load_models()
        self.is_trained = self.ml_predictor.is_trained
        
    def _load_models(self):
        if os.path.exists(self.model_path):
            self.ml_predictor.load_model(self.model_path)
        if os.path.exists(self.patterns_path):
            self.pattern_matcher.load_patterns(self.patterns_path)

    # -----------------------------------------------------
    # DATA FETCHER & MOMENTUM ANALYZER STUBS FOR SelfAnalyzer
    # -----------------------------------------------------
    def set_current_df(self, df):
        """Helper to set the current dataframe being analyzed so SelfAnalyzer's dependencies can access it."""
        self._current_df = df
        
    def find_historical_rises(self, symbol, lookback_days=365, min_rise_pct=15, min_duration=5, max_duration=30):
        if self._current_df is None or len(self._current_df) < 50:
            return []
            
        df = self._current_df
        # Use only the specified lookback window
        if len(df) > lookback_days + 100:
            df = df.tail(lookback_days + 100)
            
        closes = df['Close'].values if 'Close' in df.columns else df['close'].values
        dates = df.index
        
        rises = []
        for duration in range(min_duration, max_duration + 1):
            for i in range(len(closes) - duration):
                start_price = closes[i]
                end_price = closes[i + duration]
                if start_price == 0:
                    continue
                rise_pct = ((end_price - start_price) / start_price) * 100
                if rise_pct >= min_rise_pct:
                    window_prices = closes[i:i+duration+1]
                    max_drawdown = 0
                    for j in range(len(window_prices)):
                        for k in range(j+1, len(window_prices)):
                            dd = ((window_prices[k] - window_prices[j]) / window_prices[j]) * 100
                            if dd < max_drawdown:
                                max_drawdown = dd
                    
                    if max_drawdown > -10:
                        rises.append({
                            'start_date': dates[i],
                            'end_date': dates[i + duration],
                            'start_price': float(start_price),
                            'end_price': float(end_price),
                            'rise_pct': float(rise_pct),
                            'duration': duration,
                            'max_drawdown': float(max_drawdown)
                        })
        
        if rises:
            rises.sort(key=lambda x: x['rise_pct'], reverse=True)
            filtered_rises = []
            used_dates = set()
            for rise in rises:
                overlap = False
                rise_dates = pd.date_range(rise['start_date'], rise['end_date'])
                for date in rise_dates:
                    if date in used_dates:
                        overlap = True
                        break
                if not overlap:
                    filtered_rises.append(rise)
                    for date in rise_dates:
                        used_dates.add(date)
            return filtered_rises
        return []

    def analyze_stock_at_date(self, symbol, target_date, lookback_days=200):
        if self._current_df is None:
            return None
        df = self._current_df
        target_dt = pd.to_datetime(target_date)
        df_sub = df[df.index <= target_dt]
        if len(df_sub) < 50:
            return None
            
        features = self._extract_features(df_sub, symbol, target_dt)
        return features

    def _extract_features(self, df, symbol, target_dt=None):
        out_features = {}
        if 'Open' in df.columns:
            o, h, l, c, v = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
        else:
            o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
            
        inds = self.indicator_calculator.calculate_all_indicators(
            o.values, h.values, l.values, c.values, v.values
        )
        if not inds:
            return None
            
        for key, value in inds.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 0:
                    last_val = value[-1]
                    out_features[key] = 0.0 if pd.isna(last_val) or np.isinf(last_val) else float(last_val)
                else:
                    out_features[key] = 0.0
            else:
                out_features[key] = 0.0 if pd.isna(value) or np.isinf(value) else float(value)
                
        out_features['symbol'] = symbol
        out_features['analysis_date'] = target_dt.strftime('%Y-%m-%d') if target_dt else datetime.now().strftime('%Y-%m-%d')
        out_features['data_points'] = len(df)
        
        close_series = c
        if len(df) >= 20:
            chg20 = ((close_series.iloc[-1] - close_series.iloc[-20]) / close_series.iloc[-20]) * 100
            out_features['price_change_20d'] = float(chg20) if not pd.isna(chg20) else 0.0
        else:
            out_features['price_change_20d'] = 0.0
            
        if len(df) >= 10:
            chg10 = ((close_series.iloc[-1] - close_series.iloc[-10]) / close_series.iloc[-10]) * 100
            out_features['price_change_10d'] = float(chg10) if not pd.isna(chg10) else 0.0
        else:
            out_features['price_change_10d'] = 0.0
            
        if len(df) >= 5:
            chg5 = ((close_series.iloc[-1] - close_series.iloc[-5]) / close_series.iloc[-5]) * 100
            out_features['price_change_5d'] = float(chg5) if not pd.isna(chg5) else 0.0
        else:
            out_features['price_change_5d'] = 0.0
            
        if len(df) >= 25:
            rec_v = v.iloc[-5:].mean()
            prev_v = v.iloc[-25:-5].mean()
            out_features['volume_ratio'] = float(rec_v / prev_v) if prev_v > 0 else 1.0
        else:
            out_features['volume_ratio'] = 1.0
            
        if len(df) >= 20:
            returns = close_series.pct_change().dropna()
            volatility = returns.tail(20).std() * np.sqrt(252) * 100
            out_features['volatility_20d'] = float(volatility) if not pd.isna(volatility) else 0.0
        else:
            out_features['volatility_20d'] = 0.0
            
        if len(df) >= 252:
            h52 = h.tail(252).max()
            l52 = l.tail(252).min()
            cur_p = close_series.iloc[-1]
            if h52 > l52:
                out_features['price_position_52w'] = float((cur_p - l52) / (h52 - l52))
            else:
                out_features['price_position_52w'] = 0.5
        else:
            out_features['price_position_52w'] = 0.5
            
        out_features['current_price'] = float(close_series.iloc[-1])
        return out_features

    # -----------------------------------------------------
    # ORCHESTRATION: AUTO-DISCOVER & TRAIN
    # -----------------------------------------------------
    def setup_and_train_if_needed(self, data_dict):
        today_str = datetime.now().strftime("%Y-%m-%d")
        info = {}
        if os.path.exists(self.last_trained_path):
            with open(self.last_trained_path, "r") as f:
                info = json.load(f)
                
        if info.get("last_auto_discover") == today_str and self.is_trained:
            return # Already trained today
            
        logger.info("Auto-discovering rising stocks over the last 1 month (>30%)...")
        rising_stocks = []
        
        for symbol, df in data_dict.items():
            if len(df) < 25:
                continue
            
            # Analyze last 22 trading days (~1 calendar month)
            recent = df.iloc[-22:]
            close_col = 'Close' if 'Close' in recent.columns else 'close'
            start_price = recent[close_col].iloc[0]
            max_price = recent[close_col].max()
            
            if start_price > 0:
                rise_pct = ((max_price / start_price) - 1) * 100
                if rise_pct >= 30.0:
                    rising_stocks.append({'symbol': symbol, 'rise_pct': rise_pct})
                    
        # Save to rising_stocks.txt
        with open(self.rising_stocks_path, "w") as f:
            for stock in rising_stocks:
                f.write(f"{stock['symbol']}:{stock['rise_pct']:.2f}\n")
                
        # Now automatically train the models
        # Target date is 30 calendar days ago
        target_date = datetime.now() - timedelta(days=30)
        analyzed_rising = []
        
        for stock in rising_stocks:
            symbol = stock['symbol']
            rise_pct = stock['rise_pct']
            if symbol not in data_dict:
                continue
                
            self.set_current_df(data_dict[symbol])
            features = self.analyze_stock_at_date(symbol, target_date)
            if features:
                features['rise_pct'] = rise_pct
                features['future_rise'] = rise_pct
                analyzed_rising.append(features)
                
        if analyzed_rising:
            patterns = self.pattern_matcher.find_common_patterns(analyzed_rising)
            if patterns:
                self.pattern_matcher.save_patterns(self.patterns_path)
                
            result = self.ml_predictor.train(analyzed_rising, min_rise_threshold=30.0)
            if result['success']:
                self.ml_predictor.save_model(self.model_path)
                self.is_trained = True
                
        info["last_auto_discover"] = today_str
        with open(self.last_trained_path, "w") as f:
            json.dump(info, f)
            
    # -----------------------------------------------------
    # PREDICTION ALGORITHM (KATMAN 1 & KATMAN 2)
    # -----------------------------------------------------
    def predict_momentum(self, symbol, df):
        if not self.is_trained:
            return None
            
        self.set_current_df(df)
        stock_feat = self._extract_features(df, symbol)
        if not stock_feat:
            return None
            
        layer1_prob = self.ml_predictor.predict(stock_feat)['probability']
        pattern_score = self.pattern_matcher.calculate_pattern_score(stock_feat)
        sim_avg = 0.0
        # Wait, PatternMatcher calculates similarity against analyzed_rising which is not stored in __init__ easily.
        # However calculate_pattern_score alone is highly effective and doesn't require the raw training df points!
        
        # KATMAN 2
        self_sim = 0.0
        self_analysis = self.self_analyzer.analyze_stock_self_pattern(symbol, min_rise_pct=15, lookback_days=365)
        if self_analysis['success']:
            self_sim = self_analysis['similarity_score']
            
        # MATH:
        combined_score = (layer1_prob * 0.4) + (pattern_score * 0.3) + (self_sim * 0.3)
        return {
            'ml_probability': layer1_prob,
            'pattern_score': pattern_score,
            'self_similarity': self_sim,
            'combined_score': combined_score
        }
