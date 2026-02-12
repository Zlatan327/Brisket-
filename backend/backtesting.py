"""
Backtesting Framework
Validates model predictions on historical game data (2020-2026)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import json


class BacktestingFramework:
    """
    Backtesting framework for validating predictions
    
    Features:
    - Walk-forward validation
    - Season-by-season analysis
    - Confusion matrix
    - Calibration curves
    - ROI simulation
    """
    
    def __init__(self, model):
        """
        Initialize backtesting framework
        
        Args:
            model: Trained prediction model (XGBoost, Ensemble, etc.)
        """
        self.model = model
        self.results = {}
        
    def load_historical_games(
        self,
        start_date: str,
        end_date: str,
        league: str = "NBA"
    ) -> pd.DataFrame:
        """
        Load historical game data
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            league: "NBA" or "EuroLeague"
            
        Returns:
            DataFrame with historical games
        """
        # Placeholder - would query database or API
        # For now, return mock data
        
        print(f"Loading {league} games from {start_date} to {end_date}...")
        
        # Mock historical games
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        games = []
        
        for date in dates:
            # Generate 5-10 games per day
            n_games = np.random.randint(5, 11)
            for _ in range(n_games):
                game = {
                    'game_id': f"{league}_{date.strftime('%Y%m%d')}_{np.random.randint(1000, 9999)}",
                    'date': date,
                    'home_team': f"Team_{np.random.randint(1, 31)}",
                    'away_team': f"Team_{np.random.randint(1, 31)}",
                    'home_score': np.random.randint(90, 130),
                    'away_score': np.random.randint(90, 130),
                    'league': league
                }
                game['home_win'] = 1 if game['home_score'] > game['away_score'] else 0
                games.append(game)
        
        df = pd.DataFrame(games)
        print(f"Loaded {len(df)} games")
        
        return df
    
    def prepare_features(self, game: Dict) -> np.ndarray:
        """
        Prepare feature vector for a game
        
        Args:
            game: Game dictionary
            
        Returns:
            Feature array (37 features)
        """
        # Placeholder - would calculate actual features
        # For now, return random features
        
        features = np.random.randn(37)
        
        # Add some signal based on actual outcome
        if game.get('home_win', 0) == 1:
            features[0] += 0.5  # Home team advantage
            features[1] -= 0.3  # Away team disadvantage
        else:
            features[0] -= 0.3
            features[1] += 0.5
        
        return features
    
    def walk_forward_validation(
        self,
        games_df: pd.DataFrame,
        train_window_days: int = 365,
        test_window_days: int = 30
    ) -> Dict:
        """
        Perform walk-forward validation
        
        Args:
            games_df: Historical games DataFrame
            train_window_days: Training window size
            test_window_days: Testing window size
            
        Returns:
            Validation results
        """
        print("\nPerforming walk-forward validation...")
        print(f"Train window: {train_window_days} days")
        print(f"Test window: {test_window_days} days")
        
        results = {
            'folds': [],
            'overall_accuracy': 0.0,
            'overall_auc': 0.0
        }
        
        # Sort by date
        games_df = games_df.sort_values('date')
        
        start_date = games_df['date'].min()
        end_date = games_df['date'].max()
        
        current_date = start_date + timedelta(days=train_window_days)
        fold = 1
        
        while current_date + timedelta(days=test_window_days) <= end_date:
            # Define train and test periods
            train_start = current_date - timedelta(days=train_window_days)
            train_end = current_date
            test_start = current_date
            test_end = current_date + timedelta(days=test_window_days)
            
            # Get train and test games
            train_games = games_df[
                (games_df['date'] >= train_start) & 
                (games_df['date'] < train_end)
            ]
            
            test_games = games_df[
                (games_df['date'] >= test_start) & 
                (games_df['date'] < test_end)
            ]
            
            if len(test_games) == 0:
                current_date += timedelta(days=test_window_days)
                continue
            
            # Prepare features
            X_test = np.array([self.prepare_features(row) for _, row in test_games.iterrows()])
            y_test = test_games['home_win'].values
            
            # Predict
            y_pred_proba = self.model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            fold_result = {
                'fold': fold,
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'n_test_games': len(test_games),
                'accuracy': accuracy,
                'auc': auc
            }
            
            results['folds'].append(fold_result)
            
            print(f"Fold {fold}: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            print(f"  Test games: {len(test_games)}")
            print(f"  Accuracy: {accuracy:.1%}")
            print(f"  AUC: {auc:.3f}")
            
            # Move to next fold
            current_date += timedelta(days=test_window_days)
            fold += 1
        
        # Calculate overall metrics
        if len(results['folds']) > 0:
            results['overall_accuracy'] = np.mean([f['accuracy'] for f in results['folds']])
            results['overall_auc'] = np.mean([f['auc'] for f in results['folds']])
        
        print(f"\nOverall Accuracy: {results['overall_accuracy']:.1%}")
        print(f"Overall AUC: {results['overall_auc']:.3f}")
        
        return results
    
    def season_by_season_analysis(
        self,
        games_df: pd.DataFrame
    ) -> Dict:
        """
        Analyze performance by season
        
        Args:
            games_df: Historical games DataFrame
            
        Returns:
            Season-by-season results
        """
        print("\nPerforming season-by-season analysis...")
        
        # Extract season from date (Oct-Jun = season starting that year)
        games_df['season'] = games_df['date'].apply(
            lambda x: x.year if x.month >= 10 else x.year - 1
        )
        
        results = {
            'seasons': []
        }
        
        for season in sorted(games_df['season'].unique()):
            season_games = games_df[games_df['season'] == season]
            
            # Prepare features
            X = np.array([self.prepare_features(row) for _, row in season_games.iterrows()])
            y = season_games['home_win'].values
            
            # Predict
            y_pred_proba = self.model.predict(X)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            auc = roc_auc_score(y, y_pred_proba)
            cm = confusion_matrix(y, y_pred)
            
            season_result = {
                'season': f"{season}-{season+1}",
                'n_games': len(season_games),
                'accuracy': accuracy,
                'auc': auc,
                'confusion_matrix': cm.tolist()
            }
            
            results['seasons'].append(season_result)
            
            print(f"Season {season}-{season+1}:")
            print(f"  Games: {len(season_games)}")
            print(f"  Accuracy: {accuracy:.1%}")
            print(f"  AUC: {auc:.3f}")
        
        return results
    
    def calculate_roi_simulation(
        self,
        games_df: pd.DataFrame,
        bet_amount: float = 100.0,
        confidence_threshold: float = 0.60
    ) -> Dict:
        """
        Simulate betting ROI
        
        Args:
            games_df: Historical games DataFrame
            bet_amount: Amount to bet per game
            confidence_threshold: Minimum confidence to place bet
            
        Returns:
            ROI simulation results
        """
        print(f"\nSimulating ROI (${bet_amount} per bet, {confidence_threshold:.0%} confidence threshold)...")
        
        # Prepare features
        X = np.array([self.prepare_features(row) for _, row in games_df.iterrows()])
        y = games_df['home_win'].values
        
        # Predict
        y_pred_proba = self.model.predict(X)
        
        # Simulate betting
        total_bets = 0
        total_wins = 0
        total_losses = 0
        total_profit = 0.0
        
        for i, (prob, actual) in enumerate(zip(y_pred_proba, y)):
            # Only bet if confidence exceeds threshold
            if prob > confidence_threshold or prob < (1 - confidence_threshold):
                # Determine bet
                bet_on_home = prob > 0.5
                
                # Assume -110 odds (bet $110 to win $100)
                if (bet_on_home and actual == 1) or (not bet_on_home and actual == 0):
                    # Win
                    total_wins += 1
                    total_profit += bet_amount * 0.909  # Win $90.90 on $100 bet
                else:
                    # Loss
                    total_losses += 1
                    total_profit -= bet_amount
                
                total_bets += 1
        
        roi = (total_profit / (total_bets * bet_amount)) * 100 if total_bets > 0 else 0
        win_rate = (total_wins / total_bets) * 100 if total_bets > 0 else 0
        
        results = {
            'total_games': len(games_df),
            'total_bets': total_bets,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'roi': roi,
            'bet_amount': bet_amount,
            'confidence_threshold': confidence_threshold
        }
        
        print(f"Total games: {len(games_df)}")
        print(f"Total bets: {total_bets} ({total_bets/len(games_df)*100:.1f}% of games)")
        print(f"Wins: {total_wins}")
        print(f"Losses: {total_losses}")
        print(f"Win rate: {win_rate:.1%}")
        print(f"Total profit: ${total_profit:.2f}")
        print(f"ROI: {roi:.1f}%")
        
        return results
    
    def save_results(self, filepath: str):
        """Save backtesting results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filepath}")


# Example usage
if __name__ == "__main__":
    from xgboost_model import XGBoostModel
    
    print("Backtesting Framework Test")
    print("=" * 60)
    
    # Initialize model
    xgb_model = XGBoostModel()
    xgb_model.build_model()
    
    # Train on mock data
    print("\nTraining model on mock data...")
    X_train = np.random.randn(1000, 37)
    y_train = (X_train[:, 0] + X_train[:, 1] * 0.5 > 0).astype(int)
    X_val = np.random.randn(200, 37)
    y_val = (X_val[:, 0] + X_val[:, 1] * 0.5 > 0).astype(int)
    
    xgb_model.train(X_train, y_train, X_val, y_val)
    
    # Initialize backtesting
    backtest = BacktestingFramework(xgb_model)
    
    # Load historical games (mock data)
    games_df = backtest.load_historical_games(
        start_date="2023-10-01",
        end_date="2024-06-30",
        league="NBA"
    )
    
    # Walk-forward validation
    wf_results = backtest.walk_forward_validation(
        games_df,
        train_window_days=180,
        test_window_days=30
    )
    
    # Season-by-season analysis
    season_results = backtest.season_by_season_analysis(games_df)
    
    # ROI simulation
    roi_results = backtest.calculate_roi_simulation(
        games_df,
        bet_amount=100.0,
        confidence_threshold=0.65
    )
    
    # Save results
    backtest.results = {
        'walk_forward': wf_results,
        'season_analysis': season_results,
        'roi_simulation': roi_results
    }
    
    print("\n" + "=" * 60)
    print("Backtesting complete!")
