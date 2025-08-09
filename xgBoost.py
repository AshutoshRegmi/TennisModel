from fast_tennis_processor import FastTennisProcessor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')

def create_powerful_tennis_features(X):
    """Create only the most powerful tennis prediction features"""
    print("Creating laser-focused tennis features...")
    
    X_new = X.copy()
    
    # CORE RANKING FEATURES (these are money in tennis)
    X_new['rank_ratio'] = X_new['player2_rank'] / (X_new['player1_rank'] + 0.1)
    X_new['rank_advantage'] = np.log(X_new['player2_rank'] + 1) - np.log(X_new['player1_rank'] + 1)
    X_new['skill_gap'] = 1/(X_new['player1_rank'] + 1) - 1/(X_new['player2_rank'] + 1)
    
    # ELITE STATUS (critical breakpoints in tennis)
    X_new['p1_elite'] = (X_new['player1_rank'] <= 10).astype(int)
    X_new['p2_elite'] = (X_new['player2_rank'] <= 10).astype(int)
    X_new['p1_top20'] = (X_new['player1_rank'] <= 20).astype(int)
    X_new['p2_top20'] = (X_new['player2_rank'] <= 20).astype(int)
    X_new['p1_top50'] = (X_new['player1_rank'] <= 50).astype(int)
    X_new['p2_top50'] = (X_new['player2_rank'] <= 50).astype(int)
    
    # SKILL TIER MATCHUPS (these patterns matter hugely)
    X_new['elite_vs_regular'] = ((X_new['p1_elite'] & ~X_new['p2_top20']) | 
                                (~X_new['p1_top20'] & X_new['p2_elite'])).astype(int)
    X_new['top20_clash'] = (X_new['p1_top20'] & X_new['p2_top20']).astype(int)
    X_new['upset_alert'] = ((X_new['player1_rank'] > 50) & (X_new['player2_rank'] <= 15)).astype(int)
    
    # RANKING GAP CATEGORIES (non-linear importance)
    gap = abs(X_new['rank_difference'])
    X_new['tiny_gap'] = (gap <= 5).astype(int)
    X_new['small_gap'] = ((gap > 5) & (gap <= 15)).astype(int)
    X_new['medium_gap'] = ((gap > 15) & (gap <= 40)).astype(int)
    X_new['large_gap'] = ((gap > 40) & (gap <= 100)).astype(int)
    X_new['massive_gap'] = (gap > 100).astype(int)
    
    # TOURNAMENT IMPORTANCE (context matters)
    X_new['grand_slam'] = (X_new['tourney_level_numeric'] == 4).astype(int)
    X_new['masters'] = (X_new['tourney_level_numeric'] == 3).astype(int)
    X_new['big_event'] = (X_new['tourney_level_numeric'] >= 3).astype(int)
    
    # ONLY keep age features if they help
    if 'player1_age' in X_new.columns:
        X_new['age_advantage'] = X_new['age_difference']
        X_new['experience_gap'] = abs(X_new['age_difference'])
        # Prime age features
        X_new['p1_prime'] = ((X_new['player1_age'] >= 24) & (X_new['player1_age'] <= 29)).astype(int)
        X_new['p2_prime'] = ((X_new['player2_age'] >= 24) & (X_new['player2_age'] <= 29)).astype(int)
    
    # SURFACE (keep simple)
    # Original surface features are fine
    
    return X_new

def create_historical_indicators(df):
    """Create simple historical performance indicators"""
    print("Creating historical performance indicators...")
    
    # Sort by date for historical calculation
    df_sorted = df.sort_values(['tourney_date', 'match_num']).copy()
    
    # Simple moving averages for recent form
    player_recent_wins = {}
    historical_features = []
    
    for idx, row in df_sorted.iterrows():
        p1_id = row['player1_id']
        p2_id = row['player2_id']
        p1_wins = row['player1_wins']
        
        # Get recent form (last 10 matches)
        p1_recent = player_recent_wins.get(p1_id, [])
        p2_recent = player_recent_wins.get(p2_id, [])
        
        # Calculate form indicators
        p1_form = np.mean(p1_recent[-10:]) if len(p1_recent) >= 5 else 0.5
        p2_form = np.mean(p2_recent[-10:]) if len(p2_recent) >= 5 else 0.5
        p1_matches = len(p1_recent)
        p2_matches = len(p2_recent)
        
        historical_features.append({
            'p1_recent_form': p1_form,
            'p2_recent_form': p2_form,
            'p1_experience': min(p1_matches / 100, 1.0),  # Normalize experience
            'p2_experience': min(p2_matches / 100, 1.0),
            'form_advantage': p1_form - p2_form
        })
        
        # Update player records
        if p1_id not in player_recent_wins:
            player_recent_wins[p1_id] = []
        if p2_id not in player_recent_wins:
            player_recent_wins[p2_id] = []
        
        player_recent_wins[p1_id].append(p1_wins)
        player_recent_wins[p2_id].append(1 - p1_wins)
        
        # Keep only recent results
        if len(player_recent_wins[p1_id]) > 50:
            player_recent_wins[p1_id] = player_recent_wins[p1_id][-50:]
        if len(player_recent_wins[p2_id]) > 50:
            player_recent_wins[p2_id] = player_recent_wins[p2_id][-50:]
    
    return pd.DataFrame(historical_features)

def optimize_xgboost_70_percent():
    """Build XGBoost model"""
    
    print("XGBOOST TENNIS")
    print("========================================")
    
    # Load data
    print("Loading tennis data...")
    processor = FastTennisProcessor(data_path='./data/')
    X, y, df = processor.run_fast_pipeline(start_year=2000, end_year=2024)
    
    print(f"Dataset: {len(X):,} matches")
    
    # Create enhanced features
    X_enhanced = create_powerful_tennis_features(X)
    
    # Add historical indicators
    historical_df = create_historical_indicators(df)
    for col in historical_df.columns:
        X_enhanced[col] = historical_df[col]
    
    # Clean data
    X_enhanced = X_enhanced.replace([np.inf, -np.inf], np.nan)
    X_enhanced = X_enhanced.fillna(0)
    
    print(f"Total features: {X_enhanced.shape[1]} (added {X_enhanced.shape[1] - X.shape[1]})")
    
    # Feature selection - keep only the best features
    print("Selecting most predictive features...")
    selector = SelectKBest(score_func=f_classif, k=25)  # Keep top 25 features
    X_selected = selector.fit_transform(X_enhanced, y)
    selected_features = X_enhanced.columns[selector.get_support()]
    
    print(f"Selected features ({len(selected_features)}):")
    for i, feature in enumerate(selected_features):
        if i < 10:  # Show first 10
            print(f"  {feature}")
        elif i == 10:
            print(f"  ... and {len(selected_features) - 10} more")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining: {len(X_train):,} matches")
    print(f"Testing: {len(X_test):,} matches")
    
    # Optimized XGBoost configurations for tennis
    xgb_configs = [
        {
            'name': 'XGB-Tennis-Focused',
            'params': {
                'n_estimators': 800,
                'max_depth': 7,
                'learning_rate': 0.05,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'min_child_weight': 2,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42
            }
        },
        {
            'name': 'XGB-Deep-Learning',
            'params': {
                'n_estimators': 1000,
                'max_depth': 9,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 0.05,
                'reg_lambda': 0.05,
                'random_state': 42
            }
        },
        {
            'name': 'XGB-Ensemble-Ready',
            'params': {
                'n_estimators': 600,
                'max_depth': 8,
                'learning_rate': 0.07,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'min_child_weight': 3,
                'reg_alpha': 0.2,
                'reg_lambda': 0.1,
                'random_state': 42
            }
        }
    ]
    
    results = []
    models = []
    
    print(f"\nTesting optimized XGBoost configurations...")
    print("=" * 50)
    
    for config in xgb_configs:
        print(f"\nTraining {config['name']}...")
        
        # Create and train model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            verbosity=0,
            **config['params']
        )
        
        # Train model (simplified - no early stopping for compatibility)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        cv_mean = cv_scores.mean()
        
        print(f"  Test Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        print(f"  CV Score: {cv_mean:.4f} ± {cv_scores.std():.4f}")
        
        results.append({
            'name': config['name'],
            'accuracy': accuracy,
            'cv_score': cv_mean,
            'model': model
        })
        
        models.append(model)
    
    # Advanced hyperparameter tuning on best model
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\nAdvanced tuning on {best_result['name']}...")
    
    # Fine-grained parameter grid
    param_grid = {
        'n_estimators': [800, 1200],
        'max_depth': [7, 8, 9],
        'learning_rate': [0.03, 0.05, 0.07],
        'subsample': [0.8, 0.85, 0.9],
        'colsample_bytree': [0.8, 0.85, 0.9]
    }
    
    base_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'min_child_weight': 2,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbosity': 0
    }
    
    # Grid search with cross-validation
    print("Running intensive hyperparameter search...")
    xgb_tuned = xgb.XGBClassifier(**base_params)
    
    grid_search = GridSearchCV(
        xgb_tuned,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    # Final tuned model
    tuned_model = grid_search.best_estimator_
    tuned_pred = tuned_model.predict(X_test)
    tuned_accuracy = accuracy_score(y_test, tuned_pred)
    
    print(f"Tuned accuracy: {tuned_accuracy:.4f} ({tuned_accuracy:.1%})")
    print(f"Best params: {grid_search.best_params_}")
    
    # Model ensemble - combine top performers
    print(f"\nCreating ensemble of best models...")
    
    # Get top 3 models
    top_models = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:3]
    
    # Ensemble prediction (weighted average)
    ensemble_preds = []
    weights = [0.4, 0.35, 0.25]  # Weight better models more
    
    for i, y_test_idx in enumerate(y_test.index):
        weighted_prob = 0
        for j, result in enumerate(top_models):
            model = result['model']
            prob = model.predict_proba(X_test[i:i+1])[0][1]  # Probability of class 1
            weighted_prob += weights[j] * prob
        
        ensemble_preds.append(1 if weighted_prob > 0.5 else 0)
    
    ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
    print(f"Ensemble accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy:.1%})")
    
    # Choose best final model
    all_results = results + [
        {'name': 'XGB-Tuned', 'accuracy': tuned_accuracy, 'model': tuned_model},
        {'name': 'Ensemble', 'accuracy': ensemble_accuracy, 'model': None}
    ]
    
    final_result = max(all_results, key=lambda x: x['accuracy'])
    final_accuracy = final_result['accuracy']
    final_name = final_result['name']
    
    # Results
    print("\n" + "=" * 60)
    print("FINAL RESULTS - TARGET: 70% ACCURACY")
    print("=" * 60)
    
    print(f"Best approach: {final_name}")
    print(f"Final accuracy: {final_accuracy:.4f} ({final_accuracy:.1%})")
    
    # Check target achievement
    target = 0.70
    baseline = 0.657
    
    
    # All model comparison
    print(f"\nAll Model Results:")
    print("-" * 40)
    for result in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
        star = "⭐" if result['name'] == final_name else "  "
        print(f"{star} {result['name']:<20}: {result['accuracy']:.1%}")
    
    # Feature importance (if available)
    if final_result['model'] and hasattr(final_result['model'], 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': selected_features,
            'importance': final_result['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 15 Most Important Features:")
        print("-" * 45)
        for _, row in importance.head(15).iterrows():
            print(f"{row['feature']:<25}: {row['importance']:.4f}")
    
    print(f"\n🏆 Best XGBoost Result: {final_accuracy:.1%}")
    
    return final_result['model'], final_accuracy, selected_features

if __name__ == "__main__":
    model, accuracy, features = optimize_xgboost_70_percent()
    
    print(f"\n🚀 XGBoost Optimization Complete!")
    print(f"Final Accuracy: {accuracy:.1%}")
    