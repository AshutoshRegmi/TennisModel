from fast_tennis_processor import FastTennisProcessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_only_effective_features(X):
    """Create ONLY the most effective features - no noise"""
    print("Creating minimal but powerful features...")
    
    X_new = X.copy()
    
    # ONLY the best ranking features (proven to work)
    X_new['rank_ratio'] = X_new['player2_rank'] / (X_new['player1_rank'] + 1)  # >1 means p1 better
    X_new['rank_log_ratio'] = np.log(X_new['player2_rank'] + 1) / np.log(X_new['player1_rank'] + 1)
    X_new['rank_sqrt_ratio'] = np.sqrt(X_new['player2_rank']) / np.sqrt(X_new['player1_rank'] + 1)
    
    # Elite status - clear cutoffs
    X_new['p1_elite'] = (X_new['player1_rank'] <= 20).astype(int)
    X_new['p2_elite'] = (X_new['player2_rank'] <= 20).astype(int)
    X_new['p1_top10'] = (X_new['player1_rank'] <= 10).astype(int) 
    X_new['p2_top10'] = (X_new['player2_rank'] <= 10).astype(int)
    
    # Simple but effective gap indicators
    X_new['huge_gap'] = (abs(X_new['rank_difference']) > 100).astype(int)
    X_new['elite_vs_regular'] = ((X_new['p1_elite'] & ~X_new['p2_elite']) | 
                                (~X_new['p1_elite'] & X_new['p2_elite'])).astype(int)
    
    # Tournament importance (only if it helps)
    X_new['big_tournament'] = (X_new['tourney_level_numeric'] >= 3).astype(int)
    
    # Age features (only simple ones)
    if 'player1_age' in X_new.columns:
        X_new['age_gap'] = abs(X_new['age_difference'])
        X_new['both_experienced'] = ((X_new['player1_age'] > 28) & (X_new['player2_age'] > 28)).astype(int)
    
    # Surface (keep simple)
    # Just keep the original surface dummies
    
    return X_new

def build_lean_random_forest():
    """Build a lean Random Forest focused on what actually works"""
    
    # Load data
    print("Loading tennis data...")
    processor = FastTennisProcessor(data_path='./data/')
    X, y, df = processor.run_fast_pipeline(start_year=2000, end_year=2024)
    
    print(f"Dataset: {len(X):,} matches")
    print(f"Original features: {X.shape[1]}")
    print(f"Target: Beat 65.8% (your current best)")
    
    # Create minimal enhanced features
    X_enhanced = create_only_effective_features(X)
    
    # Clean data
    X_enhanced = X_enhanced.replace([np.inf, -np.inf], np.nan)
    X_enhanced = X_enhanced.fillna(0)
    
    print(f"Enhanced features: {X_enhanced.shape[1]} (added {X_enhanced.shape[1] - X.shape[1]})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature selection - only keep the best features
    print("Selecting best features...")
    selector = SelectKBest(score_func=f_classif, k=20)  # Keep only top 20
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_enhanced.columns[selector.get_support()]
    print(f"Selected features: {list(selected_features)}")
    
    # Test different Random Forest configurations
    configs = [
        # Baseline config
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 10},
        # More trees
        {'n_estimators': 500, 'max_depth': 12, 'min_samples_split': 15, 'min_samples_leaf': 8},
        # Deeper trees
        {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 5},
        # Many trees, conservative
        {'n_estimators': 1000, 'max_depth': 8, 'min_samples_split': 50, 'min_samples_leaf': 20},
    ]
    
    best_accuracy = 0
    best_model = None
    best_config = None
    
    print(f"\nTesting {len(configs)} configurations on selected features...")
    
    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}: {config['n_estimators']} trees, depth {config['max_depth']}")
        
        model = RandomForestClassifier(
            **config,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        # Train on selected features
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_config = config
    
    print(f"\nBest config: {best_config}")
    print(f"Best accuracy: {best_accuracy:.4f} ({best_accuracy:.1%})")
    
    # Try one more approach - use ALL features but with different RF settings
    print(f"\nTrying alternative: All features with optimized RF...")
    
    alt_model = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,  # No depth limit
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='log2',  # Different feature selection
        bootstrap=True,
        oob_score=True,  # Out-of-bag score
        random_state=42,
        n_jobs=-1
    )
    
    alt_model.fit(X_train, y_train)
    alt_pred = alt_model.predict(X_test)
    alt_accuracy = accuracy_score(y_test, alt_pred)
    
    print(f"Alternative accuracy: {alt_accuracy:.4f} ({alt_accuracy:.1%})")
    print(f"OOB Score: {alt_model.oob_score_:.4f}")
    
    # Pick the best approach
    if alt_accuracy > best_accuracy:
        final_model = alt_model
        final_accuracy = alt_accuracy
        final_features = X_enhanced.columns
        approach = "All features + optimized RF"
    else:
        final_model = best_model
        final_accuracy = best_accuracy  
        final_features = selected_features
        approach = "Feature selection + best config"
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best approach: {approach}")
    print(f"Final accuracy: {final_accuracy:.4f} ({final_accuracy:.1%})")
    
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': final_features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 15 Most Important Features:")
    print("-" * 40)
    for _, row in importance.head(15).iterrows():
        print(f"{row['feature']:<25}: {row['importance']:.4f}")
    
    # Check what's actually driving performance
    ranking_importance = importance[importance['feature'].str.contains('rank|elite|top')]['importance'].sum()
    print(f"\nRanking features total importance: {ranking_importance:.3f} ({ranking_importance*100:.1f}%)")
    

    # Quick prediction samples
    print(f"\nSample Predictions:")
    print("-" * 30)
    
    if approach == "All features + optimized RF":
        sample_X = X_test
    else:
        sample_X = X_test_selected
    
    for i in range(3):
        actual = y_test.iloc[i]
        predicted = final_model.predict(sample_X[i:i+1])[0]
        prob = final_model.predict_proba(sample_X[i:i+1])[0]
        
        p1_rank = X_test.iloc[i]['player1_rank']
        p2_rank = X_test.iloc[i]['player2_rank']
        
        result = "✓" if actual == predicted else "✗"
        winner = "Player1" if predicted == 1 else "Player2"
        conf = max(prob)
        
        print(f"  {i+1}. Rank {p1_rank:.0f} vs {p2_rank:.0f}: {winner} ({conf:.1%}) {result}")
    
    # Diagnosis
    
    
    return final_model, final_accuracy, importance

if __name__ == "__main__":
    print("🎾 TENNIS PREDICTOR")
    print("===============================")
    
    model, accuracy, importance = build_lean_random_forest()
    
    print(f"\n🏆 Final Result: {accuracy:.1%}")
