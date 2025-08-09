from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from fast_tennis_processor import FastTennisProcessor

def build_tennis_decision_tree():
    """Build decision tree model for tennis match prediction"""
    
    # Load and process data
    print("Loading tennis data...")
    processor = FastTennisProcessor(data_path='./data/')
    X, y, df = processor.run_fast_pipeline(start_year=2000, end_year=2024)
    
    print(f"\nDataset loaded: {len(X):,} matches")
    print(f"Features: {X.shape[1]}")
    print(f"Target balance: {y.mean():.1%} player1 wins")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train):,} matches")
    print(f"Test set: {len(X_test):,} matches")
    
    # Build decision tree with moderate constraints
    model = DecisionTreeClassifier(
        max_depth=8,              # Back to reasonable depth
        min_samples_split=50,     # Smaller splits
        min_samples_leaf=25,      # Smaller leaves  
        max_features=None,        # Use all features
        random_state=42
    )
    
    # Train model
    print("\nTraining decision tree...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Results
    print("\n" + "="*50)
    print("DECISION TREE RESULTS")
    print("="*50)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print("-" * 30)
    for _, row in importance.iterrows():
        if row['importance'] > 0:
            print(f"{row['feature']:<20}: {row['importance']:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print("-" * 30)
    print(classification_report(y_test, y_pred, 
                              target_names=['Player2 wins', 'Player1 wins']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print("-" * 30)
    print(f"True Negatives (Player2 wins correctly): {cm[0,0]:,}")
    print(f"False Positives (Predicted Player1, actual Player2): {cm[0,1]:,}")
    print(f"False Negatives (Predicted Player2, actual Player1): {cm[1,0]:,}")
    print(f"True Positives (Player1 wins correctly): {cm[1,1]:,}")
    
    # Sample predictions
    print(f"\nSample Predictions:")
    print("-" * 30)
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    for i, idx in enumerate(sample_indices):
        actual = y_test.iloc[idx]
        predicted = y_pred[idx]
        probability = model.predict_proba(X_test.iloc[idx:idx+1])[0]
        
        player1_rank = X_test.iloc[idx]['player1_rank']
        player2_rank = X_test.iloc[idx]['player2_rank']
        
        result = "✓ CORRECT" if actual == predicted else "✗ WRONG"
        actual_winner = "Player1" if actual == 1 else "Player2"
        pred_winner = "Player1" if predicted == 1 else "Player2"
        
        print(f"Match {i+1}: Rank {player1_rank:.0f} vs Rank {player2_rank:.0f}")
        print(f"  Actual: {actual_winner}, Predicted: {pred_winner} {result}")
        print(f"  Confidence: {max(probability):.1%}")
        print()
    
    # Model insights
    print("Model Insights:")
    print("-" * 30)
    most_important = importance.iloc[0]
    print(f"Most important feature: {most_important['feature']} ({most_important['importance']:.3f})")
    
    ranking_importance = importance[importance['feature'].str.contains('rank')]['importance'].sum()
    other_importance = importance[~importance['feature'].str.contains('rank')]['importance'].sum()
    
    print(f"Ranking features importance: {ranking_importance:.3f} ({ranking_importance*100:.1f}%)")
    print(f"Other features importance: {other_importance:.3f} ({other_importance*100:.1f}%)")
    

    
    return model, accuracy, importance

if __name__ == "__main__":
    model, accuracy, feature_importance = build_tennis_decision_tree()
    
    print(f"\n🎾 Tennis Decision Tree Model Complete!")
    print(f"Final Accuracy: {accuracy:.1%}")