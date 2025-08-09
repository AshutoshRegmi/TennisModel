import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

class FastTennisProcessor:
    def __init__(self, data_path='./data/'):
        self.data_path = data_path
        self.df = None
        self.players_df = None
        self.rankings_df = None
        self.processed_df = None
        
    def load_players_data(self):
        """Load ATP players data"""
        print("Loading players data...")
        players_file = f"{self.data_path}atp_players.csv"
        
        if os.path.exists(players_file):
            self.players_df = pd.read_csv(players_file)
            print(f"Loaded {len(self.players_df):,} players")
            return True
        else:
            print("Players file not found")
            return False
    
    def load_rankings_data(self):
        """Load ATP rankings data (optional for speed)"""
        print("Loading rankings data...")
        
        ranking_files = ['atp_rankings_00s.csv', 'atp_rankings_10s.csv', 'atp_rankings_20s.csv']
        ranking_dfs = []
        
        for file in ranking_files:
            file_path = f"{self.data_path}{file}"
            if os.path.exists(file_path):
                df_rankings = pd.read_csv(file_path)
                ranking_dfs.append(df_rankings)
                print(f"Loaded {file}: {len(df_rankings):,} records")
        
        if ranking_dfs:
            self.rankings_df = pd.concat(ranking_dfs, ignore_index=True)
            print(f"Total ranking records: {len(self.rankings_df):,}")
            return True
        else:
            print("No ranking files found - using match rankings")
            return False
    
    def load_data(self, start_year=2000, end_year=2024):
        """Load ATP match data (25 years of data)"""
        print(f"Loading match data from {start_year} to {end_year}...")
        
        dfs = []
        for year in range(start_year, end_year + 1):
            file_path = f"{self.data_path}atp_matches_{year}.csv"
            if os.path.exists(file_path):
                df_year = pd.read_csv(file_path)
                df_year['year'] = year
                dfs.append(df_year)
                print(f"Loaded {year}: {len(df_year):,} matches")
        
        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
            print(f"Total: {len(self.df):,} matches")
            return self.df
        else:
            raise Exception("No match files found")
    
    def create_features_fast(self):
        """Fast feature creation without data leakage"""
        print("Creating features without data leakage...")
        
        df = self.df.copy()
        
        # Remove data leakage by creating random player1/player2 assignments
        np.random.seed(42)
        flip_mask = np.random.random(len(df)) < 0.5
        
        # Create player1 and player2 features (NO winner/loser info)
        df['player1_id'] = np.where(flip_mask, df['loser_id'], df['winner_id'])
        df['player2_id'] = np.where(flip_mask, df['winner_id'], df['loser_id'])
        df['player1_rank'] = np.where(flip_mask, df['loser_rank'], df['winner_rank'])
        df['player2_rank'] = np.where(flip_mask, df['winner_rank'], df['loser_rank'])
        df['player1_age'] = np.where(flip_mask, df['loser_age'], df['winner_age'])
        df['player2_age'] = np.where(flip_mask, df['winner_age'], df['loser_age'])
        df['player1_hand'] = np.where(flip_mask, df['loser_hand'], df['winner_hand'])
        df['player2_hand'] = np.where(flip_mask, df['winner_hand'], df['loser_hand'])
        df['player1_height'] = np.where(flip_mask, df['loser_ht'], df['winner_ht'])
        df['player2_height'] = np.where(flip_mask, df['winner_ht'], df['loser_ht'])
        
        # Target: does player1 win?
        df['player1_wins'] = np.where(flip_mask, 0, 1)  # 1 if player1 wins, 0 if player2 wins
        
        # Create useful features
        df['rank_difference'] = df['player1_rank'] - df['player2_rank']  # positive = player1 lower ranked
        df['age_difference'] = df['player1_age'] - df['player2_age']
        df['height_difference'] = df['player1_height'] - df['player2_height']
        
        # Surface encoding
        df['surface_hard'] = (df['surface'] == 'Hard').astype(int)
        df['surface_clay'] = (df['surface'] == 'Clay').astype(int)
        df['surface_grass'] = (df['surface'] == 'Grass').astype(int)
        
        # Tournament features
        tournament_level_map = {'G': 4, 'M': 3, 'A': 2, 'D': 1, 'F': 0}
        df['tourney_level_numeric'] = df['tourney_level'].map(tournament_level_map)
        
        # Round features
        round_map = {'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3, 'R64': 2, 'R128': 1, 'RR': 0}
        df['round_numeric'] = df['round'].map(round_map).fillna(0)
        
        # Hand features
        df['both_righties'] = ((df['player1_hand'] == 'R') & (df['player2_hand'] == 'R')).astype(int)
        df['player1_lefty'] = (df['player1_hand'] == 'L').astype(int)
        df['player2_lefty'] = (df['player2_hand'] == 'L').astype(int)
        
        # Fill missing values
        df['player1_age'] = df['player1_age'].fillna(df['player1_age'].median())
        df['player2_age'] = df['player2_age'].fillna(df['player2_age'].median())
        df['player1_rank'] = df['player1_rank'].fillna(200)
        df['player2_rank'] = df['player2_rank'].fillna(200)
        
        self.processed_df = df
        print("Feature creation complete - no data leakage")
        return df
    
    def prepare_model_data(self, min_rank_threshold=None):
        """Prepare clean dataset for ML"""
        print("Preparing model data...")
        
        df = self.processed_df.copy()
        
        # No filtering - use all matches with ranking data
        df_filtered = df[
            df['player1_rank'].notna() & 
            df['player2_rank'].notna()
        ].copy()
        
        print(f"Using {len(df_filtered):,} matches (no rank filtering)")
        
        # Select clean features (NO IDs that give away winner)
        feature_columns = [
            'player1_rank', 'player2_rank', 'rank_difference',
            'player1_age', 'player2_age', 'age_difference',
            'height_difference',
            'tourney_level_numeric', 'best_of', 'round_numeric',
            'surface_hard', 'surface_clay', 'surface_grass',
            'both_righties', 'player1_lefty', 'player2_lefty'
        ]
        
        # Keep only available features
        available_features = [col for col in feature_columns if col in df_filtered.columns]
        
        X = df_filtered[available_features].copy()
        y = df_filtered['player1_wins'].copy()
        
        # Fill any remaining missing values
        X = X.fillna(X.median())
        
        print(f"Final dataset: {len(X):,} matches, {len(available_features)} features")
        print(f"Target distribution: Player1 wins: {y.sum():,} ({y.mean():.1%})")
        
        return X, y, df_filtered
    
    def run_fast_pipeline(self, start_year=2000, end_year=2024):
        """Run complete pipeline fast"""
        print("=" * 50)
        print("FAST TENNIS DATA PROCESSOR")
        print("=" * 50)
        
        # Load data
        self.load_players_data()  # Optional
        df = self.load_data(start_year=start_year, end_year=end_year)
        
        # Process without data leakage
        df_processed = self.create_features_fast()
        X, y, df_final = self.prepare_model_data()
        
        print("\n" + "=" * 50)
        print("READY FOR MACHINE LEARNING")
        print("=" * 50)
        print(f"Features: {list(X.columns)}")
        print(f"Shape: {X.shape}")
        print("No data leakage - player IDs properly handled")
        
        return X, y, df_final

# Quick usage
if __name__ == "__main__":
    processor = FastTennisProcessor(data_path='./data/')
    X, y, df = processor.run_fast_pipeline(start_year=2000, end_year=2024)
    
    print("\nSample features:")
    print(X.head())
    print(f"\nTarget: {y.head().tolist()}")