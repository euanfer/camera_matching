import pandas as pd
import numpy as np
import os

def analyze_overlap_data(left_data, right_data, overlap_x_range=(250, 350)):
    """Analyze data in the overlap region"""
    
    # First filter for the sprint frames
    left_sprint = left_data[
        (left_data['frame'] >= 44589) & 
        (left_data['frame'] <= 45372)
    ]
    right_sprint = right_data[
        (right_data['frame'] >= 46474) & 
        (right_data['frame'] <= 47162)
    ]
    
    print("\nSprint Data Summary:")
    print(f"Left sprint frames: {left_sprint['frame'].min()} - {left_sprint['frame'].max()}")
    print(f"Right sprint frames: {right_sprint['frame'].min()} - {right_sprint['frame'].max()}")
    
    # Filter for overlap region
    left_overlap = left_sprint[
        (left_sprint['pitch_x'] >= overlap_x_range[0]) & 
        (left_sprint['pitch_x'] <= overlap_x_range[1])
    ]
    right_overlap = right_sprint[
        (right_sprint['pitch_x'] >= overlap_x_range[0]) & 
        (right_sprint['pitch_x'] <= overlap_x_range[1])
    ]
    
    print("\nOverlap Region Data:")
    print(f"Left camera tracks in overlap: {left_overlap['tracking_id'].nunique()}")
    print(f"Right camera tracks in overlap: {right_overlap['tracking_id'].nunique()}")
    
    print("\nSample of left overlap data:")
    print(left_overlap[['frame', 'tracking_id', 'pitch_x', 'pitch_y']].head(10))
    
    print("\nSample of right overlap data:")
    print(right_overlap[['frame', 'tracking_id', 'pitch_x', 'pitch_y']].head(10))
    
    # Analyze track patterns
    print("\nTrack patterns in overlap region:")
    for track_id in left_overlap['tracking_id'].unique():
        track = left_overlap[left_overlap['tracking_id'] == track_id]
        print(f"\nLeft Track {track_id}:")
        print(f"Frames: {track['frame'].min()} - {track['frame'].max()}")
        print(f"X range: {track['pitch_x'].min():.1f} - {track['pitch_x'].max():.1f}")
        print(f"Y range: {track['pitch_y'].min():.1f} - {track['pitch_y'].max():.1f}")
    
    for track_id in right_overlap['tracking_id'].unique():
        track = right_overlap[right_overlap['tracking_id'] == track_id]
        print(f"\nRight Track {track_id}:")
        print(f"Frames: {track['frame'].min()} - {track['frame'].max()}")
        print(f"X range: {track['pitch_x'].min():.1f} - {track['pitch_x'].max():.1f}")
        print(f"Y range: {track['pitch_y'].min():.1f} - {track['pitch_y'].max():.1f}")

if __name__ == "__main__":
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    
    left_path = os.path.join(data_dir, 'camL_1.csv')
    right_path = os.path.join(data_dir, 'camR_1.csv')
    
    print("Loading data...")
    left_data = pd.read_csv(left_path)
    right_data = pd.read_csv(right_path)
    
    print("\nOverall Data Summary:")
    print(f"Left data shape: {left_data.shape}")
    print(f"Right data shape: {right_data.shape}")
    print(f"Left frame range: {left_data['frame'].min()} - {left_data['frame'].max()}")
    print(f"Right frame range: {right_data['frame'].min()} - {right_data['frame'].max()}")
    
    analyze_overlap_data(left_data, right_data) 