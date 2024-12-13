import pandas as pd
import numpy as np
import os

def analyze_x_ranges(left_data, right_data):
    # Filter for sprint frames
    left_sprint = left_data[
        (left_data['frame'] >= 44589) & 
        (left_data['frame'] <= 45372)
    ]
    right_sprint = right_data[
        (right_data['frame'] >= 46474) & 
        (right_data['frame'] <= 47162)
    ]
    
    print("\nX-coordinate ranges during sprints:")
    print(f"Left camera X range: {left_sprint['pitch_x'].min():.1f} - {left_sprint['pitch_x'].max():.1f}")
    print(f"Right camera X range: {right_sprint['pitch_x'].min():.1f} - {right_sprint['pitch_x'].max():.1f}")
    
    # Print distribution in bins
    print("\nLeft camera X-coordinate distribution (in 50-unit bins):")
    left_bins = np.histogram(left_sprint['pitch_x'], bins=np.arange(0, 1001, 50))
    for i in range(len(left_bins[0])):
        if left_bins[0][i] > 0:
            print(f"Range {left_bins[1][i]:.0f}-{left_bins[1][i+1]:.0f}: {left_bins[0][i]} points")
    
    print("\nRight camera X-coordinate distribution (in 50-unit bins):")
    right_bins = np.histogram(right_sprint['pitch_x'], bins=np.arange(0, 1001, 50))
    for i in range(len(right_bins[0])):
        if right_bins[0][i] > 0:
            print(f"Range {right_bins[1][i]:.0f}-{right_bins[1][i+1]:.0f}: {right_bins[0][i]} points")

if __name__ == "__main__":
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    
    left_path = os.path.join(data_dir, 'camL_1.csv')
    right_path = os.path.join(data_dir, 'camR_1.csv')
    
    print("Loading data...")
    left_data = pd.read_csv(left_path)
    right_data = pd.read_csv(right_path)
    
    analyze_x_ranges(left_data, right_data) 