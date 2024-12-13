import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import glob

def analyze_tracking_file(file_path, team_colors):
    """Comprehensive analysis of a single tracking file"""
    data = pd.read_csv(file_path)
    file_name = os.path.basename(file_path)
    
    # Calculate key statistics for title
    stats_summary = {
        "Total Tracks": data['tracking_id'].nunique(),
        "Total Frames": data['frame'].nunique(),
        "Frame Range": f"{data['frame'].min()}-{data['frame'].max()}",
        "Teams": sorted(data['team_id'].unique()),
        "Players per Team": data.groupby('team_id')['tracking_id'].nunique().to_dict(),
        "Avg Track Length": f"{len(data) / data['tracking_id'].nunique():.1f} frames"
    }
    
    # Create the title with statistics
    title_text = (
        f"Analysis: {file_name}<br>"
        f"<span style='font-size: 12px;'>"
        f"Total Tracks: {stats_summary['Total Tracks']} | "
        f"Frames: {stats_summary['Frame Range']} | "
        f"Teams: {stats_summary['Teams']} | "
        f"Avg Track Length: {stats_summary['Avg Track Length']}"
        f"</span>"
    )
    
    # Quality filtering (from team_3d_visualization.py)
    VELOCITY_THRESHOLD = 50
    MIN_TRACK_LENGTH = 30
    MAX_POSITION_JUMP = 50
    
    valid_track_ids = []
    for track_id in data['tracking_id'].unique():
        track = data[data['tracking_id'] == track_id]
        track_length = len(track)
        max_velocity = track['velocity'].max() if 'velocity' in data.columns else 0
        x_jumps = abs(track['pitch_x'].diff()).max()
        y_jumps = abs(track['pitch_y'].diff()).max()
        has_duplicates = track.groupby('frame').size().max() > 1
        
        is_valid = (
            track_length >= MIN_TRACK_LENGTH and
            max_velocity <= VELOCITY_THRESHOLD and
            x_jumps <= MAX_POSITION_JUMP and
            y_jumps <= MAX_POSITION_JUMP and
            not has_duplicates
        )
        if is_valid:
            valid_track_ids.append(track_id)
    
    filtered_data = data[data['tracking_id'].isin(valid_track_ids)]
    
    # Create main figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}]
        ],
        subplot_titles=('3D Trajectories', 'Track Length Distribution', 'Active Tracks'),
        column_widths=[0.7, 0.3]
    )
    
    # Add 3D trajectories
    for team_id in filtered_data['team_id'].unique():
        team_data = filtered_data[filtered_data['team_id'] == team_id]
        for player_id in team_data['tracking_id'].unique():
            player_data = team_data[team_data['tracking_id'] == player_id].copy()
            player_data = player_data.iloc[::3]  # Downsample for performance
            
            fig.add_trace(
                go.Scatter3d(
                    x=player_data['pitch_x'],
                    y=player_data['pitch_y'],
                    z=player_data['frame'],
                    mode='lines',
                    name=f"Player {player_id} (Team {team_id})",
                    line=dict(color=team_colors[team_id], width=2),
                    opacity=0.6
                ),
                row=1, col=1
            )
    
    # Add histograms
    track_lengths = filtered_data.groupby('tracking_id').size()
    fig.add_trace(
        go.Histogram(x=track_lengths.values, name='Track Lengths'),
        row=1, col=2
    )
    
    tracks_per_frame = filtered_data.groupby('frame')['tracking_id'].nunique()
    fig.add_trace(
        go.Scatter(x=tracks_per_frame.index, y=tracks_per_frame.values, name='Active Tracks'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1000,
        width=1500,
        title=title_text,
        showlegend=True,
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white')
    )
    
    
    # Update 3D scene
    fig.update_scenes(
        bgcolor='black',
        xaxis=dict(title='Pitch X', gridcolor='gray', showbackground=False),
        yaxis=dict(title='Pitch Y', gridcolor='gray', showbackground=False),
        zaxis=dict(title='Frame', gridcolor='gray', showbackground=False),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
    
    return fig, stats_summary

if __name__ == "__main__":
    team_colors = {
        -1: 'rgb(128, 128, 128)',    # grey
        0: 'rgb(10, 10, 10)',        # almost black
        1: 'rgb(0, 0, 230)',         # blue
        2: 'rgb(0, 255, 10)',        # green
        3: 'rgb(0, 220, 130)'        # light green
    }
    
    # Updated paths for the three camera views
    pattern_files = [
        "../data/camL_1.csv",
        "../data/camM_1.csv",
        "../data/camR_1.csv"
    ]
    
    print(f"\nAnalyzing camera files")
    
    for csv_file in pattern_files:
        try:
            print(f"\n{'='*50}")
            print(f"Analyzing: {os.path.basename(csv_file)}")
            print(f"{'='*50}")
            
            if os.path.exists(csv_file):
                fig, stats = analyze_tracking_file(csv_file, team_colors)
                
                # Print statistics
                for key, value in stats.items():
                    print(f"{key}: {value}")
                
                fig.show()
                
                # Optional: wait for user input before showing next plot
                input("\nPress Enter to continue to next file...")
                
            else:
                print(f"File not found: {csv_file}")
                
        except Exception as e:
            print(f"Error analyzing {csv_file}: {str(e)}")
            continue