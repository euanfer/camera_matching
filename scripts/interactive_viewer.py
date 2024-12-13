import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def create_interactive_view(left_data, right_data):
    # Filter for sprint frames
    left_sprint = left_data[
        (left_data['frame'] >= 44589) & 
        (left_data['frame'] <= 45372)
    ]
    right_sprint = right_data[
        (right_data['frame'] >= 46474) & 
        (right_data['frame'] <= 47162)
    ]
    
    # Create figure with slider
    fig = go.Figure()
    
    # Add frames for different offsets
    frames = []
    for offset in range(1800, 2000, 5):  # Create frames in steps of 5
        frame_data = []
        
        # Add left camera tracks (these don't change)
        for track_id in left_sprint['tracking_id'].unique():
            track = left_sprint[left_sprint['tracking_id'] == track_id]
            frame_data.append(
                go.Scatter3d(
                    x=track['pitch_x'],
                    y=track['pitch_y'],
                    z=track['frame'],
                    mode='lines',
                    name=f'Left {track_id}',
                    line=dict(color='blue'),
                    opacity=0.6
                )
            )
        
        # Add right camera tracks with current offset
        for track_id in right_sprint['tracking_id'].unique():
            track = right_sprint[right_sprint['tracking_id'] == track_id].copy()
            track['frame'] = track['frame'] - offset
            frame_data.append(
                go.Scatter3d(
                    x=track['pitch_x'],
                    y=track['pitch_y'],
                    z=track['frame'],
                    mode='lines',
                    name=f'Right {track_id}',
                    line=dict(color='red'),
                    opacity=0.6
                )
            )
        
        frames.append(go.Frame(data=frame_data, name=str(offset)))
    
    # Add initial data
    initial_offset = 1885
    for track_id in left_sprint['tracking_id'].unique():
        track = left_sprint[left_sprint['tracking_id'] == track_id]
        fig.add_trace(
            go.Scatter3d(
                x=track['pitch_x'],
                y=track['pitch_y'],
                z=track['frame'],
                mode='lines',
                name=f'Left {track_id}',
                line=dict(color='blue'),
                opacity=0.6
            )
        )
    
    for track_id in right_sprint['tracking_id'].unique():
        track = right_sprint[right_sprint['tracking_id'] == track_id].copy()
        track['frame'] = track['frame'] - initial_offset
        fig.add_trace(
            go.Scatter3d(
                x=track['pitch_x'],
                y=track['pitch_y'],
                z=track['frame'],
                mode='lines',
                name=f'Right {track_id}',
                line=dict(color='red'),
                opacity=0.6
            )
        )
    
    # Add frames to figure
    fig.frames = frames
    
    # Add slider
    fig.update_layout(
        title="Interactive Sprint Comparison",
        scene=dict(
            xaxis_title='Pitch X',
            yaxis_title='Pitch Y',
            zaxis_title='Frame',
            camera=dict(eye=dict(x=2, y=2, z=1.5))
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 100},
                    'fromcurrent': True,
                    'transition': {'duration': 0}
                }]
            }]
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Z Offset: '},
            'pad': {'t': 50},
            'steps': [
                {
                    'args': [[str(offset)], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate'
                    }],
                    'label': str(offset),
                    'method': 'animate'
                }
                for offset in range(1800, 2000, 5)
            ]
        }]
    )
    
    return fig

if __name__ == "__main__":
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    
    left_path = os.path.join(data_dir, 'camL_1.csv')
    right_path = os.path.join(data_dir, 'camR_1.csv')
    
    print("Loading data...")
    left_data = pd.read_csv(left_path)
    right_data = pd.read_csv(right_path)
    
    print("Creating interactive visualization...")
    fig = create_interactive_view(left_data, right_data)
    
    print("Opening in browser...")
    fig.show() 