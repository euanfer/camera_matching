import pandas as pd
import plotly.graph_objects as go
import os

def create_interactive_view(left_data, right_data):
    # Create figure
    fig = go.Figure()
    
    # Plot all left camera tracks
    for track_id in left_data['tracking_id'].unique():
        track = left_data[left_data['tracking_id'] == track_id]
        fig.add_trace(
            go.Scatter3d(
                x=track['pitch_x'],
                y=track['pitch_y'],
                z=track['frame'],
                mode='lines',
                name=f'Left {track_id}',
                line=dict(color='blue'),
                opacity=0.6,
                visible=True
            )
        )
    
    # Plot all right camera tracks
    for track_id in right_data['tracking_id'].unique():
        track = right_data[right_data['tracking_id'] == track_id]
        fig.add_trace(
            go.Scatter3d(
                x=track['pitch_x'],
                y=track['pitch_y'],
                z=track['frame'],
                mode='lines',
                name=f'Right {track_id}',
                line=dict(color='red'),
                opacity=0.6,
                visible=True
            )
        )
    
    # Add range sliders for filtering
    fig.update_layout(
        title="Interactive Track Explorer",
        scene=dict(
            bgcolor='black',
            xaxis=dict(title='Pitch X', gridcolor='gray', range=[0, 1000]),
            yaxis=dict(title='Pitch Y', gridcolor='gray', range=[0, 1000]),
            zaxis=dict(title='Frame', gridcolor='gray'),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        height=1000,
        width=1500,
        # Add sliders for filtering
        sliders=[
            # X-range slider
            dict(
                active=0,
                currentvalue={"prefix": "X Range: "},
                pad={"t": 50},
                steps=[
                    dict(
                        method="relayout",
                        args=[{"scene.xaxis.range": [x, x+200]}],
                        label=str(x)
                    )
                    for x in range(0, 801, 100)
                ]
            ),
            # Frame range slider
            dict(
                active=0,
                currentvalue={"prefix": "Frame Range: "},
                pad={"t": 100},
                steps=[
                    dict(
                        method="relayout",
                        args=[{"scene.zaxis.range": [f, f+5000]}],
                        label=str(f)
                    )
                    for f in range(0, 180000, 5000)
                ]
            )
        ]
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