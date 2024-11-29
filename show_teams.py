import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration parameters for filtering
EDGE_THRESHOLD = 50  # pixels from pitch edge to consider as "edge zone"
MOVEMENT_THRESHOLD = 50  # minimum standard deviation of position to be considered "moving"
MIN_TRACK_LENGTH = 10  # minimum number of detections for a valid track

def filter_static_edge_tracks(data):
    """Filter out tracks that are stationary and close to pitch edges."""
    # Group by track_id to analyze each track
    track_groups = data.groupby('tracking_id')
    valid_tracks = []
    removed_tracks = []
    
    for tracking_id, track in track_groups:
        # Calculate track statistics
        x_std = track['pitch_x'].std()
        y_std = track['pitch_y'].std()
        mean_x = track['pitch_x'].mean()
        mean_y = track['pitch_y'].mean()
        track_length = len(track)
        
        # Check if track is near edges
        is_near_x_edge = mean_x < EDGE_THRESHOLD or mean_x > (1050 - EDGE_THRESHOLD)
        is_near_y_edge = mean_y < EDGE_THRESHOLD or mean_y > (680 - EDGE_THRESHOLD)
        is_near_edge = is_near_x_edge or is_near_y_edge
        
        # Check if track is stationary
        movement = np.sqrt(x_std**2 + y_std**2)
        is_stationary = movement < MOVEMENT_THRESHOLD
        
        # Track info for diagnostics
        track_info = {
            'tracking_id': tracking_id,
            'team_id': track['team_id'].iloc[0],
            'length': track_length,
            'movement': movement,
            'mean_x': mean_x,
            'mean_y': mean_y
        }
        
        # Filter decision
        if (is_near_edge and is_stationary) or track_length < MIN_TRACK_LENGTH:
            removed_tracks.append(track_info)
        else:
            valid_tracks.append(tracking_id)
    
    # Print diagnostics
    print(f"\nFiltering Summary:")
    print(f"Total tracks: {len(track_groups)}")
    print(f"Removed tracks: {len(removed_tracks)}")
    print(f"Remaining tracks: {len(valid_tracks)}")
    
    print("\nRemoved Track Details:")
    for track in removed_tracks:
        print(f"Track {track['tracking_id']} (Team {track['team_id']}):")
        print(f"  Length: {track['length']}, Movement: {track['movement']:.2f}")
        print(f"  Position: ({track['mean_x']:.1f}, {track['mean_y']:.1f})")
    
    return data[data['tracking_id'].isin(valid_tracks)]


def draw_team_annotations(frame, positions_df, frame_number, team_colors):
    """
    Draw ellipses around players with their team colors
    
    Args:
        frame: Video frame to draw on
        positions_df: DataFrame with player positions
        frame_number: Current frame number
        team_colors: Dict mapping team names to BGR colors
    """
    # Get positions for current frame
    current_positions = positions_df[positions_df['frame'] == frame_number]
    
    for _, player in current_positions.iterrows():
        team_id = str(int(player['team_id']))
        team_color = team_colors[team_id]
        
        # Use actual bounding box coordinates from data
        x1 = int(player['topleftx'])
        y1 = int(player['toplefty'])
        x2 = x1 + int(player['width'])
        y2 = y1 + int(player['height'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
        
        # Draw black background rectangle for ID
        id_width = 35  # Width of ID background
        id_height = 15  # Height of ID background
        id_x = x1   # Align with left edge of bounding box
        id_y = y2  # Place below bounding box
        cv2.rectangle(frame, 
                     (id_x, id_y), 
                     (id_x + id_width, id_y + id_height), 
                     (0, 0, 0), 
                     -1)  # -1 fills the rectangle
        
        # Draw ID text in white, centered in black rectangle
        text = str(player['tracking_id'])
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)[0]
        text_x = id_x + (id_width - text_size[0]) // 2
        text_y = id_y + (id_height + text_size[1]) // 2
        cv2.putText(frame, text, 
                   (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.25, 
                   (255, 255, 255), 
                   1)
        
    return frame

def main():
    # Define team colors (in BGR format)
    team_colors = {
        '-1': (128, 128, 128),    # grey
        '0': (0, 0, 255),    # red
        '1': (0, 0, 0),    # black
        '2': (255, 255, 255),    # white
        '3': (130, 220, 0)    # yellow
    }
    
    # Read positions data
    positions_df = pd.read_csv('output_csv/cam1_1.csv')
    #positions_df = filter_static_edge_tracks(positions_df)
   # plt.hist(positions_df['topleftx'], bins = 106)
   # plt.show()
    # Open input video
    input_video = cv2.VideoCapture('input_videos/camera1_first_half.mp4')
    
    # Get video properties
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(
        'output_videos/cam1_1_teams.mp4',
        fourcc,
        fps,
        (width, height)
    )
    
    frame_number = 0
    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            break
            
        # Draw team annotations
        annotated_frame = draw_team_annotations(frame, positions_df, frame_number, team_colors)
        
        # Write frame to output video
        output_video.write(annotated_frame)
        
        # Optional: show progress
        if frame_number % 500 == 0:
            print(f"Processed frame {frame_number}")
            
        
        #if frame_number == 500:
        #    break    
        frame_number += 1
        
    # Clean up
    input_video.release()
    output_video.release()
    print("Video processing complete!")

if __name__ == "__main__":
    main()
