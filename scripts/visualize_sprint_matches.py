import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import deque
import os

# Visualisation parameters
#--------------------------------

team_colors = {
    -1: (128, 128, 128),  # Gray for unassigned players
    0: (10, 10, 10),       # almost_black for team 1
    1: (0, 0, 230),   # red for team 2
    2: (0, 255, 10)        # green for team 1 goalkeeper (same as team 1)
}

sprint_color = (0, 215, 255)

# Add pitch overlay parameters
PITCH_SCALE = 0.3  # Scale factor for pitch overlay
PITCH_OPACITY = 0.7  # Opacity of pitch overlay
PITCH_PADDING = 20  # Padding from corner in pixels
PERSPECTIVE_SQUEEZE = 0.85  # How much to squeeze the top of the trapezoid (0-1)


# Add velocity buffer for smoothing
VELOCITY_BUFFER_SIZE = 10  # Adjust this value to control the size of the rolling average
velocity_buffer = deque(maxlen=VELOCITY_BUFFER_SIZE)

# Add trail buffer
TRAIL_LENGTH = 30  # Adjust for longer/shorter trails
trail_positions = deque(maxlen=TRAIL_LENGTH)

speed_threshold = 5.0  # m/s

# Add pitch template loading and scaling
pitch_template = cv2.imread('pitches/pitch.png')
pitch_height, pitch_width = pitch_template.shape[:2]

# Calculate scaled pitch dimensions with wider aspect ratio
scaled_pitch_height = int(pitch_height * PITCH_SCALE)
scaled_pitch_width = int(scaled_pitch_height * 2.2)  # Make width 1.6 times the height
scaled_pitch_template = cv2.resize(pitch_template, (scaled_pitch_width, scaled_pitch_height))

# Define source points (rectangle)
src_points = np.float32([
    [0, 0],  # Top-left
    [scaled_pitch_width, 0],  # Top-right
    [scaled_pitch_width, scaled_pitch_height],  # Bottom-right
    [0, scaled_pitch_height]  # Bottom-left
])

# Define destination points (trapezoid)
squeeze_amount = int(scaled_pitch_width * (1 - PERSPECTIVE_SQUEEZE) / 2)
dst_points = np.float32([
    [squeeze_amount, 0],  # Top-left
    [scaled_pitch_width - squeeze_amount, 0],  # Top-right
    [scaled_pitch_width, scaled_pitch_height],  # Bottom-right
    [0, scaled_pitch_height]  # Bottom-left
])

# Calculate perspective transform matrix
perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Scale factors to convert from meters to pixels (for the scaled pitch)
scale_x = scaled_pitch_width / 105  # 105 meters is standard pitch length
scale_y = scaled_pitch_height / 68  # 68 meters is standard pitch width

# Drawing functions
#--------------------------------

def draw_sprint_annotation(frame, x, y, velocity, height, team_color=(0, 255, 0)):
    """Draw an ellipse around the sprinting player and show their speed"""
    
    # Add speed text below the player with background
    speed_text = f"{velocity:.1f}"
    font = cv2.FONT_HERSHEY_DUPLEX  # More formal font (alternatives: FONT_HERSHEY_TRIPLEX or FONT_HERSHEY_COMPLEX)
    font_scale = 0.5
    thickness = 1
    
    # Get text size to create properly sized background
    (text_width, text_height), baseline = cv2.getTextSize(speed_text, font, font_scale, thickness)
    text_x = int(x) - text_width // 2
    text_y = int(y) + 20
    
    # Draw ellipse
    center = (int(x), int(y))
    axes = (int(height / 4), int(height / 10))
    
    cv2.ellipse(frame, center, axes, 0, -20, 200, sprint_color, 3)
    cv2.ellipse(frame, center, axes, 0, 0, 200, (255,255,255), 1)

    # Draw black background rectangle
    padding = 2
    cv2.rectangle(frame,
                 (text_x - padding, text_y - text_height - padding),
                 (text_x + text_width + padding, text_y + padding),
                 (0, 0, 0),
                 -1)  # -1 fills the rectangle
    
    # Draw white text
    cv2.putText(frame, 
                speed_text, 
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),  # White color
                thickness)
    
    return frame

def draw_teams(frame, tracking_df, frame_number, sprint_id):
    # Draw ellipses for all players except the sprinting player
    for _, player in tracking_df[tracking_df['frame'] == frame_number].iterrows():
        if player['tracking_id'] != sprint_id:
            player_id = player['tracking_id']
            
            x = player['camera_x']
            y = player['camera_y']
            width = player['width']
            height = player['height']
            
            # Get the correct team color for this player
            player_team_id = int(player['team_id']) if not pd.isna(player['team_id']) else -1
            player_color = team_colors[player_team_id]
            
            # Draw ellipse
            center = (int(x), int(y))
            axes = (int(height / 4), int(height / 10))
            
            cv2.ellipse(frame, center, axes, 0, -20, 200, player_color, 3)
            cv2.ellipse(frame, center, axes, 0, 0, 200, (255,255,255), 1)

    return frame

def sprint_check(player_tracking, frame_number, velocity_buffer):
    """Check if player is sprinting and return position data"""
    player_pos = player_tracking[player_tracking['frame'] == frame_number]
    
    # Return default values if no data found
    if player_pos.empty:
        return None, None, False
        
    x = player_pos['camera_x'].iloc[0]
    y = player_pos['camera_y'].iloc[0]
    current_velocity = player_pos['velocity'].iloc[0]
    current_velocity = min(current_velocity, 8.5 + np.random.uniform(0,1))
    
    # Update the velocity buffer
    velocity_buffer.append(current_velocity)
    
    # Calculate the average velocity
    avg_velocity = np.mean(velocity_buffer)
    is_sprinting = avg_velocity > speed_threshold

    return x, y, is_sprinting

def draw_player_with_trail(frame, trail_positions, color, default_color=(128, 128, 128)):
    """Draw trailing path with color changing based on sprint status"""
    positions = list(trail_positions)
    # Filter out None values
    positions = [pos for pos in positions if pos[0] is not None and pos[1] is not None]
    
    if len(positions) >= 2:
        points = np.array([[int(x), int(y)] for x, y, is_sprinting in positions])
        
        # Draw trail segments with appropriate colors
        for i in range(len(points) - 1):
            start_point = tuple(points[i])
            end_point = tuple(points[i + 1])
            segment_color = color if positions[i][2] else default_color
            cv2.line(frame, start_point, end_point, segment_color, 2)
            
    return frame

def meter_to_pixel(x, y, scale_x, scale_y):
    # Convert to original pixel coordinates without perspective transform
    px = int(x * scale_x)
    py = int(y * scale_y)
    return (px, py)

def draw_team_line(frame, current_frame_data, team, edge='left', scale_x=1, scale_y=1):
    team_data = current_frame_data[current_frame_data['team_id'] == team]

    if edge == 'left':
        line_pos = team_data['pitch_x'].min()
    elif edge == 'right':
        line_pos = team_data['pitch_x'].max()  
    else:
        raise ValueError(f"Invalid edge: {edge}")

    if pd.notna(line_pos):
        
        x_px, y_px = meter_to_pixel(line_pos/10, 0, scale_x, scale_y)
        cv2.line(frame, (x_px, 0), (x_px, frame.shape[1]), team_colors[team], 2)

def filter_frame_half(frame_data):
    avg = frame_data['pitch_x'].mean()

    if avg < 525:
        return frame_data[frame_data['pitch_x'] <= 520]
    else:
        return frame_data[frame_data['pitch_x'] >= 520]

def create_pitch_overlay(pitch_template, frame_number, df1, df2, tracking_id, frame_diff, scale_x, scale_y):
    """Create pitch overlay with player positions"""
    pitch_frame = pitch_template.copy()

    current_frame_data1 = df1[df1['frame'] == frame_number]
    current_frame_data2 = df2[df2['frame'] == frame_number - frame_diff]

    current_frame_data2['tracking_id'] = current_frame_data2['tracking_id'] + 10000

    current_frame_data = pd.concat([current_frame_data1, current_frame_data2])
    # Draw team lines
    draw_team_line(pitch_frame, current_frame_data, 0, edge='left', scale_x=scale_x, scale_y=scale_y)
    draw_team_line(pitch_frame, current_frame_data, 1, 'left', scale_x=scale_x, scale_y=scale_y)
    draw_team_line(pitch_frame, current_frame_data, 0, edge='right', scale_x=scale_x, scale_y=scale_y)
    draw_team_line(pitch_frame, current_frame_data, 1, 'right', scale_x=scale_x, scale_y=scale_y)
    
    # Draw all players on pitch
    for _, player in current_frame_data.iterrows():
        # Convert coordinates from decimeters to meters and get pixel position
        x = player['pitch_x'] / 10
        y = player['pitch_y'] / 10
        x_px, y_px = meter_to_pixel(x, y, scale_x, scale_y)
        
        # Get team color
        player_team_id = int(player['team_id']) if not pd.isna(player['team_id']) else -1
        color = team_colors[player_team_id]
        
        # Highlight sprinting player
        if player['tracking_id'] == tracking_id:
            cv2.circle(pitch_frame, (x_px, y_px), int(30 * PITCH_SCALE), (255,255,255), -1)
            cv2.circle(pitch_frame, (x_px, y_px), int(20 * PITCH_SCALE), color, -1)
        else:
            cv2.circle(pitch_frame, (x_px, y_px), int(18 * PITCH_SCALE), color, -1)
            
    return pitch_frame

def apply_pitch_overlay(frame, pitch_frame, perspective_matrix, scaled_pitch_width, scaled_pitch_height):
    """Apply pitch overlay to frame with proper blending"""
    # Warp the pitch including player markers
    warped_pitch = cv2.warpPerspective(
        pitch_frame,
        perspective_matrix,
        (scaled_pitch_width, scaled_pitch_height)
    )
    
    # Create ROI in the corner of the frame
    roi = frame[-PITCH_PADDING-scaled_pitch_height:-PITCH_PADDING, 
                    PITCH_PADDING:PITCH_PADDING+scaled_pitch_width]
    
    # Create mask for warped pitch overlay
    pitch_gray = cv2.cvtColor(warped_pitch, cv2.COLOR_BGR2GRAY)
    _, pitch_mask = cv2.threshold(pitch_gray, 1, 255, cv2.THRESH_BINARY)
    pitch_mask_3ch = cv2.cvtColor(pitch_mask, cv2.COLOR_GRAY2BGR)
    
    # Apply opacity only to pitch area
    blended_roi = cv2.addWeighted(
        warped_pitch,
        PITCH_OPACITY,
        roi,
        1 - PITCH_OPACITY,
        0,
        dtype=cv2.CV_8U
    )
    
    # Combine blended pitch with original frame
    roi_final = np.where(pitch_mask_3ch > 0, blended_roi, roi)
    
    # Update frame with blended ROI
    frame[-PITCH_PADDING-scaled_pitch_height:-PITCH_PADDING, 
               PITCH_PADDING:PITCH_PADDING+scaled_pitch_width] = roi_final
    #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame

# Video functions
#--------------------------------

def get_sprint_start_frame(track, time_pad):
    slow_counter = 0
    for i in range(len(track)):
        speed = track.iloc[-i]['velocity']

        if speed < speed_threshold:
            slow_counter += 1
        else:
            slow_counter = 0

        if slow_counter > 5:
            print(f"Sprint start frame: {track.iloc[-i]['frame'] - time_pad}")
            return int(max(track.iloc[-i]['frame'] - time_pad, 0))
        
    return int(max(track.iloc[0]['frame'] - time_pad, 0))

def get_sprint_end_frame(track, time_pad):
    slow_counter = 0
    for i in range(len(track)):
        speed = track.iloc[i]['velocity']

        if speed < speed_threshold:
            slow_counter += 1
        else:
            slow_counter = 0

        if slow_counter > 5:
            return int(track.iloc[i]['frame'] + time_pad)
    
    return int(track.iloc[i]['frame'] + time_pad)

def get_frame_safely(cap, target_frame):
    """Manually seek to desired frame"""
    current_frame = 0
    while current_frame < target_frame:
        ret = cap.grab()  # Skip frames without decoding
        if not ret:
            return False, None
        current_frame += 1
    
    ret, frame = cap.retrieve()  # Only decode the frame we want
    return ret, frame

def pad_frame(frame, target_width, target_height):
    """Pad frame to match target dimensions"""
    current_height, current_width = frame.shape[:2]
    
    # Calculate padding
    pad_width = max(0, target_width - current_width)
    pad_height = max(0, target_height - current_height)
    
    # Add black padding
    if pad_width > 0 or pad_height > 0:
        frame_padded = cv2.copyMakeBorder(
            frame,
            0, pad_height,  # top, bottom
            0, pad_width,   # left, right
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        return frame_padded
    return frame

def create_sprint_match_video(match_row, df1, df2, cap1, cap2, output_path, output_dims):
    """Create enhanced sequential sprint match visualization with rich visuals"""
    
    # Get tracks
    sprint_track = df1[df1['tracking_id'] == match_row['sprint_id']]
    match_track = df2[df2['tracking_id'] == match_row['matched_track']]
    
    # Filter frame data for mini pitch
    df1_filtered = filter_frame_half(df1)
    df2_filtered = filter_frame_half(df2)
    
     # get frame difference
    frame_diff = match_row['frame_difference']
    # Setup video writer
    fps = cap1.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, output_dims)
    
    # Determine which track starts first
    sprint_start = sprint_track['frame'].min()
    match_start = match_track['frame'].min()

    # Pre-fill trail positions with None to prevent connecting to random points
    frame_number = 0
    time_pad  = 3 * fps
    
    # Reset buffers for each new sprint
    velocity_buffer = deque(maxlen=VELOCITY_BUFFER_SIZE)
    trail_positions = deque(maxlen=TRAIL_LENGTH)
    
    if sprint_start < match_start:
        # Process sprint track first (Camera 1)
        clip_start = get_sprint_start_frame(sprint_track, time_pad)
        sprint_track = sprint_track[sprint_track['frame'] >= clip_start]
        frame_number = clip_start
        print(f"Processing sprint {match_row['sprint_id']} (frames {clip_start}-{sprint_track['frame'].max()})")
        get_frame_safely(cap1, clip_start)
        
        with tqdm(total=len(sprint_track), desc="Processing sprint frames") as pbar:
            for i in range(len(sprint_track)):
                ret, frame = cap1.read()
                if not ret:
                    break

                frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                pitch_frame = create_pitch_overlay(scaled_pitch_template, frame_number, df1_filtered, df2_filtered, match_row['sprint_id'], frame_diff, scale_x, scale_y)
                frame = apply_pitch_overlay(frame, pitch_frame, perspective_matrix, scaled_pitch_width, scaled_pitch_height)                
                frame = draw_teams(frame, df1, frame_number, match_row['sprint_id'])
                frame = pad_frame(frame, output_dims[0], output_dims[1])
                
                if frame_number >= sprint_track['frame'].min():
                    row = sprint_track[sprint_track['frame'] == frame_number]

                    trail_positions.append(sprint_check(sprint_track, frame_number, velocity_buffer))
                    frame = draw_player_with_trail(frame, trail_positions, sprint_color)
                    frame = draw_sprint_annotation(frame, row['camera_x'].values[0], row['camera_y'].values[0], row['velocity'].values[0], row['height'].values[0])

                
                out.write(frame)
                frame_number += 1
                pbar.update(1)
        
        # Then process match track (Camera 2)
        print(f"Processing match {match_row['matched_track']}")
        get_frame_safely(cap2, match_start)

        clip_end = get_sprint_end_frame(match_track, time_pad)
        
        # Reset buffers for each new sprint
        velocity_buffer = deque(maxlen=VELOCITY_BUFFER_SIZE)
        trail_positions = deque(maxlen=TRAIL_LENGTH)
    
        with tqdm(total=len(match_track), desc="Processing match frames") as pbar:
            for _, row in match_track.iterrows():
                ret, frame = cap2.read()
                if not ret:
                    break
                if row['frame'] > clip_end:
                    break

                frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                
                trail_positions.append(sprint_check(match_track, row['frame'], velocity_buffer))
                frame = draw_player_with_trail(frame, trail_positions, sprint_color)
                
                pitch_frame = create_pitch_overlay(scaled_pitch_template, row['frame'], df2_filtered, df1_filtered, match_row['matched_track'],-frame_diff, scale_x, scale_y)
                frame = apply_pitch_overlay(frame, pitch_frame, perspective_matrix, scaled_pitch_width, scaled_pitch_height)
                frame = draw_sprint_annotation(frame, row['camera_x'], row['camera_y'], row['velocity'], row['height'])
                frame = draw_teams(frame, df2, row['frame'], match_row['matched_track'])
                
                frame = pad_frame(frame, output_dims[0], output_dims[1])
                
                out.write(frame)
                pbar.update(1)
            
    else:
        # Process match track first (Camera 2)
        clip_start = get_sprint_start_frame(match_track, time_pad)
        frame_number = clip_start
        print(f"Processing match {match_row['matched_track']} (frames {clip_start}-{match_track['frame'].max()})")
        get_frame_safely(cap2, clip_start)
        
        match_track = match_track[match_track['frame'] >= clip_start]
        with tqdm(total=len(match_track), desc="Processing match frames") as pbar:
            for i in range(len(match_track)):
                ret, frame = cap2.read()
                
                if not ret:
                    break
                
                frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                    
                pitch_frame = create_pitch_overlay(scaled_pitch_template, frame_number, df2_filtered, df1_filtered, match_row['matched_track'], -frame_diff, scale_x, scale_y)
                frame = apply_pitch_overlay(frame, pitch_frame, perspective_matrix, scaled_pitch_width, scaled_pitch_height)
                frame = draw_teams(frame, df2, frame_number, match_row['matched_track'])
                frame = pad_frame(frame, output_dims[0], output_dims[1])

                if frame_number >= match_track['frame'].min():

                    row = match_track[match_track['frame'] == frame_number]

                    print(row['velocity'].values[0])
                    trail_positions.append(sprint_check(match_track, frame_number, velocity_buffer))
                    frame = draw_player_with_trail(frame, trail_positions, sprint_color)
                    frame = draw_sprint_annotation(frame, row['camera_x'].values[0], row['camera_y'].values[0], row['velocity'].values[0], row['height'].values[0])
                
                out.write(frame)
                frame_number += 1
                pbar.update(1)

        clip_end = get_sprint_end_frame(sprint_track, time_pad)
        # Then process sprint track (Camera 1)
        print(f"Processing sprint {match_row['sprint_id']}")
        get_frame_safely(cap1, sprint_start)
        # Reset buffers for each new sprint
        velocity_buffer = deque(maxlen=VELOCITY_BUFFER_SIZE)
        trail_positions = deque(maxlen=TRAIL_LENGTH)
    
        with tqdm(total=len(sprint_track), desc="Processing sprint frames") as pbar:
            for _, row in sprint_track.iterrows():
                ret, frame = cap1.read()
                if not ret:
                    break
                if row['frame'] > clip_end:
                    break
                
                frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                trail_positions.append(sprint_check(sprint_track, row['frame'], velocity_buffer))
                frame = draw_player_with_trail(frame, trail_positions, sprint_color)

                pitch_frame = create_pitch_overlay(scaled_pitch_template, row['frame'], df1_filtered, df2_filtered, match_row['sprint_id'], frame_diff, scale_x, scale_y)
                frame = apply_pitch_overlay(frame, pitch_frame, perspective_matrix, scaled_pitch_width, scaled_pitch_height)

                frame = draw_sprint_annotation(frame, row['camera_x'], row['camera_y'], row['velocity'], row['height'])
                frame = draw_teams(frame, df1, row['frame'], match_row['sprint_id'])
                
                frame = pad_frame(frame, output_dims[0], output_dims[1])
        
                out.write(frame)
                pbar.update(1)
    
    out.release()

def main():
    # Load data
    matches_df = pd.read_csv('csvs/matches1_2.csv')
    df1 = pd.read_csv('csvs/cam1_2.csv')
    df2 = pd.read_csv('csvs/cam2_2.csv')
    
    video1_path = '/Users/euanferguson/Desktop/Clann/large_videos/camera1_second_half.mp4'
    video2_path = '/Users/euanferguson/Desktop/Clann/large_videos/camera2_second_half.mp4'

    # Create output directory if it doesn't exist
    output_dir = 'sprint_match_videos/matches_1_2'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video dimensions
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap1.release()
    cap2.release()
    
    output_dims = (max(width1, width2), max(height1, height2))
    print(f"Output dimensions: {output_dims}")
    print('Output shape: ', output_dims[0], output_dims[1])
    
    # Process each match pair
    for idx, match in matches_df.iterrows():

        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        print(f"\nProcessing match pair {idx + 1}/{len(matches_df)}")
        output_path = os.path.join(output_dir, f'sprint_{match["sprint_id"]}_match_{match["matched_track"]}.mp4')
        
        create_sprint_match_video(match, df1, df2, cap1, cap2, output_path, output_dims)
        
        # Release captures
        cap1.release()
        cap2.release()

if __name__ == "__main__":
    main() 