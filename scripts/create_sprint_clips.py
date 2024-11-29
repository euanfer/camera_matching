import pandas as pd
import cv2
import numpy as np
import os
from pathlib import Path
from collections import deque
from scipy.interpolate import splprep, splev

FAR_THRESHOLD = 25
# Updated team colors dictionary to handle all cases
team_colors = {
    -1: (128, 128, 128),  # Gray for unassigned players
    0: (10, 10, 10),       # almost_black for team 1
    1: (0, 0, 230),   # red for team 2
    2: (0, 255, 10)        # green for team 1 goalkeeper (same as team 1)
}

def create_pitch_overlay(pitch_template, current_frame_data, tracking_id, scale_x, scale_y):
    """Create pitch overlay with player positions"""
    pitch_frame = pitch_template.copy()
    
    # Draw team lines
    draw_team_line(pitch_frame, current_frame_data, 0, edge='left', scale_x=scale_x, scale_y=scale_y)
    draw_team_line(pitch_frame, current_frame_data, 1, 'left', scale_x=scale_x, scale_y=scale_y)
    
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

def apply_pitch_overlay(gray_frame, pitch_frame, perspective_matrix, scaled_pitch_width, scaled_pitch_height):
    """Apply pitch overlay to frame with proper blending"""
    # Warp the pitch including player markers
    warped_pitch = cv2.warpPerspective(
        pitch_frame,
        perspective_matrix,
        (scaled_pitch_width, scaled_pitch_height)
    )
    
    # Create ROI in the corner of the frame
    roi = gray_frame[-PITCH_PADDING-scaled_pitch_height:-PITCH_PADDING, 
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
    gray_frame[-PITCH_PADDING-scaled_pitch_height:-PITCH_PADDING, 
               PITCH_PADDING:PITCH_PADDING+scaled_pitch_width] = roi_final
    
    return gray_frame

def find_player_mask(frame, bbox):
    """Extract player mask from bounding box with distance-adaptive parameters"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # Extract ROI
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
        
    # Calculate ROI dimensions
    roi_height = y2 - y1
    
    # For very far players, just use the bounding box as mask
    if roi_height < FAR_THRESHOLD:
        mask = np.ones_like(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)) * 255
        return mask, (x1, y1, x2, y2)
    
    # For all other distances, use medium-range parameters
    gaussian_size = 7
    block_size = 11
    adaptive_C = 2
    morph_size = 3
    dilate_iter = 2
    erode_iter = 1
    final_blur = 5
    edge_margin = 1
    
    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply edge mask
    if edge_margin > 0:
        edge_mask = np.ones_like(gray_roi)
        edge_mask[:edge_margin, :] = 0
        edge_mask[-edge_margin:, :] = 0
        edge_mask[:, :edge_margin] = 0
        edge_mask[:, -edge_margin:] = 0
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray_roi, (gaussian_size, gaussian_size), 0)
    
    # Use adaptive thresholdin
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 
        block_size,
        adaptive_C
    )
    
    if edge_margin > 0:
        thresh = cv2.bitwise_and(thresh, thresh, mask=edge_mask)
    
    # Morphological operations
    kernel = np.ones((morph_size, morph_size), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=dilate_iter)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.squeeze().astype(np.float32)
    
    if len(contour_points) < 4:
        return None
    
    # Smooth contour with more aggressive parameters
    # Reduce number of points for initial smoothing
    num_points = max(20, len(contour_points) // 4)  # Limit minimum points but reduce total
    
    # Add more extra points for better periodic splining
    num_extra = 12  # Increased from 8
    extra_points = contour_points[:num_extra]
    extended_points = np.vstack([contour_points, extra_points])
    
    # Apply spline with smoothing factor
    tck, u = splprep([extended_points[:, 0], extended_points[:, 1]], s=100, per=True)  # Added smoothing factor
    u_new = np.linspace(0, 1, num=num_points)  # Reduced number of output points
    smooth_points = np.column_stack(splev(u_new, tck))
    
    # Second pass of smoothing for extra smoothness
    try:
        # Ensure we have enough unique points
        unique_points = np.unique(smooth_points, axis=0)
        if len(unique_points) < 4:  # Need at least 4 points for cubic spline
            smooth_points_final = smooth_points
        else:
            # Add checks for valid input
            if np.any(np.isnan(smooth_points)) or np.any(np.isinf(smooth_points)):
                smooth_points_final = smooth_points
            else:
                extra_points_2 = smooth_points[:num_extra]
                extended_points_2 = np.vstack([smooth_points, extra_points_2])
                
                # Ensure points are not too close together
                if len(np.unique(extended_points_2, axis=0)) < 4:
                    smooth_points_final = smooth_points
                else:
                    try:
                        tck_2, u_2 = splprep([extended_points_2[:, 0], extended_points_2[:, 1]], s=50, per=True)
                        smooth_points_final = np.column_stack(splev(u_new, tck_2))
                    except:
                        smooth_points_final = smooth_points
    except Exception as e:
        # If any error occurs, use results from first smoothing
        smooth_points_final = smooth_points
    
    smooth_contour = smooth_points_final.astype(np.int32).reshape((-1, 1, 2))
    
    # Create mask
    mask = np.zeros_like(gray_roi)
    cv2.drawContours(mask, [smooth_contour], -1, 255, -1)
    
    # Final smoothing with larger kernel
    final_blur = 7  # Increased from 5
    mask = cv2.GaussianBlur(mask, (final_blur, final_blur), 0)
    
    return mask, (x1, y1, x2, y2)

def draw_sprint_annotation(frame, x, y, velocity, team_color=(0, 255, 0)):
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

def draw_player_with_trail(frame, current_pos, trail_positions, color, default_color=(128, 128, 128)):
    """Draw trailing path with color changing based on sprint status"""
    positions = list(trail_positions)
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

def create_sprint_clips(video_path, sprints_df, tracking_df, output_dir, num_clips, sort_by='duration_seconds'):
    """Create video clips of sprints with annotations"""
    # Sort sprints by the specified metric
    sprints_df = sprints_df.sort_values(by=sort_by, ascending=False).head(num_clips)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    
    # Add debug prints for data inspection
    print("\nDebugging Data:")
    print(f"Tracking DataFrame columns: {tracking_df.columns.tolist()}")
    print(f"Unique tracking IDs in tracking data: {tracking_df['tracking_id'].unique()}")
    print(f"Frame range in tracking data: {tracking_df['frame'].min()} to {tracking_df['frame'].max()}")
    print("\nSprints DataFrame sample:")
    print(sprints_df[['tracking_id', 'team_id', 'start_frame', 'end_frame']].head())
    
    # Add velocity buffer for smoothing
    VELOCITY_BUFFER_SIZE = 10  # Adjust this value to control the size of the rolling average
    velocity_buffer = deque(maxlen=VELOCITY_BUFFER_SIZE)
    
    # Add trail buffer
    TRAIL_LENGTH = 30  # Adjust for longer/shorter trails
    trail_positions = deque(maxlen=TRAIL_LENGTH)
    
    PLAYER_OPACITY = 0.4  # Base opacity for non-sprinting
    SPRINT_OPACITY = 0.2  # Reduced opacity for sprinting

    # Add pitch overlay parameters
    PITCH_SCALE = 0.3  # Scale factor for pitch overlay
    PITCH_OPACITY = 0.7  # Opacity of pitch overlay
    PITCH_PADDING = 20  # Padding from corner in pixels
    PERSPECTIVE_SQUEEZE = 0.85  # How much to squeeze the top of the trapezoid (0-1)
    
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
    
    # Process each sprint
    for idx, sprint in sprints_df.iterrows():
        sprint_start_frame = int(sprint['start_frame'])
        sprint_end_frame = int(sprint['end_frame'])
        tracking_id = sprint['tracking_id']
        
        # Handle team_id properly
        team_id = int(sprint['team_id']) if not pd.isna(sprint['team_id']) else -1
        
        # Add more detailed debugging for this sprint
        print(f"\nProcessing sprint {idx}")
        print(f"Frames: {sprint_start_frame} to {sprint_end_frame}")
        print(f"Tracking ID: {tracking_id}")
        print(f"Team ID: {team_id}")
        print('suck ya mum')

        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        output_path = os.path.join(output_dir, f'sprint_{tracking_id}_{sort_by}.mp4')
        out = cv2.VideoWriter(output_path, 
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps, 
                            (frame_width, frame_height))

        time_buffer = 2 * fps
        # Get all frames for this player
        player_tracking = tracking_df[tracking_df['tracking_id'] == tracking_id].sort_values('frame')
        start_frame = max(int(sprint_start_frame - time_buffer), 0)
        end_frame = min(int(sprint_end_frame + time_buffer), max(tracking_df['frame']))
        
        # Reset buffers for each new sprint
        velocity_buffer = deque(maxlen=VELOCITY_BUFFER_SIZE)
        trail_positions = deque(maxlen=TRAIL_LENGTH)
        speed_threshold = 5.0  # m/s
        # Pre-fill trail positions with None to prevent connecting to random points
        frame_number = 0
        first_position_found = False
        
        while frame_number <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_number >= start_frame:
                # Convert frame to grayscale
                gray_frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            
                # Create and apply pitch overlay
                pitch_frame = create_pitch_overlay(scaled_pitch_template, current_frame_data, tracking_id, scale_x, scale_y)
                gray_frame = apply_pitch_overlay(gray_frame, pitch_frame, perspective_matrix, scaled_pitch_width, scaled_pitch_height)
            
                # Draw ellipses for all players except the sprinting player
                for _, player in tracking_df[tracking_df['frame'] == frame_number].iterrows():
                    if player['tracking_id'] != tracking_id:
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
                        
                        cv2.ellipse(gray_frame, center, axes, 0, -20, 200, player_color, 3)
                        cv2.ellipse(gray_frame, center, axes, 0, 0, 200, (255,255,255), 1)
                        

                player_pos = player_tracking[player_tracking['frame'] == frame_number]
                if not player_pos.empty:
                    x = player_pos['camera_x'].iloc[0]
                    y = player_pos['camera_y'].iloc[0]
                    current_velocity = player_pos['velocity'].iloc[0]
                    current_velocity = min(current_velocity, 8.5 + np.random.uniform(0,1))
                    
                    # Update the velocity buffer
                    velocity_buffer.append(current_velocity)
                    
                    # Calculate the average velocity
                    avg_velocity = np.mean(velocity_buffer)
                    
                    if avg_velocity > speed_threshold:
                        is_sprinting = True
                    else:
                        is_sprinting = False

                    # Determine opacity based on sprinting status
                    opacity = SPRINT_OPACITY if is_sprinting else PLAYER_OPACITY

                    # Extract bbox using topleft + width/height format
                    x1 = player_pos['topleftx'].iloc[0]
                    y1 = player_pos['toplefty'].iloc[0]
                    width = player_pos['width'].iloc[0]
                    height = player_pos['height'].iloc[0]
                    bbox = [x1, y1, x1 + width, y1 + height]

                    # Find player mask
                    mask_result = find_player_mask(frame, bbox)
                    if mask_result is not None:
                        mask, (x1, y1, x2, y2) = mask_result

                        # Create colored player overlay
                        player_overlay = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
                        player_overlay[:] = team_colors[team_id]

                        # Apply mask to overlay
                        mask_3d = np.stack([mask] * 3, axis=2)
                        masked_overlay = cv2.bitwise_and(player_overlay, mask_3d)

                        # Get the ROI from the original frame
                        roi = frame[y1:y2, x1:x2]

                        # Blend the player area with opacity
                        blend = cv2.addWeighted(
                            masked_overlay,
                            opacity,
                            cv2.bitwise_and(roi, mask_3d),
                            1 - opacity,
                            0
                        )

                        # Create the final ROI by combining the background and blended player
                        inv_mask_3d = np.stack([cv2.bitwise_not(mask)] * 3, axis=2)
                        background = cv2.bitwise_and(roi, inv_mask_3d)
                        final_roi = cv2.add(background, blend)

                        darker_color = tuple(int(team_colors[team_id][i] * 0.7) for i in range(3)) 
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # Draw the contour on the final ROI
                            cv2.drawContours(final_roi, contours, -1, darker_color, 1)

                        # Update the grayscale frame with the colored ROI
                        gray_frame[y1:y2, x1:x2] = final_roi

                    # Add position to trail and draw
                    current_pos = (x, y)
                    trail_positions.append((x, y, is_sprinting))
                    gray_frame = draw_player_with_trail(gray_frame, (x, y), trail_positions, team_colors[team_id])

                    frame = draw_sprint_annotation(gray_frame, x, y, avg_velocity, team_colors[team_id])
                

                out.write(gray_frame)
            
            frame_number += 1
            
        out.release()
        print(f"Created clip: {output_path}")
        cap.release()

def main():
    # Read CSVs
    sprints_df = pd.read_csv('sprint_csv/sprints_cam1_2nd.csv')
    tracking_df = pd.read_csv('processed_csv/cam1_2nd_speed.csv')
    
    # Configuration variables
    sort_by = 'distance'  # Options: 'duration_seconds', 'max_speed', 'avg_speed', 'distance'
    num_clips = 10


    video_path = '/Users/euanferguson/Desktop/Clann/large_videos/camera1_second_half.mp4'
    output_dir = 'annotated_videos/sprint_clips'
    
    create_sprint_clips(video_path, sprints_df, tracking_df, output_dir, num_clips, sort_by)

if __name__ == "__main__":
    main() 