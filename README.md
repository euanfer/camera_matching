# Track Matching Challenge

## Data Description
The data represents 2D positions of football players tracked on a pitch:
- Each CSV contains tracking data from different cameras
- There is an overlap region in the middle of the pitch
- Each track shows a player's movement over time (x, y, frame)

## Task
Match the tracking patterns from both cameras by finding where they overlap:

1. The data shows the same players viewed from different cameras
2. Your goal is to find which tracks match between the two patterns
3. The tracks should align perfectly when matched correctly

## Important Constraints
- Time axis (frame) must stay at 90° to the position plane (x,y)
- Patterns must stay "upright" - no twisting/rotation allowed
- Only use transformations that preserve the time alignment
- The overlap occurs in the middle region of the pitch

## Data Format
Each CSV contains:
- frame: Time point of detection
- tracking_id: Unique ID for each track
- pitch_x, pitch_y: 2D position on pitch
- team_id: Team identifier
- velocity: Movement speed

## Tips
1. Use the visualization tool to examine both patterns in 3D
2. Look for similar movement shapes in the overlap region
3. Consider the timing of movements
4. Remember that matching tracks should have similar motion patterns

## Evaluation
Your solution should:
1. Identify which tracks match between patterns
2. Explain your reasoning for the matches
3. Describe how you aligned the patterns 