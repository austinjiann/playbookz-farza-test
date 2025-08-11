#!/usr/bin/env python3

import argparse
import os
import json
import glob
import time
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import google.generativeai as genai
from dotenv import load_dotenv


def parse_timestamp(timestamp_str: str) -> float:
    """Convert various timestamp formats to seconds (float)."""
    if not timestamp_str:
        return 0.0
    
    # Handle MM:SS.SSS format
    if re.match(r'^\d{2}:\d{2}\.\d{3}$', timestamp_str):
        parts = timestamp_str.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    
    # Handle legacy M:SS.S format
    if ':' in timestamp_str:
        parts = timestamp_str.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    
    # Handle plain seconds
    return float(timestamp_str)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS.SSS format."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def exponential_backoff_retry(func, max_retries=5, base_delay=0.5):
    """Execute function with exponential backoff retry logic."""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            # Check if this is a retryable error
            retryable = any(term in error_str for term in [
                'rate limit', '429', '500', '502', '503', '504', 
                'timeout', 'connection', 'temporary'
            ])
            
            if not retryable or attempt == max_retries - 1:
                raise e
            
            # Calculate delay with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
            print(f"    Retry {attempt + 1}/{max_retries} in {delay:.1f}s (error: {str(e)[:50]}...)")
            time.sleep(delay)
    
    raise last_exception


def create_analysis_prompt(second_index: int, start_ts: str, end_ts: str, fps: int, tile_cols: int, tile_rows: int) -> str:
    """Create a more lenient basketball analysis prompt focused on overall second assessment."""
    frames_per_grid = tile_cols * tile_rows
    frame_dt = 1.0 / fps
    return f"""You are analyzing ONE {tile_cols}x{tile_rows} image grid of {frames_per_grid} frames ({fps} fps, {frame_dt:.3f}s each), ordered top-left → right, then next row (row-major).

Look at the ENTIRE sequence to understand what happens in this 1-second window. Be LENIENT - basketball shots happen fast and cameras miss details.

Your job: Determine if a basketball shot attempt occurs and its outcome in this second.

For EACH of the {frames_per_grid} frames, assess:
  - shot_activity: true if ANY basketball action near the rim (shooting, rebounding, ball near hoop, etc.)
  - ball_goes_in: true if the ball appears to go through the hoop (be generous - if it looks like it might have gone in, mark true)
  - unclear: true if you genuinely cannot tell what's happening due to blur/occlusion

Be GENEROUS with "ball_goes_in" - if there's any reasonable indication the shot was successful, mark it true.
Focus on the OVERALL story of the second, not pixel-perfect analysis of each frame.

Output ONLY this JSON:
{{
  "second_index": {second_index},
  "window_start_ts": "{start_ts}",
  "window_end_ts": "{end_ts}",
  "frames": [
    {{"idx": 0,  "shot_activity": true, "ball_goes_in": false, "unclear": false}},
    {{"idx": 1,  "shot_activity": false, "ball_goes_in": false, "unclear": true}},
    ...
    {{"idx": {frames_per_grid-1}, "shot_activity": false, "ball_goes_in": false, "unclear": false}}
  ]
}}"""


def coerce_boolean(value) -> bool:
    """Safely coerce values to boolean, handling string representations."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
    return False  # Default fallback


def is_attempt(frames: List[Dict], activity_threshold: int = 1) -> bool:
    """Determine if this second contains basketball activity (more lenient)."""
    activity_flags = [coerce_boolean(frame.get("shot_activity", False)) for frame in frames]
    activity_count = sum(activity_flags)
    return activity_count >= activity_threshold


def compute_final_result(frames: List[Dict], min_consecutive: int, treat_occluded_as_unknown: bool, 
                        expected_frame_count: int, activity_threshold: int = 1) -> Dict[str, Any]:
    """Compute final result with lenient shot detection and simple logic."""
    
    # Validate frame count first
    if len(frames) != expected_frame_count:
        return {
            "result": "unknown",
            "confidence": 0.2,
            "evidence": {
                "first_made_idx": None,
                "made_frames": 0,
                "activity_frames": 0,
                "unclear_frames": 0,
                "unclear_ratio": 0.0,
                "invalid_frame_count": True,
                "has_activity": False
            }
        }
    
    # Extract and coerce flags with new field names
    activity_flags = [coerce_boolean(frame.get("shot_activity", False)) for frame in frames]
    made_flags = [coerce_boolean(frame.get("ball_goes_in", False)) for frame in frames]
    unclear_flags = [coerce_boolean(frame.get("unclear", False)) for frame in frames]
    
    # Calculate statistics
    activity_count = sum(activity_flags)
    made_count = sum(made_flags)
    unclear_count = sum(unclear_flags)
    unclear_ratio = unclear_count / len(frames)
    
    # Check if this second has basketball activity
    has_activity = activity_count >= activity_threshold
    
    # If no activity, mark as no_shot
    if not has_activity:
        return {
            "result": "no_shot",
            "confidence": 0.9,
            "evidence": {
                "first_made_idx": None,
                "made_frames": made_count,
                "activity_frames": activity_count,
                "unclear_frames": unclear_count,
                "unclear_ratio": unclear_ratio,
                "invalid_frame_count": False,
                "has_activity": False
            }
        }
    
    # Find first frame where ball goes in
    first_made_idx = None
    for i, flag in enumerate(made_flags):
        if flag:
            first_made_idx = i
            break
    
    # Simple decision logic (much more lenient)
    if unclear_ratio >= 0.8:
        # Too unclear to determine
        result = "unknown"
        confidence = 0.3
    elif made_count >= 1:
        # Any frame shows ball going in = MADE
        result = "made"
        confidence = 0.8 + 0.1 * made_count - 0.2 * unclear_ratio
        confidence = max(0.7, min(0.95, confidence))
    else:
        # Has activity but no made frames = MISSED
        result = "missed"
        confidence = 0.7 - 0.2 * unclear_ratio
        confidence = max(0.4, min(0.8, confidence))
    
    return {
        "result": result,
        "confidence": confidence,
        "evidence": {
            "first_made_idx": first_made_idx,
            "made_frames": made_count,
            "activity_frames": activity_count,
            "unclear_frames": unclear_count,
            "unclear_ratio": unclear_ratio,
            "invalid_frame_count": False,
            "has_activity": True
        }
    }


def apply_cooldown_logic(results: List[Dict[str, Any]], cooldown_seconds: float = 1.0) -> List[Dict[str, Any]]:
    """Apply simple cooldown logic to prevent double-counting shots that span multiple seconds."""
    if len(results) < 2:
        return results
    
    processed_results = results.copy()
    last_make_time = -999  # Track last make timestamp
    
    for i, result in enumerate(processed_results):
        if result.get("result") != "made":
            continue
            
        # Calculate timestamp for this make
        evidence = result.get("evidence", {})
        first_made_idx = evidence.get("first_made_idx", 0)
        timestamp_seconds = i + (first_made_idx / 20)  # Assuming 20 fps for simplicity
        
        # Check cooldown against last make
        if timestamp_seconds - last_make_time < cooldown_seconds:
            # Too close to previous make - convert to missed to avoid double counting
            result["result"] = "missed"
            result["confidence"] = 0.5
            result["evidence"]["cooldown_applied"] = True
            print(f"  Applied cooldown: Grid {i+1} converted from made to missed (too close to previous shot)")
        else:
            # Valid make - update last make time
            last_make_time = timestamp_seconds
            print(f"  Valid make: Grid {i+1} at {timestamp_seconds:.2f}s")
    
    return processed_results


def analyze_batch_grids(model: genai.GenerativeModel, grid_files: List[str], fps: int, 
                       tile_cols: int, tile_rows: int, min_consecutive: int = 2, 
                       treat_occluded_as_unknown: bool = True, near_k: int = 2) -> List[Dict[str, Any]]:
    """Analyze all grid images in a single batch request with new ball-in-hoop logic."""
    print(f"Uploading {len(grid_files)} grids to Gemini for batch analysis...")
    
    def _make_request():
        # Upload all images
        uploaded_files = []
        for i, grid_file in enumerate(grid_files):
            print(f"  Uploading grid {i+1}/{len(grid_files)}: {Path(grid_file).name}")
            uploaded_file = genai.upload_file(grid_file, mime_type="image/jpeg")
            
            # Wait for processing
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(0.5)
                uploaded_file = genai.get_file(uploaded_file.name)
            
            if uploaded_file.state.name != "ACTIVE":
                raise Exception(f"File upload failed: {uploaded_file.state.name}")
            
            uploaded_files.append(uploaded_file)
        
        print(f"\nAnalyzing all {len(grid_files)} grids in single request...")
        
        # Create batch prompt
        grid_list = []
        for i, grid_file in enumerate(grid_files):
            start_ts = format_timestamp(i)
            end_ts = format_timestamp(i + 1)
            grid_list.append(f"- Grid {i+1}: {Path(grid_file).name} (second {i}, {start_ts} - {end_ts})")
        
        grid_descriptions = "\n".join(grid_list)
        
        frames_per_grid = tile_cols * tile_rows
        frame_dt = 1.0 / fps
        
        batch_prompt = f"""
You are analyzing {len(grid_files)} basketball image grids.
- Each grid is {tile_cols}x{tile_rows}, ordered **left to right, then top to bottom** (row-major order).
- Each grid covers exactly **1 second** of the game.


When a shot attempt is detected, first look for a frame where the basketball is clearly inside the net mesh.
If such a frame exists, label the shot "made" immediately.
If no such frame exists, use other cues to decide.


**Definitions:**
- **Made:** Ball fully passes downward through the hoop, OR strong visual cues (net snap, ball clearly under rim).
- **Missed:** Ball does not go through the hoop, OR clearly goes off target.

**Output:**
Return ONLY a JSON object in this exact format:
{{
  "shots": [
    {{
      "timestamp_of_outcome": "00:01.650",
      "result": "made",
      "feedback": "Great shot! Ball went through the hoop cleanly.",
      "total_shots_made_so_far": 2,
      "total_shots_missed_so_far": 0
    }}
  ]
}}

- `timestamp_of_outcome`: Time in the video when the shot result is visible.
- `feedback`: Provide encouraging, constructive feedback.
- `total_shots_made_so_far` and `total_shots_missed_so_far`: Running totals across the entire sequence.

"""

        
        # Create interleaved content with grid headers for robust ordering
        content = []
        for i, uploaded_file in enumerate(uploaded_files):
            content.append(f"GRID {i}")
            content.append(uploaded_file)
        content.append(batch_prompt)
        
        # Generate analysis
        response = model.generate_content(content)
        
        # Clean up uploaded files
        print("Cleaning up uploaded files...")
        for uploaded_file in uploaded_files:
            try:
                genai.delete_file(uploaded_file.name)
            except:
                pass  # Ignore cleanup errors
        
        return response.text.strip()
    
    try:
        response_text = exponential_backoff_retry(_make_request)
        
        # Parse JSON response
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON array found in response")
        
        json_text = response_text[json_start:json_end]
        results = json.loads(json_text)
        
        if not isinstance(results, list):
            raise ValueError("Response is not a JSON array")
        
        if len(results) != len(grid_files):
            print(f"Warning: Expected {len(grid_files)} results, got {len(results)}")
        
        # Sort results by image_index and validate completeness
        results_by_index = {}
        for result in results:
            if isinstance(result, dict) and "image_index" in result:
                results_by_index[result["image_index"]] = result
        
        # Process each grid in order
        processed_results = []
        expected_frame_count = tile_cols * tile_rows
        
        for i in range(len(grid_files)):
            if i in results_by_index:
                result = results_by_index[i]
                
                # Ensure required fields
                result.setdefault("second_index", i)
                result.setdefault("window_start_ts", format_timestamp(i))
                result.setdefault("window_end_ts", format_timestamp(i + 1))
                
                # Get frames and validate (no padding!)
                frames = result.get("frames", [])
                
                # Compute final result with proper validation and shot detection
                final_result = compute_final_result(frames, min_consecutive, treat_occluded_as_unknown, expected_frame_count, 1)
                result.update(final_result)
                
                # Log diagnostics
                evidence = result.get("evidence", {})
                print(f"  Grid {i+1}: {result['result']} (conf: {result['confidence']:.2f}, "
                      f"activity: {evidence.get('activity_frames', 0)}, "
                      f"made: {evidence.get('made_frames', 0)}, "
                      f"unclear: {evidence.get('unclear_ratio', 0):.2f})")
                
                processed_results.append(result)
            else:
                # Missing grid result - mark as error
                error_result = {
                    "second_index": i,
                    "image_index": i,
                    "window_start_ts": format_timestamp(i),
                    "window_end_ts": format_timestamp(i + 1),
                    "frames": [],
                    "result": "error",
                    "confidence": 0.0,
                    "error": f"Missing result for grid {i}",
                    "evidence": {
                        "first_made_idx": None,
                        "made_frames": 0,
                        "activity_frames": 0,
                        "unclear_frames": 0,
                        "unclear_ratio": 0.0,
                        "invalid_frame_count": True,
                        "has_activity": False
                    }
                }
                processed_results.append(error_result)
                print(f"  Grid {i+1}: ERROR - missing from batch response")
        
        return processed_results
        
    except Exception as e:
        error_msg = f"Failed to analyze grids in batch: {str(e)}"
        print(f"ERROR: {error_msg}")
        
        # Return error entries for all grids
        error_results = []
        for i in range(len(grid_files)):
            error_results.append({
                "second_index": i,
                "window_start_ts": format_timestamp(i),
                "window_end_ts": format_timestamp(i + 1),
                "frames": [],
                "result": "error",
                "confidence": 0.0,
                "error": error_msg,
                "evidence": {
                    "first_true_idx": None,
                    "num_true": 0,
                    "min_consecutive": min_consecutive,
                    "all_occluded": False
                }
            })
        return error_results


def analyze_single_grid(model: genai.GenerativeModel, image_path: str, 
                       second_index: int, fps: int, tile_cols: int, tile_rows: int,
                       min_consecutive: int = 2, treat_occluded_as_unknown: bool = True,
                       near_k: int = 2) -> Dict[str, Any]:
    """Analyze a single grid image with Gemini."""
    start_seconds = second_index
    end_seconds = second_index + 1
    start_ts = format_timestamp(start_seconds)
    end_ts = format_timestamp(end_seconds)
    
    print(f"  Analyzing grid {second_index + 1}: {Path(image_path).name}")
    
    def _make_request():
        # Upload image
        uploaded_file = genai.upload_file(image_path, mime_type="image/jpeg")
        
        # Wait for processing
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(0.5)
            uploaded_file = genai.get_file(uploaded_file.name)
        
        if uploaded_file.state.name != "ACTIVE":
            raise Exception(f"File upload failed: {uploaded_file.state.name}")
        
        # Generate analysis
        prompt = create_analysis_prompt(second_index, start_ts, end_ts, fps, tile_cols, tile_rows)
        response = model.generate_content([uploaded_file, prompt])
        
        # Clean up uploaded file
        try:
            genai.delete_file(uploaded_file.name)
        except:
            pass  # Ignore cleanup errors
        
        return response.text.strip()
    
    try:
        response_text = exponential_backoff_retry(_make_request)
        
        # Parse JSON response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in response")
        
        json_text = response_text[json_start:json_end]
        result = json.loads(json_text)
        
        # Validate required fields for new format
        required_fields = ["second_index", "window_start_ts", "window_end_ts", "frames"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        # Get frames and compute result with shot detection
        frames = result.get("frames", [])
        expected_frame_count = tile_cols * tile_rows
        final_result = compute_final_result(frames, min_consecutive, treat_occluded_as_unknown, expected_frame_count, 1)
        
        # Add computed fields to result
        result.update(final_result)
        
        # Log diagnostics
        evidence = result.get("evidence", {})
        print(f"    Result: {result['result']} (conf: {result['confidence']:.2f}, "
              f"activity: {evidence.get('activity_frames', 0)}, "
              f"made: {evidence.get('made_frames', 0)}, "
              f"unclear: {evidence.get('unclear_ratio', 0):.2f})")
        return result
        
    except Exception as e:
        error_msg = f"Failed to analyze grid {second_index + 1}: {str(e)}"
        print(f"    ERROR: {error_msg}")
        
        # Return error entry
        return {
            "second_index": second_index,
            "window_start_ts": start_ts,
            "window_end_ts": end_ts,
            "result": "error",
            "confidence": 0.0,
            "error": error_msg,
            "frames": [],
            "evidence": {
                "first_true_idx": None,
                "num_true": 0,
                "min_consecutive": min_consecutive,
                "all_occluded": False,
                "notes": f"Analysis failed: {str(e)[:50]}..."
            }
        }


def synthesize_ball_json(per_second_results: List[Dict[str, Any]], fps: int) -> Dict[str, Any]:
    """Convert per-second analysis to ball.json format - only actual shot attempts."""
    shots = []
    made_count = 0
    missed_count = 0
    
    for result in per_second_results:
        # Only include actual shot attempts (made/missed), skip no_shot, unknown, error
        if result.get("result") not in ["made", "missed"]:
            continue
        
        # Calculate precise timestamp
        start_seconds = result["second_index"]
        evidence = result.get("evidence", {})
        
        if result["result"] == "made":
            if evidence.get("first_made_idx") is not None:
                # Use first frame where ball goes in
                frame_offset = evidence["first_made_idx"]
            else:
                # Fallback for made shots without evidence
                frame_offset = 0
        else:
            # For MISSED shots: use middle of window (0.5s)
            frame_offset = fps // 2
        
        # Calculate final timestamp
        precise_seconds = start_seconds + (frame_offset / fps)
        timestamp = format_timestamp(precise_seconds)
        
        # Update counts
        if result["result"] == "made":
            made_count += 1
        else:  # missed
            missed_count += 1
        
        # Generate simple feedback
        evidence = result.get("evidence", {})
        activity_count = evidence.get("activity_frames", 0)
        made_count = evidence.get("made_frames", 0)
        
        if result["result"] == "made":
            feedback = f"Great shot! Ball went through the hoop cleanly."
            if evidence.get("cooldown_applied"):
                feedback += " (Potential duplicate shot)"
        else:
            feedback = f"Shot missed the target. Had basketball activity ({activity_count} frames) but ball didn't go in."
        
        shots.append({
            "timestamp_of_outcome": timestamp,
            "result": result["result"],
            "feedback": feedback,
            "total_shots_made_so_far": made_count,
            "total_shots_missed_so_far": missed_count
        })
    
    return {"shots": shots}


def main():
    parser = argparse.ArgumentParser(description="Analyze basketball grids with Gemini 2.5 Pro")
    parser.add_argument("--grids-dir", default="grids", help="Directory containing grid images")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model to use")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second in grids")
    parser.add_argument("--tile-cols", type=int, default=4, help="Grid columns")
    parser.add_argument("--tile-rows", type=int, default=5, help="Grid rows")
    parser.add_argument("--out", default="analysis/per_second.json", help="Output file for detailed analysis")
    parser.add_argument("--ball-out", default="ball.json", help="Output file for ball.json format")
    parser.add_argument("--keep-grids", action="store_true", help="Keep grid images after analysis")
    parser.add_argument("--min-consecutive", type=int, default=2, help="Minimum consecutive frames with ball in hoop for MADE (default: 2)")
    parser.add_argument("--treat-occluded-as-unknown", action="store_true", default=True, help="Return UNKNOWN for high occlusion scenarios (default: True)")
    parser.add_argument("--near-k", type=int, default=2, help="Minimum near_rim frames to consider shot attempt (default: 2)")
    parser.add_argument("--batch", action="store_true", help="Process all grids in single API call (faster)")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment")
        print("Create .env file with: GEMINI_API_KEY=your_key_here")
        return 1
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.0,
        "top_p": 0.2,
        "top_k": 1,
        "response_mime_type": "application/json"
    }
    
    try:
        model = genai.GenerativeModel(args.model, generation_config=generation_config)
    except Exception as e:
        print(f"ERROR: Failed to initialize model {args.model}: {e}")
        return 1
    
    # Find grid images
    grid_pattern = os.path.join(args.grids_dir, "*.jpg")
    grid_files = sorted(glob.glob(grid_pattern))
    
    if not grid_files:
        print(f"ERROR: No grid images found in {args.grids_dir}")
        print("Run ./scripts/make_grids.sh first to generate grids")
        return 1
    
    print(f"Found {len(grid_files)} grid images to analyze")
    print(f"Using model: {args.model}")
    print()
    
    # Analyze all grids
    if args.batch:
        print("Using batch mode (single API call for all grids)")
        results = analyze_batch_grids(model, grid_files, args.fps, 
                                    args.tile_cols, args.tile_rows,
                                    args.min_consecutive, args.treat_occluded_as_unknown, 
                                    getattr(args, 'near_k', 2))
    else:
        print("Using sequential mode (one API call per grid)")
        results = []
        for i, grid_file in enumerate(grid_files):
            result = analyze_single_grid(model, grid_file, i, args.fps, 
                                       args.tile_cols, args.tile_rows,
                                       args.min_consecutive, args.treat_occluded_as_unknown,
                                       getattr(args, 'near_k', 2))
            results.append(result)
    
    # Apply cooldown logic to prevent double-counting
    print("\nApplying cooldown logic to prevent double-counting...")
    results = apply_cooldown_logic(results, cooldown_seconds=1.0)
    
    errors = sum(1 for r in results if r.get("result") == "error")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Save detailed analysis
    output_data = {"per_second": results}
    with open(args.out, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Generate ball.json
    ball_data = synthesize_ball_json(results, args.fps)
    with open(args.ball_out, 'w') as f:
        json.dump(ball_data, f, indent=2)
    
    # Clean up grids unless --keep-grids specified
    if not args.keep_grids:
        print(f"\nCleaning up {len(grid_files)} grid images...")
        for grid_file in grid_files:
            try:
                os.remove(grid_file)
            except:
                pass
        
        # Remove empty grids directory
        try:
            os.rmdir(args.grids_dir)
        except:
            pass
    
    # Report results
    print(f"\nAnalysis complete!")
    print(f"  Total grids: {len(results)}")
    print(f"  Successful: {len(results) - errors}")
    print(f"  Errors: {errors}")
    print(f"  Detailed analysis: {args.out}")
    print(f"  Ball.json format: {args.ball_out}")
    
    # Count shots
    shots_made = sum(1 for r in results if r.get("result") == "made")
    shots_missed = sum(1 for r in results if r.get("result") == "missed")
    shots_unknown = sum(1 for r in results if r.get("result") == "unknown")
    
    print(f"  Shots detected: {shots_made} made, {shots_missed} missed, {shots_unknown} unknown")
    
    # Determine exit code
    error_rate = errors / len(results) if results else 1.0
    if error_rate >= 0.05:  # 5% threshold
        print(f"ERROR: High failure rate ({error_rate:.1%}), check API key and connectivity")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())