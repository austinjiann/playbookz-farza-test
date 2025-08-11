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


def create_analysis_prompt(second_index: int, start_ts: str, end_ts: str) -> str:
    """Create the basketball analysis prompt."""
    return f"""You are a basketball shot outcome reviewer. Analyze ONLY the single 4x5 image grid I provide.
Each grid contains 20 frames sampled at 20 fps (0.05s per frame), laid out in strict chronological order:
top-left → right, then next row; i.e., row-major order.

Your job is to decide whether a *single* shot in this one-second window is MADE, MISSED, or UNKNOWN.

Strict definitions:
- MADE: Ball fully passes downward through the rim cylinder and emerges below the net, or the net shows a clear downward snap consistent with a make.
- MISSED: Ball departs outward/upward without passing fully below the rim plane, clear rim/backboard reject, or visible airball long/short.
- UNKNOWN: View obstructed or frames insufficient to be decisive.

Output ONLY this JSON (no extra text):
{{
  "second_index": {second_index},
  "window_start_ts": "{start_ts}",
  "window_end_ts": "{end_ts}",
  "result": "made" | "missed" | "unknown",
  "confidence": 0.0-1.0,
  "evidence": {{
    "frame_index_below_rim": null | 0-19,
    "frame_index_emerge_below_net": null | 0-19,
    "net_snap_observed": true | false,
    "notes": "one short sentence"
  }}
}}"""


def analyze_single_grid(model: genai.GenerativeModel, image_path: str, 
                       second_index: int, fps: int) -> Dict[str, Any]:
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
        prompt = create_analysis_prompt(second_index, start_ts, end_ts)
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
        
        # Validate required fields
        required_fields = ["second_index", "window_start_ts", "window_end_ts", "result", "confidence"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        print(f"    Result: {result['result']} (confidence: {result['confidence']:.2f})")
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
            "evidence": {
                "frame_index_below_rim": None,
                "frame_index_emerge_below_net": None,
                "net_snap_observed": False,
                "notes": f"Analysis failed: {str(e)[:50]}..."
            }
        }


def synthesize_ball_json(per_second_results: List[Dict[str, Any]], fps: int) -> Dict[str, Any]:
    """Convert per-second analysis to ball.json format."""
    shots = []
    made_count = 0
    missed_count = 0
    
    for result in per_second_results:
        if result.get("result") not in ["made", "missed"]:
            continue  # Skip unknown and error results
        
        # Calculate precise timestamp
        start_seconds = result["second_index"]
        
        # Use evidence for sub-second precision
        evidence = result.get("evidence", {})
        frame_offset = 0
        
        if evidence.get("frame_index_emerge_below_net") is not None:
            frame_offset = evidence["frame_index_emerge_below_net"]
        elif evidence.get("frame_index_below_rim") is not None:
            frame_offset = evidence["frame_index_below_rim"]
        else:
            frame_offset = 10  # Use middle of window (0.5s)
        
        # Calculate final timestamp
        precise_seconds = start_seconds + (frame_offset / fps)
        timestamp = format_timestamp(precise_seconds)
        
        # Update counts
        if result["result"] == "made":
            made_count += 1
        else:  # missed
            missed_count += 1
        
        shots.append({
            "timestamp_of_outcome": timestamp,
            "result": result["result"],
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
        "temperature": 0.1,
        "top_p": 0.9,
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
    results = []
    errors = 0
    
    for i, grid_file in enumerate(grid_files):
        result = analyze_single_grid(model, grid_file, i, args.fps)
        results.append(result)
        
        if result.get("result") == "error":
            errors += 1
    
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