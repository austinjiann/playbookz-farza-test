# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a basketball shot analysis demo that uses AI to analyze basketball shots in video footage. It's based on a viral demo showing real-time basketball coaching feedback. The project consists of:

- `ball.json` - Contains shot analysis data (timestamps, results, running totals)
- `ball.py` - OpenCV-based video processor and visualizer
- `final_ball.mov` - Input video file for analysis
- `README.md` - Project documentation and setup instructions

## Key Architecture

### Data Flow
1. **Input**: Video file (`final_ball.mov`) containing basketball footage
2. **Analysis**: Shot data is stored in `ball.json` with AI-generated analysis
3. **Processing**: `ball.py` processes the video using OpenCV and MediaPipe for pose detection
4. **Output**: Generates annotated video with shot statistics and feedback overlays

### Core Components

**Shot Analysis (`ball.json`)**
- Contains timestamped shot data with results (made/missed)
- Includes running totals for made and missed shots
- Schema format:
  ```json
  {
    "shots": [
      {
        "timestamp_of_outcome": "MM:SS.sss",
        "result": "made" or "missed",
        "total_shots_made_so_far": number,
        "total_shots_missed_so_far": number
      }
    ]
  }
  ```

**Video Processing (`ball.py`)**
- Uses MediaPipe for pose detection and head tracking
- Overlays player name and arrow pointer
- Displays real-time shot statistics with color animations
- Processes dual video streams (one for processing, one for display)
- Key features:
  - Head tracking with red arrow overlay
  - Animated statistics (green for made shots, red for misses)
  - Text wrapping and positioning for feedback display
  - Dual resolution processing for performance optimization

### Dependencies
- OpenCV (`cv2`) - Video processing and computer vision
- MediaPipe (`mediapipe`) - Pose detection and tracking
- NumPy - Numerical operations for video frames
- JSON - Data parsing for shot analysis

## Running the Application

### Basic Usage
```bash
python ball.py
```

### Prerequisites
- Ensure video files exist at the expected paths (update paths in `ball.py` lines 21-22)
- `ball.json` must contain valid shot analysis data
- Install dependencies: `pip install opencv-python mediapipe numpy`

### Key Configuration
The script expects specific file paths:
- `process_video_path` (line 21) - Lower resolution video for pose detection
- `display_video_path` (line 22) - Higher resolution video for final output
- Output video: `final.mp4`

## Development Architecture

### Video Processing Pipeline
1. **Dual Video Input**: Separate streams for processing (pose detection) and display (final output)
2. **Frame Processing**: Processes every nth frame (20 FPS) for pose detection while maintaining display quality
3. **Shot Timing**: Converts timestamps from `ball.json` to frame numbers for precise timing
4. **Animation System**: Color-coded feedback system with smooth transitions (1.25s duration)
5. **Overlay System**: Real-time head tracking with arrow and text overlays

### Performance Optimizations
- Dual resolution: Lower resolution for pose detection, higher for display
- Frame skipping: Processes pose detection at 20 FPS regardless of input framerate
- Efficient overlay rendering with border/fill text rendering
- Pre-calculated shot timing based on frame numbers

## Grid + Gemini Analysis Pipeline

The repository now includes a complete automated analysis pipeline:

### Core Pipeline
- **Grid Generation**: `scripts/make_grids.sh` creates 4×5 image grids (20 frames at 20fps per second)
- **AI Analysis**: `tools/analyze_grids.py` sends grids to Gemini 2.5 Pro for shot outcome detection
- **Data Integration**: Outputs both detailed per-second analysis and legacy ball.json format

### Key Features
- **Exponential backoff retry** with jitter for API resilience
- **Error handling** with 5% failure threshold before exit
- **Timestamp standardization** (MM:SS.SSS format) with frame-level precision
- **Automatic cleanup** of temporary grid files (configurable)
- **Dual output formats** for analysis and visualization compatibility

### Extension Points
To make this production-ready (as noted in README):
1. Implement smart frame sampling for AI analysis (1 FPS limit consideration)
2. Integrate real-time AI API calls for shot analysis
3. Add proper video input/output handling
4. Consider mobile app implementation (iOS suggested)

## File Structure
- `ball.json` - Shot analysis data (JSON format)
- `ball.py` - Main video processing and visualization script
- `final_ball.mov` - Input video file
- `final.mp4` - Generated output video with overlays
- `README.md` - Project documentation

## Development Notes

### Grid Pipeline Usage
**Grid Regeneration**: Claude Code can regenerate grids and re-run analysis by executing:
```bash
./scripts/make_grids.sh
python tools/analyze_grids.py --grids-dir grids
```

**Environment Setup**: `.env` file must be present locally with valid GEMINI_API_KEY (never committed to repo). Use `.env.example` as template.

### Legacy System
- The current implementation uses pre-processed shot analysis data
- Video processing is optimized to run at 20 FPS for pose detection while maintaining display quality
- The system supports animated feedback with precise timestamp-based triggering
- Text rendering includes proper border/fill techniques for visibility over video