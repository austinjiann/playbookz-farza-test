# gemini-bball

Hi this is the code from [this](https://x.com/FarzaTV/status/1928484483076087922) viral demo.

<img width="543" alt="Screenshot 2025-07-02 at 1 31 28 PM" src="https://github.com/user-attachments/assets/8d317156-f187-470c-8e26-5b7f7f60d6f2" />

Please read `ball.json`. That's where the magic is. `ball.py` is mostly just an OpenCV visualizer.

To make this a real time product you'll need to:

1) Smartly send frames to Gemini (Gemini Video can only handle 1 FPS)
2) Use Gemini API to return content.
3) Render it.

This would make a killer iOS app.

Good luck.

## Grid + Gemini Analysis

This repo now includes an automated pipeline to analyze basketball videos using per-second image grids and Gemini 2.5 Pro.

### Setup

1. **Environment setup**:
   ```bash
   cp .env.example .env
   # Edit .env and set your GEMINI_API_KEY
   ```

2. **Install FFmpeg** (required for grid generation):
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian  
   apt install ffmpeg
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Generate grids** (creates one 4×5 image per second):
   ```bash
   ./scripts/make_grids.sh
   ```
   
   Or with custom parameters:
   ```bash
   IN=your_video.mp4 OUT=custom_grids ./scripts/make_grids.sh
   ```

2. **Run analysis** (sends grids to Gemini 2.5 Pro):
   ```bash
   python tools/analyze_grids.py --grids-dir grids
   ```

### Outputs

- **`analysis/per_second.json`**: Detailed per-second analysis with evidence and confidence scores
- **`ball.json`**: Compatible format for existing `ball.py` visualization with running totals
- **Timestamps**: Standardized MM:SS.SSS format with frame-level precision

### Options

- `--keep-grids`: Preserve grid images after analysis (default: deleted to save space)
- `--no-pad-last-second`: Skip partial seconds instead of padding to 20 frames
- `--model gemini-2.5-pro`: Change Gemini model (default: gemini-2.5-pro)

### Simple Ball-in-Hoop Heuristic

The analysis uses a simplified frame-by-frame approach for shot detection:

**Rule**: Any frame showing the ball inside the rim cylinder → MADE; otherwise MISSED

**Options**:
- `--min-consecutive N`: Require N consecutive frames with ball in hoop (default: 1)
- `--treat-occluded-as-unknown`: Return UNKNOWN if all 20 frames are occluded (default: treat as MISSED)

**Behavior**:
- MADE timestamps use the first positive frame for precise timing
- MISSED timestamps use the window midpoint (0.5s offset)  
- Temperature 0.0 eliminates AI creativity for consistent results

### Error Handling

The analysis tool includes robust error handling:
- Exponential backoff retry for API rate limits
- Graceful handling of failed grids (continues processing)
- Exit code reflects success rate (fails if >5% of grids fail)
