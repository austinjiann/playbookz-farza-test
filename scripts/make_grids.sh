#!/bin/bash

# Default configuration
IN="${IN:-final_ball.mov}"
OUT="${OUT:-grids}"
FPS="${FPS:-20}"
TILE_COLS="${TILE_COLS:-4}"
TILE_ROWS="${TILE_ROWS:-5}"
TILE_WIDTH="${TILE_WIDTH:-300}"
PAD_LAST_SECOND="${PAD_LAST_SECOND:-true}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input|-i)
      IN="$2"
      shift 2
      ;;
    --output|-o)
      OUT="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --tile-cols)
      TILE_COLS="$2"
      shift 2
      ;;
    --tile-rows)
      TILE_ROWS="$2"
      shift 2
      ;;
    --tile-width)
      TILE_WIDTH="$2"
      shift 2
      ;;
    --pad-last-second)
      PAD_LAST_SECOND="true"
      shift
      ;;
    --no-pad-last-second)
      PAD_LAST_SECOND="false"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --input, -i FILE       Input video file (default: final_ball.mov)"
      echo "  --output, -o DIR       Output directory (default: grids)"
      echo "  --fps NUM             Frames per second (default: 20)"
      echo "  --tile-cols NUM        Grid columns (default: 4)"
      echo "  --tile-rows NUM        Grid rows (default: 5)"
      echo "  --tile-width NUM       Width of each tile (default: 300)"
      echo "  --pad-last-second      Pad partial last second (default)"
      echo "  --no-pad-last-second   Skip partial last second if <20 frames"
      echo "  --help, -h             Show this help"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate input file exists
if [[ ! -f "$IN" ]]; then
  echo "Error: Input file '$IN' not found"
  exit 1
fi

# Check for required tools
for tool in ffmpeg ffprobe; do
  if ! command -v "$tool" &> /dev/null; then
    echo "Error: $tool is required but not installed"
    echo "Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)"
    exit 1
  fi
done

# Create output directory
mkdir -p "$OUT"

# Get video duration
echo "Analyzing video: $IN"
duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$IN")
if [[ -z "$duration" ]]; then
  echo "Error: Could not determine video duration"
  exit 1
fi

# Calculate number of full seconds
secs=$(echo "$duration" | awk '{print int($1)}')
remaining=$(echo "$duration $secs" | awk '{print $1 - $2}')

echo "Video duration: ${duration}s (${secs} full seconds + ${remaining}s remaining)"

# Determine total grids to create
if [[ "$PAD_LAST_SECOND" == "true" ]]; then
  total_grids=$((secs + 1))
  echo "Creating $total_grids grids (padding last ${remaining}s to full second)"
else
  # Only create grid for partial second if it has enough frames
  min_frames_needed=$((TILE_COLS * TILE_ROWS))
  frames_in_remaining=$(echo "$remaining $FPS" | awk '{print int($1 * $2)}')
  
  if [[ $frames_in_remaining -ge $min_frames_needed ]]; then
    total_grids=$((secs + 1))
    echo "Creating $total_grids grids (partial second has $frames_in_remaining frames)"
  else
    total_grids=$secs
    echo "Creating $total_grids grids (skipping partial second with only $frames_in_remaining frames)"
  fi
fi

# Generate grids
successful_grids=0
for ((s=0; s<total_grids; s++)); do
  output_file="$OUT/grid_$(printf "%02d" $((s+1)))_${FPS}fps.jpg"
  
  # Determine if this is the last partial second that needs padding
  if [[ $s -eq $secs && "$PAD_LAST_SECOND" == "true" && $(echo "$remaining > 0" | bc -l) -eq 1 ]]; then
    # Pad the last partial second
    pad_duration=$(echo "1.0 - $remaining" | awk '{print $1 - $2}')
    vf_filter="fps=${FPS},scale=${TILE_WIDTH}:-2,tpad=stop_mode=clone:stop_duration=${pad_duration},tile=${TILE_COLS}x${TILE_ROWS}"
  else
    # Regular 1-second window
    vf_filter="fps=${FPS},scale=${TILE_WIDTH}:-2,tile=${TILE_COLS}x${TILE_ROWS}"
  fi
  
  echo "Creating grid $((s+1))/${total_grids}: ${output_file##*/}"
  
  if ffmpeg -y -hide_banner -loglevel error \
    -ss "$s" -t 1 -i "$IN" \
    -vf "$vf_filter" \
    "$output_file"; then
    ((successful_grids++))
  else
    echo "Warning: Failed to create grid $((s+1))"
  fi
done

echo ""
echo "Grid generation complete!"
echo "Created: $successful_grids/$total_grids grids"
echo "Output directory: $OUT"

if [[ $successful_grids -eq $total_grids ]]; then
  echo "All grids created successfully"
  exit 0
else
  echo "Warning: $((total_grids - successful_grids)) grids failed"
  exit 1
fi