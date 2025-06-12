# Emotion Heatmap Generator from YouTube Interviews

A powerful AI-driven tool that analyzes emotions in YouTube interviews and generates interactive heatmaps to visualize emotional patterns over time.

## Features

- 🎥 **YouTube Video Processing**: Download and process YouTube videos automatically
- 🎤 **Speech-to-Text**: Convert audio to text using Google Speech Recognition
- 🤖 **AI Emotion Analysis**: Analyze emotions using state-of-the-art transformer models
- 📊 **Interactive Heatmaps**: Generate beautiful emotion heatmaps and timelines
- 🌐 **Web Interface**: User-friendly Streamlit web application
- 💾 **Export Options**: Download results as CSV and PNG files

## Supported Emotions

- Joy
- Sadness
- Anger
- Fear
- Surprise
- Disgust
- Neutral

## Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/yourusername/emotion-heatmap-generator.git
cd emotion-heatmap-generator
\`\`\`

2. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Install additional system dependencies:
\`\`\`bash
# For Ubuntu/Debian
sudo apt-get install ffmpeg

# For macOS
brew install ffmpeg

# For Windows
# Download ffmpeg from https://ffmpeg.org/download.html
\`\`\`

## Usage

### Web Application

1. Start the Streamlit app:
\`\`\`bash
streamlit run main.py
\`\`\`

2. Open your browser and navigate to `http://localhost:8501`

3. Enter a YouTube URL and configure analysis parameters

4. Click "Generate Emotion Heatmap" to start the analysis

### Command Line Usage

```python
from src.youtube_processor import YouTubeProcessor
from src.emotion_analyzer import EmotionAnalyzer
from src.heatmap_generator import HeatmapGenerator

# Initialize components
youtube_processor = YouTubeProcessor()
emotion_analyzer = EmotionAnalyzer()
heatmap_generator = HeatmapGenerator()

# Process video
audio_path, video_info = youtube_processor.process_video("YOUR_YOUTUBE_URL")

# Extract speech segments
segments = youtube_processor.extract_speech_segments(audio_path, segment_duration=30)

# Analyze emotions
emotion_results = []
for segment in segments:
    emotions = emotion_analyzer.analyze_emotions(segment['text'])
    emotion_results.append({
        'timestamp': segment['timestamp'],
        'text': segment['text'],
        **emotions
    })

# Generate heatmap
import pandas as pd
df = pd.DataFrame(emotion_results)
fig = heatmap_generator.create_heatmap(df, video_info['title'])
