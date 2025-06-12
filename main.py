"""
Emotion Heatmap Generator from YouTube Interviews
Main application entry point
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import tempfile

from src.youtube_processor import YouTubeProcessor
from src.emotion_analyzer import EmotionAnalyzer
from src.heatmap_generator import HeatmapGenerator
from src.utils import setup_directories, validate_youtube_url

def main():
    st.set_page_config(
        page_title="Emotion Heatmap Generator",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Emotion Heatmap Generator from YouTube Interviews")
    st.markdown("Analyze emotions in YouTube interviews and generate interactive heatmaps using AI")
    
    # Setup directories
    setup_directories()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # YouTube URL input
    youtube_url = st.text_input(
        "Enter YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    # Analysis parameters
    segment_duration = st.sidebar.slider(
        "Segment Duration (seconds)", 
        min_value=10, 
        max_value=60, 
        value=30
    )
    
    emotion_model = st.sidebar.selectbox(
        "Emotion Analysis Model",
        ["cardiffnlp/twitter-roberta-base-emotion", "j-hartmann/emotion-english-distilroberta-base"]
    )
    
    if st.button("🚀 Generate Emotion Heatmap", type="primary"):
        if not youtube_url or not validate_youtube_url(youtube_url):
            st.error("Please enter a valid YouTube URL")
            return
            
        try:
            with st.spinner("Processing video and analyzing emotions..."):
                # Initialize processors
                youtube_processor = YouTubeProcessor()
                emotion_analyzer = EmotionAnalyzer(model_name=emotion_model)
                heatmap_generator = HeatmapGenerator()
                
                # Process YouTube video
                st.info("📥 Downloading and processing video...")
                audio_path, video_info = youtube_processor.process_video(youtube_url)
                
                # Display video information
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📹 Video Information")
                    st.write(f"**Title:** {video_info['title']}")
                    st.write(f"**Duration:** {video_info['duration']} seconds")
                    st.write(f"**Channel:** {video_info['uploader']}")
                
                # Extract and analyze audio
                st.info("🎤 Extracting speech and analyzing emotions...")
                segments = youtube_processor.extract_speech_segments(
                    audio_path, 
                    segment_duration
                )
                
                # Analyze emotions for each segment
                emotion_results = []
                progress_bar = st.progress(0)
                
                for i, segment in enumerate(segments):
                    emotions = emotion_analyzer.analyze_emotions(segment['text'])
                    emotion_results.append({
                        'timestamp': segment['timestamp'],
                        'text': segment['text'],
                        **emotions
                    })
                    progress_bar.progress((i + 1) / len(segments))
                
                # Create DataFrame
                df = pd.DataFrame(emotion_results)
                
                # Generate heatmap
                st.info("📊 Generating emotion heatmap...")
                fig = heatmap_generator.create_heatmap(df, video_info['title'])
                
                # Display results
                with col2:
                    st.subheader("📈 Analysis Summary")
                    emotion_cols = [col for col in df.columns if col not in ['timestamp', 'text']]
                    avg_emotions = df[emotion_cols].mean()
                    dominant_emotion = avg_emotions.idxmax()
                    st.write(f"**Dominant Emotion:** {dominant_emotion.title()}")
                    st.write(f"**Total Segments:** {len(segments)}")
                
                # Display heatmap
                st.subheader("🎭 Emotion Heatmap")
                st.pyplot(fig)
                
                # Display detailed results
                st.subheader("📋 Detailed Analysis")
                
                # Emotion timeline
                timeline_fig = heatmap_generator.create_timeline(df)
                st.pyplot(timeline_fig)
                
                # Data table
                st.subheader("📊 Raw Data")
                st.dataframe(df, use_container_width=True)
                
                # Download options
                st.subheader("💾 Download Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        f"emotion_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                
                with col2:
                    # Save heatmap as image
                    img_buffer = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    fig.savefig(img_buffer.name, dpi=300, bbox_inches='tight')
                    
                    with open(img_buffer.name, 'rb') as f:
                        st.download_button(
                            "Download Heatmap",
                            f.read(),
                            f"emotion_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            "image/png"
                        )
                
                # Cleanup
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check the YouTube URL and try again.")

if __name__ == "__main__":
    main()
