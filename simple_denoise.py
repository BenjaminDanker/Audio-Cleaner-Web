#!/usr/bin/env python3
"""
Simple video denoising script for local testing
This bypasses the Azure infrastructure and directly processes videos
"""

import os
import sys
import argparse
from pathlib import Path
from video_handler import VideoProcessor

class SimpleFileUpload:
    """Mock file upload object to work with VideoProcessor"""
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.filename = self.file_path.name
    
    def save(self, target_path):
        """Copy the file to target location"""
        import shutil
        shutil.copy2(self.file_path, target_path)

def main():
    parser = argparse.ArgumentParser(description='Denoise audio in video files')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('-o', '--output', help='Output video file path (optional)')
    parser.add_argument('-a', '--attenuation', type=int, default=30, 
                       help='Attenuation limit in dB (default: 30)')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_video)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_denoised{input_path.suffix}"
    
    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Attenuation: {args.attenuation} dB")
    print()
    
    try:
        # Create mock upload object
        mock_upload = SimpleFileUpload(input_path)
        
        # Process the video
        print("Initializing video processor...")
        processor = VideoProcessor(mock_upload, args.attenuation)
        
        print("Starting video processing...")
        print("1. Extracting audio...")
        print("2. Applying AI denoising...")
        print("3. Replacing audio in video...")
        
        temp_output = processor.process()
        
        # Copy result to final location
        import shutil
        shutil.copy2(temp_output, output_path)
        
        # Cleanup
        processor.immediate_cleanup(None)
        
        print(f"✅ Success! Denoised video saved to: {output_path}")
        print(f"📁 File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
