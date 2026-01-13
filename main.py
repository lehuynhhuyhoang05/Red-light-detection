"""
Red Light Violation Detection System
Main entry point
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils import load_config, setup_logging, create_directory_structure
from src.detector import create_detector
from src.tracker import ObjectTracker
from src.violation_logic import ViolationDetector
from src.gui import run_gui
from loguru import logger


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description='Red Light Violation Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GUI application
  python main.py --gui
  
  # Process video file (CLI mode)
  python main.py --video path/to/video.mp4
  
  # Use specific model
  python main.py --gui --model yolov11
  
  # Specify config file
  python main.py --gui --config custom_config.yaml
        """
    )
    
    parser.add_argument('--gui', action='store_true',
                       help='Launch GUI application')
    parser.add_argument('--video', type=str,
                       help='Process video file (CLI mode)')
    parser.add_argument('--model', type=str,
                       choices=['yolov11', 'yolo-nas', 'rt-detr'],
                       help='Model type (overrides config)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--output', type=str, default='data/sessions',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Please ensure config.yaml exists in the project root.")
        sys.exit(1)
    
    # Override model if specified
    if args.model:
        config['model']['type'] = args.model
    
    # Setup logging
    setup_logging(config)
    logger.info("=" * 60)
    logger.info("Red Light Violation Detection System")
    logger.info("=" * 60)
    
    # Create directory structure
    create_directory_structure(Path.cwd())
    
    # Initialize components
    try:
        logger.info("Initializing detector...")
        detector = create_detector(config)
        
        logger.info("Initializing tracker...")
        tracker = ObjectTracker(config)
        
        logger.info("Initializing violation detector...")
        violation_detector = ViolationDetector(config)
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        logger.error("Make sure all dependencies are installed:")
        logger.error("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Run application
    if args.gui:
        logger.info("Launching GUI...")
        run_gui(config, detector, tracker, violation_detector)
    
    elif args.video:
        logger.info(f"Processing video: {args.video}")
        process_video_cli(args.video, detector, tracker, violation_detector, config, args.output)
    
    else:
        parser.print_help()
        print("\nPlease specify --gui or --video")
        sys.exit(1)


def process_video_cli(video_path: str, detector, tracker, violation_detector, 
                     config: dict, output_dir: str):
    """Process video in CLI mode"""
    import cv2
    from datetime import datetime
    from tqdm import tqdm
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        sys.exit(1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {total_frames} frames, {fps} FPS, {width}x{height}")
    
    # Create session directory
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = Path(output_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Session directory: {session_dir}")
    
    # Output video
    output_video_path = session_dir / 'output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Process frames
    frame_number = 0
    
    with tqdm(total=total_frames, desc="Processing") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            timestamp = datetime.now()
            
            # Detect
            detections = detector.detect(frame)
            
            # Track
            tracked_vehicles = tracker.update(detections)
            
            # Check violations
            violations = violation_detector.update(
                tracked_vehicles, detections, frame, frame_number, timestamp
            )
            
            # Draw on frame
            annotated = detector.draw_detections(frame, detections)
            
            # Draw tracking IDs
            for vehicle in tracked_vehicles:
                x1, y1, x2, y2 = vehicle.detection.bbox
                cv2.putText(annotated, f"ID:{vehicle.track_id}",
                           (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 0), 2)
            
            # Write frame
            out.write(annotated)
            
            # Update progress
            pbar.update(1)
            pbar.set_postfix({
                'vehicles': len(tracked_vehicles),
                'violations': len(violation_detector.violations)
            })
    
    cap.release()
    out.release()
    
    # Save violations
    logger.info(f"Total violations detected: {len(violation_detector.violations)}")
    
    if violation_detector.violations:
        # Save evidence images
        violations_dir = session_dir / 'violations'
        violations_dir.mkdir(exist_ok=True)
        
        for violation in violation_detector.violations.values():
            violation_detector.save_violation_evidence(violation, violations_dir)
        
        # Save JSON
        from src.utils import save_violations_json
        json_path = session_dir / 'violations.json'
        save_violations_json(violation_detector.violations, json_path)
        
        # Generate PDF report
        try:
            from src.report_generator import ViolationReportGenerator
            report_gen = ViolationReportGenerator(config)
            pdf_path = session_dir / 'report.pdf'
            report_gen.generate_report(
                list(violation_detector.violations.values()), 
                str(pdf_path)
            )
            logger.info(f"Report saved: {pdf_path}")
        except Exception as e:
            logger.warning(f"Failed to generate PDF: {e}")
    
    logger.info(f"Output video: {output_video_path}")
    logger.info(f"Session saved to: {session_dir}")
    logger.info("Processing complete!")


if __name__ == '__main__':
    main()
