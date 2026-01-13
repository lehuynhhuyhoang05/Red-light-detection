"""
GUI Application using PySide6
Main interface for Red Light Violation Detection System
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QFileDialog, QTableWidget,
    QTableWidgetItem, QSlider, QSpinBox, QComboBox, QTextEdit,
    QGroupBox, QFormLayout, QLineEdit, QProgressBar, QMessageBox,
    QSplitter, QHeaderView
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QFont

from loguru import logger


class VideoProcessor(QThread):
    """Thread for processing video"""
    
    frame_processed = Signal(np.ndarray, dict)  # frame, stats
    progress_updated = Signal(int, int)  # current, total
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, video_path: str, detector, tracker, violation_detector):
        super().__init__()
        self.video_path = video_path
        self.detector = detector
        self.tracker = tracker
        self.violation_detector = violation_detector
        self.is_running = True
        self.is_paused = False
    
    def run(self):
        """Process video"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = 0
            
            while self.is_running and cap.isOpened():
                if not self.is_paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_number += 1
                    timestamp = datetime.now()
                    
                    # Detect objects
                    detections = self.detector.detect(frame)
                    
                    # Track objects
                    tracked_vehicles = self.tracker.update(detections)
                    
                    # Check violations
                    violations = self.violation_detector.update(
                        tracked_vehicles, detections, frame, frame_number, timestamp
                    )
                    
                    # Draw on frame
                    annotated = self.detector.draw_detections(frame, detections)
                    
                    # Draw tracking IDs
                    for vehicle in tracked_vehicles:
                        x1, y1, x2, y2 = vehicle.detection.bbox
                        cv2.putText(annotated, f"ID:{vehicle.track_id}",
                                   (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5, (255, 255, 0), 2)
                    
                    # Statistics
                    stats = {
                        'frame': frame_number,
                        'total_frames': total_frames,
                        'vehicles': len(tracked_vehicles),
                        'violations': len(violations),
                        'light_state': self.violation_detector.current_light_state
                    }
                    
                    self.frame_processed.emit(annotated, stats)
                    self.progress_updated.emit(frame_number, total_frames)
                
                self.msleep(int(1000 / fps))
            
            cap.release()
            self.finished.emit()
            
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            self.error.emit(str(e))
    
    def pause(self):
        self.is_paused = True
    
    def resume(self):
        self.is_paused = False
    
    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, config: dict, detector, tracker, violation_detector):
        super().__init__()
        
        self.config = config
        self.detector = detector
        self.tracker = tracker
        self.violation_detector = violation_detector
        
        self.video_processor: Optional[VideoProcessor] = None
        self.current_frame: Optional[np.ndarray] = None
        
        self.init_ui()
        self.setup_connections()
        
        logger.info("GUI initialized")
    
    def init_ui(self):
        """Initialize UI components"""
        gui_config = self.config.get('gui', {})
        
        self.setWindowTitle(gui_config.get('window_title', 'Red Light Violation Detection'))
        self.setGeometry(100, 100, 
                        gui_config.get('window_width', 1920),
                        gui_config.get('window_height', 1080))
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Tab widget
        self.tabs = QTabWidget()
        
        self.video_tab = self.create_video_tab()
        self.violations_tab = self.create_violations_tab()
        self.comparison_tab = self.create_comparison_tab()
        self.stats_tab = self.create_stats_tab()
        self.settings_tab = self.create_settings_tab()
        
        self.tabs.addTab(self.video_tab, "📹 Video")
        self.tabs.addTab(self.violations_tab, "⚠️ Vi phạm")
        self.tabs.addTab(self.comparison_tab, "🔄 So sánh Models")
        self.tabs.addTab(self.stats_tab, "📊 Thống kê")
        self.tabs.addTab(self.settings_tab, "⚙️ Cài đặt")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.status_label = QLabel("Sẵn sàng")
        self.statusBar().addWidget(self.status_label)
    
    def create_control_panel(self) -> QWidget:
        """Create control panel"""
        panel = QGroupBox("Điều khiển")
        layout = QHBoxLayout()
        
        # File selection
        self.file_label = QLabel("Chưa chọn file")
        self.file_label.setMinimumWidth(300)
        layout.addWidget(QLabel("Video:"))
        layout.addWidget(self.file_label)
        
        self.btn_select = QPushButton("📂 Chọn video")
        self.btn_play = QPushButton("▶️ Phát")
        self.btn_pause = QPushButton("⏸️ Tạm dừng")
        self.btn_stop = QPushButton("⏹️ Dừng")
        
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        
        layout.addWidget(self.btn_select)
        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_stop)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(200)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_video_tab(self) -> QWidget:
        """Create video display tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        self.video_label.setMinimumSize(1280, 720)
        
        layout.addWidget(self.video_label)
        
        # Info panel
        info_layout = QHBoxLayout()
        
        self.lbl_frame = QLabel("Frame: 0/0")
        self.lbl_vehicles = QLabel("Xe: 0")
        self.lbl_light = QLabel("Đèn: -")
        self.lbl_violations = QLabel("Vi phạm: 0")
        
        font = QFont()
        font.setPointSize(12)
        for lbl in [self.lbl_frame, self.lbl_vehicles, self.lbl_light, self.lbl_violations]:
            lbl.setFont(font)
            info_layout.addWidget(lbl)
        
        info_layout.addStretch()
        layout.addLayout(info_layout)
        
        widget.setLayout(layout)
        return widget
    
    def create_violations_tab(self) -> QWidget:
        """Create violations list tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Violations table
        self.violations_table = QTableWidget()
        self.violations_table.setColumnCount(7)
        self.violations_table.setHorizontalHeaderLabels([
            "ID", "Thời gian", "Frame", "Loại xe", "Trạng thái đèn", "Độ tin cậy", "Tình trạng"
        ])
        self.violations_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(self.violations_table)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_view_evidence = QPushButton("👁️ Xem bằng chứng")
        self.btn_export_pdf = QPushButton("📄 Xuất biên bản PDF")
        self.btn_export_json = QPushButton("💾 Xuất JSON")
        
        btn_layout.addWidget(self.btn_view_evidence)
        btn_layout.addWidget(self.btn_export_pdf)
        btn_layout.addWidget(self.btn_export_json)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        
        widget.setLayout(layout)
        return widget
    
    def create_comparison_tab(self) -> QWidget:
        """Create model comparison tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("🔄 So sánh YOLOv11 vs RT-DETR")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header)
        
        # Info text
        info = QLabel("Chức năng so sánh hiệu năng 2 models trên cùng video input.")
        layout.addWidget(info)
        
        # Comparison controls
        control_layout = QHBoxLayout()
        
        self.btn_run_comparison = QPushButton("▶️ Chạy so sánh")
        self.btn_run_comparison.setMinimumHeight(40)
        control_layout.addWidget(self.btn_run_comparison)
        
        self.comparison_progress = QProgressBar()
        control_layout.addWidget(self.comparison_progress)
        
        layout.addLayout(control_layout)
        
        # Results table
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(4)
        self.comparison_table.setHorizontalHeaderLabels([
            "Metric", "YOLOv11", "RT-DETR", "Winner"
        ])
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Pre-fill metrics
        metrics = [
            ("Model Size (MB)", "5.24", "121.53", "YOLOv11 ⭐"),
            ("Load Time (s)", "0.13", "N/A", "YOLOv11 ⭐"),
            ("Inference Time (ms)", "~30", "N/A", "TBD"),
            ("mAP@50", "87.9%", "TBD", "TBD"),
            ("FPS (avg)", "~30", "N/A", "TBD"),
            ("Violations Detected", "-", "-", "-"),
            ("False Positives", "-", "-", "-"),
        ]
        
        self.comparison_table.setRowCount(len(metrics))
        for row, (metric, yolo, rtdetr, winner) in enumerate(metrics):
            self.comparison_table.setItem(row, 0, QTableWidgetItem(metric))
            self.comparison_table.setItem(row, 1, QTableWidgetItem(yolo))
            self.comparison_table.setItem(row, 2, QTableWidgetItem(rtdetr))
            self.comparison_table.setItem(row, 3, QTableWidgetItem(winner))
        
        layout.addWidget(self.comparison_table)
        
        # Notes
        notes = QTextEdit()
        notes.setReadOnly(True)
        notes.setMaximumHeight(150)
        notes.setText("""
📝 GHI CHÚ SO SÁNH:

• YOLOv11: Model nhỏ gọn (5.24 MB), inference nhanh, phù hợp edge deployment
• RT-DETR: Model lớn (121.53 MB), cần GPU mạnh, độ chính xác cao với objects nhỏ
• RF-DETR (Roboflow variant): Cần wrapper riêng để load, chưa implement

🎯 KHUYẾN NGHỊ: Sử dụng YOLOv11 cho production vì hiệu quả tài nguyên tốt hơn.
        """)
        layout.addWidget(notes)
        
        # Export button
        self.btn_export_comparison = QPushButton("📄 Xuất báo cáo so sánh (PDF)")
        layout.addWidget(self.btn_export_comparison)
        
        widget.setLayout(layout)
        return widget
    
    def create_stats_tab(self) -> QWidget:
        """Create statistics tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Statistics display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Courier", 10))
        
        layout.addWidget(QLabel("📊 Thống kê tổng quan"))
        layout.addWidget(self.stats_text)
        
        # Refresh button
        btn_refresh = QPushButton("🔄 Làm mới")
        btn_refresh.clicked.connect(self.update_statistics)
        layout.addWidget(btn_refresh)
        
        widget.setLayout(layout)
        return widget
    
    def create_settings_tab(self) -> QWidget:
        """Create settings tab"""
        widget = QWidget()
        layout = QFormLayout()
        
        # Model selection
        self.combo_model = QComboBox()
        self.combo_model.addItems(['yolov11', 'yolo-nas', 'rt-detr'])
        self.combo_model.setCurrentText(self.config['model']['type'])
        layout.addRow("Mô hình:", self.combo_model)
        
        # Confidence threshold
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 95)
        self.slider_conf.setValue(50)
        self.lbl_conf = QLabel("0.50")
        
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(self.slider_conf)
        conf_layout.addWidget(self.lbl_conf)
        
        layout.addRow("Ngưỡng tin cậy:", conf_layout)
        
        # Stop line position
        self.spin_stopline = QSpinBox()
        self.spin_stopline.setRange(0, 2000)
        self.spin_stopline.setValue(500)
        layout.addRow("Vị trí vạch dừng (Y):", self.spin_stopline)
        
        # Location info
        self.edit_location = QLineEdit(self.config['location']['intersection'])
        layout.addRow("Địa điểm:", self.edit_location)
        
        # Camera ID
        self.edit_camera = QLineEdit(self.config['location']['camera_id'])
        layout.addRow("Mã camera:", self.edit_camera)
        
        # Save button
        btn_save = QPushButton("💾 Lưu cài đặt")
        btn_save.clicked.connect(self.save_settings)
        layout.addRow("", btn_save)
        
        widget.setLayout(layout)
        return widget
    
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.btn_select.clicked.connect(self.select_video)
        self.btn_play.clicked.connect(self.play_video)
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_stop.clicked.connect(self.stop_video)
        
        self.slider_conf.valueChanged.connect(
            lambda v: self.lbl_conf.setText(f"{v/100:.2f}")
        )
        
        # Violations tab buttons
        self.btn_view_evidence.clicked.connect(self.view_evidence)
        self.btn_export_pdf.clicked.connect(self.export_pdf)
        self.btn_export_json.clicked.connect(self.export_json)
    
    @Slot()
    def select_video(self):
        """Select video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.file_label.setText(Path(file_path).name)
            self.btn_play.setEnabled(True)
            logger.info(f"Video selected: {file_path}")
    
    @Slot()
    def play_video(self):
        """Start video processing"""
        if not hasattr(self, 'video_path'):
            return
        
        self.video_processor = VideoProcessor(
            self.video_path, self.detector, self.tracker, self.violation_detector
        )
        
        self.video_processor.frame_processed.connect(self.on_frame_processed)
        self.video_processor.progress_updated.connect(self.on_progress_updated)
        self.video_processor.finished.connect(self.on_processing_finished)
        self.video_processor.error.connect(self.on_error)
        
        self.video_processor.start()
        
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("Đang xử lý video...")
    
    @Slot()
    def pause_video(self):
        """Pause video processing"""
        if self.video_processor:
            if self.video_processor.is_paused:
                self.video_processor.resume()
                self.btn_pause.setText("⏸️ Tạm dừng")
            else:
                self.video_processor.pause()
                self.btn_pause.setText("▶️ Tiếp tục")
    
    @Slot()
    def stop_video(self):
        """Stop video processing"""
        if self.video_processor:
            self.video_processor.stop()
            self.video_processor.wait()
        
        self.reset_controls()
    
    @Slot(np.ndarray, dict)
    def on_frame_processed(self, frame: np.ndarray, stats: dict):
        """Handle processed frame"""
        # Convert to QPixmap
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update info labels
        self.lbl_frame.setText(f"Frame: {stats['frame']}/{stats['total_frames']}")
        self.lbl_vehicles.setText(f"Xe: {stats['vehicles']}")
        self.lbl_light.setText(f"Đèn: {stats.get('light_state', '-')}")
        self.lbl_violations.setText(f"Vi phạm: {len(self.violation_detector.violations)}")
        
        # Update violations table
        self.update_violations_table()
    
    @Slot(int, int)
    def on_progress_updated(self, current: int, total: int):
        """Update progress bar"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
    
    @Slot()
    def on_processing_finished(self):
        """Handle processing finished"""
        self.status_label.setText("Xử lý hoàn tất!")
        self.reset_controls()
        QMessageBox.information(self, "Hoàn tất", 
                               f"Đã phát hiện {len(self.violation_detector.violations)} vi phạm")
    
    @Slot(str)
    def on_error(self, error_msg: str):
        """Handle error"""
        QMessageBox.critical(self, "Lỗi", f"Lỗi xử lý video:\n{error_msg}")
        self.reset_controls()
    
    def reset_controls(self):
        """Reset control buttons"""
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("⏸️ Tạm dừng")
        self.status_label.setText("Sẵn sàng")
    
    def update_violations_table(self):
        """Update violations table"""
        violations = list(self.violation_detector.violations.values())
        self.violations_table.setRowCount(len(violations))
        
        for row, violation in enumerate(violations):
            self.violations_table.setItem(row, 0, QTableWidgetItem(violation.violation_id))
            self.violations_table.setItem(row, 1, 
                QTableWidgetItem(violation.timestamp.strftime('%Y-%m-%d %H:%M:%S')))
            self.violations_table.setItem(row, 2, 
                QTableWidgetItem(str(violation.frame_number)))
            self.violations_table.setItem(row, 3, 
                QTableWidgetItem(violation.vehicle_class))
            self.violations_table.setItem(row, 4, 
                QTableWidgetItem(violation.light_state))
            self.violations_table.setItem(row, 5, 
                QTableWidgetItem(f"{violation.vehicle_confidence:.2f}"))
            self.violations_table.setItem(row, 6, 
                QTableWidgetItem(violation.status))
    
    def update_statistics(self):
        """Update statistics display"""
        stats = self.violation_detector.get_statistics()
        
        text = f"""
╔══════════════════════════════════════════╗
║     THỐNG KÊ VI PHẠM VƯỢT ĐÈN ĐỎ        ║
╚══════════════════════════════════════════╝

📊 Tổng số vi phạm: {stats['total_violations']}

🚗 Phân loại theo loại xe:
"""
        
        for vehicle_class, count in stats['by_vehicle_class'].items():
            text += f"   • {vehicle_class}: {count}\n"
        
        text += f"""
🚦 Trạng thái đèn hiện tại: {stats.get('current_light', 'N/A')}

📍 Địa điểm: {self.config['location']['intersection']}
📹 Camera: {self.config['location']['camera_id']}
"""
        
        self.stats_text.setText(text)
    
    def save_settings(self):
        """Save settings"""
        # Update config
        self.config['model']['type'] = self.combo_model.currentText()
        
        QMessageBox.information(self, "Thành công", "Đã lưu cài đặt!")
        logger.info("Settings saved")
    
    @Slot()
    def view_evidence(self):
        """View evidence for selected violation"""
        selected = self.violations_table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một vi phạm từ danh sách!")
            return
        
        violations = list(self.violation_detector.violations.values())
        if selected < len(violations):
            violation = violations[selected]
            if violation.evidence_frames:
                # Show first evidence frame
                evidence_data = violation.evidence_frames[0]
                # Handle both dict (new format) and numpy array (old format)
                if isinstance(evidence_data, dict):
                    frame = evidence_data.get('frame')
                else:
                    frame = evidence_data
                
                if frame is None:
                    QMessageBox.warning(self, "Lỗi", "Không thể đọc bằng chứng hình ảnh!")
                    return
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                # Show in dialog
                from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout
                dialog = QDialog(self)
                dialog.setWindowTitle(f"Bằng chứng - {violation.violation_id}")
                layout = QVBoxLayout()
                lbl = QLabel()
                lbl.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
                layout.addWidget(lbl)
                dialog.setLayout(layout)
                dialog.exec()
            else:
                QMessageBox.information(self, "Thông báo", "Chưa có bằng chứng hình ảnh cho vi phạm này.")
    
    @Slot()
    def export_pdf(self):
        """Export violations to PDF"""
        violations = list(self.violation_detector.violations.values())
        if not violations:
            QMessageBox.warning(self, "Cảnh báo", "Không có vi phạm nào để xuất!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu báo cáo PDF", "violations_report.pdf", "PDF Files (*.pdf)"
        )
        
        if file_path:
            try:
                from src.report_generator import ViolationReportGenerator
                generator = ViolationReportGenerator(self.config)
                generator.generate_report(violations, file_path)
                QMessageBox.information(self, "Thành công", f"Đã xuất báo cáo: {file_path}")
                logger.info(f"PDF exported: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể xuất PDF: {e}")
                logger.error(f"PDF export failed: {e}")
    
    @Slot()
    def export_json(self):
        """Export violations to JSON"""
        violations = list(self.violation_detector.violations.values())
        if not violations:
            QMessageBox.warning(self, "Cảnh báo", "Không có vi phạm nào để xuất!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu JSON", "violations.json", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                import json
                data = {
                    'export_time': datetime.now().isoformat(),
                    'total_violations': len(violations),
                    'violations': [v.to_dict() for v in violations]
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "Thành công", f"Đã xuất JSON: {file_path}")
                logger.info(f"JSON exported: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể xuất JSON: {e}")
                logger.error(f"JSON export failed: {e}")


def run_gui(config: dict, detector, tracker, violation_detector):
    """Run GUI application"""
    app = QApplication(sys.argv)
    
    # Set style
    app.setStyle('Fusion')
    
    window = MainWindow(config, detector, tracker, violation_detector)
    window.show()
    
    sys.exit(app.exec())
