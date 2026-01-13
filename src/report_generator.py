"""
PDF Report Generator
Generate violation reports in PDF format
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
from pathlib import Path
from typing import List
from loguru import logger

from .violation_logic import Violation


class ViolationReportGenerator:
    """Generate PDF reports for violations"""
    
    def __init__(self, config: dict):
        self.config = config
        self.location_config = config.get('location', {})
        
        # Try to register Vietnamese font (optional)
        try:
            # You may need to download and place a Vietnamese font
            # pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
            pass
        except:
            logger.warning("Vietnamese font not available, using default")
    
    def generate_report(self, violations: List[Violation], output_path: str) -> str:
        """
        Generate PDF report for violations
        
        Args:
            violations: List of Violation objects
            output_path: Output PDF path
            
        Returns:
            Path to generated PDF
        """
        try:
            # Create document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=20*mm,
                leftMargin=20*mm,
                topMargin=20*mm,
                bottomMargin=20*mm
            )
            
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=colors.HexColor('#1a1a1a'),
                spaceAfter=12,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=12,
                textColor=colors.HexColor('#333333'),
                spaceAfter=6,
                fontName='Helvetica-Bold'
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=10,
                leading=14,
                alignment=TA_LEFT
            )
            
            # Header
            story.append(Paragraph("CONG HOA XA HOI CHU NGHIA VIET NAM", title_style))
            story.append(Paragraph("Doc lap - Tu do - Hanh phuc", normal_style))
            story.append(Spacer(1, 10*mm))
            
            # Title
            story.append(Paragraph("BIEN BAN VI PHAM GIAO THONG", title_style))
            story.append(Paragraph("(Vi pham vuot den do)", heading_style))
            story.append(Spacer(1, 5*mm))
            
            # Summary info
            summary_data = [
                ['Dia diem:', self.location_config.get('intersection', 'N/A')],
                ['Thanh pho:', self.location_config.get('city', 'N/A')],
                ['Ma camera:', self.location_config.get('camera_id', 'N/A')],
                ['Thoi gian lap:', datetime.now().strftime('%d/%m/%Y %H:%M:%S')],
                ['Tong so vi pham:', str(len(violations))]
            ]
            
            summary_table = Table(summary_data, colWidths=[50*mm, 100*mm])
            summary_table.setStyle(TableStyle([
                ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
                ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 10*mm))
            
            # Violations
            story.append(Paragraph("DANH SACH VI PHAM", heading_style))
            story.append(Spacer(1, 3*mm))
            
            for i, violation in enumerate(violations, 1):
                # Violation header
                story.append(Paragraph(f"<b>Vi pham {i}:</b> {violation.violation_id}", heading_style))
                
                # Violation details
                details_data = [
                    ['Thoi gian:', violation.timestamp.strftime('%d/%m/%Y %H:%M:%S')],
                    ['Frame so:', str(violation.frame_number)],
                    ['Loai phuong tien:', violation.vehicle_class],
                    ['Trang thai den:', violation.light_state],
                    ['Do tin cay:', f"{violation.confidence:.2%}"],
                    ['Muc phat:', self.location_config.get('fine_amount', 'Theo quy dinh')]
                ]
                
                details_table = Table(details_data, colWidths=[45*mm, 105*mm])
                details_table.setStyle(TableStyle([
                    ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
                    ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 9),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                
                story.append(details_table)
                story.append(Spacer(1, 3*mm))
                
                # Evidence images
                if violation.evidence_paths:
                    story.append(Paragraph("<b>Bang chung:</b>", normal_style))
                    story.append(Spacer(1, 2*mm))
                    
                    # Add first evidence image
                    img_path = violation.evidence_paths[0]
                    if Path(img_path).exists():
                        try:
                            img = Image(img_path, width=150*mm, height=85*mm)
                            story.append(img)
                        except Exception as e:
                            logger.error(f"Failed to add image: {e}")
                
                story.append(Spacer(1, 5*mm))
                
                # Separator
                if i < len(violations):
                    story.append(Spacer(1, 3*mm))
                    # story.append(PageBreak())  # Uncomment for one violation per page
            
            # Footer
            story.append(Spacer(1, 10*mm))
            story.append(Paragraph("Nguoi lap bien ban", heading_style))
            story.append(Spacer(1, 15*mm))
            story.append(Paragraph("(Ky ten)", normal_style))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            raise
    
    def generate_single_violation_report(self, violation: Violation, output_path: str) -> str:
        """Generate report for a single violation"""
        return self.generate_report([violation], output_path)
