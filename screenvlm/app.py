import sys
import threading
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                               QCheckBox, QLabel, QScrollArea, QFrame)
from PySide6.QtCore import Qt, Signal, QObject, Slot
from PySide6.QtGui import QFont, QKeySequence, QShortcut

from .config import settings
from .capture import capture_fullscreen
from .vlm.worker import VLMWorker

class WorkerSignals(QObject):
    response_ready = Signal(object)

class ScreenVLMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScreenVLM")
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.resize(400, 600)
        
        # Initialize components
        self.worker = VLMWorker()
        self.worker.start()
        
        # Setup UI
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Output Area
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.layout.addWidget(self.output_area)
        
        # Status Label
        self.status_label = QLabel("Ready")
        self.layout.addWidget(self.status_label)
        
        # Input Layout
        self.input_layout = QHBoxLayout()
        self.layout.addLayout(self.input_layout)
        
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Ask a question about your screen...")
        self.input_box.returnPressed.connect(self.handle_ask)
        self.input_layout.addWidget(self.input_box)
        
        self.ask_btn = QPushButton("Ask")
        self.ask_btn.clicked.connect(self.handle_ask)
        self.input_layout.addWidget(self.ask_btn)
        
        # Options
        self.options_layout = QHBoxLayout()
        self.layout.addLayout(self.options_layout)
        
        self.rag_checkbox = QCheckBox("Use RAG")
        self.options_layout.addWidget(self.rag_checkbox)
        
        self.ingest_btn = QPushButton("Ingest Docs")
        self.ingest_btn.clicked.connect(self.handle_ingest)
        self.options_layout.addWidget(self.ingest_btn)

        # Worker signals
        self.signals = WorkerSignals()
        self.signals.response_ready.connect(self.update_ui)
        
        # Timer for polling worker
        self.timer = threading.Timer(0.1, self.poll_worker)
        self.timer.start() # Start polling loop
        
        # Shortcuts
        self.shortcut_hide = QShortcut(QKeySequence("Ctrl+Shift+H"), self)
        self.shortcut_hide.activated.connect(self.toggle_visibility)
        
        self.shortcut_hide.activated.connect(self.toggle_visibility)


    def handle_ask(self):
        question = self.input_box.text().strip()
        if not question:
            return
            
        self.input_box.clear()
        self.output_area.append(f"User: {question}")
        self.status_label.setText("Capturing & Thinking...")
        
        # Capture screen
        try:
            # Hide window to capture clean screenshot
            self.hide()
            QApplication.processEvents()
            time.sleep(0.2)  # Give OS time to repaint
            
            screenshot = capture_fullscreen()
        except Exception as e:
            self.output_area.append(f"System: Capture failed: {e}")
            self.status_label.setText("Error")
            return
        finally:
            self.show()
            self.activateWindow()

        # Submit to worker
        self.status_label.setText("Thinking...")
        rag_enabled = self.rag_checkbox.isChecked()
        self.worker.submit_task(screenshot, question, rag_enabled=rag_enabled)

    def handle_ingest(self):
        self.output_area.append("System: Ingestion triggers via CLI for now. Run `screenvlm ingest`.")

    def poll_worker(self):
        # Check for results
        result = self.worker.get_result(block=False)
        if result:
            self.signals.response_ready.emit(result)
        
        # Reschedule
        self.timer = threading.Timer(0.1, self.poll_worker)
        self.timer.start()

    @Slot(object)
    def update_ui(self, result):
        self.status_label.setText("Ready")
        if "response" in result:
             self.output_area.append(f"Assistant: {result['response']}\n")
             self.output_area.ensureCursorVisible()
        elif "error" in result:
             self.output_area.append(f"System Error: {result['error']}\n")

    def toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.activateWindow()

    def closeEvent(self, event):
        self.timer.cancel()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Splash Screen
    from PySide6.QtWidgets import QSplashScreen
    from PySide6.QtGui import QPixmap, QPainter, QColor
    
    # Create a simple splash image programmatically
    pixmap = QPixmap(400, 300)
    pixmap.fill(QColor("black"))
    painter = QPainter(pixmap)
    painter.setPen(QColor("white"))
    painter.setFont(QFont("Arial", 20))
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "ScreenVLM\nLoading Model...")
    painter.end()
    
    splash = QSplashScreen(pixmap)
    splash.show()
    app.processEvents()
    
    # Init main window
    window = ScreenVLMApp()
    
    # Close splash when window is ready
    window.show()
    splash.finish(window)
    
    sys.exit(app.exec())
