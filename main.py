import sys
import cv2
import numpy as np
import os
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QSize

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QFrame, QTabWidget, QSpacerItem, QSizePolicy,
    QVBoxLayout, QWidget, QMessageBox, QComboBox, QSpinBox, QHBoxLayout, QDoubleSpinBox
)


from src.thresholding.optimal_thresholding import optimal_threshold
from src.thresholding.otsu_thresholding import otsu_threshold
from src.thresholding.spectral_thresholding import spectral_threshold

from src.segmentation.mean_shift_segmentation import mean_shift_segmentation_with_extra
from src.segmentation.mean_shift_segmentation import mean_shift_segmentation_without_boundries
from src.segmentation.mean_shift_segmentation import create_feature_space


class thresholdTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)

        
        self.function_selector = QComboBox()
        self.function_selector.addItems(["Optimal Threshold", "Otsu Threshold", "Spectral Threshold"])
        self.function_selector.currentTextChanged.connect(self.update_parameter_fields)
        self.layout.addWidget(self.function_selector)

        
        self.parameter_container = QFrame()
        self.parameter_layout = QVBoxLayout(self.parameter_container)
        self.layout.addWidget(self.parameter_container)

        
        self.apply_button = QPushButton("Apply Threshold")
        self.apply_button.clicked.connect(self.apply_threshold_function)
        self.layout.addWidget(self.apply_button)

        
        self.update_parameter_fields(self.function_selector.currentText())

    def update_parameter_fields(self, selected_function):
        
        for i in reversed(range(self.parameter_layout.count())):
            widget = self.parameter_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        
        if selected_function in ["Optimal Threshold", "Otsu Threshold", "Spectral Threshold"]:
            
            self.add_combobox("Method Type", ["global", "local"])
            
            method_type_combobox = self.parameter_container.findChild(QComboBox)
            if method_type_combobox and method_type_combobox.currentText() == "local":
                self.add_spinbox("Block Size", 1, 100, 1, default_value=11)
            
            if selected_function == "Spectral Threshold":
                self.add_spinbox("Number of Levels", 2, 20, 1, default_value=3)
                self.add_spinbox("Offset Range", 1, 50, 1, default_value=10)

    def add_spinbox(self, label_text, min_value, max_value, step, default_value=0):
        label = QLabel(label_text)
        spinbox = QSpinBox()
        spinbox.setRange(min_value, max_value)
        spinbox.setSingleStep(step)
        spinbox.setValue(default_value)
        self.parameter_layout.addWidget(label)
        self.parameter_layout.addWidget(spinbox)

    def add_combobox(self, label_text, options):
        label = QLabel(label_text)
        combobox = QComboBox()
        combobox.addItems(options)
        
        if label_text == "Method Type":
            combobox.currentTextChanged.connect(self.update_block_size_visibility)
        self.parameter_layout.addWidget(label)
        self.parameter_layout.addWidget(combobox)

    def update_block_size_visibility(self, method_type):
        
        block_size_label = None
        block_size_spinbox = None
        for i in range(self.parameter_layout.count()):
            widget = self.parameter_layout.itemAt(i).widget()
            if isinstance(widget, QLabel) and widget.text() == "Block Size":
                block_size_label = widget
            elif isinstance(widget, QSpinBox) and i > 0 and self.parameter_layout.itemAt(i-1).widget().text() == "Block Size":
                block_size_spinbox = widget

        if block_size_label:
            block_size_label.deleteLater()
        if block_size_spinbox:
            block_size_spinbox.deleteLater()

        
        if method_type == "local":
            
            self.add_spinbox("Block Size", 1, 100, 1, default_value=11)
            
            for i in range(self.parameter_layout.count()):
                widget = self.parameter_layout.itemAt(i).widget()
                if isinstance(widget, QLabel) and widget.text() == "Method Type":
                    block_size_label = self.parameter_layout.itemAt(self.parameter_layout.count()-2).widget()
                    block_size_spinbox = self.parameter_layout.itemAt(self.parameter_layout.count()-1).widget()
                    self.parameter_layout.insertWidget(i+2, block_size_label)
                    self.parameter_layout.insertWidget(i+3, block_size_spinbox)
                    break

    def apply_threshold_function(self):
        if self.parent.image is None:
            QMessageBox.warning(self, "Error", "No image loaded. Please load an image first.")
            return

        selected_function = self.function_selector.currentText()
        params = {}

        
        for i in range(0, self.parameter_layout.count(), 2):  
            label_widget = self.parameter_layout.itemAt(i).widget()
            value_widget = self.parameter_layout.itemAt(i + 1).widget()
            if isinstance(label_widget, QLabel) and value_widget:
                param_name = label_widget.text()
                if isinstance(value_widget, QSpinBox):
                    params[param_name] = value_widget.value()
                elif isinstance(value_widget, QComboBox):
                    params[param_name] = value_widget.currentText()

        
        mapped_params = {}
        if "Method Type" in params:
            mapped_params["method_type"] = params.pop("Method Type")
        if "Block Size" in params:
            mapped_params["block_size"] = params.pop("Block Size")
        if "Number of Levels" in params:
            mapped_params["num_levels"] = params.pop("Number of Levels")
        if "Offset Range" in params:
            mapped_params["offset_range"] = params.pop("Offset Range")

        
        image = self.parent.image
        if len(image.shape) == 3:  
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        try:
            
            if selected_function == "Optimal Threshold":
                result = optimal_threshold(image, **mapped_params)
            elif selected_function == "Otsu Threshold":
                result = otsu_threshold(image, **mapped_params)
            elif selected_function == "Spectral Threshold":
                result = spectral_threshold(image, **mapped_params)
            else:
                QMessageBox.warning(self, "Error", "Invalid function selected.")
                return

            
            self.parent.modified_image = result
            self.parent.display_image(result)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to apply threshold: {str(e)}")


class clusterTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)

        self.function_selector = QComboBox()
        self.function_selector.addItems(["Shift Segmentation Without Boundaries", "Shift Segmentation With Extra"])
        self.function_selector.currentTextChanged.connect(self.update_parameter_fields)
        self.layout.addWidget(self.function_selector)

        self.parameter_container = QFrame()
        self.parameter_layout = QVBoxLayout(self.parameter_container)
        self.layout.addWidget(self.parameter_container)

        self.apply_button = QPushButton("Apply Clustering")
        self.apply_button.clicked.connect(self.apply_clustering_function)
        self.layout.addWidget(self.apply_button)

        self.update_parameter_fields(self.function_selector.currentText())

    def update_parameter_fields(self, selected_function):
        for i in reversed(range(self.parameter_layout.count())):
            widget = self.parameter_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        if selected_function == "Shift Segmentation Without Boundaries":
            self.add_spinbox("Threshold", 1, 300, 5, default_value=90)
            self.add_spinbox("Convergence Threshold", 0.1, 10.0, 0.1, default_value=2.0)
            self.add_spinbox("Max Iterations", 1, 5000, 1, default_value=1500)
        elif selected_function == "Shift Segmentation With Extra":
            self.add_spinbox("Threshold", 1, 300, 5, default_value=150)
            self.add_spinbox("Convergence Threshold", 0.1, 10.0, 0.1, default_value=1.0)
            self.add_spinbox("Max Iterations", 1, 5000, 1, default_value=1000)

    def add_spinbox(self, label_text, min_value, max_value, step, default_value=0):
        label = QLabel(label_text)
        spinbox = QDoubleSpinBox() if isinstance(step, float) else QSpinBox()
        spinbox.setRange(min_value, max_value)
        spinbox.setSingleStep(step)
        spinbox.setValue(default_value)
        self.parameter_layout.addWidget(label)
        self.parameter_layout.addWidget(spinbox)

    def apply_clustering_function(self):
        if self.parent.image is None:
            QMessageBox.warning(self, "Error", "No image loaded. Please load an image first.")
            return

        selected_function = self.function_selector.currentText()
        params = {}

        # Collect parameter values from input fields
        for i in range(0, self.parameter_layout.count(), 2):
            label_widget = self.parameter_layout.itemAt(i).widget()
            value_widget = self.parameter_layout.itemAt(i + 1).widget()
            if isinstance(label_widget, QLabel) and value_widget:
                param_name = label_widget.text().replace(" ", "_").lower()
                params[param_name] = value_widget.value()

        # Ensure the image is 3D (convert grayscale to RGB if necessary)
        image = self.parent.image
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply the selected clustering function
        try:
            if selected_function == "Shift Segmentation Without Boundaries":
                result = mean_shift_segmentation_without_boundries(image, **params)
            elif selected_function == "Shift Segmentation With Extra":
                feature_space, row, col = create_feature_space(image)
                result = mean_shift_segmentation_with_extra(feature_space, row, col, **params)
            else:
                QMessageBox.warning(self, "Error", "Invalid function selected.")
                return

            # Update the modified image in the MainWindow
            self.parent.modified_image = result
            self.parent.display_image(result)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply clustering: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Image Processing App")
        self.setGeometry(50, 50, 1200, 800)

        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout()
        central_widget.setLayout(self.main_layout)
        
        self.init_ui(self.main_layout)

        
        self.params = {
            "noise_filter": {},
            "filtering": {},
            "edge_detection": {},
            "thresholding": {},
            "frequency_filter": {},
            "hybrid_image": {},
            "shape_detection": {},
            "active_contour": {}
        }
        
        
        self.image = None
        self.original_image = None
        self.modified_image = None
        self.processors = {}  

    def init_ui(self, main_layout):
        
        left_frame = QFrame()
        left_frame.setFixedWidth(500)
        left_frame.setObjectName("left_frame")
        left_layout = QVBoxLayout(left_frame)
        
        tab_widget = QTabWidget()
        tab_widget.setObjectName("tab_widget")

        self.threshold_tab = thresholdTab(self)

        self.cluster_tab = clusterTab(self)

        tab_widget.addTab(self.threshold_tab, "Thresholding")
        tab_widget.addTab(self.cluster_tab, "Clustering")

        left_layout.addWidget(tab_widget)
        main_layout.addWidget(left_frame)
        
        
        self.right_frame = QFrame()
        self.right_frame.setObjectName("right_frame")
        self.right_layout = QVBoxLayout(self.right_frame)
        self.right_layout.setAlignment(Qt.AlignTop)  

        
        control_frame = QFrame()
        control_frame.setMaximumHeight(100)
        control_layout = QHBoxLayout(control_frame)

        control_layout.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.btn_confirm = QPushButton()
        self.btn_confirm.setIcon(QIcon(os.path.join(os.path.dirname(__file__), './resources/confirm.png')))
        self.btn_confirm.setIconSize(QSize(28, 28))
        self.btn_confirm.clicked.connect(self.confirm_edit)
        control_layout.addWidget(self.btn_confirm)

        self.btn_discard = QPushButton()
        self.btn_discard.setIcon(QIcon(os.path.join(os.path.dirname(__file__), './resources/discard.png')))
        self.btn_discard.setIconSize(QSize(28, 28))
        self.btn_discard.clicked.connect(self.discard_edit)
        control_layout.addWidget(self.btn_discard)

        self.btn_reset = QPushButton()
        self.btn_reset.setIcon(QIcon(os.path.join(os.path.dirname(__file__), './resources/reset.png')))
        self.btn_reset.setIconSize(QSize(28, 28))
        self.btn_reset.clicked.connect(self.reset_image)
        control_layout.addWidget(self.btn_reset)

        self.right_layout.addWidget(control_frame)

        
        self.image_display_frame = QFrame()  
        self.image_display_frame.setFixedSize(1390, 880)
        self.image_display_layout = QVBoxLayout(self.image_display_frame)

        self.lbl_image = QLabel("No Image Loaded")
        self.lbl_image.setObjectName("lbl_image")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.image_display_layout.addWidget(self.lbl_image)
        self.lbl_image.mouseDoubleClickEvent = self.on_image_label_double_click

        self.right_layout.addWidget(self.image_display_frame)
        self.right_layout.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        main_layout.addWidget(self.right_frame)

    def display_image(self, img):
        """
        Convert a NumPy image (RGB or grayscale) to QImage and display it in lbl_image.
        """
        if img is None:
            self.lbl_image.setText("No Image Loaded")
            return

        if len(img.shape) == 3:  
            img_rgb = img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:  
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qimg)
        self.lbl_image.setPixmap(pixmap.scaled(
            self.lbl_image.width(), self.lbl_image.height(), Qt.KeepAspectRatio
        ))

    def load_image(self):
        """
        Open a file dialog to load an image.
        """
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp *.jpeg)")
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is None:
                QMessageBox.critical(self, "Error", "Failed to load image.")
                return
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.original_image = self.image.copy()
            self.display_image(self.image)
        else:
            QMessageBox.information(self, "Info", "No file selected.")

    def on_image_label_double_click(self, event):
        self.load_image()

    def confirm_edit(self):
        """
        Confirm the edit.
        """
        if self.modified_image is not None:
            self.image = self.modified_image.copy()
            for processor in self.processors.values():
                processor.set_image(self.image)
            self.modified_image = None
            self.display_image(self.image)
        else:
            QMessageBox.warning(self, "Warning", "No modified image to confirm.")

    def discard_edit(self):
        """
        Discard the edit.
        """
        if self.modified_image is not None:
            self.modified_image = None
            self.display_image(self.image)
        else:
            QMessageBox.warning(self, "Warning", "No modified image to discard.")

    def reset_image(self):
        """
        Reset the image to the original.
        """
        if self.original_image is not None:
            self.image = self.original_image.copy()
            for processor in self.processors.values():
                processor.set_image(self.image)
            self.modified_image = None
            self.display_image(self.image)
        else:
            QMessageBox.warning(self, "Warning", "No original image available.")

def main():
    app = QApplication(sys.argv)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    qss_path = os.path.join(script_dir, "resources\\styles.qss")
    
    with open(qss_path, "r") as file:
        app.setStyleSheet(file.read())
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


import cv2
import numpy as np

if __name__ == "__main__":
    main()
