import os
import time

import tensorflow as tf

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject, QVariant

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QWidget
from dafne_dl import DynamicDLModel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from tensorflow.keras.callbacks import Callback

from .ModelTrainer_Ui import Ui_ModelTrainerUI
from ..bin.create_model import load_data, get_model_info, create_model_source, get_model_functions, train_model, \
    prepare_data
from ..utils.ThreadHelpers import separate_thread_decorator

PATIENCE = 5
VALIDATION_SPLIT = 0.2


class PredictionUICallback(Callback, QObject):
    fit_signal = pyqtSignal(float, float, np.ndarray)

    def __init__(self, test_image=None):
        Callback.__init__(self)
        QObject.__init__(self)
        self.min_val_loss = np.inf
        self.n_val_loss_increases = 0
        self.test_image = test_image
        self.do_stop = False
        self.best_weights = None

    @pyqtSlot(np.ndarray)
    def set_test_image(self, image):
        print('Setting test image...')
        self.test_image = image

    @pyqtSlot()
    def stop(self):
        self.do_stop = True
        print('Stopping training...')

    def on_epoch_end(self, epoch, logs=None):
        if self.do_stop:
            self.model.stop_training = True
            return

        if 'val_loss' in logs:
            val_loss = logs['val_loss']
        else:
            val_loss = np.inf

        loss = logs['loss']

        if epoch >= 20 and val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.n_val_loss_increases = 0
            self.best_weights = self.model.get_weights()
        elif val_loss > self.min_val_loss:
            self.n_val_loss_increases += 1

        if self.n_val_loss_increases >= PATIENCE:
            self.model.stop_training = True

        if self.test_image is None:
            self.fit_signal.emit(loss, val_loss, np.zeros((10,10)))
            return

        segmentation = self.model.predict(np.expand_dims(self.test_image, 0))
        label = np.argmax(np.squeeze(segmentation[0, :, :, :-1]), axis=2)
        self.fit_signal.emit(loss, val_loss, label)


class ModelTrainer(QWidget, Ui_ModelTrainerUI):

    set_progress_signal = pyqtSignal(int, str)
    start_fitting_signal = pyqtSignal()
    end_fitting_signal = pyqtSignal()

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setupUi(self)

        self.setWindowTitle('Dafne Model Trainer')

        self.fit_output_box.hide()
        self.adjustSize()
        self.pyplot_layout = QtWidgets.QVBoxLayout(self.fit_output_box)
        self.pyplot_layout.setObjectName("pyplot_layout")
        self.data_dir = None
        self.fig = plt.figure()
        self.fig.tight_layout()
        self.fig.set_tight_layout(True)
        self.canvas = FigureCanvas(self.fig)
        self.ax_left = self.fig.add_subplot(121)
        self.ax_right = self.fig.add_subplot(122)
        self.ax_right.set_title('Current output')
        self.ax_right.axis('off')

        self.pyplot_layout.addWidget(self.canvas)
        self.slice_select_slider = QtWidgets.QSlider(self.fit_output_box)
        self.slice_select_slider.setOrientation(QtCore.Qt.Horizontal)
        self.pyplot_layout.addWidget(self.slice_select_slider)
        self.slice_select_slider.valueChanged.connect(self.val_slice_changed)
        self.slice_select_slider.setRange(0, 0)

        self.choose_Button.clicked.connect(self.choose_data)
        self.save_choose_Button.clicked.connect(self.choose_save_location)
        self.set_progress_signal.connect(self.set_progress)
        self.start_fitting_signal.connect(self.start_fitting_slot)
        self.end_fitting_signal.connect(self.stop_fitting_slot)
        self.fit_Button.clicked.connect(self.fit_clicked)
        self.is_fitting = False
        self.fitting_ui_callback = None
        self.loss_list = []
        self.val_loss_list = []
        self.current_val_slice = 0
        self.val_image_list = []

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.is_fitting:
            event.ignore()

    def decide_enable_fit(self):
        if self.location_Text.text() and self.model_location_Text.text():
            self.fit_Button.setEnabled(True)
        else:
            self.fit_Button.setEnabled(False)
            self.set_progress(0, '')

    @pyqtSlot()
    def choose_data(self):
        def invalid():
            self.save_choose_Button.setEnabled(False)
            self.location_Text.setText("")
            self.decide_enable_fit()
            self.data_dir = None

        npz_file, _ = QFileDialog.getOpenFileName(self, "Choose data files", "", "Numpy files (*.npz);;All Files(*)")

        if not npz_file:
            invalid()
            return

        self.data_dir = os.path.dirname(npz_file)

        npz_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        if len(npz_files) > 0:
            self.save_choose_Button.setEnabled(True)
            self.location_Text.setText(self.data_dir)
            self.decide_enable_fit()
        else:
            invalid()
            # show a warning dialog
            QMessageBox.warning(self, "Warning", "No npz files found in the selected directory")

    @pyqtSlot()
    def choose_save_location(self):
        def invalid():
            self.save_choose_Button.setEnabled(False)
            self.model_location_Text.setText("")
            self.decide_enable_fit()

        fileName, _ = QFileDialog.getSaveFileName(self, "Save Model As", "",
                                                  "Model Files (*.model);;All Files (*)")
        if fileName:
            self.model_dir = os.path.dirname(fileName)
            self.model_name = os.path.basename(fileName)
            if self.model_name.endswith('.model'):
                self.model_name = self.model_name[:-6]
            self.model_location_Text.setText(fileName)
            self.decide_enable_fit()
        else:
            invalid()

    @pyqtSlot()
    @separate_thread_decorator
    def fit(self):
        self.is_fitting = True
        self.start_fitting_signal.emit()
        self.set_progress_signal.emit(0, 'Loading data')
        data_list = load_data(self.data_dir)

        self.set_progress_signal.emit(10, 'Getting model info')
        common_resolution, model_size, label_dict = get_model_info(data_list)

        self.set_progress_signal.emit(20, 'Creating model')
        source, model_uuid = create_model_source(self.model_name, common_resolution, model_size, label_dict)

        # write the new model generator script
        with open(os.path.join(self.model_dir, f'generate_{self.model_name}_model.py'), 'w') as f:
            f.write(source)

        create_model_function, apply_model_function, incremental_learn_function = get_model_functions(source)
        model = create_model_function()

        n_datasets = len(data_list)
        n_validation = int(n_datasets * VALIDATION_SPLIT)

        if n_validation == 0:
            print("WARNING: No validation data will be used")

        validation_data_list = data_list[:n_validation]
        training_data_list = data_list[n_validation:]

        self.set_progress_signal.emit(30, 'Preparing data')

        print('preparing data')

        training_generator, steps, x_val_list, y_val_list = prepare_data(training_data_list, validation_data_list,
                                                                         common_resolution, model_size, label_dict)

        self.set_progress_signal.emit(50, 'Training model')

        self.fitting_ui_callback = PredictionUICallback()
        self.val_image_list = x_val_list
        self.fitting_ui_callback.fit_signal.connect(self.update_plot)
        self.set_test_image.connect(self.fitting_ui_callback.set_test_image)
        self.stop_fitting_signal.connect(self.fitting_ui_callback.stop)
        self.slice_select_slider.setMaximum(len(x_val_list) - 1)
        if len(x_val_list) > 0:
            self.set_test_image.emit(x_val_list[0])
            self.slice_select_slider.setEnabled(True)
        else:
            self.slice_select_slider.setEnabled(False)

        trained_model, history = train_model(model, training_generator, steps, x_val_list, y_val_list, [self.fitting_ui_callback])
        if self.fitting_ui_callback.best_weights is not None:
            trained_model.set_weights(self.fitting_ui_callback.best_weights)

        self.fitting_ui_callback.deleteLater()
        self.fitting_ui_callback = None

        self.set_progress_signal.emit(90, 'Saving model')

        model_object = DynamicDLModel(model_uuid,
                                     create_model_function,
                                     apply_model_function,
                                     incremental_learn_function=incremental_learn_function,
                                     weights=trained_model.get_weights(),
                                     timestamp_id=int(time.time())
                                     )

        with open(os.path.join(self.model_dir, f'{self.model_name}.model'), 'wb') as f:
            model_object.dump(f)

        # save weights
        os.makedirs(os.path.join(self.model_dir, 'weights'), exist_ok=True)
        trained_model.save_weights(os.path.join(self.model_dir, 'weights', f'weights_{self.model_name}.hdf5'))
        self.set_progress_signal.emit(100, 'Done')
        self.end_fitting_signal.emit()
        self.is_fitting = False


    @pyqtSlot()
    def val_slice_changed(self):
        try:
            self.fitting_ui_callback.set_test_image(self.val_image_list[self.slice_select_slider.value()])
        except IndexError:
            print("Validation slice out of range")

    @pyqtSlot(float, float, np.ndarray)
    def update_plot(self, loss, val_loss, label):
        self.loss_list.append(loss)
        self.val_loss_list.append(val_loss)
        self.ax_left.clear()
        self.ax_left.plot(self.loss_list)
        self.ax_left.plot(self.val_loss_list)
        self.ax_left.legend(['train', 'validation'], loc='upper right')
        self.ax_right.clear()
        self.ax_right.set_title('Current output')
        self.ax_right.imshow(label)
        self.ax_right.axis('off')
        self.canvas.draw()

    @pyqtSlot(int, str)
    def set_progress(self, progress, message=''):
        self.progressBar.setValue(progress)
        self.progress_Label.setText(message)

    @pyqtSlot()
    def start_fitting_slot(self):
        self.loss_list = []
        self.val_loss_list = []
        self.fit_Button.setText('Stop fitting')
        self.fit_output_box.show()
        self.adjustSize()
        self.choose_Button.setEnabled(False)
        self.save_choose_Button.setEnabled(False)

    @pyqtSlot()
    def stop_fitting_slot(self):
        self.fit_Button.setText('Fit model')

    @pyqtSlot()
    def fit_clicked(self):
        if self.is_fitting:
            #stop the fitting
            if self.fitting_ui_callback is not None:
                self.fitting_ui_callback.stop()
                self.fit_Button.setText('Stopping...')
        else:
            self.fit()


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('Dafne Model trainer')
    window = ModelTrainer()
    window.show()
    sys.exit(app.exec_())
