# main.py — безопасный запуск вычислительного потока
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout
from PyQt6.QtCore import QTimer
from ui import ControlPanel, TalbotCanvas
from worker import ComputeThread

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Talbot Carpet Viewer")
        self.panel = ControlPanel()
        self.canvas = TalbotCanvas(self.panel)

        lay = QHBoxLayout(self)
        lay.addWidget(self.panel)
        lay.addWidget(self.canvas, 1)

        self.thread: ComputeThread | None = None

        # дебаунс 200 мс — не ставим задачи при каждом «микро-движении» слайдера
        self.debounce = QTimer(interval=200, singleShot=True)
        self.debounce.timeout.connect(self._start_compute)
        self.panel.changed.connect(self.debounce.start)

        self._start_compute()  # первый расчёт

    # ---------- запуск / перезапуск потока ---------------------------
    def _start_compute(self):
        # если поток ещё считает — дождёмся окончания
        if self.thread is not None and self.thread.isRunning():
            return

        params = self.panel.params()
        self.canvas.set_busy(True)

        self.thread = ComputeThread(params)
        self.thread.finished.connect(self.canvas.update_image)
        self.thread.finished.connect(lambda _arr: self.canvas.set_busy(False))
        self.thread.finished.connect(self._cleanup_thread)
        self.thread.start()

    def _cleanup_thread(self):
        # поток завершил работу → можно создать новый
        self.thread = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())
