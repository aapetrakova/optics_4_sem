
from PyQt6.QtCore import QThread, pyqtSignal
from compute import talbot_carpet

class ComputeThread(QThread):
    finished = pyqtSignal(object)
    def __init__(self, params):
        super().__init__()
        self.params = params
    def run(self):
        arr = talbot_carpet(**self.params)
        self.finished.emit(arr)
