# ui.py — полный интерфейс (PyQt6 + Matplotlib QtAgg)

import pathlib
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QMovie, QFont
from PyQt6.QtWidgets import (
    QWidget, QSlider, QFormLayout, QCheckBox, QLabel, QVBoxLayout,
    QHBoxLayout, QPushButton, QDoubleSpinBox, QDialog, QDialogButtonBox,
    QSpacerItem, QSizePolicy, QProgressBar
)

# Константы для масштабирования шрифтов
BASE_MULT = 20        # базовый размер шрифта (pt)
GRAPH_FONT_SCALE = 1.25  # множитель для шрифтов графика
SP = QSizePolicy.Policy
SPINNER_PX = 64  # размер spinner.gif


# ────────── helper slider ─────────────────────────────────────────────
class LabeledSlider(QWidget):
    """
    Горизонтальный QSlider с живой подписью текущего значения.
    Позволяет менять диапазон и шаг через set_range().
    """
    valueChanged = pyqtSignal(int)

    def __init__(self, mn: float, mx: float, val: float, step: float, suffix: str = ""):
        super().__init__()
        self._step = step
        self._suffix = suffix

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(mn / step), int(mx / step))
        self.slider.setValue(int(val / step))
        self.slider.setTickInterval(
            max(1, (self.slider.maximum() - self.slider.minimum()) // 10)
        )
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.label = QLabel()
        self.slider.valueChanged.connect(self._update_label)
        self.slider.valueChanged.connect(self.valueChanged.emit)
        self._update_label(self.slider.value())

        # применяем базовый шрифт к подписи
        font = QFont()
        font.setPointSize(BASE_MULT)
        self.label.setFont(font)

        lay = QHBoxLayout(self)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.label)

    def _update_label(self, raw: int):
        self.label.setText(f"{raw * self._step:g}{self._suffix}")

    def current(self) -> float:
        return self.slider.value() * self._step

    def set_range(self, mn: float, mx: float, step: float):
        self._step = step
        self.slider.setRange(int(mn / step), int(mx / step))
        v = self.slider.value()
        v = max(self.slider.minimum(), min(v, self.slider.maximum()))
        self.slider.setValue(v)
        self._update_label(v)


# ────────── RangeDialog ─────────────────────────────────────────────────
class RangeDialog(QDialog):
    """
    Диалог для редактирования min/max/step каждого слайдера.
    Размер окна установлен так, чтобы все поля были видны.
    """
    def __init__(self, sliders: dict[str, LabeledSlider], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройка диапазонов и шага")
        # задаём размер, достаточный для отображения всех полей
        self.resize(600, 400)

        font = QFont()
        font.setPointSize(BASE_MULT)
        self.setFont(font)

        self.widgets: dict[str, tuple[QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox]] = {}
        form = QFormLayout(self)

        for name, sl in sliders.items():
            mn_box = QDoubleSpinBox(decimals=4)
            mx_box = QDoubleSpinBox(decimals=4)
            st_box = QDoubleSpinBox(decimals=4)

            mn_box.setRange(-1e6, 1e6)
            mx_box.setRange(-1e6, 1e6)
            st_box.setRange(1e-6, 1e6)

            mn_box.setValue(sl.slider.minimum() * sl._step)
            mx_box.setValue(sl.slider.maximum() * sl._step)
            st_box.setValue(sl._step)

            # применяем базовый шрифт к spinbox-ам и меткам
            for w in (mn_box, mx_box, st_box):
                w.setFont(font)

            row = QHBoxLayout()
            for lbl, w in (("min", mn_box), ("max", mx_box), ("step", st_box)):
                label = QLabel(lbl)
                label.setFont(font)
                row.addWidget(label)
                row.addWidget(w)
            wrap = QWidget()
            wrap.setLayout(row)
            form.addRow(name, wrap)

            self.widgets[name] = (mn_box, mx_box, st_box)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

        self.setLayout(form)

    def values(self) -> dict[str, tuple[float, float, float]]:
        return {
            name: (mn.value(), mx.value(), st.value())
            for name, (mn, mx, st) in self.widgets.items()
        }


# ────────── ControlPanel ───────────────────────────────────────────────
class ControlPanel(QWidget):
    """
    Панель с ползунками и кнопками.
    """
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setSizePolicy(SP.Preferred, SP.Expanding)

        font = QFont()
        font.setPointSize(BASE_MULT)
        self.setFont(font)

        self.a     = LabeledSlider(0.2, 3.0, 1.0, 0.05, " a.u.")
        self.lam   = LabeledSlider(0.1, 2.0, 1.0, 0.05, " λ")
        self.duty  = LabeledSlider(0.05, 0.9, 0.2, 0.05)
        self.nslit = LabeledSlider(5, 60, 20, 1)
        self.res   = LabeledSlider(60, 500, 150, 10, " px/u")
        self.zmax  = LabeledSlider(0.5, 6.0, 3.0, 0.5, " z/zT")

        for sl in (self.a, self.lam, self.duty, self.nslit, self.res, self.zmax):
            sl.valueChanged.connect(self.changed.emit)

        self.gpu = QCheckBox("GPU (torch)")
        self.gpu.stateChanged.connect(self.changed.emit)

        btn_plus, btn_minus = QPushButton("A+"), QPushButton("A-")
        btn_plus.clicked.connect(lambda: self._bump_font(+1))
        btn_minus.clicked.connect(lambda: self._bump_font(-1))

        btn_wider, btn_narrow = QPushButton("⟷"), QPushButton("⟶⟵")
        btn_wider.clicked.connect(lambda: self._adjust_width(+50))
        btn_narrow.clicked.connect(lambda: self._adjust_width(-50))

        btn_gear = QPushButton("⚙️")
        btn_gear.clicked.connect(self._open_ranges)

        for b in (btn_plus, btn_minus, btn_wider, btn_narrow, btn_gear):
            b.setFont(font)

        btn_row = QHBoxLayout()
        for w in (btn_minus, btn_plus, btn_narrow, btn_wider, btn_gear):
            btn_row.addWidget(w)
        btn_row.addItem(QSpacerItem(10, 10, SP.Expanding, SP.Minimum))
        btn_wrap = QWidget()
        btn_wrap.setLayout(btn_row)

        form = QFormLayout(self)
        form.addRow(btn_wrap)
        form.addRow("Период a", self.a)
        form.addRow("Длина λ", self.lam)
        form.addRow("Duty", self.duty)
        form.addRow("Щелей", self.nslit)
        form.addRow("Разр-е", self.res)
        form.addRow("Z / zT", self.zmax)
        form.addRow(self.gpu)
        self.setLayout(form)

    def _bump_font(self, delta: int):
        f = self.font()
        new_size = max(6, f.pointSize() + delta)
        f.setPointSize(new_size)
        self.setFont(f)

    def _adjust_width(self, delta: int):
        self.setFixedWidth(max(250, self.width() + delta))

    def _open_ranges(self):
        dlg = RangeDialog({
            "Период a": self.a,
            "Длина λ": self.lam,
            "Duty": self.duty,
            "Щелей": self.nslit,
            "Разр-е": self.res,
            "Z / zT": self.zmax,
        }, self)
        if dlg.exec():
            for name, (mn, mx, st) in dlg.values().items():
                attr = {
                    "Период a": "a",
                    "Длина λ": "lam",
                    "Duty": "duty",
                    "Щелей": "nslit",
                    "Разр-е": "res",
                    "Z / zT": "zmax",
                }[name]
                getattr(self, attr).set_range(mn, mx, st)
            self.changed.emit()

    def params(self) -> dict:
        return dict(
            a=self.a.current(),
            wavelength=self.lam.current(),
            duty=self.duty.current(),
            nslits=int(self.nslit.current()),
            x_min=-3.0,
            x_max=3.0,
            z_max=self.zmax.current(),
            res=int(self.res.current()),
            use_gpu=self.gpu.isChecked(),
        )


# ────────── TalbotCanvas ───────────────────────────────────────────────
class TalbotCanvas(QWidget):
    """
    Canvas с постоянной цветовой шкалой 0–1 и крупными подписями.
    Цветовая шкала прижата к правому краю, оси занимают оставшуюся область.
    """

    def __init__(self, ctrl: ControlPanel):
        super().__init__()
        self.ctrl = ctrl
        self.fig = plt.Figure()
        # Разбиваем фигуру: оставляем место справа для colorbar
        self.ax = self.fig.add_axes([0.10, 0.1, 0.78, 0.85])  # [left, bottom, width, height]
        size = BASE_MULT * GRAPH_FONT_SCALE
        self.ax.set_xlabel("$x/a$", fontsize=size)
        self.ax.set_ylabel("$z/z_T$", fontsize=size)
        self.ax.tick_params(axis="both", labelsize=size)

        # Ось для colorbar: занимает узкий слот справа внизу от 0.95 до 0.98
        self.cax = self.fig.add_axes([0.89, 0.1, 0.03, 0.85])

        self.canvas = Canvas(self.fig)
        self.im = None
        self.cbar = None

        gif = pathlib.Path(__file__).with_name("spinner.gif")
        if gif.exists():
            self.overlay = QLabel(self.canvas)
            mv = QMovie(str(gif))
            mv.setScaledSize(QSize(SPINNER_PX, SPINNER_PX))
            self.overlay.setMovie(mv)
            self._movie = mv
        else:
            self.overlay = QProgressBar(self.canvas, maximum=0, textVisible=False)
            self.overlay.setFixedHeight(4)
            self._movie = None
        self.overlay.hide()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.overlay.resize(self.canvas.size())

    def set_busy(self, flag: bool):
        self.overlay.setVisible(flag)
        if self._movie:
            if flag:
                self._movie.start()
            else:
                self._movie.stop()

    def update_image(self, arr):
        zmax = self.ctrl.zmax.current()
        if self.im is None:
            size = BASE_MULT * GRAPH_FONT_SCALE
            # отрисовываем данные на ax
            self.im = self.ax.imshow(
                arr,
                cmap="viridis",
                aspect="auto",
                vmin=0,
                vmax=1,
                extent=[-3, 3, zmax, 0],
            )
            # создаём colorbar на заранее определённой оси cax
            self.cbar = self.fig.colorbar(self.im, cax=self.cax, orientation="vertical")
            self.cbar.set_label("Интенсивность (0–1)", fontsize=size)
            self.cbar.ax.tick_params(labelsize=size)
        else:
            self.im.set_data(arr)
            self.im.set_extent([-3, 3, zmax, 0])
        self.canvas.draw_idle()
