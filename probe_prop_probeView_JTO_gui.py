from PyQt5.QtWidgets import QWidget
from PyQt5 import QtWidgets, uic
import pyqtgraph as pg
import numpy as np

import os

ui_path = os.path.dirname(os.path.abspath(__file__))

class probeViewWindow(QWidget):
    def __init__(self, probe_in):
        super(probeViewWindow, self).__init__()
        uic.loadUi(os.path.join(ui_path,'probe_prop_probeView_JTO_gui.ui'), self)

        self.probe_in = probe_in
        self.probe_in[np.isnan(self.probe_in)] = 0.
        self.Ny, self.Nx, _, _ = probe_in.shape

        self._setup_grid()
        self._setup_shared_lut()
        self._link_views()


    def _setup_grid(self):
        layout = QtWidgets.QGridLayout(self.probeImageGridWidget)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_items = []
        self.views = []

        for iy in range(self.Ny):
            row_views = []
            row_imgs = []

            for ix in range(self.Nx):
                iv = pg.ImageView(view=pg.PlotItem())
                iv.ui.histogram.hide()
                iv.ui.roiBtn.hide()
                iv.ui.menuBtn.hide()

                iv.setImage(np.abs(self.probe_in[iy, ix]), autoLevels=False)
                iv.getView().setTitle("mode %d, OPR %d" % (ix, iy), size="8pt")  # very small


                layout.addWidget(iv, iy, ix)

                row_views.append(iv.getView())
                row_imgs.append(iv.imageItem)

            self.views.append(row_views)
            self.image_items.append(row_imgs)


    def _setup_shared_lut(self):
        # QWidget wrapper for pyqtgraph items
        self.hist_widget = pg.GraphicsLayoutWidget(parent=self.histogramWidget)
        self.hist = pg.HistogramLUTItem()
        self.hist.sigLevelsChanged.connect(self._apply_lut_to_all)
        self.hist.gradient.sigGradientChanged.connect(self._apply_lut_to_all)

        self.hist_widget.addItem(self.hist)

        # Attach histogram to reference image
        self.hist.setImageItem(self.image_items[0][0])

        # Layout for the placeholder widget
        layout = QtWidgets.QVBoxLayout(self.histogramWidget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.hist_widget)

        # Optional cosmetics
        self.hist.gradient.loadPreset("viridis")

        # Optional fixed initial levels
        # vmin, vmax = np.percentile(self.data, (1, 99))
        # self.hist.setLevels(vmin, vmax)
        
        self._apply_lut_to_all()


    def _apply_lut_to_all(self):
        lut = self.hist.gradient.getLookupTable(256)
        levels = self.hist.getLevels()

        for row in self.image_items:
            for img in row:
                img.setLookupTable(lut)
                img.setLevels(levels)


    def _link_views(self):
        ref_view = self.views[0][0]

        for row in self.views:
            for view in row:
                if view is not ref_view:
                    view.setXLink(ref_view)
                    view.setYLink(ref_view)
