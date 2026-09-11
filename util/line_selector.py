from typing import Optional, Union, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams.update({'font.size': 12})

##################
#  Line Selector #
##################


class LineSelector:
    """
    Creates an interactive matplotlib plot allowing users
    to define a line by selecting 2 points
    """

    def __init__(
        self, image: np.ndarray,
        figsize: Optional[Tuple[int]] = (10, 8),
        cmap: Optional[matplotlib.colors.LinearSegmentedColormap] = plt.cm.gist_gray,
        vmin: Optional[Union[float, int]] = None,
        vmax: Optional[Union[float, int]] = None
    ):
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None

        self.pnt1 = None
        self.pnt2 = None

        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, visible=False)
        self.rect = patches.Rectangle(
            (0.0, 0.0), figsize[0], figsize[1],
            fill=False, clip_on=False, visible=False)
        
        self.rect_patch = self.ax.add_patch(self.rect)
        self.cid = self.rect_patch.figure.canvas.mpl_connect('button_press_event',
                                                             self)
        self.cmap = cmap
        self.image = image
        self.plot = self.gray_plot(self.fig, vmin=vmin, vmax=vmax, return_ax=True)
        self.plot.set_title('Select 2 Points of Interest')

    def gray_plot(self,
                  fig: matplotlib.figure.Figure,
                  vmin: Optional[Union[float, int]] = None,
                  vmax: Optional[Union[float, int]] = None,
                  return_ax: Optional[bool] = False):
        """
        Takes: a matplotlib.figure.Figure object and optional vmin and vmax

        Calculates reasonable vmin, vmax if not passed

        Returns: axes if return_ax == True
        """
        if vmin is None:
            vmin = np.nanpercentile(self.image, 1)
        if vmax is None:
            vmax = np.nanpercentile(self.image, 99)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.imshow(self.image, cmap=self.cmap, vmin=vmin, vmax=vmax)
        if return_ax:
            return (ax)

    def __call__(self, event: matplotlib.backend_bases.Event):
        """
        Takes: a click event

        Maintains a stack of 2 points and one line:
        Adding a new point deletes the line and oldest point, and
        creates a new line between the new point and the remaining old point.
        """

        self.x1 = event.xdata
        self.y1 = event.ydata

        if len(self.plot.get_lines()) == 3:
            self.plot.get_lines()[2].remove()

        plt.plot(self.x1, self.y1, 'ro')

        for i, pnt in enumerate(self.plot.get_lines()):
            if len(self.plot.get_lines()) == 3 and i == 0:
                pnt.remove()

        self.line_x = [pnt.get_xdata() for pnt in self.plot.get_lines()]
        self.line_y = [pnt.get_ydata() for pnt in self.plot.get_lines()]
        if len(self.plot.get_lines()) > 1:
            plt.plot(self.line_x, self.line_y)

        for i, pnt in enumerate(self.plot.get_lines()):
            if i == 0:
                self.pnt1 = pnt.get_xydata()
            elif i == 1:
                self.pnt2 = pnt.get_xydata()