"""PyPower model graphic"""

from ppmodel import PPModel
import matplotlib.pyplot as plt

class PPPlots:

    figsize = (15,8)

    def __init__(self,
        model:PPModel,
        ):

        self.model = model

    def voltage(self,
        figsize=(15,8)):

        bus = self.model.get_data("bus")

        #
        # Plot voltage errors by bus
        #
        fig = plt.figure(figsize=self.figsize if figsize is None else figsize)

        plt.subplot(2,1,1)
        plt.plot(bus.VM,label=self.model.name)
        plt.ylabel("Voltage Magnitude (pu.kV)")
        plt.grid()
        plt.xticks(rotation=90)
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(bus.VA,label=self.model.name)
        plt.ylabel("Voltage Angle (deg)")
        plt.xlabel("Bus ID")
        plt.grid()
        plt.xticks(rotation=90)
        plt.legend()

        plt.suptitle(f"{self.model.name} Voltage")

        return fig

    def generation(self,
        figsize=None):

        bus = self.model.get_data("gen")

        #
        # Plot voltage errors by bus
        #
        fig = plt.figure(figsize=self.figsize if figsize is None else figsize)

        plt.subplot(2,1,1)
        plt.plot(bus.PG,label=self.model.name)
        plt.ylabel("Real power generation (pu.MVA)")
        plt.grid()
        plt.xticks(rotation=90)
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(bus.QG,label=self.model.name)
        plt.ylabel("Reactive power generation (pu.MVA)")
        plt.xlabel("Bus ID")
        plt.grid()
        plt.xticks(rotation=90)
        plt.legend()

        plt.suptitle(f"{self.model.name} Generation")

        return fig

    def load(self,
        figsize=(15,8)):

        bus = self.model.get_data("bus")

        #
        # Plot voltage errors by bus
        #
        fig = plt.figure(figsize=self.figsize if figsize is None else figsize)

        plt.subplot(2,1,1)
        plt.plot(bus.PD,label=self.model.name)
        plt.ylabel("Real power demand (pu.MVA)")
        plt.grid()
        plt.xticks(rotation=90)
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(bus.QD,label=self.model.name)
        plt.ylabel("Reactive power demand (pu.MVA)")
        plt.xlabel("Bus ID")
        plt.grid()
        plt.xticks(rotation=90)
        plt.legend()

        plt.suptitle(f"{self.model.name} Generation")

        return fig


if __name__ == "__main__":
    from wecc240 import wecc240
    model = PPModel("WECC 240",case=wecc240(options=["SCHEDULING"]))
    plotter = PPPlots(model)
    plotter.voltage().show()
