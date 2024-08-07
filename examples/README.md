# Files

* `.gitignore` should list `case14_network.glm` to avoid saving the output of
  the converter from `py` format.
* `case14.glm` is the GLM file linking the IEEE 14 bus model to the controller
  and model objects.
* `case14_network.py` is the PyPower version of the IEEE 14 bus model use to
  generate the network GLM file.
* `case14_loads.glm` is the load model
* `case14_plants.glm` is the generation model
* `case14_recorders.glm` is the data collection model
* `case14_modify.glm` is the model parameter settings file.
* `controllers.glm` is the Python module that define the controller object and
  event handlers.

# References

* Recording of GridLAB-D introduction given by David Chassin on 3/13/24: [Zoom link](https://stanford.zoom.us/rec/share/9VuLIQs_Mqc7QxqMq0jwlF7koQvsq8s_K_ojcYjcxbJVY4zoLHEJuq6MEgKnRFOM.KYGNtoDGLC5ahU1E) (Passcode: !guPpY1n)
