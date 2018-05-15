# Rendering from Fire Image
###Overview
The purpose of this repository is to explore the application of Machine Learning tools to assist in rendering Computer Generated fire volumes. Specifically this entails training a MLP to learn appropriate render parameters for render time based on an input image of actual fire and other data relevant to the simulation.

The MLP will be trained using images from a number of sources. These can be viewed in the "fire_images" folder. The fire images are masked to select only those pixels which are part of the fire in the image.

Examples of such masks can be found in the "masks" folder. The pixel temperature values are then calculated based on the black body radiation scale, and statistically significant values for these temperatures (such as the mean, median, range, etc.) are used as the input vector for training the MLP. These inputs will be added to as more relevant features are found. Example datasets can be found in the "train_data" folder.

As the model trains it will create ifd's which are then used by the render engine to create an image of fire. The same masking technique described above is used and the output values are compared to the input vector to define the loss.

###Usage
#####Masking Fire Images

The code to mask fire image pixels is found in "mask_fire.py"

#####Training the Network

You can train the network by running "render_fire.py"
