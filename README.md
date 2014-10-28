# Scene reconstruction from light-field data

This is a prototype for scene reconstruction from light-field data. It will load a series a Lytro's 'raw.lfp' files. After processing, an interactive reconstruction is available.

The current version of the prototype is still faulty and does not provide a useful reconstruction. However, it's an example of how to render images from Lytro's files, perform depth estimation from light-field data and reconstruct a scene from it.

## How to use it

The prototype works with 'raw.lfp' files generated by either a Lytro plenoptic camera (first generation) or its companion software Lytro Desktop. It uses a number of libraries. Except for OpenCV, they are all contained in the repository. OpenCV 2.4.9.0 including the modules ocl and viz is required. Both modules must be compiled explicitly and have further dependencies. The current prototype is partially OpenCL-accellerated and requires an AMD graphics chip.

For information on OpenCV ocl see http://docs.opencv.org/2.4.3/modules/ocl/doc/introduction.html

For instructions on how to compile OpenCV with viz, see http://answers.opencv.org/question/32502/opencv-249-viz-module-not-there/

With OpenCV set up, change the variables *lfpCount* and *lfpPaths* in `main.cpp` to match the LFP files you want to use. You can use `main.cpp` as a sandbox to explore the API.

## Licence information

See [LICENSE.txt](./LICENSE.txt) for license information.
