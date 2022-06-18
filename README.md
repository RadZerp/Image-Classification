# Image-Classification
Project for PROG2051 2021 Spring course

Full report: [report.pdf](report.pdf)

Dataset: http://www.josiahwang.com/dataset/leedsbutterfly/

### Usage
- Clone repository to a local folder
- Run main.py in your IDE of choice - we recommend VSCode or spyder
- NOTE: The program will download a fairly large dataset to work with by default!
- Parameters for variables such as target size for image resizing, train/test splitting, number of epochs and whether to work with color or grayscale images can be found in dictionary.py. Feel free to test modifying these parameters and observe how the model performs.
- To swap between color and greyscale specifically, or activate/deactivate cross validation, flip the bools in dictionary.py.
