# Star Classification
This is a project made for school in order to practice support vector classification. It is composed of two files: _star_classification.csv_ and _StarSVC.py_.

**star_classification.csv**: data from the [Stellar Classification Dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17), including 100,000 samples from the SDSS, labeling each as a star, quasar, or galaxy. I dropped most identifier columns before saving the csv file, but a couple are dropped in the program.
The remaining features of the data are defined below:
```
alpha = Right Ascension angle (at J2000 epoch)
delta = Declination angle (at J2000 epoch)
u = Ultraviolet filter in the photometric system
g = Green filter in the photometric system
r = Red filter in the photometric system
i = Near Infrared filter in the photometric system
z = Infrared filter in the photometric system
redshift = redshift value based on the increase in wavelength
MJD = Modified Julian Date, used to indicate when a given piece of SDSS data was taken
class = object class (galaxy, star or quasar object)
```
Note that the data is inbalanced. There are 584 GALAXY samples, 232 STAR samples, and 184 QSO samples.

**StarSVC.py**: the python program which I run the classifier. In order:
1. Load the data. I sample 1000 of the 100,000 rows for runtime purposes, so more accuracy and less variance can be achieved by increasing this number.
2. Perform grid search on SVC with a radial basis function to find the optimal C and gamma parameters.
3. Run SVC using the optimal parameters.
4. Display confusion matrix.

Below is an example of the produced confusion matrix. Along the diagonal are how many samples were correctly classified.

![image](https://github.com/user-attachments/assets/792d2ffe-9728-4f9a-9124-c30547e438ea)

_This sample run through the classifier produced a margin which scored 0.930 (**93% accuracy**) using C=60 and gamma=0.023. These values differ per sample._
