
# Vehicle Detection Project

## The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier

* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.

* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.

* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.

* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

* Estimate a bounding box for vehicles detected.

## Feature extraction and classifier training

The code for this section is defined in the "Feature extraction and classifier training" section of the notebook.

### Loading training data

I have loaded the training data and analyzed the total number of samples in both vehicles vs non-vehicles categories.
There were 8792 vehicle sample and 8969 non-vehicles samples, i.e. the initial training set is fairly balanced.

Samples of the training data from both categories look like following:

    Samples of vehicles



![png](output_8_1.png)


    Samples of non-vehicles



![png](output_8_3.png)


### Histogram of oriented gradients


I have mostrly reused the code from Udacity materials to extract hog features from the images.
These utility functions are defined in the "Set of utility functions mostly copied from the Udacity class materials"
section of the notebook.

To get better feeling of how HOG features will look like for cars vs non-cars I have plotted samples
of both categories with corresponding HOG features:

![png](output_13_2.png)



![png](output_13_3.png)



### Feature extraction

To understand which feature set will be best for the purpose of classification of cars vs non-cars, I have
decided to try multiple feature set configurations. The corresponding code for that is defined in the
"Definition of different feature extraction configurations" section of the notebook.

I have run the training of LinearSVC classifier using feature extracted based on the configurations.
The results are summarized in the following table:

| Color  | Spatial | Hist feature | Hog   | Orientiations | Cell per block | Hog channels | hog pix per cell | Accuracy |
|--------|---------|--------------|-------|---------------|----------------|--------------|------------------|----------|
| RGB'   | FALSE   | FALSE        | TRUE  | 9             | 2              | ALL'         | 8                | 0.973255 |
| RGB'   | FALSE   | FALSE        | TRUE  | 11            | 2              | ALL'         | 8                | 0.973255 |
| RGB'   | FALSE   | FALSE        | TRUE  | 9             | 2              | 0            | 8                | 0.958333 |
| RGB'   | FALSE   | FALSE        | TRUE  | 11            | 2              | 0            | 8                | 0.958052 |
| RGB'   | FALSE   | FALSE        | TRUE  | 9             | 2              | 1            | 8                | 0.963401 |
| RGB'   | FALSE   | FALSE        | TRUE  | 11            | 2              | 1            | 8                | 0.962838 |
| RGB'   | FALSE   | FALSE        | TRUE  | 9             | 2              | 2            | 8                | 0.960304 |
| RGB'   | FALSE   | FALSE        | TRUE  | 11            | 2              | 2            | 8                | 0.965653 |
| RGB'   | FALSE   | TRUE         | FALSE | None          | None           | None         | None             | 0.861205 |
| RGB'   | FALSE   | TRUE         | TRUE  | 9             | 2              | ALL'         | 8                | 0.904842 |
| RGB'   | FALSE   | TRUE         | TRUE  | 11            | 2              | ALL'         | 8                | 0.864865 |
| RGB'   | FALSE   | TRUE         | TRUE  | 9             | 2              | 0            | 8                | 0.884291 |
| RGB'   | FALSE   | TRUE         | TRUE  | 11            | 2              | 0            | 8                | 0.875845 |
| RGB'   | FALSE   | TRUE         | TRUE  | 9             | 2              | 1            | 8                | 0.879786 |
| RGB'   | FALSE   | TRUE         | TRUE  | 11            | 2              | 1            | 8                | 0.847128 |
| RGB'   | FALSE   | TRUE         | TRUE  | 9             | 2              | 2            | 8                | 0.891047 |
| RGB'   | FALSE   | TRUE         | TRUE  | 11            | 2              | 2            | 8                | 0.907658 |
| RGB'   | FALSE   | FALSE        | TRUE  | None          | None           | None         | None             | 0.916385 |
| RGB'   | TRUE    | FALSE        | TRUE  | 9             | 2              | ALL'         | 8                | 0.967342 |
| RGB'   | TRUE    | FALSE        | TRUE  | 11            | 2              | ALL'         | 8                | 0.971565 |
| RGB'   | TRUE    | FALSE        | TRUE  | 9             | 2              | 0            | 8                | 0.9558   |
| RGB'   | TRUE    | FALSE        | TRUE  | 11            | 2              | 0            | 8                | 0.959741 |
| RGB'   | TRUE    | FALSE        | TRUE  | 9             | 2              | 1            | 8                | 0.960867 |
| RGB'   | TRUE    | FALSE        | TRUE  | 11            | 2              | 1            | 8                | 0.956926 |
| RGB'   | TRUE    | FALSE        | TRUE  | 9             | 2              | 2            | 8                | 0.962275 |
| RGB'   | TRUE    | FALSE        | TRUE  | 11            | 2              | 2            | 8                | 0.961149 |
| RGB'   | FALSE   | TRUE         | TRUE  | None          | None           | None         | None             | 0.929054 |
| RGB'   | TRUE    | TRUE         | TRUE  | 9             | 2              | ALL'         | 8                | 0.942849 |
| RGB'   | TRUE    | TRUE         | TRUE  | 11            | 2              | ALL'         | 8                | 0.927365 |
| RGB'   | TRUE    | TRUE         | TRUE  | 9             | 2              | 0            | 8                | 0.938345 |
| RGB'   | TRUE    | TRUE         | TRUE  | 11            | 2              | 0            | 8                | 0.914414 |
| RGB'   | TRUE    | TRUE         | TRUE  | 9             | 2              | 1            | 8                | 0.943412 |
| RGB'   | TRUE    | TRUE         | TRUE  | 11            | 2              | 1            | 8                | 0.934122 |
| RGB'   | TRUE    | TRUE         | TRUE  | 9             | 2              | 2            | 8                | 0.947072 |
| RGB'   | TRUE    | TRUE         | TRUE  | 11            | 2              | 2            | 8                | 0.916385 |
| HSV'   | FALSE   | FALSE        | TRUE  | 9             | 2              | ALL'         | 8                | 0.983671 |
| HSV'   | FALSE   | FALSE        | TRUE  | 11            | 2              | ALL'         | 8                | 0.981982 |
| HSV'   | FALSE   | FALSE        | TRUE  | 9             | 2              | 0            | 8                | 0.935529 |
| HSV'   | FALSE   | FALSE        | TRUE  | 11            | 2              | 0            | 8                | 0.939189 |
| HSV'   | FALSE   | FALSE        | TRUE  | 9             | 2              | 1            | 8                | 0.907095 |
| HSV'   | FALSE   | FALSE        | TRUE  | 11            | 2              | 1            | 8                | 0.919764 |
| HSV'   | FALSE   | FALSE        | TRUE  | 9             | 2              | 2            | 8                | 0.957207 |
| HSV'   | FALSE   | FALSE        | TRUE  | 11            | 2              | 2            | 8                | 0.961712 |
| HSV'   | FALSE   | TRUE         | FALSE | None          | None           | None         | None             | 0.817568 |
| HSV'   | FALSE   | TRUE         | TRUE  | 9             | 2              | ALL'         | 8                | 0.869369 |
| HSV'   | FALSE   | TRUE         | TRUE  | 11            | 2              | ALL'         | 8                | 0.902309 |
| HSV'   | FALSE   | TRUE         | TRUE  | 9             | 2              | 0            | 8                | 0.727477 |
| HSV'   | FALSE   | TRUE         | TRUE  | 11            | 2              | 0            | 8                | 0.873029 |
| HSV'   | FALSE   | TRUE         | TRUE  | 9             | 2              | 1            | 8                | 0.877534 |
| HSV'   | FALSE   | TRUE         | TRUE  | 11            | 2              | 1            | 8                | 0.879505 |
| HSV'   | FALSE   | TRUE         | TRUE  | 9             | 2              | 2            | 8                | 0.719595 |
| HSV'   | FALSE   | TRUE         | TRUE  | 11            | 2              | 2            | 8                | 0.849662 |
| HSV'   | FALSE   | FALSE        | TRUE  | None          | None           | None         | None             | 0.757601 |
| HSV'   | TRUE    | FALSE        | TRUE  | 9             | 2              | ALL'         | 8                | 0.73705  |
| HSV'   | TRUE    | FALSE        | TRUE  | 11            | 2              | ALL'         | 8                | 0.748874 |
| HSV'   | TRUE    | FALSE        | TRUE  | 9             | 2              | 0            | 8                | 0.741836 |
| HSV'   | TRUE    | FALSE        | TRUE  | 11            | 2              | 0            | 8                | 0.735079 |
| HSV'   | TRUE    | FALSE        | TRUE  | 9             | 2              | 1            | 8                | 0.74634  |
| HSV'   | TRUE    | FALSE        | TRUE  | 11            | 2              | 1            | 8                | 0.749155 |
| HSV'   | TRUE    | FALSE        | TRUE  | 9             | 2              | 2            | 8                | 0.750282 |
| HSV'   | TRUE    | FALSE        | TRUE  | 11            | 2              | 2            | 8                | 0.743806 |
| HSV'   | FALSE   | TRUE         | TRUE  | None          | None           | None         | None             | 0.853322 |
| HSV'   | TRUE    | TRUE         | TRUE  | 9             | 2              | ALL'         | 8                | 0.863176 |
| HSV'   | TRUE    | TRUE         | TRUE  | 11            | 2              | ALL'         | 8                | 0.851351 |
| HSV'   | TRUE    | TRUE         | TRUE  | 9             | 2              | 0            | 8                | 0.853604 |
| HSV'   | TRUE    | TRUE         | TRUE  | 11            | 2              | 0            | 8                | 0.849381 |
| HSV'   | TRUE    | TRUE         | TRUE  | 9             | 2              | 1            | 8                | 0.849662 |
| HSV'   | TRUE    | TRUE         | TRUE  | 11            | 2              | 1            | 8                | 0.849099 |
| HSV'   | TRUE    | TRUE         | TRUE  | 9             | 2              | 2            | 8                | 0.855856 |
| HSV'   | TRUE    | TRUE         | TRUE  | 11            | 2              | 2            | 8                | 0.844313 |
| YUV'   | FALSE   | FALSE        | TRUE  | 9             | 2              | ALL'         | 8                | 0.982545 |
| YUV'   | FALSE   | FALSE        | TRUE  | 11            | 2              | ALL'         | 8                | 0.984516 |
| YUV'   | TRUE    | FALSE        | FALSE | None          | None           | None         | None             | 0.92286  |
| YUV'   | TRUE    | FALSE        | TRUE  | 9             | 2              | ALL'         | 8                | 0.97973  |
| YUV'   | TRUE    | FALSE        | TRUE  | 11            | 2              | ALL'         | 8                | 0.980574 |
| YCrCb' | FALSE   | FALSE        | TRUE  | 9             | 2              | ALL'         | 8                | 0.983671 |
| YCrCb' | FALSE   | FALSE        | TRUE  | 11            | 2              | ALL'         | 8                | 0.984797 |
| YCrCb' | TRUE    | FALSE        | FALSE | None          | None           | None         | None             | 0.931588 |
| YCrCb' | TRUE    | FALSE        | TRUE  | 9             | 2              | ALL'         | 8                | 0.9817   |
| YCrCb' | TRUE    | FALSE        | TRUE  | 11            | 2              | ALL'         | 8                | 0.983953 |


Based on the results of these runs, I have concluded that both histogram of color and spatial binning of color did not
provide good features to classify cars from non-cars. Through some more experiments I have chosen the following configuration
for further progress:

feature_config = {
    'color_space': 'YUV',
    'spatial_feat': False,
    'hog_feat': True,
    'hog_channel': 'ALL',
    'hog_orientations': 11,
    'hog_pix_per_cell': 8,
    'hog_cell_per_block': 2,
    'hist_feat': False,
}

So, I have used YUV color space and HOG features on all color channels. I have used 11 hog orientations.
The choice is mostly driven through the experiments I have conducted on test images.

## Finding cars in images

I have reused the *find_cars* function from Udacity class to search for cars on the images.

To make the identification of the cars more robust, I have run the find_cars function on the same image
using several strips of the images and different scale factors.

I have overlayed identified windows with cars over the test images:


![png](output_31_3.png)



![png](output_31_4.png)



![png](output_31_5.png)



![png](output_31_6.png)



![png](output_31_7.png)



![png](output_31_8.png)


As demonstrated on the images above, the method robust enough to identify all cars on the images, however,
it also produces some number of false positive identifications.


## Using heatmap to filter out results and draw single box

To filter out false positives and combine multiple windows corresponding to the same car
I have used the heatmap method described in Udacity materials. The code for this section
is defined in the section "Using heatmap to filter out results and draw single box" of the notebook.

Application of the method to the test images produced the following results:



![png](output_34_3.png)



![png](output_34_4.png)



![png](output_34_5.png)



![png](output_34_6.png)



![png](output_34_7.png)



![png](output_34_8.png)


So, it has successfully removed most of the false positives and nicely combined multiple windows corresponding to the same car.

## Processing video input

To process the video input I have defined addition function *process_frame_with_history*.
This function joins previously defined steps into the single pipeline. It also adds additional
step by recording the identified windows from the last *N* processed frames. All recorded windows
are added to single heatmap to identify the cars on the current frame.

The code for this section is defined at "Collecting the history of previously identified cars" section of the
notebook.

The produced video is [project_video_out.mp4](project_video_out.mp4)


## Discussion

The test of the pipeline on the project video demonstrates several short-comings of the pipeline:

- Higher than expected rate of false positives. Trying to stricten filtering criteria lead to false negatives results.
- Cars overdriving from behind are not immediately picked up by the pipeline
- Cars distant enough tend to avoid detection

Having more time to work on the project, I would improve the algorithm using the following approaches:

- Using more training data to improve the classification robustness
- Improve usage of previous frames to smooth the identification of the cars
- Investigate completely different approach - using CNNs to identify cars on the images.
