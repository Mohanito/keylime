# Project Notes

## Plan:

1. Single-site Binary Classifier: Cat vs. No Cat.
2. Single-site Cat A vs. Cat B ...
3. Multiple-site Binary: Given video data from ANY site, would this project be able to tell if it's a cat video?
4. Cat A vs. Cat B vs. Cat C ... Classify Cat ID for ANY video data.

## 11/1 Generating training data
### Collecting positive (with cat) images (prepare_dataset.py)
From the 60 longest videos from the site I selected, for every video: 
- sample 1 out of 20 frames
- detect objects with pre-trained model
- if the model detects a cat with over 80% confidence, save that frame.
### Collecting negative images (prepare_negative_dataset.py)
Very naive approach: From 3 videos I selected: (1)cleaning, (2)opening door, (3)opening door with a dog in the corner, extract every frame.
### Known issues:
- Converting to grayscale has a different effect from the security camera's night mode.
- The negative dataset only includes one angle.
