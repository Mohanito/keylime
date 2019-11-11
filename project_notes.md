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


## 11/5 Single-site Binary Classifier 
- 7205 training data. Another 1000 were used for validation.
- To speed up the process, all the images were resized initially from 1280*720 to 640*360, then to 320*180
- The training process is still very slow. The first epoch took 25 minutes.
### Next step:
- It might be too early to conclude that Cat vs No Cat is an easy task for our CNN.
- Training still takes too long. Need to think about Quality vs. Efficiency.
- What would be a reasonable resolution for the training images?

## 11/11 Update on results:
- ~90% validation accuracy (10 epochs) for cat vs. no cat, and 100% (questionable) for cat A vs. B.
- The cause might be the naive dataset collection for the second experiment.