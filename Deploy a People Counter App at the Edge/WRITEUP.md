# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves using these custom layers as extensions to the model optimizer, replace subgraph with another, offload computation to the subgraph.

Some of the potential reasons for handling custom layers are in case that our model topology that we would like to convert it to an IR using the model optimizer is contains layers that are not in the list of known layers which leads the model optimizer to fail in converting this model to an IR for ineference.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were:
Before:
Use tensorflow to handle and load the frozen model graph and test accuracy.
Note: i managed to test 3 models from tensorflow object detection model zoo (ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz, ssd_mobilenet_v2_coco_2018_03_29.tar.gz, ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)
and decided to use : ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
After:
load the IR into the inferene engine and test accuracy (probability threshold)

The difference between model accuracy pre- and post-conversion was nearly 12% improvement in the post-conversion.

The size of the model pre- and post-conversion was:
pre-convertion ~ 19 MB
post-convertion ~ 17 MB

The inference time of the model pre- and post-conversion was:
pre-conversion ~ 0.07 second most of the time (most of the frames).
post-conversion ~ 0.03 second most of the time (most of the frames).

## Assess Model Use Cases

Some of the potential use cases of the people counter app are for example deploying it at a supermarkets/groceries/local stores or for surveillance/security cameras.

Each of these use cases would be useful because the app could spot any person in the frame and log back the duration of staying in front of the camera and total spots which can help assess & orient statistics.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:
1- When it comes to lighting the model i used is optimised for lite-resources edge devices (usually mobile phones) so indeed i think this aspect will win.
2- When it comes to accuracy there is a noticeable drop/loss in some situations/events
3- when it comes to camera focal length/image size i think that cannot affect much.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
