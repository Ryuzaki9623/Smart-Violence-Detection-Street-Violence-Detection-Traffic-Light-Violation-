# Automatic Traffic Light Running Violation Detection
## Abstract
An automatic traffic red-light violation detection system was implemented, which may play a big role in transportation management in smart cities. The system mainly relies on modern computer vision techniques, which was implemented in OpenCV under Python environment. Mainly, the system consists of object detector and object tracker which work in an integrated manner in order to precisely keep position of the existing cars. The primary task of the system is to eventually indicate locations of violating vehicles. The output showed accurate results, as all violating vehicles were detected and distinguished precisely.

## Design Specification
In this project, the system was implemented to accept the input as video of crossroad, which needed to observe and detect traffic red-light violations. System consists mainly of three components: violation line estimator, vehicles detector and vehicles tracker.

### Violation line estimation
A crucial part in building a system for vehicle violation detection is specifying the area for which a vehicle is considered to be in a violation state. Such task can be quite challenging and nearly unattainable due to the high levels of variability present in the input videos such as the depth of view, video capture perspective, location and distribution of traffic lights along with the specific description of the road. Consequently, it is clear that in order to obtain the desired region of violation, one must set direct assumptions which narrow down the generality of the problem and bring it closer to a technical level. Therefore, two main assumption regarding the input were made in order to gain knowledge about the violation region and proceed properly with the needed operations, these assumptions are: 
  • First, the region is to be represented as any vertical area within the video which surpasses a certain threshold horizontal line, referred to as the violation line, which splits the road into a regular and violating zone, as the main goal of the system evidentially becomes to determine the vertical location of this horizontal line. 
  • Second, in order to obtain clear info and indication towards the approximate vertical location of the violation line, it is required that a crosswalk exists between the regular and violating area, within the vicinity of the traffic light.


### Vehicle detection
At each five frames, a detection is done using YOLOv3 pretrained model on COCO dataset. The detection output is formulated by several steps, from filtering the bounding boxes with low confidence rate and filtering any bounding box that isn’t a vehicle to finally doing non-maximum suppression to the detected boxes, so that each vehicle has only one bounding box.

### Vehicle tracking
Keeping an accurate localization on each vehicle can be very tedious task, tracking the first frame vehicles is not enough, due to incoming and outgoing vehicles of the scope of the images. To resolve this, a merging between tracking and detection tasks is needed. The tracking of the vehicles is done at every frame, but every five frames a new detection occurs, then for each detected bounding box a measure of IoU (Intersection over Union) is done with the current tracking bounding boxes. If a detected box matches a tracking box with a relatively good percentage then it's the same box that in the tracking list, but the new detected bounding box has to be more accurate than the tracking one, so an adjustment operation occurs to the tracking object bounding box.


However, if there are detected boxes with no good matches with any of the tracking boxes, then they have to be a new incoming vehicle, when so, the tracking list is updated with those new vehicles, in contrast if there’s a bounding box that disappeared in the detection boxes, then it’s an outgoing car, so its tracker is removed. As mentioned above, at each iteration an update operation to the tracking list is done, when so, each bounding box is applied to a validator to check the violation, if it violates the red light, it’s tracking object is moved to the violated tracking list so it can be visualized in the output video.

### Violation detection
A vehicle violates if it satisfies two conditions, if its center y value crosses the violation line AND if the current status of the detected traffic light is red, when those two conditions present the vehicle violates the red light and its tracker is moved to the violated tracking list. When the traffic light is detected, recognizing its status can be done greedily using color histogram. Each pixel in the traffic light is mapped to either red, yellow/orange, green or other, after this the maximum summation of the pins is simply chosen as the status of the traffic light.

The Following is the proposed flow for violence Detection

<img width="327" alt="proposed flow" src="https://github.com/Ryuzaki9623/Smart-Violence-Detection-Street-Violence-Detection-Traffic-Light-Violation-/assets/45362890/e10b3430-a6d7-4004-a8c2-059b23500d00">

## System Evaluation
An important aspect in implementing a violation system is to thoroughly asses the outputs and results and reflect on the performance of the system in general based on common metrics. First, the system is to be evaluated on the processing time which will be categorized mainly into violation line approximation time, traffic light localization time and tracking and violation detection processing time (per frame). Finally, the metrics of average precision and average recall will be calculated for the input videos present and based on whether a vehicle is classified to be violating or not, namely, a false positive classification would be a non-violating vehicle being classified as violating, inversely, a false negative violation would be a violating vehicle being classified as non-violating.




