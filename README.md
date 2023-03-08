# Smart Violence Detection(Street Violence + Traffic Light Violation Detection)

Smart violence detection using deep learning is a cutting-edge technology that aims to detect violent actions or behaviors and traffic light violations in real-time using advanced deep learning algorithms.

Violence detection systems are often used in security and surveillance applications, such as in airports, public spaces, or prisons, to help prevent violent incidents and ensure public safety. They can also be used in social media monitoring to detect and flag potentially violent or harmful content. However, there are ethical and privacy concerns related to the use of violence detection technology, as it can infringe on individual rights and freedoms if not used appropriately.

Traffic violation detection systems typically use cameras or sensors to capture images or video of traffic and then analyze the data using algorithms to detect and classify traffic violations. The most commonly used computer vision algorithms for traffic violation detection are object detection, tracking, and recognition algorithms.

# The idea

# Algorithms Used

The Algorithms used for the detection of Street Violence is CNN-LSTM.
For traffic-Light violation we are using RCNN,YOLOV5 and ALPR.

# Packages Required

For Violence detection 

> Tensorflow version 2.0.0 ,numpy,skimage.io,opencv,PIL,BytesIO,time

For Traffic-Light Violation

> Will be updating soon

# Datasets used

> Violence Detection

Hockey Fight Dataset - https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89

Real Life Violence Situations Dataset - https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset

# How To Run

Run **ViolenceDetection.ipynb** on jupyter notebook

If you like to change the model weights change the name of the arguments where the model name exists such as **modelweight** in both **localfiletesting.py** and **violencedetection.ipynb**.

I will upload my model once i finish the training and testing of the model.

# Remarks

So far i have only implemented street Violence Detection and I will be finishing of the traffic-light violation detection and combining these 2 models so that we will get both desired output from the input/Live feed.Along with this a UI will also be implemented.
