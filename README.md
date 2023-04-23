# Smart Violence Detection(Street Violence + Traffic Light Violation Detection)

Smart violence detection using deep learning is a cutting-edge technology that aims to detect violent actions or behaviors and traffic light violations in real-time using advanced deep learning algorithms.

Violence detection systems are often used in security and surveillance applications, such as in airports, public spaces, or prisons, to help prevent violent incidents and ensure public safety. They can also be used in social media monitoring to detect and flag potentially violent or harmful content. However, there are ethical and privacy concerns related to the use of violence detection technology, as it can infringe on individual rights and freedoms if not used appropriately.

Traffic violation detection systems typically use cameras or sensors to capture images or video of traffic and then analyze the data using algorithms to detect and classify traffic violations. The most commonly used computer vision algorithms for traffic violation detection are object detection, tracking, and recognition algorithms.

# The idea

> Violence Detection

The idea behind violence detection is that whenever a violence is detected the part where the violence occurs will be stored in the disk as a .avi/.mp4 file.Along with that a picture will also be captured.An alert will be sent to the officials along with the time,date and the saved image.A proposal for the detection of face of the violator has also been made.

> Traffic-Light Violation Detection

The idea behind Traffic-Light violence detection is that whenever a car passes the crossing whenever the light is red the video will be captured along with the plate number and an alert will be sent to the official.

# Algorithms Used

The Algorithms used for the detection of Street Violence is CNN-LSTM.
For traffic-Light violation we are using YOLOV3 and ANPR.

# Packages Required

For Violence detection 

> Tensorflow version 2.0.0 ,numpy,skimage.io,opencv,PIL,BytesIO,time,tkinter,MTCNN(for face detection)

For Traffic-Light Violation

> Tensorflow >= 2.0.0,
> opencv >= 4.5,
> numpy,
> TKinter

# Datasets used

> Violence Detection

Hockey Fight Dataset - https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89

Real Life Violence Situations Dataset - https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset

# How To Run

Run **ViolenceDetection.ipynb** on jupyter notebook

If you like to change the model weights change the name of the arguments where the model name exists such as **modelweight** in both **localfiletesting.py** and **ViolenceDetection.ipynb**.

I will upload my model once I finish the training and testing of the model.

To run traffic light violation detection just run **main.py**

# Remarks

So far I have implemented street Violence Detection and traffic light violation detection and .Along with this a UI will also be implemented.The Alert system will be implemented in the coming days.The things which are left to do are number plate detection in traffic light violation detection and the alert system for both.
