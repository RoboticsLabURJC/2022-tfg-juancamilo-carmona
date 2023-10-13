In today's post, I'm excited to delve deeper into one of the most critical aspects of autonomous driving: lane detection. The primary idea behind my Final Degree Project was to create a lane detector based on Convolutional Neural Networks (CNN). However, to demonstrate the effectiveness of this method, I designed three other algorithms based on traditional computer vision techniques. I'll discuss each of these in detail before diving into the issues encountered and how I resolved them.

## HSV color filter
The first method I implemented was a simple HSV (Hue, Saturation, Value) color filter. This algorithm separates the color components and thresholds the HSV image to get binary images, highlighting the lane markings effectively against the road. This is a very simple method that demands very lower computational resources but that gives very poor results in a realistic enviroment in consecuence to the change in lighting conditions since the same hsv threshold doesnt work in dark scenarios and conditions with a lot of light.

## HSV color filter and canny edge detection
To improve upon the first method, I combined the HSV color filter with the Canny edge detection algorithm. The introduction of Canny edge detection enhanced detection capabilities by identifying sharp changes in color gradients, signifying potential edges. Despite issues with the color filter under varying lighting conditions, the combined method remained computationally lightweight while providing modest improvements in lane detection.

## Sliding Window and artificial vision lane detection pipeline

Next, I utilized a computer vision pipeline, incorporating a sliding window algorithm for lane detection. This method, while more computationally intensive than the previous approaches, offered more robust lane detection irrespective of lighting conditions. However, as vehicle speeds increased or during encounters with tight, rapid curves, this algorithm struggled to maintain accurate lane detection.

## Convolutional Neural Network (CNN) based lane detection

After exploring traditional computer vision techniques, I turned to deep learning, implementing a lane detector based on convolutional Neural Networks (CNN) for lane detection. Despite the computational weight, the CNN-based lane detector demonstrated superior performance over all other methods. It managed to maintain robust lane detection across varying lighting conditions, vehicle speeds, and curve angles. This outcome affirmed the power of deep learning applications in autonomous driving, confirming its suitability for my project.

![image](https://github.com/RoboticsLabURJC/2022-tfg-juancamilo-carmona/assets/78978326/10ee5757-b1eb-4ad5-85a3-2170c2f76233)

## Problems and Solutions
Despite the success of the CNN-based lane detector, it was not without its challenges. The main issue was the bottleneck created by ROSBridge when using CARLA. This limitation capped graphical applications at around 13 to 14 frames per second (FPS), which, combined with the inference time of the algorithms, provided functional but somewhat inadequate FPS when the vehicle exceeded certain speeds.

To mitigate this issue, I spent a lot of time optimizing the code and tuning the parameters to ensure that the system could handle higher vehicle speeds and maintain robust lane detection. Despite this challenge, it served as another valuable lesson in the journey, reinforcing the importance of considering all system components' interactions and their potential impacts on the final application.

## Demostration
The following link will take you to a youtube video where you can see a demostration of the final CNN lane detection 
[Link to video](https://youtu.be/JNjXhbmbLmg)
