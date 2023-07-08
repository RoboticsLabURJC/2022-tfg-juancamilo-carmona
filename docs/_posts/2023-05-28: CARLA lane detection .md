# CARLA lane detection

In today's post, I'm excited to delve deeper into one of the most critical aspects of autonomous driving: lane detection. The primary idea behind my Final Degree Project was to create a lane detector based on Convolutional Neural Networks (CNN). However, to demonstrate the effectiveness of this method, I designed three other algorithms based on traditional computer vision techniques. I'll discuss each of these in detail before diving into the issues encountered and how I resolved them.

## HSV Color Filter
The first method I implemented was a simple HSV (Hue, Saturation, Value) color filter. This algorithm separates the color components and thresholds the HSV image to get binary images, highlighting the lane markings effectively against the road. This is a very simple method that demands very lower computational resources but that gives very poor results in a realistic enviroment in consecuence to the change in lighting conditions since the same hsv threshold doesnt work in dark scenarios and conditions with a lot of light.

## HSV Color Filter and Canny Edge Detection
Next, I combined the HSV color filter with the Canny edge detection algorithm. The Canny algorithm is a multi-stage process that involves blurring, gradient calculation, non-maximum suppression, and finally, a double threshold for edge tracking. This combination improved the detection capabilities, but it was still not as robust as I would have liked.

## Sliding Window and artificial vision Lane Detection Pipeline
The third method involved a traditional lane detection pipeline with a sliding window algorithm. This process began by applying a perspective transformation to get a bird's eye view of the road, which was followed by histogram analysis and sliding window search to locate and track the lanes. While this method was more effective, it was computationally expensive and struggled with sharp turns and varying road conditions.

## Convolutional Neural Network (CNN) Based Lane Detection
After exploring traditional computer vision techniques, I turned to deep learning, implementing a lane detector based on Convolutional Neural Networks (CNN). The CNN-based detector proved to be very effective and robust, capable of handling various road and lighting conditions. This method was the clear winner, demonstrating the power and flexibility of deep learning.

## Problems and Solutions
Despite the success of the CNN-based lane detector, it was not without its challenges. The main issue was the bottleneck created by ROSBridge when using CARLA. This limitation capped graphical applications at around 13 to 14 frames per second (FPS), which, combined with the inference time of the algorithms, provided functional but somewhat inadequate FPS when the vehicle exceeded certain speeds.

To mitigate this issue, I spent a lot of time optimizing the code and tuning the parameters to ensure that the system could handle higher vehicle speeds and maintain robust lane detection. Despite this challenge, it served as another valuable lesson in the journey, reinforcing the importance of considering all system components' interactions and their potential impacts on the final application.
