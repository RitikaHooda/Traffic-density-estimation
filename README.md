# Traffic-density-estimation
This project uses OpenCV functions to estimate traffic density from traffic videos. Queue density is density of all vehicles queued (either standing or moving). Dynamic density is the density of those vehicles which are not standing but moving in that same stretch. Queue density needs background subtraction whereas dynamic density needs optical flow across subsequent frames, to detect which pixels moved across frames. 
1. Firstly, perspective correction is done using homography and then frame is cropped for further processing.
2. The frame is processed for calculating density with a comparative analysis of different methods in analysis folder. 
Usage 
