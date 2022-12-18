#include <iostream>
#include <string>

#include <darknet.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

// g++ -o example faceDetectionVideo.cpp -I /home/ulas/darknet/include/ -L . -ldarknet `pkg-config --cflags --libs opencv4` && ./example

using namespace std;
using namespace cv;

int main() {
    
    // Path to configuration file.
    static char *cfg_file = const_cast<char *>("yolo/yolov4-tiny-3l.cfg");
    // Path to weight file.
    static char *weight_file = const_cast<char *>("yolo/yolov4-tiny-3l_best.weights");
    // Path to a file describing classes names.
    static char *names_file = const_cast<char *>("yolo/obj.data");
 
    network *net = load_network_custom(cfg_file, weight_file, 0, 1);
    std::cout<<"w "<<net->w<<std::endl;

    metadata met = get_metadata(names_file);

    std::cout<<"classes "<<met.classes<<std::endl;
    // for this example there is only one class 
    char* className;
    for (char ** p = met.names; *p; ++p) // or "*p != NULL"  
    {
	className = *p;
	std::cout<<className<<std::endl;
    }

    std::cout<<network_width(net)<<std::endl;
    std::cout<<network_height(net)<<std::endl;
    
    int widthFrame = 800;
    int heightFrame = 480;
    std::string pipeline = "nvarguscamerasrc sensor-id=" +std::to_string(0) +
		" ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(widthFrame) +
		", height=(int)" + std::to_string(heightFrame) +
		", format=(string)NV12, framerate=(fraction)" + std::to_string(30) +
		"/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

 
    
    image im = make_image(network_width(net), network_height(net), 3);
    cv::VideoCapture cap;
    cap.open(pipeline, cv::CAP_GSTREAMER);
	
    float wS = net->w;
    float hS = net->h;
    float wT = 800;
    float hT = 480;

    float wScale = wT/wS;
    float hScale = hT/hS;
    std::cout<<"wScale "<<wScale<<std::endl;
    std::cout<<"hScale "<<hScale<<std::endl;
 
    while(true)
    {

    	cv::Mat frame;
    	cap.read(frame);
  
    	Mat imageRgb;
    	cvtColor(frame, imageRgb, cv::COLOR_BGR2RGB);
    	Mat imageResized;
    	resize(imageRgb, imageResized, Size(network_width(net), network_height(net)), INTER_LINEAR);
    
    	int size = imageResized.total() * imageResized.elemSize();
    	char* bytes = new char[size];
    	std::memcpy(bytes, imageResized.data, size*sizeof(char));
    	copy_image_from_bytes(im, bytes);   
    
    	float* fp = network_predict_image(net, im);
    	int number_boxes;
    	int map;
    	detection* det = get_network_boxes(net, im.w, im.h, 0.5, 0.5, nullptr, 0, &number_boxes, 0);
 
    	do_nms_sort(det, number_boxes, 1, 0.45);
	if(0 != number_boxes)
    	std::cout<<"Detection "<<number_boxes<<" obj, class"<<det->classes<<std::endl;
       
    	for(int i=0; i<number_boxes; ++i)
    	{
    		for(int j=0; j<1; ++j)
		{
	    		if(det[i].prob[j] > 0.5) 
	    		{	
	        		// std::cout<<"x :"<<det[i].bbox.x<<" y :"<<det[i].bbox.y<<" w :"<<det[i].bbox.w<<" h :"<<det[i].bbox.h<<std::endl; 
	                        
				int x = det[i].bbox.x * wScale;
				int y = det[i].bbox.y * hScale; 
	        	        int w = det[i].bbox.w * wScale; 	
				int h = det[i].bbox.h * hScale;
			       std::cout<<"xS :"<<x<<" yS :"<<y<<" wS :"<<w<<" hS :"<<h<<std::endl; 
	                        
	
				cv::rectangle(frame, cv::Rect((x - (w/2)),
			                             (y - (h/2)), 
			                              w, h), 
		                                      cv::Scalar(0, 255, 0), 2);

	   			std::cout<<(int)(det[i].prob[j]*100)<<std::endl;
	   		}
		}
    	}
    
    	imshow("video", frame);
    
    	if(waitKey(1) == 27)
	    	break;
   
   	free_detections(det, number_boxes);
   	delete[] bytes;
   }


    free_image(im);  
    free_network_ptr(net);

    return 0;
}
