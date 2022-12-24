#include <iostream>
#include <string>
#include <chrono>

#include <darknet.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

// g++ -o example faceDetectionVideo.cpp -I /home/ulas/darknet/include/ -L . -ldarknet `pkg-config --cflags --libs opencv4` && ./example

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
    
    const int widthFrame = 800;
    const int heightFrame = 480;
    const std::string pipeline = "nvarguscamerasrc sensor-id=" +std::to_string(0) +
		" ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(widthFrame) +
		", height=(int)" + std::to_string(heightFrame) +
		", format=(string)NV12, framerate=(fraction)" + std::to_string(30) +
		"/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

 
    
    image im = make_image(network_width(net), network_height(net), 3);
    cv::VideoCapture cap;
    cap.open(pipeline, cv::CAP_GSTREAMER);
	
    const float wS = net->w;
    const float hS = net->h;
    const float wT = 800;
    const float hT = 480;

    const float wScale = wT/wS;
    const float hScale = hT/hS;
    std::cout<<"wScale "<<wScale<<std::endl;
    std::cout<<"hScale "<<hScale<<std::endl;
 
    while(true)
    {

	cv::Mat frame;
    	cap.read(frame);
  
	cv::Mat imageRgb;
	cv::cvtColor(frame, imageRgb, cv::COLOR_BGR2RGB);
	cv::Mat imageResized;
	cv::resize(imageRgb, imageResized, cv::Size(network_width(net), network_height(net)), cv::INTER_LINEAR);
    
    	const int size = imageResized.total() * imageResized.elemSize();
    	char* bytes = new char[size];
    	std::memcpy(bytes, imageResized.data, size*sizeof(char));
    	copy_image_from_bytes(im, bytes);   
    
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    	float* fp = network_predict_image(net, im);
    	int number_boxes;
    	detection* det = get_network_boxes(net, im.w, im.h, 0.5, 0.5, nullptr, 0, &number_boxes, 0);
 
    	do_nms_sort(det, number_boxes, 1, 0.45);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	if(0 != number_boxes)
    	std::cout<<"Detection "<<number_boxes<<" obj, class"<<det->classes<<std::endl;
       
    	for(int i=0; i<number_boxes; ++i)
    	{
    		for(int j=0; j<1; ++j)
		{
	    		if(det[i].prob[j] > 0.5) 
	    		{	
	        		// std::cout<<"x :"<<det[i].bbox.x<<" y :"<<det[i].bbox.y<<" w :"<<det[i].bbox.w<<" h :"<<det[i].bbox.h<<std::endl; 
	                        
				const int x = det[i].bbox.x * wScale;
				const int y = det[i].bbox.y * hScale; 
	        	        const int w = det[i].bbox.w * wScale; 	
				const int h = det[i].bbox.h * hScale;
			        std::cout<<"xS :"<<x<<" yS :"<<y<<" wS :"<<w<<" hS :"<<h<<std::endl; 
	                        
	
				cv::rectangle(frame, cv::Rect((x - (w/2)), (y - (h/2)), w, h), cv::Scalar(0, 255, 0), 2);
				cv::putText(frame, "Detection time: "+std::to_string(time)+" msec, fps: "+std::to_string(1000.0/(time)), cv::Point(20,40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
				std::cout<<(int)(det[i].prob[j]*100)<<std::endl;
	   		}
		}
    	}
    
	cv::imshow("Face Detection Yolo", frame);
    
    	if(cv::waitKey(1) == 27)
	    	break;
   
   	free_detections(det, number_boxes);
   	delete[] bytes;
   }


    free_image(im);  
    free_network_ptr(net);

    return 0;
}
