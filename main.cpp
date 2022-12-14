#include <iostream>
#include <darknet.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

// g++ -o example main.cpp -I /home/ulas/darknet/include/ -L . -ldarknet `pkg-config --cflags --libs opencv4` && ./example

using namespace std;
using namespace cv;

int main() {
    
    // Path to configuration file.
    static char *cfg_file = const_cast<char *>("yolo/yolov4-tiny-3l.cfg");
    // Path to weight file.
    static char *weight_file = const_cast<char *>("yolo/yolov4-tiny-3l_best.weights");
    // Path to a file describing classes names.
    static char *names_file = const_cast<char *>("yolo/obj.data");
    float thresh = 0.5;
    float hier_thresh = 0.5;
 
    network *net = load_network_custom(cfg_file, weight_file, 0, 1);
    std::cout<<"w "<<net->w<<std::endl;

    metadata met = get_metadata(names_file);

    std::cout<<"classes "<<met.classes<<std::endl;
    char* className;
    for (char ** p = met.names; *p; ++p) // or "*p != NULL"  
    {
	className = *p;
	std::cout<<className<<std::endl;
    }

    std::cout<<network_width(net)<<std::endl;
    std::cout<<network_height(net)<<std::endl;
    image im = make_image(network_width(net), network_height(net), 3);

    //Mat image;
    Mat image = imread("/home/ulas/FaceDetection-YOLO4-jetsonNano/face3.jpg");
    Mat imageRgb;
    cvtColor(image, imageRgb, cv::COLOR_BGR2RGB);
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
    std::cout<<"Detection "<<number_boxes<<" obj, class"<<det->classes<<std::endl;

    for(int i=0; i<number_boxes; ++i)
    {
    	for(int j=0; j<1; ++j)
	{
	    if(det[i].prob[j] > 0.5) 
	    {	
	        std::cout<<"x :"<<det[i].bbox.x<<" y :"<<det[i].bbox.y<<" w :"<<det[i].bbox.w<<" h :"<<det[i].bbox.h<<std::endl; 
	   
	        cv::rectangle(imageResized, cv::Rect((det[i].bbox.x-det[i].bbox.w/2),
			                             (det[i].bbox.y - det[i].bbox.h/2), 
			                              det[i].bbox.w, det[i].bbox.h), 
		                                      cv::Scalar(0, 255, 0), 2);

	   	std::cout<<(int)(det[i].prob[j]*100)<<std::endl;
	   }
	}
    }
    
    free_detections(det, number_boxes);
    free_image(im);

    String windowName = "Face Detection Yolo";
    namedWindow(windowName);
    imshow(windowName, imageResized);  
    waitKey(0);
    destroyWindow(windowName);

    return 0;
}
