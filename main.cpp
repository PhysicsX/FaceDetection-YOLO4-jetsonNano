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
    cout << "Darknet application" << endl;

    // -----------------------------------------------------------------------------------------------------------------
    // Define constants that were used when Darknet network was trained.
    // This is pretty much hardcoded code zone, just to give an idea what is needed.
    // -----------------------------------------------------------------------------------------------------------------

    // Path to configuration file.
    static char *cfg_file = const_cast<char *>("yolo/yolov4-tiny-3l.cfg");
    // Path to weight file.
    static char *weight_file = const_cast<char *>("yolo/yolov4-tiny-3l_best.weights");
    // Path to a file describing classes names.
    static char *names_file = const_cast<char *>("yolo/obj.data");
    // This is an image to test.
    // Define thresholds for predicted class.
    float thresh = 0.5;
    float hier_thresh = 0.5;
 



    // -----------------------------------------------------------------------------------------------------------------
    // Do actual logic of classes prediction.
    // -----------------------------------------------------------------------------------------------------------------

    // Load Darknet network itself.
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
    Mat image = imread("/home/ulas/FaceDetection-YOLO4-jetsonNano/face.png");
    Mat imageRgb;
    cvtColor(image, imageRgb, cv::COLOR_BGR2RGB);
    Mat imageResized;
    resize(imageRgb, imageResized, Size(network_width(net), network_height(net)), INTER_LINEAR);
    
    int size = imageResized.total() * imageResized.elemSize();
    char* bytes = new char[size];
    std::memcpy(bytes, imageResized.data, size*sizeof(char));
    copy_image_from_bytes(im, bytes);   
    // detect_image(net, className, im, 0.5);
    float* fp = network_predict_image(net, im);
    int map ;
    int map2;
    detection* det = get_network_boxes(net, im.w, im.h, 0.5, 0.5, &map2, 0, &map, 0);
    std::cout<<"x :"<<det->bbox.x<<std::endl;

    cv::rectangle(imageResized, cv::Rect((det->bbox.x-det->bbox.w/2),
			    (det->bbox.y - det->bbox.h/2), 
			    det->bbox.w, det->bbox.h), 
		            cv::Scalar(0, 255, 0));


    String windowName = "Example";
    namedWindow(windowName);
    imshow(windowName, imageResized);  
    waitKey(0);
    destroyWindow(windowName);

//    free_detections(detections, num_boxes);
//    free_image(im);
//    free_image(sized);
//    free(labels);

    return 0;
}
