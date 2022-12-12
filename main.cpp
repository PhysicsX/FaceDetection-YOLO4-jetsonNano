#include <iostream>
#include <darknet.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
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
    static char *input = const_cast<char *>("/home/yurii/Pictures/road_signs/max_50.jpg");
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
    for (char ** p = met.names; *p; ++p) // or "*p != NULL"  
    {
	char * temp = *p;
	std::cout<<temp<<std::endl;
    }

    std::cout<<network_width(net)<<std::endl;
    std::cout<<network_height(net)<<std::endl;
    image im = make_image(network_width(net), network_height(net), 3);

    //Mat image;
    Mat image = imread("/home/ulas/FaceDetection-YOLO4-jetsonNano/face.png");
	String windowName = "Example";
	namedWindow(windowName);
	imshow(windowName, image);
	waitKey(0);
	destroyWindow(windowName);

//    free_detections(detections, num_boxes);
//    free_image(im);
//    free_image(sized);
//    free(labels);

    return 0;
}
