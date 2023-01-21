#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QDebug>
#include <QImage>
#include "../darknet.h"

//#include </home/ulas/FaceDetection-YOLO4-jetsonNano/darknet.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <QPainter>

#include <QObject>
#include <QQuickPaintedItem>
#include "qmlType.h"
#include <vector>
#include <utility>
#include <mutex>
#include <iostream>
#include <thread>

// sudo su
// export LD_LIBRARY_PATH=/home/ulas/FaceDetection-YOLO4-jetsonNano
// ./QtOpencvQml -platform eglf
//

// rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_PREFIX_PATH=/usr/local/QtPath/ && make

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);

    qRegisterMetaType<cv::Mat>("cv::Mat");
    qmlRegisterType<qmlType>("Painter", 1, 0, "QmlType");

    QQmlApplicationEngine engine;

    const QUrl url(QStringLiteral("qrc:/main.qml"));

    engine.load(url);
    QObject *object = engine.rootObjects()[0];

    QObject *inputCamera = object->findChild<QObject*>("inputCamera");

    qmlType *ptrQmlType = qobject_cast<qmlType*>(inputCamera);
    
    QTimer timer;

    // Path to configuration file.
    static char *cfg_file = const_cast<char *>("/home/ulas/FaceDetection-YOLO4-jetsonNano/yolo/yolov4-tiny-3l.cfg");
    // Path to weight file.
    static char *weight_file = const_cast<char *>("/home/ulas/FaceDetection-YOLO4-jetsonNano/yolo/yolov4-tiny-3l_best.weights");
    // Path to a file describing classes names.
    static char *names_file = const_cast<char *>("/home/ulas/FaceDetection-YOLO4-jetsonNano/yolo/obj.data");


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

    cv::Mat frameClone;
    std::vector<std::vector<int>> coordinates, vecCopy;
    std::mutex m;

    //QObject::connect(&timer, &QTimer::timeout, [&]()
    std::thread([&]()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(4000));
        
	while(1){
    	cv::Mat imageRgb;
	
	m.lock();
	cv::cvtColor(frameClone, imageRgb, cv::COLOR_BGR2RGB);
        m.unlock();

	cv::Mat imageResized;
	cv::resize(imageRgb, imageResized, cv::Size(network_width(net), network_height(net)), cv::INTER_LINEAR);
    	const int size = imageResized.total() * imageResized.elemSize();
    	char* bytes = new char[size];
    	std::memcpy(bytes, imageResized.data, size*sizeof(char));
    	copy_image_from_bytes(im, bytes);   

	float* fp = network_predict_image(net, im);
    	int number_boxes;
    	detection* det = get_network_boxes(net, im.w, im.h, 0.5, 0.5, nullptr, 0, &number_boxes, 0);

    	do_nms_sort(det, number_boxes, 1, 0.45);

    	for(int i=0; i<number_boxes; ++i)
    	{
    		for(int j=0; j<1; ++j)
		{
			if(det[i].prob[j] > 0.5)
	    		{
				const int x = det[i].bbox.x * wScale;
				const int y = det[i].bbox.y * hScale;
	        	        const int w = det[i].bbox.w * wScale;
				const int h = det[i].bbox.h * hScale;
			        std::cout<<"xS :"<<x<<" yS :"<<y<<" wS :"<<w<<" hS :"<<h<<std::endl; 
				m.lock();
				coordinates.push_back({x, y, w, h});
				m.unlock();
			}
		}
	}

	free_detections(det, number_boxes);
   	delete[] bytes;
	}

    }).detach();
    //timer.start(10);
    //
    int cnt {0};
    while(true)
    { 
        cv::Mat frame;
	cap >> frame;
	
	m.lock();
	cnt ++;
	frameClone = frame.clone();
	if(!coordinates.empty())
	{  
	 	cnt = 0;
		vecCopy = coordinates;
		coordinates.clear();
	}
	if(cnt == 10)
	{
		vecCopy.clear();
	}
	m.unlock();

        if(frame.empty())
	{
		// to clean lsat frame
		const int width = 100;
		const int height = 200;
		QImage image = QPixmap(width, height).toImage();
		
		ptrQmlType->updateImage(image);
		//timer.stop();
        	break;
	}

	for(const auto& v : vecCopy)
	{
		cv::rectangle(frame, cv::Rect((v[0] - (v[2]/2)), (v[1] - (v[3]/2)), v[2], v[3]), cv::Scalar(0, 255, 0), 2);
	
	}
	
	// Opencv use BGR but QImage use RGB so color information
	// should be swapped for this case Mat->QImage
	// no need to swap when Mat->QImage->Mat
	cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);	
	QImage imgIn= QImage((uchar*) frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
	//.rgbSwapped();

 	ptrQmlType->updateImage(imgIn);
    	if(cv::waitKey(1) == 27)
	    	break;

    }


    free_image(im);  
    free_network_ptr(net);

//    QObject::connect(&timer, &QTimer::timeout, [&]()
//    {
//
//	cv::Mat frame;	
//	cap >> frame;
//	
//        if(frame.empty())
//	{
//		// to clean lsat frame
//		int width = 100;
//		int height = 200;
//		QImage image = QPixmap(width, height).toImage();
//		
//		ptrQmlType->updateImage(image);
//		timer.stop();
//        	return;
//	}
//	// Opencv use BGR but QImage use RGB so color information
//	// should be swapped for this case Mat->QImage
//	// no need to swap when Mat->QImage->Mat
//	cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);	
//	QImage imgIn= QImage((uchar*) frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
//	//.rgbSwapped();
//
// 	ptrQmlType->updateImage(imgIn);
//
//    }
//   );
//    timer.start(60);
 
    return app.exec();
}
