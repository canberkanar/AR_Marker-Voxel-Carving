#include <iostream>

#include <aruco/aruco.h>
#include <aruco/markerdetector.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;

int
main(int argc, char** argv)
{

	if (argc < 3)
	{
		cout << "Usage: " << argv[0] << " [Camera Num] [Camera Config XML]\n";
		return 0;
	}

	int cameraNum = atoi(argv[1]);

	cv::VideoCapture inputVideo;

	inputVideo.open(cameraNum);
	cv::Mat cameraMatrix, distCoeffs;

	while (inputVideo.grab())
	{
		cv::Mat image, imageCopy;
		inputVideo.retrieve(image);
		image.copyTo(imageCopy);
		std::vector<int> ids;
		std::vector<std::vector<cv::Point2f>> corners;
		aruco::CameraParameters camera;
		camera.readFromXMLFile(argv[2]);
		aruco::MarkerDetector Detector;


		Detector.setDictionary("ARUCO");
		auto markers = Detector.detect(imageCopy, camera, 0.05);
		for (auto m : markers)
		{
			aruco::CvDrawingUtils::draw3dAxis(imageCopy, m, camera);
			cout << m.Rvec << " " << m.Tvec << endl;
		}
		cv::imshow("out", imageCopy);
		char key = (char) cv::waitKey(100);
		if (key == 27)
			break;
	}

}


