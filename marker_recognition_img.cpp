#include <iostream>
//
//#include <aruco/aruco.h>
//#include <aruco/markerdetector.h>

#include <aruco/aruco.h>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <direct.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "PointsObjectCoord.h"
using namespace cv;
using namespace std;
//Put in "Line;" to print the program line number
#define LINE std::cout<<__LINE__ << "\n"



cv::Mat resizeImg(cv::Mat preimg) {
	cv::Mat img;
	double newwidth = (540.0 / preimg.size().height) * preimg.size().width;
	cv::Size s = cv::Size((int)newwidth, 540);
	cv::resize(preimg, img, s);
	return img;
}



void subtract_background(cv::Mat img) {

	//create Background Subtractor objects
	Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();
	Mat fgMask;
	
	//update the background model
	pBackSub->apply(img, fgMask);
	//get the frame number and write it on the current frame
	rectangle(img, cv::Point(10, 2), cv::Point(100, 20),
		cv::Scalar(255, 255, 255), -1);
	stringstream ss;
	/*ss << img.get(CAP_PROP_POS_FRAMES);
	string frameNumberString = ss.str();
	putText(img, frameNumberString.c_str(), cv::Point(15, 15),
		FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));*/
	//show the current frame and the fg masks
	imshow("Frame", img);
	imshow("FG Mask", fgMask);
	cv::waitKey(0);
	
	return;
}


int
main(int argc, char** argv)
{
	try
	{
		cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);

		Mat frames[7];

		for (int i = 0; i < 7; i++) {
			std::stringstream path;
			path << "../../../images/" << "image_" << (i+1) << ".jpg";
			std::string image_path = cv::samples::findFile(path.str());
			cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
			if (img.empty())
			{
				std::cout << "Could not read the image: " << path.str() << std::endl;
				return 1;
			}
			frames[i] = resizeImg(img);
		}

		if (argc >= 4)
		{
			int dimX = atoi(argv[2]);
			int dimY = atoi(argv[3]);
		}
		for (int i = 0; i < 7; i++) {
			
			// Detect markers in frame
			std::vector<int> ids;
			std::vector<std::vector<cv::Point2f> > corners;
			cv::aruco::detectMarkers(frames[i], dictionary, corners, ids);

			if (!ids.empty())
			{
				if (argc == 3)
				{
					::aruco::CameraParameters camera;
					camera.readFromXMLFile(argv[2]);
					cv::Mat cameraMatrix = camera.CameraMatrix;
					cv::Mat distCoeffs = camera.Distorsion;

					//Uncomment to see the 3d orientations of the arUco markers
					/*std::vector<cv::Vec3d> rvecs, tvecs;
					cv::aruco::estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);
					for (int i = 0; i < rvecs.size(); ++i)
					{
						auto rvec = rvecs[i];
						auto tvec = tvecs[i];
						cv::aruco::drawAxis(frames[i], cameraMatrix, distCoeffs, rvec, tvec, 0.1);
					}*/

					//Perform PNP
					std::vector<cv::Point2d> imagePoints;
					std::vector<cv::Point3d> objectPoints;
					for (int& id : ids)
					{
						for (unsigned x = 0; x < 4; x++)
						{
							objectPoints.push_back(objectCoordMap[id][x]);
						}
					}
					for (unsigned idx = 0; idx < ids.size(); idx++)
					{
						for (unsigned x = 0; x < 4; x++)
						{
							imagePoints.push_back(corners[idx][x]);
						}
					}

					cv::Mat cameraRVec, cameraTVec;
					cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, cameraRVec, cameraTVec);

					std::cout << "Camera Rotation:\n" << cameraRVec << "\n";
					std::cout << "Camera Translation:\n" << cameraTVec << "\n";

				}
				cv::aruco::drawDetectedMarkers(frames[i], corners, ids);
			}
			subtract_background(frames[i]);
			cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
			imshow("Display window", frames[i]);
			cv::waitKey(0);

		}	
		
	}
	catch (std::exception& ex)
	{
		std::cout << "Exception :" << ex.what() << std::endl;
	}
}
