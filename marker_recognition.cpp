#include <iostream>
//
//#include <aruco/aruco.h>
//#include <aruco/markerdetector.h>

#include <aruco/aruco.h>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "PointsObjectCoord.h"

//Put in "Line;" to print the program line number
#define LINE std::cout<<__LINE__ << "\n"

int
main(int argc, char** argv)
{
	try
	{
		if (argc < 2)
		{
			std::cout << "Usage: ./" << argv[0] << " [Camera Num] [Optional: Camera Parameters XML]\n";
			return 1;
		}

		int camNum = atoi(argv[1]);

		cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);
		// Read the web cam

		cv::Mat frame;
		cv::VideoCapture cap;

		if (argc >= 4)
		{
			int dimX = atoi(argv[2]);
			int dimY = atoi(argv[3]);
			cap.set(cv::CAP_PROP_FRAME_WIDTH, dimX);
			cap.set(cv::CAP_PROP_FRAME_HEIGHT, dimY);
			if (argc == 5)
			{
				int fps = atoi(argv[4]);
				cap.set(cv::CAP_PROP_FPS, fps);
			}
		}

		if (!cap.open(camNum, cv::CAP_ANY))
			std::cout << "Could not open camera!\n";
		while (cap.isOpened() && cap.read(frame))
		{
			// Detect markers in frame
			std::vector<int> ids;
			std::vector<std::vector<cv::Point2f> > corners;
			cv::aruco::detectMarkers(frame, dictionary, corners, ids);

			// if at least one marker detected
			if (!ids.empty())
			{
				if (argc == 3)
				{
					::aruco::CameraParameters camera;
					camera.readFromXMLFile(argv[2]);
					cv::Mat cameraMatrix = camera.CameraMatrix;
					cv::Mat distCoeffs = camera.Distorsion;

					//Uncomment to see the 3d orientations of the arUco markers
//					std::vector<cv::Vec3d> rvecs, tvecs;
//					cv::aruco::estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);
//					for (int i = 0; i < rvecs.size(); ++i)
//					{
//						auto rvec = rvecs[i];
//						auto tvec = tvecs[i];
//						cv::aruco::drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
//					}

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
				cv::aruco::drawDetectedMarkers(frame, corners, ids);
			}
			cv::imshow("out", frame);
			char key = (char) cv::waitKey(10);
			if (key == 27)
				break;
		}
	}
	catch (std::exception& ex)
	{
		std::cout << "Exception :" << ex.what() << std::endl;
	}

}
