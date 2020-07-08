//
// Created by seedship on 7/8/20.
//

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#define MAX_ARUCO_IMAGE_IDX 1023

int
main(int argc, char** argv)
{
	if (argc < 6)
	{
		std::cout << "Usage: ./" << argv[0] << " [X_RES] [Y_RES] [MARKER_RES] [INTERMARKER SPACING] [OUTPUT PATH]\n";
		return 0;
	}
	int x = atoi(argv[1]);
	int y = atoi(argv[2]);
	int d = atoi(argv[3]);
	int s = atoi(argv[4]);
	cv::Mat markerImage;
	cv::Mat output(y, x, CV_8UC1, 255);

	unsigned numX = x / (s + d);
	unsigned numY = y / (s + d);
	if ((s + d) * numX + d != x)
	{
		std::cout << "Illegal Dimensions: X dimension will have truncation!\n";
		return 0;
	}
	if ((s + d) * numY + d != y)
	{
		std::cout << "Illegal Dimensions: Y dimension will have truncation!\n";
		return 0;
	}

	int arUco_idx = 0;
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);

	//Putting markers on top row
	for (int id = 0; id < numX + 1; id++)
	{
		if (arUco_idx > MAX_ARUCO_IMAGE_IDX)
		{
			std::cout << "Too many markers\n";
		}
		cv::aruco::drawMarker(dictionary, arUco_idx, d, markerImage, 1);
		arUco_idx++;
		markerImage.copyTo(output(cv::Rect(id * (d + s), 0, markerImage.cols, markerImage.rows)));
	}

	//Putting markers on columns
	for (int id = 1; id < numY; id++)
	{
		if (id > MAX_ARUCO_IMAGE_IDX)
		{
			std::cout << "Too many markers\n";
		}
		cv::aruco::drawMarker(dictionary, arUco_idx, d, markerImage, 1);
		arUco_idx++;
		markerImage.copyTo(output(cv::Rect(0, id * (d + s), markerImage.cols, markerImage.rows)));
		cv::aruco::drawMarker(dictionary, arUco_idx, d, markerImage, 1);
		arUco_idx++;
		markerImage.copyTo(output(cv::Rect(x - 1 - d, id * (d + s), markerImage.cols, markerImage.rows)));
	}

	//Putting markers on bottom row
	for (int id = 0; id < numX + 1; id++)
	{
		if (arUco_idx > MAX_ARUCO_IMAGE_IDX)
		{
			std::cout << "Too many markers\n";
		}
		cv::aruco::drawMarker(dictionary, arUco_idx, d, markerImage, 1);
		arUco_idx++;
		markerImage.copyTo(output(cv::Rect(id * (d + s), y - d, markerImage.cols, markerImage.rows)));
	}
	std::cout << "Saving image as: " << argv[5] << "\n";
	cv::imwrite(argv[5], output);
}