//
// Created by seedship on 7/19/20.
//
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <aruco/aruco.h>

#include "utils.h"

std::pair<cv::Mat, cv::Mat>
findCameraPos(const std::unordered_map<int, std::vector<cv::Point3d>>& objectCoordMap,
			  const std::vector<std::vector<cv::Point2f>>& corners, const std::vector<int>& ids,
			  const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs)
{
	std::vector<cv::Point2d> imagePoints;
	std::vector<cv::Point3d> objectPoints;
	for (const int& id : ids)
	{
		for (unsigned x = 0; x < 4; x++)
		{
			objectPoints.push_back(objectCoordMap.at(id)[x]);
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

	// Must convert from camera translation to world translation: https://stackoverflow.com/questions/18637494/camera-position-in-world-coordinate-from-cvsolvepnp
//	std::cout  << "Translation:\n" << cameraTVec << "\n";
//	std::cout  << "Translation:\n" << cameraTVec.size << "\n";

	cv::Mat pose = cv::Mat::eye(4, 4, cameraRVec.type());
//	std::cout << "Initial pose:\n" << pose << "\n";

	cv::Mat rotation;
	cv::Rodrigues(cameraRVec, rotation);

//	std::cout << "Rotation:\n" << rotation << "\n";

	for (int x = 0; x < 3; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			pose.at<double>(x, y) = rotation.at<double>(x, y);
		}
	}
//	std::cout  << "Assigning rotation:\n" << pose << "\n";

	for (int x = 0; x < 3; x++)
	{
		pose.at<double>(x, 3) = cameraTVec.at<double>(x, 0);
	}
//	std::cout  << "Assigning Translation:\n" << pose << "\n";

	pose = pose.inv();
//	std::cout  << "Inversion:\n" << pose << "\n";

	for (int x = 0; x < 3; x++)
	{
		cameraTVec.at<double>(x, 0) = pose.at<double>(x, 3);
	}

	rotation = pose(cv::Range(0, 3), cv::Range(0, 3));
//	std::cout  << "Extracted Rotation:\n" << rotation << "\n";


	cv::Rodrigues(rotation, cameraRVec);

//	std::cout << "Translation: " << cameraTVec << "\n";
//	std::cout << "Rotation: " << cameraRVec << "\n";


	return std::pair<cv::Mat, cv::Mat>(cameraRVec, cameraTVec);
}

std::pair<std::vector<int>, std::vector<std::vector<cv::Point2f>>>
detectArucoMarkers(const cv::Ptr<cv::aruco::Dictionary> dictionary, const cv::Mat& frame)
{
	std::vector<int> ids;
	std::vector<std::vector<cv::Point2f>> corners;
	cv::aruco::detectMarkers(frame, dictionary, corners, ids);
	return std::pair<std::vector<int>, std::vector<std::vector<cv::Point2f>>>(ids, corners);
}

cv::Mat
subtractBackground(const cv::Mat& img, cv::Ptr<cv::BackgroundSubtractor> pBackSub)
{

	//create Background Subtractor objects
	cv::Mat fgMask;

	//update the background model
	pBackSub->apply(img, fgMask);

	return fgMask;
}

std::pair<cv::Mat, cv::Mat>
readCameraConfigFromFile(const std::string& path)
{
	::aruco::CameraParameters camera;
	camera.readFromXMLFile(path);
	cv::Mat cameraMatrix = camera.CameraMatrix;
	cv::Mat distCoeffs = camera.Distorsion;
	return std::pair<cv::Mat, cv::Mat>(cameraMatrix, distCoeffs);
}

//https://answers.opencv.org/question/162932/create-a-stereo-projection-matrix-using-rvec-and-tvec/
cv::Mat
computeProjMat(const cv::Mat& camMat, const cv::Mat& rotVec, const cv::Mat& transVec)
{
	cv::Mat rotMat(3, 3, CV_64F), rotTransMat(3, 4, CV_64F); //Init.
	//Convert rotation vector into rotation matrix
	cv::Rodrigues(rotVec, rotMat);
	//Append translation vector to rotation matrix
	cv::hconcat(rotMat, transVec, rotTransMat);
	//Compute projection matrix by multiplying intrinsic parameter
	//matrix (A) with 3 x 4 rotation and translation pose matrix (RT).
	//Formula: Projection Matrix = A * RT;
//	std::cout<< "camMat type " << camMat.type() << "\n";
	rotTransMat.convertTo(rotTransMat, camMat.type());
//	std::cout<< "rotTransMat type " << rotTransMat.type() << "\n";
//	std::cout<< "rotTransMat:\n" << rotTransMat << "\n";
	return (camMat * rotTransMat);
}