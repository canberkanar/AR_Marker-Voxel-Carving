//
// Created by seedship on 7/19/20.
//

#ifndef HELLOAR_UTILS_H
#define HELLOAR_UTILS_H

#include <utility>
#include <opencv2/core/mat.hpp>
#include <unordered_map>

// First is RVec, 2nd is TVec
std::pair<cv::Mat, cv::Mat>
findCameraPos(const std::unordered_map<int, std::vector<cv::Point3d>>& objectCoordMap,
			  const std::vector<std::vector<cv::Point2f>>& corners, const std::vector<int>& ids,
			  const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs = cv::Mat());


std::pair<std::vector<int>, std::vector<std::vector<cv::Point2f>>>
detectArucoMarkers(const cv::Ptr<cv::aruco::Dictionary> dictionary, const cv::Mat& frame);

//cv::Mat
//backgroundSubtraction()cv::Mat
//backgroundSubtraction()

#endif //HELLOAR_UTILS_H
