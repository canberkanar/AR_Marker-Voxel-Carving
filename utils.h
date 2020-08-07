//
// Created by seedship on 7/19/20.
//

#ifndef HELLOAR_UTILS_H
#define HELLOAR_UTILS_H

#include <utility>
#include <unordered_map>
#include <opencv2/core/mat.hpp>
#include <opencv2/video.hpp>
#include <opencv2/aruco.hpp>


// First is RVec, 2nd is TVec
std::pair<cv::Mat, cv::Mat>
findCameraPos(const std::unordered_map<int, std::vector<cv::Point3d>>& objectCoordMap,
			  const std::vector<std::vector<cv::Point2f>>& corners, const std::vector<int>& ids,
			  const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs = cv::Mat());

std::pair<std::vector<int>, std::vector<std::vector<cv::Point2f>>>
detectArucoMarkers(const cv::Ptr<cv::aruco::Dictionary> dictionary, const cv::Mat& frame);

cv::Mat
subtractBackground(const cv::Mat& img, cv::Ptr<cv::BackgroundSubtractor> pBackSub);

std::pair<cv::Mat, cv::Mat>
readCameraConfigFromFile(const std::string& path);

cv::Mat
computeProjMat(const cv::Mat& camMat, const cv::Mat& rotVec, const cv::Mat& transVec);

//cv::Mat
//backgroundSubtraction()cv::Mat
//backgroundSubtraction()

#endif //HELLOAR_UTILS_H
