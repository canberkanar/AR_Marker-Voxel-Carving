//
// Created by seedship on 7/9/20.
//

#ifndef HELLOAR_POINTSOBJECTCOORD_H
#define HELLOAR_POINTSOBJECTCOORD_H

#include <unordered_map>
#include <opencv2/opencv.hpp>
// Please note that this data is specifically for the image: images/1100_700_100_100.png
// It is generated by executing ./marker_generation 1100 700 100 100 out.png and then adding 100 pixels of padding on all 4 sides
// NOTE: each entry in the map corresponds to the 4 corners of the arUco marker. We define the top left corner of voxel 0 to be (0,0,0) in object space
// The first coordinate is the top left corner. The remaining 3 corners have been defined to be 100 away in the +x and +y directions. They are added in clockwise order.
// Perhaps we could not specify the last 3 coordinates and simply add +100 to the x and y coordinates of the 1st coordinate, but I wrote them in for convienence.
// All Z coordinates are set to 0, as we consider the surface that has the arUco markers to be our XY plane.

// I have arbitrarily defined the distances between voxels to be 100, because in the original image they are 100 pixels apart. We may need to find new offsets.
// It seems that moving the camera 1 cm in the X-Y direction produces a different distance change than moving it in the Z direction.
static std::unordered_map<int, std::vector<cv::Point3d>> objectCoordMap {{0, {cv::Point3d(0, 0, 0), cv::Point3d(100, 0, 0), cv::Point3d(100, 100, 0), cv::Point3d(0, 100, 0)}},
															{1, {cv::Point3d(200, 0, 0), cv::Point3d(300, 0, 0), cv::Point3d(300, 100, 0), cv::Point3d(200, 100, 0)}},
															{2, {cv::Point3d(400, 0, 0), cv::Point3d(500, 0, 0), cv::Point3d(500, 100, 0), cv::Point3d(400, 100, 0)}},
															{3, {cv::Point3d(600, 0, 0), cv::Point3d(700, 0, 0), cv::Point3d(700, 100, 0), cv::Point3d(600, 100, 0)}},
															{4, {cv::Point3d(800, 0, 0), cv::Point3d(900, 0, 0), cv::Point3d(900, 100, 0), cv::Point3d(800, 100, 0)}},
															{5, {cv::Point3d(1000, 0, 0), cv::Point3d(1100, 0, 0), cv::Point3d(1100, 100, 0), cv::Point3d(1000, 100, 0)}},
															{6, {cv::Point3d(0, 200, 0), cv::Point3d(100, 200, 0), cv::Point3d(100, 300, 0), cv::Point3d(0, 300, 0)}},
															{7, {cv::Point3d(1000, 200, 0), cv::Point3d(1100, 200, 0), cv::Point3d(1100, 300, 0), cv::Point3d(1000, 300, 0)}},
															{8, {cv::Point3d(0, 400, 0), cv::Point3d(100, 400, 0), cv::Point3d(100, 500, 0), cv::Point3d(0, 500, 0)}},
															{9, {cv::Point3d(1000, 400, 0), cv::Point3d(1100, 400, 0), cv::Point3d(1100, 500, 0), cv::Point3d(1000, 500, 0)}},
															{10, {cv::Point3d(0, 600, 0), cv::Point3d(100, 600, 0), cv::Point3d(100, 700, 0), cv::Point3d(0, 600, 0)}},
															{11, {cv::Point3d(200, 600, 0), cv::Point3d(300, 600, 0), cv::Point3d(300, 700, 0), cv::Point3d(200, 700, 0)}},
															{12, {cv::Point3d(400, 600, 0), cv::Point3d(500, 600, 0), cv::Point3d(500, 700, 0), cv::Point3d(400, 700, 0)}},
															{13, {cv::Point3d(600, 600, 0), cv::Point3d(700, 600, 0), cv::Point3d(700, 700, 0), cv::Point3d(600, 700, 0)}},
															{14, {cv::Point3d(800, 600, 0), cv::Point3d(900, 600, 0), cv::Point3d(900, 700, 0), cv::Point3d(800, 700, 0)}},
															{15, {cv::Point3d(1000, 600, 0), cv::Point3d(1100, 600, 0), cv::Point3d(1100, 700, 0), cv::Point3d(1000, 700, 0)}}
};

#endif //HELLOAR_POINTSOBJECTCOORD_H
