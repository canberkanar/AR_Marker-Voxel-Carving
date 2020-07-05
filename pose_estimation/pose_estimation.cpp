#include <iostream>

#include <aruco/aruco.h>
#include <aruco/markerdetector.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>


using namespace cv;
using namespace std;

static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}

int main(int argc, char** argv) {

    cv::VideoCapture inputVideo;
    inputVideo.open(2);
    cv::Mat cameraMatrix, distCoeffs;


    // camera parameters are read from somewhere
    readCameraParameters("/home/canberk/Downloads/3d/AR_Marker-Voxel-Carving/pose_estimation/calibration_params.yml", cameraMatrix, distCoeffs);

    std::cout << "camera_matrix\n" << cameraMatrix << std::endl;
    std::cout << "\ndist coeffs\n" << distCoeffs << std::endl;

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        cv::aruco::detectMarkers(image, dictionary, corners, ids);
        // if at least one marker detected
        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);
            // draw axis for each marker
            for(int i=0; i<ids.size(); i++)
                cv::aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
        }
        cv::imshow("out", imageCopy);
        char key = (char) cv::waitKey(100);
        if (key == 27)
            break;
    }

}


