#include <iostream>
//
//#include <aruco/aruco.h>
//#include <aruco/markerdetector.h>

#include <aruco/aruco.h>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "PointsObjectCoord.h"
#include <vtkSmartPointer.h>
#include <vtkStructuredPoints.h>
#include <vtkPointData.h>
#include <vtkPLYWriter.h>
#include <vtkFloatArray.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkMarchingCubes.h>
#include <vtkCleanPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkAutoInit.h>
#include <opencv2/imgproc/types_c.h>

#include "utils.h"

VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);

using namespace cv;
using namespace std;
//Put in "Line;" to print the program line number
#define LINE std::cout<<__LINE__ << "\n"

int IMG_WIDTH = 504;
const int IMG_HEIGHT = 378;
const int VOXEL_DIM = 128;
const int VOXEL_SIZE = VOXEL_DIM * VOXEL_DIM * VOXEL_DIM;
const int NUM_IMAGE = 8;

struct voxel {
	float xpos;
	float ypos;
	float zpos;
	float res;
	float value;
};

struct coord {
	int x;
	int y;
};

struct startParams {
	float startX;
	float startY;
	float startZ;
	float voxelWidth;
	float voxelHeight;
	float voxelDepth;
};

struct camera {
	cv::Mat Image;
	cv::Mat P;
	cv::Mat K;
	cv::Mat R;
	cv::Mat t;
	cv::Mat Silhouette;
};

cv::Mat resizeImg(cv::Mat preimg) {
    cv::Mat img;
    double newwidth = ((double)IMG_HEIGHT / preimg.size().height) * preimg.size().width;
    IMG_WIDTH = newwidth;
    cv::Size s = cv::Size((int) newwidth, IMG_HEIGHT);
    cv::resize(preimg, img, s);
    return img;
}

coord project(camera cam, voxel v) {

	coord im;

	// voxel projection with to image
	float z = cam.P.at<float>(2, 0) * v.xpos +
			  cam.P.at<float>(2, 1) * v.ypos +
			  cam.P.at<float>(2, 2) * v.zpos +
			  cam.P.at<float>(2, 3);

	im.y = (cam.P.at<float>(1, 0) * v.xpos +
			cam.P.at<float>(1, 1) * v.ypos +
			cam.P.at<float>(1, 2) * v.zpos +
			cam.P.at<float>(1, 3)) / z;

	im.x = (cam.P.at<float>(0, 0) * v.xpos +
			cam.P.at<float>(0, 1) * v.ypos +
			cam.P.at<float>(0, 2) * v.zpos +
			cam.P.at<float>(0, 3)) / z;

	return im;
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
	return;
}

void carve(float fArray[], startParams params, camera cam) {

	cv::Mat silhouette, distImage;
	cv::threshold(cam.Silhouette, silhouette, 100, 255, THRESH_BINARY);
	//Calculates the distance to the closest zero pixel for each pixel of the source image.
	//using CV_DIST_L2 as 3rd argument
	cv::distanceTransform(silhouette, distImage, CV_DIST_L2, 3);
	// show images for debugging
	//cv::imshow("sil cam", cam.Silhouette);
	//cv::imshow("sil", silhouette);
	//cv::normalize(distImage, distImage, 0, 1.0, NORM_MINMAX);
	//cv::imshow("dist", distImage);

	for (int i = 0; i < VOXEL_DIM; i++) {
		for (int j = 0; j < VOXEL_DIM; j++) {
			for (int k = 0; k < VOXEL_DIM; k++) {

				/* calc voxel position inside camera view frustum */
				voxel v;
				v.xpos = params.startX + i * params.voxelWidth;
				v.ypos = params.startY + j * params.voxelHeight;
				v.zpos = params.startZ + k * params.voxelDepth;
				v.value = 1.0f;

				coord im = project(cam, v);
				//if pixel is not in the image
				float dist = -1.0f;

				/* test if projected voxel is within image coords */
				if (im.x > 0 && im.y > 0 && im.x < IMG_WIDTH && im.y < IMG_HEIGHT) {
					dist = distImage.at<float>(im.y, im.x);
					//Optional: filter out single pixels that are accidentally mapped to foreground
					//if (dist < 0.05)
						//dist = 0;
					if (cam.Silhouette.at<uchar>(im.y, im.x) == 0) {
						dist *= -1.0f;
					}
				}
				//carve away if part of background
				if (dist < fArray[i * VOXEL_DIM * VOXEL_DIM + j * VOXEL_DIM + k]) {
					fArray[i * VOXEL_DIM * VOXEL_DIM + j * VOXEL_DIM + k] = dist;
				}

			}
		}
	}

}

void renderModel(float fArray[], startParams params) {

	/* create vtk visualization pipeline from voxel grid (float array) */

	vtkSmartPointer<vtkStructuredPoints> sPoints = vtkSmartPointer<vtkStructuredPoints>::New();
	sPoints->SetDimensions(VOXEL_DIM, VOXEL_DIM, VOXEL_DIM);
	sPoints->SetSpacing(params.voxelDepth, params.voxelHeight, params.voxelWidth);
	sPoints->SetOrigin(params.startZ, params.startY, params.startX);

	vtkSmartPointer<vtkFloatArray> vtkFArray = vtkSmartPointer<vtkFloatArray>::New();
	vtkFArray->SetNumberOfValues(VOXEL_SIZE);
	vtkFArray->SetArray(fArray, VOXEL_SIZE, 1);

	sPoints->GetPointData()->SetScalars(vtkFArray);
	sPoints->GetPointData()->Update();

	//use marching cubes algorithm from lecture
	vtkSmartPointer<vtkMarchingCubes> mcSource = vtkSmartPointer<vtkMarchingCubes>::New();
	mcSource->SetInputData(sPoints);
	mcSource->SetNumberOfContours(1);
	mcSource->SetValue(0, 0.5);
	mcSource->Update();

	vtkSmartPointer<vtkCleanPolyData> cleanPolyData = vtkSmartPointer<vtkCleanPolyData>::New();
	cleanPolyData->SetInputConnection(mcSource->GetOutputPort());
	cleanPolyData->Update();
	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(cleanPolyData->GetOutputPort());

	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();

	actor->SetMapper(mapper);

	vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
	renderer->GradientBackgroundOn();
	renderer->SetBackground(.45, .45, .8);
	renderer->SetBackground2(.0, .0, .0);

	vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	renderWindowInteractor->SetRenderWindow(renderWindow);

	actor->GetProperty()->SetSpecular(0.2);
	renderer->AddActor(actor);

	//Render the scene
	renderWindow->Render();
	renderWindowInteractor->Start();
}

int main(int argc, char* argv[]) {
	std::vector<camera> cameras;


	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);

	Mat frames[NUM_IMAGE];
	
		for (int i = 0; i < NUM_IMAGE; i++) {
			std::stringstream path, path_bg;
			path << "../../../images/maya3/" << "img_" << i << ".jpg";
			path_bg << "../../../images/maya3/" << "bg_" << i << ".jpg";

			//path << "C:/Users/mayay/source/repos/ARVoxelCarving/images/maya3/" << "img_" << i << ".jpg";
			//path_bg << "C:/Users/mayay/source/repos/ARVoxelCarving/images/maya3/" << "bg_" << i << ".jpg";
			std::string image_path = cv::samples::findFile(path.str());
			std::string image_bg_path = cv::samples::findFile(path_bg.str());

			cv::Mat img = cv::imread(image_path);
			cv::Mat bg = cv::imread(image_bg_path);

			if (img.empty() || bg.empty())
			{
				std::cout << "Could not read the image: " << path.str() << std::endl;
				return 1;
			}
			frames[i] = resizeImg(img);

			cv::Mat silhouette;
			cv::Ptr<cv::BackgroundSubtractor> subtractor = cv::createBackgroundSubtractorMOG2();
			subtractor->apply(bg, silhouette);
			subtractor->apply(img, silhouette);
			//cvtColor(bg, silhouette, CV_BGR2GRAY);
			
			// Detect markers in frame
			std::vector<int> ids;
			std::vector<std::vector<cv::Point2f> > corners;
			cv::aruco::detectMarkers(frames[i], dictionary, corners, ids);

			if (!ids.empty())
			{
				
				::aruco::CameraParameters cam;
				cam.readFromXMLFile("../../../images/maya3/out_camera_data.xml");
				//cam.readFromXMLFile("C:/Users/mayay/source/repos/ARVoxelCarving/images/maya3/out_camera_data.xml");
				cv::Mat cameraMatrix = cam.CameraMatrix;
				cv::Mat distCoeffs = cam.Distorsion;

				//Uncomment to see the 3d orientations of the arUco markers
				/*std::vector<cv::Vec3d> rvecs, tvecs;
				cv::aruco::estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);
				for (int i = 0; i < rvecs.size(); ++i)
				{
					auto rvec = rvecs[i];
					auto tvec = tvecs[i];
					cv::aruco::drawAxis(frames[i], cameraMatrix, distCoeffs, rvec, tvec, 0.1);
				}*/

                auto cameraPos = findCameraPos(objectCoordMap, corners, ids, cameraMatrix, distCoeffs);

				std::cout << "Camera Matrix:\n" << cameraMatrix << "\n";
                std::cout << "Camera Rotation:\n" << cameraPos.first << "\n";
                std::cout << "Camera Translation:\n" << cameraPos.second << "\n";

				camera c;
				c.Image = resizeImg(img);
				c.K = cameraMatrix;
				c.R = cameraPos.first;
				c.t = cameraPos.second;
				hconcat(c.R, c.t, c.P);
				c.K.convertTo(c.K, CV_32FC1);
				c.P.convertTo(c.P, CV_32FC1);
				c.P = c.K * c.P;

				c.Silhouette = resizeImg(silhouette);
				cameras.push_back(c);

				
				cv::aruco::drawDetectedMarkers(frames[i], corners, ids);
			}
			subtract_background(frames[i]);

		}	
        /* bounding box dimensions of object */
        //Original dimensions:
        //float xmin = -6.21639, ymin = -10.2796, zmin = -14.0349;
        //float xmax = 7.62138, ymax = 12.1731, zmax = 12.5358;

		float xStart = 0, xEnd = 60;
		float yStart = 0, yEnd = 70;
        float zStart = 0, zEnd = 50;

        startParams params;
        //original:
        //params.startX = xmin - bbwidth;
        params.startX = xStart;
        //original:
        //params.startY = ymin - bbheight;
        params.startY = yStart;
        params.startZ = zStart;

		float bbwidth = std::abs(xEnd - xStart);
		float bbheight = std::abs(yEnd - yStart);
		float bbdepth = std::abs(zEnd - zStart);

        params.voxelWidth = bbwidth / VOXEL_DIM;
        params.voxelHeight = bbheight / VOXEL_DIM;
        params.voxelDepth = bbdepth / VOXEL_DIM;

		//define voxel grid
		float* fArray = new float[VOXEL_SIZE];
		std::fill_n(fArray, VOXEL_SIZE, 1000.0f);

		/* carving model for every given camera image */
		for (int i = 0; i < NUM_IMAGE; i++) {
			std::cout << cameras.at(i).P;
			std::cout << "\n";
			carve(fArray, params, cameras.at(i));
		}
		/* show example of segmented image */
		cv::Mat original, segmented;
		original = resizeImg(cameras.at(1).Image);
		segmented = resizeImg(cameras.at(1).Silhouette);
		cv::imshow("Object", original);
		cv::imshow("Silhouette", segmented);
        renderModel(fArray, params);
		cv::waitKey(0);
		return 0;
           
	}
