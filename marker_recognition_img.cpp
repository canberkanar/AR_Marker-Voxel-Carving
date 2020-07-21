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

int IMG_WIDTH = 640;
const int IMG_HEIGHT = 480;
const int VOXEL_DIM = 300;
const int VOXEL_SIZE = VOXEL_DIM * VOXEL_DIM * VOXEL_DIM;
const int VOXEL_SLICE = VOXEL_DIM * VOXEL_DIM;
const int OUTSIDE = 0;

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

	/* project voxel into camera image coords */
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
	/*ss << img.get(CAP_PROP_POS_FRAMES);
	string frameNumberString = ss.str();
	putText(img, frameNumberString.c_str(), cv::Point(15, 15),
		FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));*/
	//show the current frame and the fg masks
	//imshow("Frame", img);
	//imshow("FG Mask", fgMask);
	//cv::waitKey(0);

	return;
}

void carve(float fArray[], startParams params, camera cam) {

	cv::Mat silhouette, distImage;
	//edge detector, output in silhouette
	//cv::Canny(cam.Silhouette, silhouette, 0, 255);
	cv::threshold(cam.Silhouette, silhouette, 100, 255, THRESH_BINARY);
	//silhouette = cam.Silhouette;
	//inverts every bit
	//cv::bitwise_not(silhouette, silhouette);
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
				float dist = -1.0f;

				/* test if projected voxel is within image coords */
				if (im.x > 0 && im.y > 0 && im.x < IMG_WIDTH && im.y < IMG_HEIGHT) {
					dist = distImage.at<float>(im.y, im.x);
					if (cam.Silhouette.at<uchar>(im.y, im.x) == OUTSIDE) {
						dist *= -1.0f;
					}
				}

				if (dist < fArray[i * VOXEL_SLICE + j * VOXEL_DIM + k]) {
					fArray[i * VOXEL_SLICE + j * VOXEL_DIM + k] = dist;
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
	//sPoints->SetScalarTypeToFloat();

	vtkSmartPointer<vtkFloatArray> vtkFArray = vtkSmartPointer<vtkFloatArray>::New();
	vtkFArray->SetNumberOfValues(VOXEL_SIZE);
	vtkFArray->SetArray(fArray, VOXEL_SIZE, 1);

	sPoints->GetPointData()->SetScalars(vtkFArray);
	sPoints->GetPointData()->Update();
	//sPoints->Update();

	/* create iso surface with marching cubes algorithm */

	vtkSmartPointer<vtkMarchingCubes> mcSource = vtkSmartPointer<vtkMarchingCubes>::New();
	mcSource->SetInputData(sPoints);
	mcSource->SetNumberOfContours(1);
	mcSource->SetValue(0, 0.5);
	mcSource->Update();

	/* recreate mesh topology and merge vertices */

	vtkSmartPointer<vtkCleanPolyData> cleanPolyData = vtkSmartPointer<vtkCleanPolyData>::New();
	cleanPolyData->SetInputConnection(mcSource->GetOutputPort());
	cleanPolyData->Update();

	/* usual render stuff */

	vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
	renderer->SetBackground(.45, .45, .9);
	renderer->SetBackground2(.0, .0, .0);
	renderer->GradientBackgroundOn();

	vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	interactor->SetRenderWindow(renderWindow);

	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(cleanPolyData->GetOutputPort());
	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();

	actor->SetMapper(mapper);

	/* visible light properties */

	actor->GetProperty()->SetSpecular(0.15);
	actor->GetProperty()->SetInterpolationToPhong();
	renderer->AddActor(actor);

	renderWindow->Render();
	interactor->Start();
}

void exportModel(char* filename, vtkPolyData* polyData) {



	/* exports 3d model in ply format */

	vtkSmartPointer<vtkPLYWriter> plyExporter = vtkSmartPointer<vtkPLYWriter>::New();

	plyExporter->SetFileName(filename);

	plyExporter->SetInputData(polyData);

	plyExporter->Update();

	plyExporter->Write();

}

// For comparison
//int main(int argc, char* argv[]) {
//
//	/* acquire camera images, silhouettes and camera matrix */
//	std::vector<camera> cameras;
//	cv::FileStorage fs("/tmp/Voxel-Carving/assets/viff.xml", cv::FileStorage::READ);
//	for (int i=0; i<36; i++) {
//
//		/* camera image */
//		std::stringstream simg;
//		simg << "/tmp/Voxel-Carving/assets/image_" << i << ".jpg";
//		cv::Mat img = cv::imread(simg.str());
//
//		/* silhouette */
//		cv::Mat silhouette;
//		cv::cvtColor(img, silhouette, CV_BGR2HSV);
//		cv::inRange(silhouette, cv::Scalar(0, 0, 30), cv::Scalar(255,255,255), silhouette);
//
//		/* camera matrix */
//		std::stringstream smat;
//		smat << "viff" << std::setfill('0') << std::setw(3) << i << "_matrix";
//		cv::Mat P;
//		fs[smat.str()] >> P;
//
//		/* decompose proj matrix to cam- and rot matrix and trans vect */
//		cv::Mat K, R, t;
//		cv::decomposeProjectionMatrix(P, K, R, t);
//		K = cv::Mat::eye(3, 3, CV_32FC1);
//		K.at<float>(0,0) = 1680.2631413061415; /* fx */
//		K.at<float>(1,1) = 1676.1202869984309; /* fy */
//		K.at<float>(0,2) = 621.59194200994375; /* cx */
//		K.at<float>(1,2) = 467.7223229477861; /* cy */
//
//		camera c;
//		c.Image = img;
//		c.P = P;
//		std::cout << "P:\n" << P << "\n";
//		c.K = K;
//		std::cout << "K:\n" << K << "\n";
//		c.R = R;
//		std::cout << "R:\n" << R << "\n";
//		c.t = t;
//		std::cout << "T:\n" << t << "\n";
//		c.Silhouette = silhouette;
//
//		cameras.push_back(c);
//	}
//
//	/* bounding box dimensions of squirrel */
//	float xmin = -6.21639, ymin = -10.2796, zmin = -14.0349;
//	float xmax = 7.62138, ymax = 12.1731, zmax = 12.5358;
//
//	float bbwidth = std::abs(xmax-xmin)*1.15;
//	float bbheight = std::abs(ymax-ymin)*1.15;
//	float bbdepth = std::abs(zmax-zmin)*1.05;
//
//	startParams params;
//	params.startX = xmin-std::abs(xmax-xmin)*0.15;
//	params.startY = ymin-std::abs(ymax-ymin)*0.15;
//	params.startZ = 0.0f;
//	params.voxelWidth = bbwidth/VOXEL_DIM;
//	params.voxelHeight = bbheight/VOXEL_DIM;
//	params.voxelDepth = bbdepth/VOXEL_DIM;
//
//	/* 3 dimensional voxel grid */
//	float *fArray = new float[VOXEL_SIZE];
//	std::fill_n(fArray, VOXEL_SIZE, 1000.0f);
//
//	/* carving model for every given camera image */
//	for (int i=0; i<36; i++) {
//		carve(fArray, params, cameras.at(i));
//	}
//
//	/* show example of segmented image */
//	cv::Mat original, segmented;
//	cv::resize(cameras.at(1).Image, original, cv::Size(640, 480));
//	cv::resize(cameras.at(1).Silhouette, segmented, cv::Size(640, 480));
//	cv::imshow("Squirrel" , original);
//	cv::imshow("Squirrel Silhouette", segmented);
//
//	renderModel(fArray, params);
//
//	return 0;
//}


int main(int argc, char* argv[]) {
	std::vector<camera> cameras;


	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);

	Mat frames[7];
	//auto camParams = readCameraConfigFromFile("../images/bosebag/out_camera_data.xml");
	//std::cout<< "Camera Matrix\n" << camParams.first << "\n";
	//std::cout<< "Distortion Coeffs\n" << camParams.second << "\n";

		for (int i = 0; i < 7; i++) {
			std::stringstream path, path_bg;
			path << "../../../images/" << "image_" << (i + 1) << ".jpg";
			path_bg << "../../../images/" << "image_" << (i + 1) << ".jpg";
			//path << "C:/Users/mayay/source/repos/ARVoxelCarving/images/" << "image_" << (i+8) << ".jpg";
			//path_bg << "C:/Users/mayay/source/repos/ARVoxelCarving/images/" << "image_" << (i+1) << ".jpg";
			std::string image_path = cv::samples::findFile(path.str());
			std::string image_bg_path = cv::samples::findFile(path_bg.str());

			cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
			cv::Mat bg = cv::imread(image_bg_path, cv::IMREAD_COLOR);

			if (img.empty() || bg.empty())
			{
				std::cout << "Could not read the image: " << path.str() << std::endl;
				return 1;
			}
			frames[i] = resizeImg(img);

			/* silhouette */
			
			cv::Mat silhouette;
			//using CV_BGR2HSV for 40
			//cv::cvtColor(img, silhouette, 40);
			//Checks if silhouette array elements lie between the (0,0,30) and 255,255,255
			//cv::inRange(silhouette, cv::Scalar(0, 0, 30), cv::Scalar(255, 255, 255), silhouette);
			cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorMOG2();
			pBackSub->apply(bg, silhouette);
			pBackSub->apply(img, silhouette);
			
			// Detect markers in frame
			std::vector<int> ids;
			std::vector<std::vector<cv::Point2f> > corners;
			cv::aruco::detectMarkers(frames[i], dictionary, corners, ids);

			if (!ids.empty())
			{
				
				::aruco::CameraParameters cam;
				cam.readFromXMLFile("../../../extern/out_camera_data.xml");
				//cam.readFromXMLFile("C:/Users/mayay/source/repos/ARVoxelCarving/extern/out_camera_data.xml");
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
			//cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
			//imshow("Display window", frames[i]);
			//cv::waitKey(0);

		}	
        /* bounding box dimensions of object */
        //Original dimensions:
        //float xmin = -6.21639, ymin = -10.2796, zmin = -14.0349;
        //float xmax = 7.62138, ymax = 12.1731, zmax = 12.5358;

        float xmin = -200, ymin = -150, zmin = -100;
        float xmax = 200, ymax = 150, zmax = 100;

        float bbwidth = std::abs(xmax - xmin) * 1.15;
        float bbheight = std::abs(ymax - ymin) * 1.15;
        float bbdepth = std::abs(zmax - zmin) * 1.05;

        startParams params;
        //original:
        //params.startX = xmin - bbwidth;
        params.startX = xmin;
        //original:
        //params.startY = ymin - bbheight;
        params.startY = ymin;
        params.startZ = 0.0f;
		//params.startZ = zmin-bbdepth;
        params.voxelWidth = bbwidth / VOXEL_DIM;
        params.voxelHeight = bbheight / VOXEL_DIM;
        params.voxelDepth = bbdepth / VOXEL_DIM;
		/* 3 dimensional voxel grid */
		float* fArray = new float[VOXEL_SIZE];
		std::fill_n(fArray, VOXEL_SIZE, 1000.0f);
		//unsigned char* fArray = new unsigned char[VOXEL_SIZE];
		/* carving model for every given camera image */
		for (int i = 0; i < 7; i++) {
			std::cout << cameras.at(i).P;
			std::cout << "\n";
			carve(fArray, params, cameras.at(i));
		}
        for (int i = 128 * 128 * 70; i < 128 * 128 * 70 + 200; i++) {
            std::cout << fArray[i];
            std::cout << "\n";
        }
		/* show example of segmented image */
		cv::Mat original, segmented;
		original = resizeImg(cameras.at(4).Image);
		segmented = resizeImg(cameras.at(4).Silhouette);
		cv::imshow("Squirrel", original);
		cv::imshow("Squirrel Silhouette", segmented);
        renderModel(fArray, params);
		cv::waitKey(0);
		return 0;
           
	}
