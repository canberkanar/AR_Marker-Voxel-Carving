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

// Resize the image for showing it in the opencv window
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

	// Project voxel to image with the projection matrix
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

void carve(float fArray[], startParams params, camera cam) {

	cv::Mat silhouette, distImage;
	cv::threshold(cam.Silhouette, silhouette, 100, 255, THRESH_BINARY);

	// Calculates the signed distance values for each pixel of the source image.
	cv::distanceTransform(silhouette, distImage, CV_DIST_L2, 3);

	for (int i = 0; i < VOXEL_DIM; i++) {
		for (int j = 0; j < VOXEL_DIM; j++) {
			for (int k = 0; k < VOXEL_DIM; k++) {

				// Calculate voxel position 
				voxel v;
				v.xpos = params.startX + i * params.voxelWidth;
				v.ypos = params.startY + j * params.voxelHeight;
				v.zpos = params.startZ + k * params.voxelDepth;
				v.value = 1.0f;

				// Project 3d voxel point to 2d
				coord im = project(cam, v);

				float dist = -1.0f;

				// Check if projected voxel is within image coords
				if (im.x > 0 && im.y > 0 && im.x < IMG_WIDTH && im.y < IMG_HEIGHT) {
					dist = distImage.at<float>(im.y, im.x);
					// Optional: filter out single pixels that are accidentally mapped to foreground
					//if (dist < 0.05)
						//dist = 0;
					// If pixel is in the background
					if (cam.Silhouette.at<uchar>(im.y, im.x) == 0) {
						dist *= -1.0f;
					}
				}
				// Update signed distance values
				if (dist < fArray[i * VOXEL_DIM * VOXEL_DIM + j * VOXEL_DIM + k]) {
					fArray[i * VOXEL_DIM * VOXEL_DIM + j * VOXEL_DIM + k] = dist;
				}

			}
		}
	}

}

void renderModel(float fArray[], startParams params) {

	// Create vtk visualization pipeline from voxel grid 
	vtkSmartPointer<vtkStructuredPoints> sPoints = vtkSmartPointer<vtkStructuredPoints>::New();
	sPoints->SetDimensions(VOXEL_DIM, VOXEL_DIM, VOXEL_DIM);
	sPoints->SetSpacing(params.voxelDepth, params.voxelHeight, params.voxelWidth);
	sPoints->SetOrigin(params.startZ, params.startY, params.startX);

	vtkSmartPointer<vtkFloatArray> vtkFArray = vtkSmartPointer<vtkFloatArray>::New();
	vtkFArray->SetNumberOfValues(VOXEL_SIZE);
	vtkFArray->SetArray(fArray, VOXEL_SIZE, 1);

	sPoints->GetPointData()->SetScalars(vtkFArray);
	sPoints->GetPointData()->Update();

	// Use marching cubes algorithm from lecture
	vtkSmartPointer<vtkMarchingCubes> mcSource = vtkSmartPointer<vtkMarchingCubes>::New();
	mcSource->SetInputData(sPoints);
	mcSource->SetNumberOfContours(1);
	mcSource->SetValue(0, 0.5);
	mcSource->Update();

	// Clean mesh topology: remove unused points and merge duplicates
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

	// Render the carved model
	renderWindow->Render();
	renderWindowInteractor->Start();
}

int main(int argc, char* argv[]) {
	std::vector<camera> cameras;


	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);

	Mat frames[NUM_IMAGE];
	
		for (int i = 0; i < NUM_IMAGE; i++) {
			std::stringstream path, path_bg;
			path << "../../../images/final_db/" << "img_" << i << ".jpg";
			path_bg << "../../../images/final_db/" << "bg_" << i << ".jpg";

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

			// Apply background segmentation on image
			// This step can be improved. Optionally, images can be converted to grayscale images
			cv::Mat silhouette;
			cv::Ptr<cv::BackgroundSubtractor> subtractor = cv::createBackgroundSubtractorMOG2();
			subtractor->apply(bg, silhouette);
			subtractor->apply(img, silhouette);
			
			// Detect markers in frame
			std::vector<int> ids;
			std::vector<std::vector<cv::Point2f> > corners;
			cv::aruco::detectMarkers(frames[i], dictionary, corners, ids);

			if (!ids.empty())
			{
				// Read camera matrix obtained by camera calibration
				::aruco::CameraParameters cam;
				cam.readFromXMLFile("../../../images/final_db/out_camera_data.xml");
				cv::Mat cameraMatrix = cam.CameraMatrix;
				cv::Mat distCoeffs = cam.Distorsion;

				// Estimate camera pose with camera matrix and detected marker coordinates
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
				//Calculate the projection matrix
				c.P = c.K * c.P;

				c.Silhouette = resizeImg(silhouette);
				cameras.push_back(c);
			}
		}	

        //Define bounding box dimensions of object
		float xStart = 0, xEnd = 60;
		float yStart = 0, yEnd = 70;
        float zStart = 0, zEnd = 50;

		// Parameters for the dimension of voxel grid
        startParams params;
        params.startX = xStart;
        params.startY = yStart;
        params.startZ = zStart;

		float bbwidth = std::abs(xEnd - xStart);
		float bbheight = std::abs(yEnd - yStart);
		float bbdepth = std::abs(zEnd - zStart);

        params.voxelWidth = bbwidth / VOXEL_DIM;
        params.voxelHeight = bbheight / VOXEL_DIM;
        params.voxelDepth = bbdepth / VOXEL_DIM;

		// Define voxel grid
		float* fArray = new float[VOXEL_SIZE];
		std::fill_n(fArray, VOXEL_SIZE, 1000.0f);

		// Carve model for every given camera image
		for (int i = 0; i < NUM_IMAGE; i++) {
			std::cout << cameras.at(i).P;
			std::cout << "\n";
			carve(fArray, params, cameras.at(i));
		}

		// Show example of segmented image
		cv::Mat original, segmented;
		original = resizeImg(cameras.at(1).Image);
		segmented = resizeImg(cameras.at(1).Silhouette);
		cv::imshow("Object", original);
		cv::imshow("Silhouette", segmented);

		// Render carved model
        renderModel(fArray, params);
		cv::waitKey(0);
		return 0;
           
	}
