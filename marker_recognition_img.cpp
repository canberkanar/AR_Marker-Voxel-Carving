#include <iostream>
//
//#include <aruco/aruco.h>
//#include <aruco/markerdetector.h>

#include <aruco/aruco.h>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
//#include <direct.h>

#include <opencv2/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/imgproc/types_c.h>

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
#include <vtkOutlineFilter.h>
#include <vtkSphereSource.h>

VTK_MODULE_INIT(vtkRenderingOpenGL2);

VTK_MODULE_INIT(vtkInteractionStyle);

#define N_OF_IMAGES 7

using namespace cv;
using namespace std;
//Put in "Line;" to print the program line number
#define LINE std::cout<<__LINE__ << "\n"

int getIndexMarker(int id, std::vector<int> ids) {
    for (int i = 0; i < ids.size(); i++) {
        if (id == ids[i])
            return i;
    }

    return -1;
}

int IMG_WIDTH = 1280;
const int IMG_HEIGHT = 540;
const int VOXEL_DIM = 128;
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
    double newwidth = (540.0 / preimg.size().height) * preimg.size().width;
    IMG_WIDTH = newwidth;
    cv::Size s = cv::Size((int) newwidth, 540);
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
    cv::Canny(cam.Silhouette, silhouette, 0, 255);
    //inverts every bit
    cv::bitwise_not(silhouette, silhouette);
    //Calculates the distance to the closest zero pixel for each pixel of the source image.
    //using CV_DIST_L2 as 3rd argument
    cv::distanceTransform(silhouette, distImage, 2, 3);

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
//    sPoints->SetSpacing(params.voxelDepth, params.voxelHeight, params.voxelWidth);
//    sPoints->SetOrigin(params.startZ, params.startY, params.startX);
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

    // Create a sphere
    vtkSmartPointer<vtkSphereSource> sphereSource =
            vtkSmartPointer<vtkSphereSource>::New();
    sphereSource->SetCenter(0.0, 0.0, 0.0);
    sphereSource->SetRadius(5.0);
    sphereSource->Update();

    vtkPolyData *sphere = sphereSource->GetOutput();
    // Create the outline
    vtkSmartPointer<vtkOutlineFilter> outline =
            vtkSmartPointer<vtkOutlineFilter>::New();
#if VTK_MAJOR_VERSION <= 5
    outline->SetInput(sphere);
#else
    outline->SetInputData(sphere);
#endif
    vtkSmartPointer<vtkPolyDataMapper> outlineMapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    outlineMapper->SetInputConnection(outline->GetOutputPort());
    vtkSmartPointer<vtkActor> outlineActor =
            vtkSmartPointer<vtkActor>::New();
    outlineActor->SetMapper(outlineMapper);
    outlineActor->GetProperty()->SetColor(0, 0, 0);


    /* visible light properties */

    actor->GetProperty()->SetSpecular(0.15);
    actor->GetProperty()->SetInterpolationToPhong();

    // Add the actors to the scene
    renderer->AddActor(actor);
    renderer->AddActor(outlineActor);
    renderer->SetBackground(1,1,1); // Background color white

    renderWindow->Render();
    interactor->Start();
}

void exportModel(char *filename, vtkPolyData *polyData) {



    /* exports 3d model in ply format */

    vtkSmartPointer<vtkPLYWriter> plyExporter = vtkSmartPointer<vtkPLYWriter>::New();

    plyExporter->SetFileName(filename);

    plyExporter->SetInputData(polyData);

    plyExporter->Update();

    plyExporter->Write();

}


int main(int argc, char **argv) {
    try {
        std::vector<camera> cameras;


        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);

        Mat frames[N_OF_IMAGES];

        for (int i = 0; i < N_OF_IMAGES; i++) {
            std::stringstream path;
            path << "images/" << "image_" << (i + 1) << ".jpg";
            std::string image_path = cv::samples::findFile(path.str());
            cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cout << "Could not read the image: " << path.str() << std::endl;
                return 1;
            }
            frames[i] = resizeImg(img);


            /* silhouette */

            cv::Mat silhouette;
            //using CV_BGR2HSV for 40
            cv::cvtColor(img, silhouette, 40);
            //Checks if silhouette array elements lie between the (0,0,30) and 255,255,255
            cv::inRange(silhouette, cv::Scalar(0, 0, 30), cv::Scalar(255, 255, 255), silhouette);


            // Detect markers in frame
            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f> > corners;
            cv::aruco::detectMarkers(frames[i], dictionary, corners, ids);

            if (!ids.empty()) {
//                if (getIndexMarker(0, ids) != -1 && getIndexMarker((15), ids) != -1 && getIndexMarker((5), ids) != -1 &&
//                    getIndexMarker((10), ids) != -1) {
//
//                    int m1 = getIndexMarker(0, ids);
//                    int m2 = getIndexMarker((15), ids);
//                    int m3 = getIndexMarker((5), ids);
//                    int m4 = getIndexMarker((10), ids);
//
//                    cv::Point2f a1 = corners[m1][2];
//                    cv::Point2f a2 = corners[m2][0];
//                    cv::Point2f a3 = corners[m3][3];
//                    cv::Point2f a4 = corners[m4][1];
//
//                    vector<cv::Point> point;
//
//                    point.push_back(Point(a1.x, a1.y));
//                    point.push_back(Point(a3.x, a3.y));
//                    point.push_back(Point(a2.x, a2.y));
//                    point.push_back(Point(a4.x, a4.y));
//
//                    cvtColor(frames[i], frames[i], CV_BGR2GRAY);
//
//                    // Mask is black with white where our ROI is
//                    Mat mask = Mat::zeros(frames[i].rows, frames[i].cols, CV_8UC1);
//                    vector<vector<Point>> pts{point};
//                    fillPoly(mask, pts, Scalar(255, 255, 255));
//
//                    cv::Mat white_background(frames[i].rows, frames[i].cols, CV_8UC1, cv::Scalar(255, 255, 255));
//                    cv::bitwise_and(frames[i], mask, white_background, mask);
//
//                    cv::cvtColor(white_background, white_background, CV_GRAY2RGB);
//
//                    cv::cvtColor(white_background, silhouette, 40);
//                    //Checks if silhouette array elements lie between the (0,0,30) and 255,255,255
//                    cv::inRange(silhouette, cv::Scalar(0, 0, 30), cv::Scalar(255, 255, 255), silhouette);
//
////                    imshow("Display window", silhouette);
////                    cv::waitKey(0);
//                }

                ::aruco::CameraParameters cam;
                cam.readFromXMLFile("extern/out_camera_data.xml");
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

                //Perform PNP
                std::vector<cv::Point2d> imagePoints;
                std::vector<cv::Point3d> objectPoints;
                for (int &id : ids) {
                    for (unsigned x = 0; x < 4; x++) {
                        objectPoints.push_back(objectCoordMap[id][x]);
                    }
                }
                for (unsigned idx = 0; idx < ids.size(); idx++) {
                    for (unsigned x = 0; x < 4; x++) {
                        imagePoints.push_back(corners[idx][x]);
                    }
                }

                cv::Mat cameraRVec, cameraTVec;
                cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, cameraRVec, cameraTVec);

                std::cout << "Camera Rotation:\n" << cameraRVec << "\n";
                std::cout << "Camera Translation:\n" << cameraTVec << "\n";


                camera c;
                c.Image = img;
                c.R = cameraMatrix;
                c.t = cameraTVec;
                c.t.convertTo(c.t, 5);
                hconcat(c.R, c.t, c.P);
                cv::Mat lowerRank = cv::Mat(1, 4, CV_32F, {0, 0, 0, 1});
                vconcat(c.P, lowerRank, c.P);
                c.Silhouette = silhouette;
                cameras.push_back(c);

                cv::aruco::drawDetectedMarkers(frames[i], corners, ids);
            }

            subtract_background(frames[i]);
            //cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
            //imshow("Display window", frames[i]);
            //cv::waitKey(0);

        }
        //NEEWW
        /* bounding box dimensions of squirrel */
        float xmin = -550, ymin = -350, zmin = -250;
        float xmax = 550, ymax = 350, zmax = 250;

        float bbwidth = std::abs(xmax - xmin) * 1.15;
        float bbheight = std::abs(ymax - ymin) * 1.15;
        float bbdepth = std::abs(zmax - zmin) * 1.05;

        startParams params;
        params.startX = xmin - std::abs(xmax - xmin) * 0.15;
        params.startY = ymin - std::abs(ymax - ymin) * 0.15;
        params.startZ = 0.0f;
        params.voxelWidth = bbwidth / VOXEL_DIM;
        params.voxelHeight = bbheight / VOXEL_DIM;
        params.voxelDepth = bbdepth / VOXEL_DIM;

        /* 3 dimensional voxel grid */
        float *fArray = new float[VOXEL_SIZE];
        std::fill_n(fArray, VOXEL_SIZE, 1000.0f);

        /* carving model for every given camera image */
        for (int i = 0; i < N_OF_IMAGES; i++) {
            std::cout << cameras.at(i).P;
            carve(fArray, params, cameras.at(i));
        }


        /* show example of segmented image */
        cv::Mat original, segmented;
        original = resizeImg(cameras.at(4).Image);
        segmented = resizeImg(cameras.at(4).Silhouette);
        cv::imshow("Squirrel", original);
        cv::imshow("Squirrel Silhouette", segmented);

        renderModel(fArray, params);
        cv::waitKey(0);
        //NEWEND
    }
    catch (std::exception &ex) {
        std::cout << "Exception :" << ex.what() << std::endl;
    }
}
