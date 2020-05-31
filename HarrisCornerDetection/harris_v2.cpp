#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat createRMatrix(Mat grX, Mat grY)
{
	Mat iX2 = Mat::zeros(grX.rows, grX.cols, CV_32F);
	Mat iY2 = Mat::zeros(grY.rows, grY.cols, CV_32F);
	Mat iXY = Mat::zeros(grY.rows, grY.cols, CV_32F);

	multiply(grX, grX, iX2);
	multiply(grY, grY, iY2);
	multiply(grX, grY, iXY);

	Mat result = Mat::zeros(grY.rows, grY.cols, CV_32F);

	GaussianBlur(iX2, iX2, Size(5,5), 1.732);
	GaussianBlur(iY2, iY2, Size(5,5), 1.732);
	GaussianBlur(iXY, iXY, Size(5,5), 1.732);

	double detC;
	double trace;
	float kr = 0.04;

	for(int i = 0; i < result.rows; i++)
	{
		for(int j = 0; j < result.cols; j++)
		{
			detC = (iX2.at<float>(i,j) * iY2.at<float>(i,j)) - (iXY.at<float>(i,j) * iXY.at<float>(i,j));
			trace = iX2.at<float>(i,j) + iY2.at<float>(i,j);
			result.at<float>(i,j) = detC - kr*(trace * trace);
		}
	}

	return result;
}


void threshold(Mat source, Mat& result, Mat& canny)
{

	Mat output = source.clone();
	cvtColor(source, output, COLOR_GRAY2BGR);


	for(int i = 0; i < result.rows; i++)
	{
		for(int j = 0; j < result.cols; j++)
		{
			if(result.at<float>(i,j) > 0.99 && canny.at<uchar>(i,j) > uchar(250))
			{
				circle(output, Point(j,i), 9, Scalar(0, 255, 0), 1 );
			}
		}
	}

	imshow("Output", output);
	waitKey(0);
}

void harris_corner_detector(Mat& source, int tLow, int tHigh)
{
	Mat gaussian = source.clone();

	GaussianBlur(source, gaussian, Size(5,5), 1.732);

	Mat grX = Mat::zeros(source.rows, source.cols, CV_32F);
	Mat grY = Mat::zeros(source.rows, source.cols, CV_32F);

	Sobel(gaussian, grX, CV_32F, 1, 0, 3);
	Sobel(gaussian, grY, CV_32F, 0, 1, 3);

	Mat canny;
	Canny(gaussian, canny, tLow, tHigh, 3);

	imshow("Canny", canny);
	waitKey(0);

	Mat rMat = createRMatrix(grX, grY);

	threshold(source, rMat, canny);

}

int main(int argc, char* argv[])
{
	if(argc != 4)
	{
		cout << "Error on input. \n Usage: ./exename image_path  Low_int_for canny_thres  High_int" << endl;
		return -1;
	}

	Mat source = imread(argv[1], IMREAD_GRAYSCALE);

	if(source.data == nullptr)
	{
		cout << "Error on loading image" << endl;
		return -2;
	}

	imshow("Image", source);
	waitKey(0);

	harris_corner_detector(source, atoi(argv[2]), atoi(argv[3]));

	return 0;

}
