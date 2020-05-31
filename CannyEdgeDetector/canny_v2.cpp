#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat isteresi(Mat& res, Mat sum)
{
	Mat output = Mat::zeros(res.rows, res.cols, CV_8UC1);
	bool value = true;

	while(value)
	{

		value = false;

		for(int i = 0; i < output.rows; i++)
		{
			for(int j = 0; j < output.cols; j++)
			{
				if(res.at<float>(i,j) == 255.0)
				{
					res.at<float>(i,j) = 100.0;

					for(int k = -1; k <= 1; k++)
					{
						for(int m = -1; m <= 1; m++)
						{
							if(res.at<float>(i+k, j+m) == 80.0)
							{
								value = true;
								res.at<float>(i+k, j+m) = 255.0;
							}
						}
					}
				}
			}
		}
	}

	for(int i = 0; i < output.rows; i++)
	{
		for(int j = 0; j < output.cols; j++)
		{
			if(res.at<float>(i,j) == 100.0)
				output.at<uchar>(i,j) = uchar(255);
		}
	}

	return output;
}



Mat canny_edge_detector(Mat source, float tLow, float tHigh, double size = 3)
{

	Mat workImg = source.clone();
	GaussianBlur(source, workImg, Size(5,5), 1.732);

	Mat magX = Mat(source.rows, source.cols, CV_32F);
	Mat magY = Mat(source.rows, source.cols, CV_32F);

	Sobel(workImg, magX, CV_32F, 1, 0, size);
	Sobel(workImg, magY, CV_32F, 0, 1, size);

	Mat direction = Mat(source.rows, source.cols, CV_32F);

	for(int i= 0; i < direction.rows; i++)
		for(int j= 0; j < direction.cols; j++)
			direction.at<float>(i,j) = (atan2(magY.at<float>(i,j), magX.at<float>(i,j)) * 180.0)/CV_PI;


	Mat prodX = Mat(source.rows, source.cols, CV_64F);
	Mat prodY = Mat(source.rows, source.cols, CV_64F);
	multiply(magX, magX, prodX);
	multiply(magY, magY, prodY);

	Mat sum = Mat(source.rows, source.cols, CV_64F);
	sum = prodX + prodY;
	cv::sqrt(sum, sum);

	copyMakeBorder(direction, direction, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0.0));
	copyMakeBorder(sum, sum, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0.0));

	Mat res = Mat::zeros(workImg.rows+2, workImg.cols+2, CV_32F);

	for(int i = 1; i < direction.rows -1; i++)
	{
		for(int j = 1; j < direction.cols -1; j++)
		{
			while(direction.at<float>(i,j) < 0.0)
				direction.at<float>(i,j) += 180.0;
			while(direction.at<float>(i,j) > 180.0)
				direction.at<float>(i,j) -= 180.0;

			if(tHigh < sum.at<float>(i,j))
			{

				if( 67.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 112.5) // gradiente verticale
				{
					if(sum.at<float>(i,j) >= sum.at<float>(i-1,j) && sum.at<float>(i,j) >= sum.at<float>(i+1,j))
						res.at<float>(i,j) = 255.0;
				}
				else if( 22.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 67.5) //gradiente da sx alto a dx basso
				{
					if(sum.at<float>(i,j) >= sum.at<float>(i-1,j-1) && sum.at<float>(i,j) >= sum.at<float>(i+1,j+1))
						res.at<float>(i,j) = 255.0;
				}
				else if( 112.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 157.5) // gradiente da dx alto a sx basso
				{
					if(sum.at<float>(i,j) >= sum.at<float>(i-1,j+1) && sum.at<float>(i,j) >= sum.at<float>(i+1,j-1))
						res.at<float>(i,j) = 255.0;
				}
				else // gradiente orizzontale
				{
					if(sum.at<float>(i,j) >= sum.at<float>(i,j-1) && sum.at<float>(i,j) >= sum.at<float>(i,j+1))
						res.at<float>(i,j) = 255.0;
				}
			}
			else if(tLow < sum.at<float>(i,j) && sum.at<float>(i,j) <= tHigh)
			{

				if( 67.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 112.5) // gradiente verticale
				{
					if(sum.at<float>(i,j) >= sum.at<float>(i-1,j) && sum.at<float>(i,j) >= sum.at<float>(i+1,j))
						res.at<float>(i,j) = 80.0;
				}
				else if( 22.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 67.5) //gradiente da sx alto a dx basso
				{
					if(sum.at<float>(i,j) >= sum.at<float>(i-1,j-1) && sum.at<float>(i,j) >= sum.at<float>(i+1,j+1))
						res.at<float>(i,j) = 80.0;
				}
				else if( 112.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 157.5) // gradiente da dx alto a sx basso
				{
					if(sum.at<float>(i,j) >= sum.at<float>(i-1,j+1) && sum.at<float>(i,j) >= sum.at<float>(i+1,j-1))
						res.at<float>(i,j) = 80.0;
				}
				else // gradiente orizzontale
				{
					if(sum.at<float>(i,j) >= sum.at<float>(i,j-1) && sum.at<float>(i,j) >= sum.at<float>(i,j+1))
						res.at<float>(i,j) = 80.0;
				}
			}
		}
	}

	Mat output = isteresi(res, sum);

	return output; 
}

int main(int argc, char*argv[])
{
	if(argc != 4)
	{
		cout << "Error on input. \n Usage: ./exename  image_path tLow_float tHigh_float" <<endl;
		return -1;
	}

	Mat source = imread(argv[1], IMREAD_GRAYSCALE);

	if(source.data == nullptr)
	{
		cout << "Error on loading image." << endl;
		return -2;
	}


	imshow("Input Image", source);

	waitKey(0);

	Mat cannyResult = canny_edge_detector(source, atof(argv[2]), atof(argv[3]));

	imshow("Canny Edge Detector", cannyResult);
	waitKey(0);

	return 0;

}
