#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

///Più regioni eliminando i pallini arancioni

using namespace std;
using namespace cv;

vector<vector<int>> matrix;

int dist_euclid(Vec3b one, Vec3b two)
{
	int b = one.val[0] - two.val[0];
	int g = one.val[1] - two.val[1];
	int r = one.val[2] - two.val[2];

	int dist = int(sqrt(b*b + g*g + r*r));
	return dist;
}

void reset_matrice()
{
	for(int i = 0; i < matrix.size(); i++)
	{
		for(int j = 0; j < matrix.at(0).size(); j++)
		{
			matrix.at(i).at(j) = 0;
		}
	}
}

void region_growing(Mat source, Point punto, int soglia, int k, int l)
{
	vector<Point> coda;
	coda.push_back(Point(k,l));
	while(coda.size() != 0)
	{
		Point pt = coda.at(coda.size()-1);
		coda.pop_back();

		for(short i = -1; i <= 1; i++)
		{
			for(short j = -1; j <= 1; j++)
			{
				if(pt.x + i >= 0 && pt.y + j >= 0 && pt.x + i < source.rows && pt.y + j < source.cols)
				{
					if(matrix.at(pt.x + i).at(pt.y + j) == 0)
					{
						int dst = dist_euclid(source.at<Vec3b>(pt.x + i, pt.y + j), source.at<Vec3b>(punto.x, punto.y));
						if( dst <= soglia)
						{
							matrix.at(pt.x + i).at(pt.y + j) = 1;
							coda.push_back(Point(pt.x + i, pt.y + j));
						}
					}
				}
			}
		}
	}
}

Mat ricolorazione(Mat source)
{
	Mat temp = source.clone();

	for(short i = 0; i < source.rows; i++)
	{
		for(short j = 0; j < source.cols; j++)
		{
			if(matrix.at(i).at(j) != 1)
			{
				temp.at<Vec3b>(i,j) = Vec3b(0,0,0);
			}
		}
	}
	return temp;
}

int main(int argc, char* argv[])
{
	Mat img = imread(argv[1], IMREAD_COLOR);
	if(!img.data || argc != 2)
	{
		cout<<"Error on input or on image opening"<<endl;
		return -1;
	}

	matrix.resize(img.rows);

	vector<Point> seeds;
	vector<int> soglie;

	seeds.push_back(Point(216,397)); // verde
	seeds.push_back(Point(301,345)); // rosso
	seeds.push_back(Point(197,336)); // blu?
	seeds.push_back(Point(125,505)); // giallo?

	soglie.push_back(80);
	soglie.push_back(40);
	soglie.push_back(80);
	soglie.push_back(50);
	

	Point aranc(368,257);
	// aranc.x = 345; 257
	// aranc.y = 244; 368
	int sogl_ar = 37;

	for(int i = 0; i < img.rows; i++)
	{
		matrix.at(i).resize(img.cols);
		for(int j = 0; j < img.cols; j++)
		{
			matrix.at(i).at(j) = 0;
		}
	}

	Mat output = Mat::zeros(img.size(), img.type());

	for(int k = 0; k < seeds.size(); k++)
	{
		for(int i = 0; i < img.rows; i+=3) //richiamo a piccole regioni di tre consentendo l'allargamento delle regioni
		{
			for(int j = 0; j < img.cols; j+=3)
			{
				region_growing(img,seeds.at(k),soglie.at(k), i, j);
			}
		}
		output += ricolorazione(img);
		reset_matrice();
	}


	Mat output_aranc = Mat::zeros(img.size(), img.type());

	for(int i = 0; i < img.rows; i+=3) //richiamo a piccole regioni di tre consentendo l'allargamento delle regioni
	{
		for(int j = 0; j < img.cols; j+=3)
		{
			region_growing(img,aranc,sogl_ar, i, j);
		}
	}
	output_aranc += ricolorazione(img);

	output -= output_aranc;

	imshow("Image", img);
	imshow("Risult", output);
	imshow("Arancione", output_aranc);
	waitKey(0);
	return 0;
}