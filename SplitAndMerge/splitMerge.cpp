#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

/**
	Lista input:
	- nome immagine
	- coordinata x del Pixel di riferimento
	- coordinata y del Pixel di riferimento
	- soglia
**/
Mat immaginePrincipale;
Mat output;

vector<float> mediaRegione(int rigaIniziale, int rigaFinale, int colonnaIniziale, int colonnaFinale);
vector<float> varianzaRegione(int rigaIniziale, int rigaFinale, int colonnaIniziale, int colonnaFinale);
void coloraRegione(Mat image, int rigaIniziale, int rigaFinale, int colonnaIniziale, int colonnaFinale, Vec3b colore);
int distanzaEuclidea(vector<float> X, Point y);
void splitMerge(int rigaIniziale, int rigaFinale, int colonnaIniziale, int colonnaFinale, int soglia);


int main(int argc, char **argv)
{
	immaginePrincipale = imread(argv[1]);
	if(immaginePrincipale.data == nullptr)
	{
		cerr << "error open image" << endl;
		return(-1);
	}
	//resize(immaginePrincipale, immaginePrincipale, Size(640, 320));
	namedWindow("Immagine Principale", WINDOW_AUTOSIZE);
	imshow("Immagine Principale", immaginePrincipale);

	output = immaginePrincipale.clone();

	splitMerge(0, immaginePrincipale.rows, 0, immaginePrincipale.cols, atoi(argv[2]));
	namedWindow("Immagine output", WINDOW_AUTOSIZE);
	imshow("Immagine output", output);

	waitKey(0);
	return(0);
}

vector<float> mediaRegione(int rigaIniziale, int rigaFinale, int colonnaIniziale, int colonnaFinale)
{
	vector<float> media(3, 0.0F);
	long int numeroPixel = 0;

	#pragma omp parallel for
	for(int i = rigaIniziale; i < rigaFinale; i++)
		for(int j = colonnaIniziale; j < colonnaFinale; j++)
		{
			media.at(0) += immaginePrincipale.at<Vec3b>(i,j).val[0];
			media.at(1) += immaginePrincipale.at<Vec3b>(i,j).val[1];
			media.at(2) += immaginePrincipale.at<Vec3b>(i,j).val[2];
			numeroPixel++;
		}
	for(short i = 0; i < 3; i++)
		media.at(i) /= numeroPixel;
	return(media);
}

vector<float> varianzaRegione(int rigaIniziale, int rigaFinale, int colonnaIniziale, int colonnaFinale)
{
	vector<float> media = mediaRegione(rigaIniziale, rigaFinale, colonnaIniziale, colonnaFinale);
	vector<float> varianza(3, 0.0F);
	long numeroPixel = 0;

	#pragma omp parallel for
	for(int i = rigaIniziale; i < rigaFinale; i++)
		for(int j = colonnaIniziale; j < colonnaFinale; j++)
		{
			float diffB = immaginePrincipale.at<Vec3b>(i,j).val[0] - media.at(0);
			float diffG = immaginePrincipale.at<Vec3b>(i,j).val[1] - media.at(1);
			float diffR = immaginePrincipale.at<Vec3b>(i,j).val[2] - media.at(2);

			numeroPixel++;

			varianza.at(0) += pow(diffB, 2);
			varianza.at(1) += pow(diffG, 2);
			varianza.at(2) += pow(diffR, 2);
		}
	for(short i = 0; i < 3; i++)
		varianza.at(i) /= numeroPixel;
	return(varianza);
}

int distanzaEuclidea(vector<float> X, Point y)
{
	double diffB = X.at(0) - immaginePrincipale.at<Vec3b>(y.x, y.y).val[0];
	double diffG = X.at(1) - immaginePrincipale.at<Vec3b>(y.x, y.y).val[1];
	double diffR = X.at(2) - immaginePrincipale.at<Vec3b>(y.x, y.y).val[2];
	return sqrt(diffB*diffB + diffG*diffG + diffR*diffR);
}

void coloraRegione(Mat image, int rigaIniziale, int rigaFinale, int colonnaIniziale, int colonnaFinale, Vec3b colore)
{
	#pragma omp parallel for
	for(int i = rigaIniziale; i < rigaFinale; i++)
		for(int j = colonnaIniziale; j < colonnaFinale; j++)
			image.at<Vec3b>(i,j) = colore;
}


/*
	La fase di merging viene effettuata contestualmente in quanto ogni regione viene colorata tramite la media delle regioni. Regioni 
	papabili per il merge avranno stessa media, e dunque stesso colore.
*/
void splitMerge(int rigaIniziale, int rigaFinale, int colonnaIniziale, int colonnaFinale, int soglia)
{
	Vec3b nero(0, 0, 0);
	vector<float> varianza = varianzaRegione(rigaIniziale, rigaFinale, colonnaIniziale, colonnaFinale);
	vector<float> media = mediaRegione(rigaIniziale, rigaFinale, colonnaIniziale, colonnaFinale);

	if(varianza.at(0) < 15 && varianza.at(1) < 15 && varianza.at(2) < 15 ||
		(rigaFinale - rigaIniziale) < 8 || (colonnaFinale - colonnaIniziale) < 8)
	{
			coloraRegione(output, rigaIniziale, rigaFinale, colonnaIniziale, colonnaFinale, Vec3b(media.at(0), media.at(1), media.at(2)));
	}
	else
	{
		int riga = (rigaFinale - rigaIniziale)/2;
		int colonna = (colonnaFinale - colonnaIniziale)/2;
		splitMerge(rigaIniziale, rigaIniziale + riga, colonnaIniziale, colonnaIniziale + colonna, soglia);
		splitMerge(rigaIniziale, rigaIniziale + riga, colonnaIniziale + colonna, colonnaFinale, soglia);
		splitMerge(rigaIniziale + riga, rigaFinale, colonnaIniziale, colonnaIniziale + colonna, soglia);
		splitMerge(rigaIniziale + riga, rigaFinale, colonnaIniziale + colonna, colonnaFinale, soglia);
	}
}
