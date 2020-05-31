#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

int pad;


/*
	Algortimo per la ricerca di intersezioni di linee, pattern di corner nelle immagini, ecc.
	Intuizione per l'algoritmo:
	1) Regione Flat: regione in cui non vi è nessuna variazione in qualsiasi direzione del gradiente
	2) Contorno (edge): Nessuna variazione lungo la direzione del bordo/contorno
	3) Spigolo (corner, angolo): variazione significativa in tutte le direzioni

	La matematica di Harris Corner detection
	Cerchiamo le differenze di intensità in tutte le direzioni di un pixel con uno spiazzamento di dimesione (u,v)

	E(u,v) =   SOMMAT(x, y) di { w(u, v) * [I(x+u, y+v) - I(x, y)]^2 }

	w(u,v) è una funzione finestra che assegna un peso al pixel in posizione (x,y). La più semplice funzione finestra assegna a tutti i pixel
	che ricadono nella finestra peso unitario.

	In regioni FLAT E(u,v) risulterà bassa mentre sarà alta latrove. Attraverso l'espansione di Taylor del secondo termine della funzione E(u,v)

					 [ u ]
	E(u,v) =~ [u,v]*M[ v ]


                                           [ I^(2)x     IxIy    ]
	Dove  M = SOMMAT (x, y ) di {    w(x,y)[                    ]      }
										   [ IxIy       I^(2)y  ]2x2


	matrice 2x2 costruita a partire dalle derivate parziali rispetto ad X e rispetto ad Y  (ottenibili con il filtro di Sobel)

	Rispetto a questa formulazione:        R = det(M) - k(trace(M))^2

	dove:  1)  det(M) = lamb1 * lamb2               2)  trace(M) = lamb1 + lamb2
		   3) k è un valore molto piccolo calcolato empiricamente e compreso tra 0,04 e 0,06
		   4) lamb1 e lamb2 sono gli autovalori di M
	
	Si può determinare se un pixel è corner in base alle seguenti condizioni:
	1) Se |R| è piccolo (lamb1 e lamb2 sono piccoli) -> la regione è Flat
	2) Se |R| < 0 (se lamb1 << lamb2 e viceversa) -> la regione è contorno/edge
	3) Se |R| è grande (quando lamb1 e lamb2 sono grandi similarmente) -> la regione è corner

	Gli autovalori rappresentano gli assi maggiori e minori dell'approssimazione ellittica della distribuzione dei vettori gradiente.

	N.B. per rendere robusto il risultato applicare uno smoothing gaussiano a I^(2)x, I^(2)y e IxIy prima di calcolare M.
*/



void mediano(Mat &image)
{
	vector<uchar> median_filt;
	for(int i = pad; i < image.rows -pad; i++)
    {
        for(int j = pad; j < image.cols -pad; j++)
        {
            median_filt.push_back(image.at<uchar>(i,j));
            if((i-1) >= 0 && (j-1) >= 0) //pixel in alto a sinistra NO
                median_filt.push_back(image.at<uchar>(i-1,j-1));
            if((i-1) >= 0)// N
               median_filt.push_back(image.at<uchar>(i-1,j));
            if((i-1) >= 0 && (j+1) < image.cols) //NE
                median_filt.push_back(image.at<uchar>(i-1,j+1));
            if((j-1) >= 0) // O
                median_filt.push_back(image.at<uchar>(i,j-1));
            if((j+1) < image.cols)//E
                median_filt.push_back(image.at<uchar>(i,j+1));
            if((i+1)< image.rows && (j-1)>= 0)//SO
                median_filt.push_back(image.at<uchar>(i+1,j-1));
            if((i+1)< image.rows)//S
               median_filt.push_back(image.at<uchar>(i+1,j));
            if((i+1) < image.rows && (j+1) < image.cols)//SE
                median_filt.push_back(image.at<uchar>(i+1,j+1));

            sort(median_filt.begin(), median_filt.end());
            image.at<uchar>(i,j) = median_filt.at((int)(median_filt.size()/2));
            median_filt.clear();
        }
    }
}

float compute_value(float sigma, int i, int j)
{
	float x1 = pow(i,2)/(2*pow(sigma,2.0));
	float x2 = pow(j,2)/(2*pow(sigma,2.0));
	float value = exp(-x1 + x2);
	return value;
}


Mat gaussian_filter(Mat source, int kernel_size, float sigma)
{
	float minimum = 10000.0;
	float sum = 0.0;
	pad = (kernel_size - 1)/2;

	vector<vector<float>> kernel;
	for(int i = 0; i < kernel_size; i++)
	{
		vector<float> temp;
		for(int j = 0; j < kernel_size; j++)
		{
			temp.push_back(0.0);
		}
		kernel.push_back(temp);
		temp.clear();
	}

	for(int i = -pad; i <= pad ; i++ )
	{
		for(int j = -pad; j <= pad; j++)
		{
			float temp_val = compute_value(sigma, i, j);
			if(temp_val < minimum)
				minimum = temp_val;
			kernel.at(i+pad).at(j+pad) = temp_val;
		}
	}

	minimum = 1.0/minimum;
	for(int i = 0; i < kernel_size; i++)
	{
		for(int j = 0; j < kernel_size; j++)
		{
			kernel.at(i).at(j) = round(kernel.at(i).at(j)*minimum);
			sum += kernel.at(i).at(j); 
		}
	}

	Mat result;

	if(source.type() == CV_32F) // se la matrice di input è di tipo float, vuol dire che è o Ix2 o Iy2 o IxIy, e quindi immagini già sottoposte a pad
	{
		result = Mat::zeros(source.rows, source.cols, source.type());
		for(int i = pad; i < source.rows -pad; i++)
		{
			for(int j = pad; j < source.cols -pad; j++)
			{
				float cnt = 0.0;
				for(int l = -pad; l <= pad; l++)
				{
					for(int m = -pad; m <= pad; m++)
					{
						cnt += source.at<float>(i+l, j+m) * kernel.at(l+pad).at(m+pad);
					}
				}
				result.at<float>(i,j) = round(cnt/sum);
			}
		}
	}
	else if(source.type() == CV_8UC1)
	{
		Mat padded;
		result = Mat::zeros(source.rows + 2*pad, source.cols +2*pad, source.type());
		copyMakeBorder(source, padded, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0));

		for(int i = pad; i <= source.rows; i++)
		{
			for(int j = pad; j <= source.cols; j++)
			{
				int cnt = 0;
				for(int l = -pad; l <= pad; l++)
				{
					for(int m = -pad; m <= pad; m++)
					{
						cnt += (int)padded.at<uchar>(i+l, j+m) * kernel.at(l+pad).at(m+pad);
					}
				}
				result.at<uchar>(i,j) = round(cnt/sum);
			}
		}
	}
	return result;
}

void sobel(Mat source, Mat& iX, Mat& iY, Mat& magnitudo, Mat& orientation)
{
	vector<vector<int>> mx;
	vector<vector<int>> my;

	vector<int> tmp;

	tmp.push_back(-1);tmp.push_back(0);tmp.push_back(1);
	mx.push_back(tmp); tmp.clear();
	tmp.push_back(-2);tmp.push_back(0);tmp.push_back(2);
	mx.push_back(tmp); tmp.clear();
	tmp.push_back(-1);tmp.push_back(0);tmp.push_back(1);
	mx.push_back(tmp); tmp.clear();

	tmp.push_back(-1);tmp.push_back(-2);tmp.push_back(-1);
	my.push_back(tmp); tmp.clear();
	tmp.push_back(0);tmp.push_back(0);tmp.push_back(0);
	my.push_back(tmp); tmp.clear();
	tmp.push_back(1);tmp.push_back(2);tmp.push_back(1);
	my.push_back(tmp); tmp.clear();

	for(int i = pad; i < source.rows - pad; i++)
	{
		for(int j = pad; j < source.cols - pad; j++)
		{
			float valx = 0.0;
			float valy = 0.0;
			for(int l = -1; l <= 1; l++)
			{
				for(int m = -1; m <= 1; m++)
				{
					valx += source.at<uchar>(i+l,j+m) * mx.at(l+1).at(m+1);
					valy += source.at<uchar>(i+l,j+m) * my.at(l+1).at(m+1);
				}
			}

			if(valx > 255.0)
				valx = 255.0;

			if(valy > 255.0)
				valy = 255.0;

			iX.at<float>(i,j) = valx;
			iY.at<float>(i,j) = valy;

			float sumT = round(sqrt(pow(valx,2.0) + pow(valy, 2.0)));
            if(sumT > 255.0)
                sumT = 255.0;

			magnitudo.at<float>(i,j) = sumT;

			orientation.at<float>(i,j) = (atan2(valy, valx) * 180.0)/M_PI;
		}
	}
}


Mat non_maximum_suppression(Mat& magnitudo, Mat& direction, float Tlow, float Thigh)
{
	Mat res = Mat::zeros(magnitudo.rows, magnitudo.cols, CV_32F);

	for(int i = pad; i < magnitudo.rows - pad; i++)
	{
		for(int j = pad; j < magnitudo.cols -pad; j++)
		{

			while(direction.at<float>(i,j) < 0) //porto la direzione in angoli positivi, se ha valore negativo
				direction.at<float>(i,j) += 180.0;
			while(direction.at<float>(i,j) > 180) //porto la direzione in angoli positivi, se ha valore negativo
				direction.at<float>(i,j) -= 180.0;

			if( Thigh < magnitudo.at<float>(i,j)) // controllo se il pixel ha una magnitudo maggiore di Thigh, per poterlo quindi utilizzare
			{

				if(67.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 112.5) //orizzontale (gradiente verticale)
				{
					if( magnitudo.at<float>(i,j) >= magnitudo.at<float>(i-1,j) && magnitudo.at<float>(i,j) >= magnitudo.at<float>(i+1,j))
					{
						res.at<float>(i,j) = 255.0;
					}

				}

				else if(22.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 67.5) // diagonale da sx a dx
				{
					if( magnitudo.at<float>(i,j) >= magnitudo.at<float>(i-1,j-1) && magnitudo.at<float>(i,j) >= magnitudo.at<float>(i+1,j+1))
					{
						// res.at<uchar>(i,j) = uchar(255.0);
						res.at<float>(i,j) = 255.0;
					}
				}

				else if(112.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 157.5) // diagonale da dx a sx
				{
					if( magnitudo.at<float>(i,j) >= magnitudo.at<float>(i-1,j+1) && magnitudo.at<float>(i,j) >= magnitudo.at<float>(i+1,j-1))
					{
						// res.at<uchar>(i,j) = uchar(255.0);
						res.at<float>(i,j) = 255.0;
					}
				}

				else// if( ( 0.0 <= direction.at<float>(i,j) && direction.at<float>(i,j) <= 22.5 ) || (157.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 180.0)) // verticale
				{
					if( magnitudo.at<float>(i,j) >= magnitudo.at<float>(i,j-1) && magnitudo.at<float>(i,j) >= magnitudo.at<float>(i,j+1))
					{
						// res.at<uchar>(i,j) = uchar(255.0);
						res.at<float>(i,j) = 255.0;
					}					
				}
			}
			else if( Tlow < magnitudo.at<float>(i,j) && magnitudo.at<float>(i,j) <= Thigh)
			{
				if(67.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 112.5) //orizzontale (gradiente verticale)
				{
					if( magnitudo.at<float>(i,j) >= magnitudo.at<float>(i-1,j) && magnitudo.at<float>(i,j) >= magnitudo.at<float>(i+1,j))
					{
						res.at<float>(i,j) = 80.0;
					}

				}

				else if(22.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 67.5) // diagonale da sx a dx
				{
					if( magnitudo.at<float>(i,j) >= magnitudo.at<float>(i-1,j-1) && magnitudo.at<float>(i,j) >= magnitudo.at<float>(i+1,j+1))
					{
						// res.at<uchar>(i,j) = uchar(255.0);
						res.at<float>(i,j) = 80.0;
					}
				}

				else if(112.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 157.5) // diagonale da dx a sx
				{
					if( magnitudo.at<float>(i,j) >= magnitudo.at<float>(i-1,j+1) && magnitudo.at<float>(i,j) >= magnitudo.at<float>(i+1,j-1))
					{
						// res.at<uchar>(i,j) = uchar(255.0);
						res.at<float>(i,j) = 80.0;
					}
				}

				else// if( ( 0.0 <= direction.at<float>(i,j) && direction.at<float>(i,j) <= 22.5 ) || (157.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 180.0)) // verticale
				{
					if( magnitudo.at<float>(i,j) >= magnitudo.at<float>(i,j-1) && magnitudo.at<float>(i,j) >= magnitudo.at<float>(i,j+1))
					{
						// res.at<uchar>(i,j) = uchar(255.0);
						res.at<float>(i,j) = 80.0;
					}					
				}
			}
		}

	}
	return res;
}


Mat isteresi(Mat& nonmaxsub, Mat magnitudo, Mat direction, int Tlow, int Thigh)
{
	Mat result = Mat::zeros(nonmaxsub.rows, nonmaxsub.cols, CV_8UC1);

	bool value = true;

	while(value)
	{
		value = false;

		for(int i = 0; i < nonmaxsub.rows; i++)
		{
			for(int j = 0; j < nonmaxsub.cols; j++)
			{
				if(nonmaxsub.at<float>(i,j) == 255.0)
				{	
					nonmaxsub.at<float>(i,j) = 100.0; //metto un valore arbitrario

					for(int k = -1; k <= 1; k++)
					{
						for(int m = -1; m <= 1; m++)
						{
							if(nonmaxsub.at<float>(i+k, j+m) == 80.0)
							{
								value = true;
								nonmaxsub.at<float>(i+k, j+m) = 255.0;
							}
						}
					}
			  	}	
			}
		}
	}
	
	for(int i = 0; i < nonmaxsub.rows; i++)
	{
		for(int j = 0; j < nonmaxsub.cols; j++)
		{
			if(nonmaxsub.at<float>(i,j) == 100.0)// || nonmaxsub.at<float>(i,j) == 255.0)
			{
				result.at<uchar>(i,j) = uchar(255.0);
			}
		}
	}

	return result;
}

Mat canny_edge_detector(Mat& magnitudo, Mat& direction)
{
	int tLow = -1;
	int tHig = -1;

	cout << "Insert positive Tlow and Thigh values:"<<endl;

	while(tLow > tHig || tLow < 0 || tHig > 255 )
	{
		cout << "Tlow: " <<endl;
		cin >> tLow;
		cout << "Thigh: " <<endl;
		cin >> tHig;
	}

	Mat temp = non_maximum_suppression(magnitudo, direction, tLow, tHig);
	Mat risult = isteresi(temp, magnitudo, direction, tLow, tHig);

	return risult;
}

/*
	Funzione per la creazione della matrice che contiene i valori R per i pixel.
	Vengono calcolate le matrici temporanee per I^(2)x I^(2)y e IxIy, che poi vengono sottoposte a blurring gaussiano.
	per ogni pixel considerato, viene calcolato il rispettivo valore detC e trace, che poi vengono utilizzati per calcolare il
	valore di R per quel pixel e inserirlo nell'apposita matrice risultato. 
*/
Mat createR(Mat iX, Mat iY, int kernel_size, float sigma)
{
	Mat rMat = Mat::zeros(iX.rows, iX.cols, iX.type());
	Mat tmpx = Mat::zeros(iX.rows, iX.cols, iX.type());
	Mat tmpy = Mat::zeros(iX.rows, iX.cols, iX.type());
	Mat tmpxy = Mat::zeros(iX.rows, iX.cols, iX.type());

	for(int i = 0; i < iX.rows; i++)
	{
		for(int j = 0; j < iX.cols; j++)
		{
			tmpx.at<float>(i,j) = iX.at<float>(i,j) * iX.at<float>(i,j);
			tmpy.at<float>(i,j) = iY.at<float>(i,j) * iY.at<float>(i,j);
			tmpxy.at<float>(i,j) = iY.at<float>(i,j) * iX.at<float>(i,j);
		}
	}

	Mat iX2 = gaussian_filter(tmpx, kernel_size, sigma);
	Mat iY2 = gaussian_filter(tmpy, kernel_size, sigma);
	Mat iXY = gaussian_filter(tmpxy, kernel_size, sigma);

	float detC;
	float trace;
	float kr = 0.04;

	for(int i = pad; i < iX.rows - pad; i++)
	{
		for(int j = pad; j < iX.cols - pad; j++)
		{
			detC = (iX2.at<float>(i,j) * iY2.at<float>(i,j)) - (iXY.at<float>(i,j)*iXY.at<float>(i,j));
			trace = iX2.at<float>(i,j) + iY2.at<float>(i,j);
			rMat.at<float>(i,j) = detC - kr*(trace*trace);
		}
	}

	return rMat;
}

/*
	funzione per la determinazione dei pixel di corner.
	Viene prima portata l'immagine ottenuta in input in BGR (oichè ottenuta in scala di grigi)
	Poi per ogni pixel della matrice, che non faccia parte di righe o colonne di padding,
	viene valutato il corrispettivo valore della matrice degli R. Se questo valore è maggiore di 0.99 , e se il corrispettivo 
	pixel sottoposto alla soppressione dei non massimi corrisponde ad un pixel di edge (valore pari a 255, bianco), allora in quel pixel vi è
	un corner, che viene visualizzato in rosso sull'immagine, tramite un cerchio. 	
*/
void threshold(Mat& source, Mat nonMs, Mat rMatrix)
{
	Mat colored;
	cvtColor(source, colored, COLOR_GRAY2BGR);

	for(int i = pad; i < rMatrix.rows -pad; i++)
	{
		for(int j = pad; j < rMatrix.cols - pad; j++)
		{
			if(rMatrix.at<float>(i,j) > 0.99 && nonMs.at<uchar>(i,j) == uchar(255))
			{
				circle(colored,Point(j-pad,i-pad),9,Scalar(0,0,250),1);
			}
		}
	}

	imshow("Corners", colored);
}


/*
	Vengono create le matrici che conterranno il gradiente X il gradiente Y, la direzione del gradiente e la matrice risultato di Soble (che non verrà usata).
	Viene sottoposta l'immagine sorgente di input al'operatore di Sobel e poi viene invocata la funzione che crea la matrice che contiene
	i valori di R per ogni singolo pixel.
	Successivamente viene invocata la funzione per l'operatore Canny per il riconoscimento dei bordi, in modo tale che invocando la funzione threshold
	si possano determinare al meglio i punti di corner.  
*/
void harris_corner_det(Mat& source, Mat& blurred_image, int gauss_kern_dim, float sigma)
{

	Mat grX = Mat::zeros(blurred_image.rows, blurred_image.cols, CV_32F);
	Mat grY = Mat::zeros(blurred_image.rows, blurred_image.cols, CV_32F);
	Mat alpha = Mat::zeros(blurred_image.rows, blurred_image.cols, CV_32F);
	Mat sum = Mat::zeros(blurred_image.rows, blurred_image.cols, CV_32F);

	sobel(blurred_image, grX, grY, sum, alpha);

	Mat rMat = createR(grX, grY, gauss_kern_dim, sigma);

	Mat nonMsupp = canny_edge_detector(sum, alpha); //non_max_supp(sum,alpha);

	imshow("Canny", nonMsupp);
	threshold(source, nonMsupp, rMat);
}

int main(int argc, char* argv[])
{
	Mat image = imread(argv[1], IMREAD_GRAYSCALE);

	bool med = false;
	bool gaus = false;

	int rispo;
	cout << "Eseguire anche gaussiano? 1 si 0 no" <<endl;
	cin >> rispo;

	if(rispo == 1)
		gaus = true;

	if(gaus)
	{
		int risp;
		cout << "Eseguire anche mediano? 1 si 0 no" <<endl;
		cin >> risp;

		if(risp == 1)
			med = true;

		int gauss_kern_dim;
		float sigma;
		cout << "Insert Gauss' kernel dimension (odd value): " << endl;
		cin >> gauss_kern_dim;

		while((gauss_kern_dim%2) != 1 || gauss_kern_dim == 1)
		{
			cout << "Insert an odd value different from one: " << endl;
			cin >> gauss_kern_dim;
		}

		pad = (gauss_kern_dim -1)/2;

		cout << "Insert sigma value" <<endl;
		cin >> sigma;

		int loops;
		Mat gaussian = gaussian_filter(image, gauss_kern_dim, sigma);

	// Mat sourc_temp = source.clone();
		if(med)
		{
			cout << "How many times do the median filter?" <<endl;
			cin >> loops;
			for(int k = 0; k < loops; k++)
				mediano(gaussian);//sourc_temp);
		}	
		harris_corner_det(image, gaussian, gauss_kern_dim, sigma);
	}
	else
	{
		pad = 1;
		harris_corner_det(image, image, 3, 0.5);	
	}
	

	imshow("image", image);

	waitKey(0);

	return 0;
}