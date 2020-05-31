#define _USE_MATH_DEFINES

#include <cmath>
#include <iostream>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

int pad;


/*
	Funzione per il calcolo degli elementi contenuti nelle locazioni
	del kernel che rappresenta il filtro gaussiano.
	dati i due indici dell'elemento: i e j
	il valore viene calcolato come:  e^(  - (i^2 / 2*sigma^2) + (j^2 / 2*sigma^2))
*/
float compute_value(float sigma, int i, int j)
{
    float val = 0.0;
    float x1;
    float x2;
    x1 = (pow(i,2) / (2 * pow(sigma,2)));
    x2 =  (pow(j,2) / (2 * pow(sigma,2)));
    val = exp(-x1 + x2);
    return val;
}


/*
	Filtro Gaussiano, per lo smoothing dell'immagine.
	I dati in input corrispondono all'immagine da sottoporre a blurring, la dimensione del kernel 
	di Gauss che si vuole utilizzare, ed il sigma da utilizzare per la costruzione del kernel.
*/
Mat gaussian_filter(Mat image, int kernel_size, float sigma)
{
    Mat immPadded;

    // valore che definisce il numero di righe/colonne da aggiungere ad ogni lato per effettuare zero padding
    int sub = (kernel_size - 1)/2;

    //matrice risultante, inizializzata a zero e di dimensioni pari a quelle della matrice in input, più le righe/colonne di padding
    Mat result = Mat::zeros(image.rows + 2*sub, image.cols + 2*sub, image.type());
    
    //variabili d'appoggio
    float summ = 0.0;
    float minimum = 1000.0;

    //Dichiarazione ed inizializzazione del kernel
    vector<vector<float>> kernel_temp;

     for(int i = 0; i < kernel_size; i++)
    {
        vector<float> temp;
        for(int j = 0; j < kernel_size; j++)
        {
            temp.push_back(0.0);
        }
        kernel_temp.push_back(temp);
        temp.clear();
    }

   // popolamento del kernel utilizzando la funzione "compute_value", e individuazione del valore
    // minimo contenuto nel kernel, valore che verrà utilizzato in seguito.
   for(int i = 0; i < kernel_size; i++)
    {
        for(int j = 0; j < kernel_size; j++)
        {
            float tmp = compute_value(sigma, i-sub, j-sub);
            if( tmp < minimum)
                minimum = tmp;
            kernel_temp.at(i).at(j) = tmp; 
        }
    }

    // aggiornamento della variabile di minimo
    minimum = 1.0 / minimum;

    summ = 0.0;
    // aggiornamento del kernel di gauss prima della convoluzione. 
    // ogni elemento calcolato in precedenza viene moltiplicato per la variabile "minimum"
    // viene poi calcolata la somma di tutti gli elementi del kernel
    for(int i = 0; i < kernel_size; i++)
    {
        for(int j = 0; j < kernel_size; j++)
        {
            kernel_temp.at(i).at(j) = round(kernel_temp.at(i).at(j) * minimum);
            summ += kernel_temp.at(i).at(j);
        }
    }

    // Copia della matrice di input nella matrice immPadded ed in aggiunta vengono inserite le righe/colonne di padding
    // che vengono poste a zero. Le righe/col aggiunte sono, in ordine, rig_top, rig_bott, col_left, col_right
    copyMakeBorder(image, immPadded, sub, sub, sub, sub, BORDER_CONSTANT, Scalar(0));

    //convoluzione del kernel di gauss con la matrice sottoposta a padding.
    // Poichè l'immagine è sottoposta a padding, gli indici che verranno scansionati nei for esterni, che scorrono sull'immagine
    // partono dal valore di "sub" ovvero del numero di rig/col di padding. Se il kernel è 3*3 allora ci saranno due righe e due colonne di
    // padding (risp una sopra e una sotto, una a sinistra e una a destra). Quindi l'indice che scorre le righe, così come l'indice
    // che scorre le colonne partiranno dall'indice 1 e non da 0.
    // Per ogni elemento controllato Si considera l'intorno della stessa dimensione del kernel.
    // Si calcola così una variabile "cnt" che costituisce la somma delle moltiplicazioni degli elementi dell'intorno
    // con i corrispettivi elementi del kernel di gauss. Gli indici dei due cicli for innestati partono da -sub e arrivano a sub
    // in modo tale da avere il pixel (i,j) centrato nel kernel. Una volta calcolato cnt, il valore del pixel (i,j)
    // nella matrice risultato sarà uguale al rapporto di cnt e della somma degli elementi del kernel di gauss (il tutto sottoposto a round)
    for(int i = sub; i < immPadded.rows - sub; i++)
    {
        for(int j = sub; j < immPadded.cols - sub; j++)
        {
            int cnt = 0;
            for(int l = -sub; l <= sub; l++)
            {
                for(int m = -sub; m <= sub; m++)
                {
                    cnt += (int)immPadded.at<uchar>(i+l,j+m)*kernel_temp.at(l+sub).at(m+sub);
                }
            }
            result.at<uchar>(i,j) = round(cnt/summ);
        }
    }
    //imshow("gauss", result);
    return result;
}


/*
	Operatore locale per l'estrazione dei bordi.
	Esso è costituito da due kernel M1 ed M2
	M1 è pari a -1 0 1    ed M2 è pari a  -1 -2 -1
				-2 0 2					   0  0  0
				-1 0 1					   1  2  1

	In input abbiamo l'immagine a cui applicare Sobel (già sottoposta a padding), poi le matrici di input/output che corrispondono
	Alla matrice del gradiente per X, alla matrice del gradiente di Y, Alla matrice della direzione del gradiente, e la matrice
	che corrisponderà alla combinazione delle matrici dei due gradienti X ed Y.
*/
void sobel(Mat source, Mat& gradientX, Mat& gradientY, Mat& alpha, Mat& sum)
{
	vector<vector<int>> m1;
	vector<vector<int>> m2;

    vector<int> temp;
    temp.push_back(-1); temp.push_back(0); temp.push_back(1);
    m1.push_back(temp); temp.clear();
    temp.push_back(-2); temp.push_back(0); temp.push_back(2);
    m1.push_back(temp); temp.clear();
    temp.push_back(-1); temp.push_back(0); temp.push_back(1);
    m1.push_back(temp); temp.clear();

    temp.push_back(-1);	temp.push_back(-2);	temp.push_back(-1);
    m2.push_back(temp);	temp.clear();
    temp.push_back(0); 	temp.push_back(0); 	temp.push_back(0);
    m2.push_back(temp);	temp.clear();
    temp.push_back(1); 	temp.push_back(2); 	temp.push_back(1);
    m2.push_back(temp);	temp.clear();

    float val = 0.0;
    float val2 = 0.0;
    float sumT = 0.0;


    /*
		Partendo dall'indice della matrice di input che corrisponde alla prima riga e alla prima colonna non di padding,
		Viene effettuata la convoluzione, separata, per ognuno dei due kernel M1 ed M2, il cui valore risultato è salvato in, risp,
		val e val2. Terminate le convoluzioni per il pixel considerato, vengono controllati questi due valori in modo tale che non
		superino il valore 255 (massimo valore rappresentabile in scala di grigio).
		Vengono poi inseriti questi due valori nelle corrispettive matrici dei gradienti X ed Y, risp..
		Nella matrice della somma, invece viene inserito il valore, arrotondato, della radice quadrata della somma dei quadrati dei due
		valori dei gradienti.
		Il valore della direzione del gradiente (alpha) per il pixel considerato viene calcolato portando in gradi il valore risultante
		dalla funzione atan2 che calcola l'arcotangente di Gy/Gx
    */
    for(int i = pad; i < source.rows - pad; i++)
    {
        for(int j = pad; j < source.cols - pad; j++)
        {
            val = 0.0;
            val2 = 0.0;
            sumT = 0.0;
            for(int k = 0; k < 3; k++)
            {
                for(int l = 0; l < 3; l++)
                {
                    val += source.at<uchar>(i+k-1,j+l-1) * m1.at(k).at(l);
                    val2 +=  source.at<uchar>(i+k-1,j+l-1) * m2.at(k).at(l);
                }
            }
           
            //val = abs(val);
            if(val > 255.0)
                val = 255.0;
            // else if(val < 0.0)
            // 	val = 0.0;

            gradientX.at<float>(i,j) = val;

            //val2 = abs(val2);
            if(val2 > 255.0)
                val2 = 255.0;
            // else if(val2 < 0.0)
            // 	val2 = 0.0;

            gradientY.at<float>(i,j) = val2;

            sumT = round(sqrt(pow(val,2.0) + pow(val2, 2.0)));
            if(sumT > 255.0)
                sumT = 255.0;
            sum.at<float>(i,j) = sumT;//abs(val) + abs(val2);//sumT;

            alpha.at<float>(i,j) = (atan2(val2, val) * 180.0)/3.142; // - val2 perchè la y nelle immagini va verso il basso?
        }
    }
}


/*
	La sola applicazione dei filtri di Sobel, Prewitt ecc. producono degli edge visibili ma non direttamente utilizzabili,
	in quanto una binarizzazione di un loro risultato produrrebbe degli edge spezzati.
	L'operatore di Canny è risultato essere ottimale per risolvere il compromesso tra la localizzazione accurata dei bordi
	e l'influenza del rumore.
	Il rapporto segnale rumore del gradiente è massimizzato per ottenere la minore probabilità di errore nella determinazione degli edge reali.
	I punti identificati come bordi sono i più vicini. Inoltre l'operatore dovrebbe produrre una unica risposta per uno stesso bordo.

	L'approccio prevede 4 fasi principali:
	1) Smoothuing Gaussiano dell'immagine
	2) Calcolo del gradiente con l'operatore di Roberts/Sobel
	3) soppressione di non massimi
	4) thresholding con isteresi.



	Di seguito vedremo la soppressione dei non massimi.
	L'obiettivo è di eliminare dall'immagine modulo-gradiente i pixel che non sono massimi locali rispetto all'orientazione del gradiente.
	Se un punto appartiene a un bordo, il valore del gradiente in esso è superiore del valore del gradiente dei vicini NELLA DIREZIONE DEL GRADIENTE.
	Se questo punto non soddisfa questa condizione viene azzerato (soppressione dei non massimi).

	Quattro possibili casi della soppressione dei non massimi.
	Dato un intorno 3x3 di un pixel ci sono quattro possibili edge: uno da N a S, Da W a E , da NE a SW, da NW a SE
	Sopprimere i non massimi deve quindi comunque tenere conto di questi casi, e valuta in corrispondenza di quali pixel vicini, il pixel considerato
	deve essere soppresso o meno.
	Se l'edge è da NE a SW, la direzione del gradiente (Matrice direction) si troverà nel range 22.5 e 67.5
	W a E: 67.5 a 112.5
	NW a SE:  112.5 a 157.5
	N a S : 0 a 22.5 && 157.5 a 180
																								0 0 *
	Prendendo per esempio il primo caso specificato, ovvero l'edge da NE a SW quindi un edge    0 * 0  il gradiente del pixel centrale
																							    * 0 0 
	verrà confrontato con quello dei vicini che si trovano rispettivamente in alto a sinistra ed in basso a destra per verificare che sia massimo.
	Simili osservazioni si applicano agli altri casi.

	LA matrice risultante sarà una matrice di float.
*/
Mat non_maximum_suppression(Mat& magnitudo, Mat& direction, float Tlow, float Thigh)
{
	Mat res = Mat::zeros(magnitudo.rows, magnitudo.cols, CV_32F);

	for(int i = pad; i < magnitudo.rows - pad; i++)
	{
		for(int j = pad; j < magnitudo.cols -pad; j++)
		{

			// La direzione del gradiente riportata in angoli positivi tra 0 e 180 gradi
			while(direction.at<float>(i,j) < 0) //porto la direzione in angoli positivi, se ha valore negativo
				direction.at<float>(i,j) += 180.0;
			while(direction.at<float>(i,j) > 180) //porto la direzione in angoli positivi, se ha valore negativo
				direction.at<float>(i,j) -= 180.0;


			// Poichè per l'algoritmo vengono utilizzate anche due soglie, per determinare un range di magnitudo da considerare per gli edge deboli
			// se la magnitudo del pixel considerato è maggiore della soglia alta (Thigh) allora esso è un potenziale edge forte.
			// e dunque se successivamente soddisfa le condizioni per la soppressione dei non massimi di cui prima,
			// verrà impostato a 255.0 nella matrice risultato della soopressione.

			if( Thigh < magnitudo.at<float>(i,j)) // controllo se il pixel ha una magnitudo maggiore di Thigh, per poterlo quindi utilizzare
			{

				if(67.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 112.5) //orizzontale (gradiente verticale)
				{
					if( magnitudo.at<float>(i,j) >= magnitudo.at<float>(i-1,j) && magnitudo.at<float>(i,j) >= magnitudo.at<float>(i+1,j))
					{
						res.at<float>(i,j) = 255.0;
					}

				}

				else if(22.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 67.5) // diagonale da sx a dx per il gradiente
				{
					if( magnitudo.at<float>(i,j) >= magnitudo.at<float>(i-1,j-1) && magnitudo.at<float>(i,j) >= magnitudo.at<float>(i+1,j+1))
					{
						res.at<float>(i,j) = 255.0;
					}
				}

				else if(112.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 157.5) // diagonale da dx a sx per il gradiente
				{
					if( magnitudo.at<float>(i,j) >= magnitudo.at<float>(i-1,j+1) && magnitudo.at<float>(i,j) >= magnitudo.at<float>(i+1,j-1))
					{
						res.at<float>(i,j) = 255.0;
					}
				}

				else// if( ( 0.0 <= direction.at<float>(i,j) && direction.at<float>(i,j) <= 22.5 ) || (157.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 180.0)) // verticale
				{
					if( magnitudo.at<float>(i,j) >= magnitudo.at<float>(i,j-1) && magnitudo.at<float>(i,j) >= magnitudo.at<float>(i,j+1))
					{
						res.at<float>(i,j) = 255.0;
					}					
				}
			}

			// nel caso in cui non sia un edge forte, ma la magnitudo ricade nel range di soglie definite (Tlow e Thigh)
			// allora si procede a controllare se esso è un potenziale edge debole che verrà poi connesso eventualmente con gli edge forti.
			// vengono quindi controllate le stesse condizioni per la soppressione dei non massimi, e se vengono soddisfatte, il suo valore 
			// e è impostato in maniera arbitraria a 80.0
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
						res.at<float>(i,j) = 80.0;
					}
				}

				else if(112.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 157.5) // diagonale da dx a sx
				{
					if( magnitudo.at<float>(i,j) >= magnitudo.at<float>(i-1,j+1) && magnitudo.at<float>(i,j) >= magnitudo.at<float>(i+1,j-1))
					{
						res.at<float>(i,j) = 80.0;
					}
				}

				else// if( ( 0.0 <= direction.at<float>(i,j) && direction.at<float>(i,j) <= 22.5 ) || (157.5 < direction.at<float>(i,j) && direction.at<float>(i,j) <= 180.0)) // verticale
				{
					if( magnitudo.at<float>(i,j) >= magnitudo.at<float>(i,j-1) && magnitudo.at<float>(i,j) >= magnitudo.at<float>(i,j+1))
					{
						res.at<float>(i,j) = 80.0;
					}					
				}
			}
		}

	}
	return res;
}


/*
	Il processo di isteresi consiste nel collegare tra loro edge deboli connessi con edge forti.
	Il processo di ricerca e connessione continua finchè ci sono edge deboli vicini ad edge forti.
	PEr ogni ciclo del while si scorre tutta la matrice sottoposta a soppressione dei non massimi in precedenza.
	Se si trova un pixel impostato a 255.0, ovvero un edge forte, esso viene posto a 100.0 in modo tale da non essere riconsiderato nelle iterazioni
	successive, si va poi a controllare l'intorno 3x3 del pixel alla ricerca degli edge deboli. Se vi sono edge deboli, essi vengono
	promossi ad edge forti, e si procede a impostare la variabile di validità per il ciclo while a true.
	Terminato il ciclo while, si costruiscde la nuova matrice risultato (in scala di grigi) ponendo a bianco ( uchar(255.0) )
	i pixel segnati come 100.0 nella matrice della soppressione dei non massimi. 
*/
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
				// if(magnitudo.at<float>(i,j) >= Thigh)
				// {
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
			if(nonmaxsub.at<float>(i,j) == 100.0)
			{
				result.at<uchar>(i,j) = uchar(255.0);
			}
		}
	}

	return result;
}



int main(int argc, char* argv[])
{
	if(argc != 2)
	{
		cout << "Error: when launching program, write the name of an image" << endl;
		return -1;
	}

	Mat source = imread(argv[1], IMREAD_GRAYSCALE);
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

	//Mat immPadded;
	// copyMakeBorder(source, immPadded, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0));
	Mat gaussian = gaussian_filter(source, gauss_kern_dim, sigma);
	// cv::GaussianBlur(immPadded, gaussian, cv::Size(gauss_kern_dim, gauss_kern_dim), sigma);

	Mat grX = Mat::zeros(gaussian.rows, gaussian.cols, CV_32F);
	Mat grY = Mat::zeros(gaussian.rows, gaussian.cols, CV_32F);
	Mat alpha = Mat::zeros(gaussian.rows, gaussian.cols, CV_32F);
	Mat sum = Mat::zeros(gaussian.rows, gaussian.cols, CV_32F);

	sobel(gaussian, grX, grY, alpha, sum);

	float Tlow = -1.0;
	float Thigh = -1.0;

//////////////////////////////////////////////////////////////////////////////////////
	cout << "Insert positive Tlow and Thigh values:"<<endl;

	while(Tlow > Thigh || Tlow < 0 || Thigh > 255 )
	{
		cout << "Tlow: " <<endl;
		cin >> Tlow;
		cout << "Thigh: " <<endl;
		cin >> Thigh;
	}

	Mat nonMaxS = non_maximum_suppression(sum, alpha, Tlow, Thigh);

	Mat risultat = isteresi(nonMaxS, sum, alpha, Tlow, Thigh);

	Mat tmp;
	Canny(gaussian, tmp, Tlow, Thigh, 3);

	imshow("Input Image", source);
	imshow("Canny Edge detector", risultat);
	imshow("Canny OpenCv", tmp);
	waitKey(0);

	return 0;
}
