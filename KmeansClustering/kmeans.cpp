#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <time.h>
#include <limits>
#include <map>

using namespace std;
using namespace cv;


vector<vector<int>> clustering(Mat image, int K, int soglia, int num_iter)
{

	//Matrice di interi che rappresenterà i cluster
	vector<vector<int>> matrice;
	matrice.resize(image.rows);

	// resize e popolamento
	for(int i = 0; i < image.rows; i++)
	{
		matrice.at(i).resize(image.cols);
		for(int j = 0; j < image.cols; j++)
		{
			matrice.at(i).at(j) = 0;
		}
	}

	srand(time(NULL));

	//Array di K elementi di Vec3b che contengono i valorio RGB dei semi dei cluster.
	Vec3b punti_iniziali[K], punti_finali[K];
	// Array che contiene il numero di elementi per ogni cluster
	int sizeCluster[K];

	// scelta randomica dei seed dei cluster iniziali
	for(int k = 0; k < K; k++)
	{
		punti_iniziali[k] = image.at<Vec3b>(rand()%image.rows, rand()%image.cols);
		punti_finali[k] = punti_iniziali[k];
		sizeCluster[k] = 0;
	}

	// contatore per le iterazioni
	short ind = 0;
	while(ind != num_iter)
	{
		// per ogni pixel nell'immagine
		for(int i = 0; i < image.rows; i++)
		{
			for(int j = 0 ; j < image.cols; j++)
			{
				int min = numeric_limits<int>::max();
				
				int cluster_minimo = 0;
				// calcolo la distanza euclidea del pixel considerato con ogni elemento dei punti iniziali
				// se la distanza euclidea è minore del minimo, aggiorniamo la variabile minimo, e indichiamo l'indice 
				// dell'array di punti iniziali in modo tale che contenga l'indice del cluster con distanza minima
				for(int k = 0; k < K; k++)
				{
					int diffb = punti_iniziali[k].val[0] - image.at<Vec3b>(i, j).val[0];
					int diffg = punti_iniziali[k].val[1] - image.at<Vec3b>(i, j).val[1];
					int diffr = punti_iniziali[k].val[2] - image.at<Vec3b>(i, j).val[2];
					int distanza = sqrt(diffb*diffb + diffg*diffg + diffr*diffr);
					if(distanza < min)
					{
						min = distanza;
						cluster_minimo = k;
					}
				}

				// assegnamo il pixel considerato al relativo cluster con distanza minima
				matrice.at(i).at(j) = cluster_minimo;

				// incrementiamo il numero di pixel apartenenti al cluster
				sizeCluster[cluster_minimo]++;
			}
		}

		// vector che conterranno le medie dei valori di BGR per ogni cluster (conterranno le somme dei valori di BGR per ogni cluster)
		// valori che serviranno poi per calcolare il valore medio.
		vector<int> mediab(K, 0);
		vector<int> mediag(K, 0);
		vector<int> mediar(K, 0);

		// per ogni pixel dell'immagine, trovo l'indice del vector da aggiornare guardando il valore del cluster assegnato nella matrice
		// e aggiungo all'elemento del vector i corrispettivi valori di BGR di ogni pixel del cluster
		for(int i = 0; i < image.rows; i++)
		{
			for(int j = 0 ; j < image.cols; j++)
			{
				mediab.at(matrice.at(i).at(j)) += image.at<Vec3b>(i, j).val[0];
				mediag.at(matrice.at(i).at(j)) += image.at<Vec3b>(i, j).val[1];
				mediar.at(matrice.at(i).at(j)) += image.at<Vec3b>(i, j).val[2];
			}
		}

		// Per ogni cluster il cui size è maggiore di zero, creo dei nuovi valori di seed per i punti finali, i cui valori BGR sono ugali ai valori medi dei 
		// corrispettivi cluster
		for(int k = 0; k < K; k++)
		{
			if(sizeCluster[k] > 0)
			{
				punti_finali[k].val[0] = mediab.at(k) / sizeCluster[k];
				punti_finali[k].val[1] = mediag.at(k) / sizeCluster[k];
				punti_finali[k].val[2] = mediar.at(k) / sizeCluster[k];
			}
		}

		int differenza = 0;

		// per ogni cluster calcolo le differenze dei valori BGR dei punti finali calcolati,
		// con i punti iniziali trovati. Ne calcolo la distanza euclidea. 
		// Se la distanza così calcolata è minore della soglia passata in input, incremento la variabile contatore 
		// che indica il numero di cluster la cui distanza tra i punti finali (medie dei cluster) ed i punti iniziali
		// è minore della soglia
		for(int k = 0; k < K; k++)
		{
			int diffb = punti_finali[k].val[0] - punti_iniziali[k].val[0];
			int diffg = punti_finali[k].val[1] - punti_iniziali[k].val[1];
			int diffr = punti_finali[k].val[2] - punti_iniziali[k].val[2];

			int disTot = sqrt(diffb*diffb + diffg*diffg + diffr*diffr);

			if(disTot < soglia)
			{
				differenza++;
			}
		}

		cout << differenza << endl;

		// Se la variabile contatore è pari al numero di cluster. Vuol dire che non è più necessario continuare a modificarli
		// spostando i pixel da un cluster all'altro. E quindi si esce dal ciclo while.
		// in caso contrario, si impostano i nuovi punti iniziali pari ai punti finali trovati in precedenza.
		if(differenza == K)
			break;
		else
		{
			for(int k = 0; k < K; k++)
			{
				punti_iniziali[k] = punti_finali[k];
			}
		}

		cout << "iterazione numero: "<< ind << endl;

		// incremento contatore per il numero di iterazioni
		ind++;
	}

	return matrice;
}


// Creazione della matrice di output colorando in maniera randomica i cluster
Mat colorazione(Mat image, vector<vector<int>> mat, int K)
{
	srand(time(NULL));

	// creazione di una map per associare i colori ai cluster
	map<int, Vec3b> colori;

	for(int k = 0; k < K; k++)
	{
		colori.insert(make_pair(k, Vec3b(rand()%255, rand()%255, rand()%255)));
	}

	Mat output = image.clone();

	for(int i = 0; i < image.rows; i++)
	{
		for(int j = 0 ; j < image.cols; j++)
		{
			output.at<Vec3b>(i, j) = colori[mat.at(i).at(j)];
		}
	}

	return output;
}

int main(int argc, char* argv[])
{
	if(argc != 5)
	{
		cout << " ERROR ON INPUT" <<endl;
		return -1;
	}

	Mat source = imread(argv[1], IMREAD_COLOR);
	imshow("Source", source);
	waitKey(0);
 	
	vector<vector<int>> mappa = clustering(source, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));

	Mat risult = colorazione(source, mappa, atoi(argv[2]));
	imshow("Kmeans", risult);
	waitKey(0);


	return 0;
}