#include<iostream>
#include<opencv2/opencv.hpp>
#define GRADI 180
#define RAD CV_PI/180

/*
    HOUGH PER RETTE
    Insensibilità all'occlusione.
    Il trucco della Trasformata di Hough è quello di rappresentare
    le linee in un sistema di coordinate polari (Spazio di Hough)
    Ogni possibile linea può essere rappresentata da una unica coppia di rho e theta.
    Inoltre ogni pixel (x,y) su di una linea verrà convertito negli stessi rho e theta.
*/

using namespace std;
using namespace cv;

void HoughTransform( Mat img, int threshold, vector<pair<Point, Point>>& punti);
void PolarToCartesian(double rho, double theta, Point& P1, Point& P2);

int main(int argc, char* argv[])
{
    if(argc != 3)
    {
        cerr << "Utilizza: <nome eseguibile> <path immagine> <threshold hough>" << endl;
        exit(1);
    }

    Mat image = imread( argv[1], IMREAD_GRAYSCALE );
    Mat canny, gauss, result;
    int threshold = atoi( argv[2] );
    vector<pair<Point, Point>> punti;

    if( image.data == nullptr)
    {
        cerr << "Errore apertura immagine" << endl;
        exit(1);
    }

    namedWindow("Immagine Originale", WINDOW_NORMAL);
    imshow("Immagine Originale", image);

    GaussianBlur( image, gauss, Size(5,5), 1.4 );
    Canny( gauss, canny, 100, 150, 3 );

    HoughTransform( canny, threshold, punti);

    cvtColor(image, result, COLOR_GRAY2BGR);

    cout << punti.size() << endl;

    for(int i=0; i<punti.size(); i++)
        line(result, punti.at(i).first, punti.at(i).second, Scalar(0, 255, 0), 1, LINE_AA);

    
    namedWindow("Hough Rette", WINDOW_NORMAL);
    imshow("Hough Rette", result);
    waitKey();


    return 0;
}

void HoughTransform( Mat img, int threshold, vector<pair<Point, Point>>& punti)
{

    // calcolo la diagonale dell'immagine, l'ipotenusa
    int distanza = hypot( img.rows, img.cols );
    //int distanza = max(img.rows, img.cols) * sqrt(2);

    cout << distanza << endl;
    
    /* inizializzazione dell'accumulatore.
        L'accumulatore rappresenta lo spazio di Hough. Sull'asse dele X abbiamo
        Ogni possibile theta e sul'asse delle Y abbiamo tutti i possibili valori di rho.
        Ogni punto in questo array bidimensionale corrisponde ad una linea.
        Viene chiamato accumulatore perchè raccoglie prove su quali linee esistono nell'immagine.
        Il numero di righe viene imposto pari al doppio della distanza in modo da prevenire dei segmentation fault.
    */
    Mat accumulatore(2 * distanza, GRADI, CV_32SC1, Scalar(0) );

    //partendo dal centro dell'immagine, tracciamo idealmente una linea
    // che dal centro cade perpendicolarmente sulla retta che stiamo considerando.
    // la distanza dunque dal centro all'intersezione con la retta viene chiamata "rho"
    double rho;
    
    // theta è l'angolo che viene formato tra rho e l'asse X
    double theta;


    /*
        Per ogni pixel nell'immagine, poichè l'immagine di input è una immagine già sottoposta a 
        edge detection, consideriamo solo i pixel di bordo ( valore > 250).
        Per ognuno di questi pixel di edge calcoliamo il rho, di cui prima, di tutte le possibili 
        rette su cui giace il pixel considerato utilizzando la formula:
        rho = x * cos(theta) + y*sin(theta) dove theta è un angolo compreso tra 0 e 180 gradi e che viene
        portato in radianti nel calcolo del rho.
        Ad ogni rho sommiamo il valore della distanza, che rappresenta una costante che in seguito ci consentirà di 
        trovare il rho giusto per la conversione da coordinate polari a cartesiane.
        Una volta calcolato rho, viene incrementata la cella dell'accumulatore corrispondente alle coordinate (rho, theta)
    */
    for(int i=0; i<img.rows; i++)
        for(int j=0; j<img.cols; j++)
        {
            if(img.at<uchar>(i,j) > 250 ) 
            {
                for(int t=0; t<180; t++) // <=
                {
                    rho = round( j * cos(( t-90 )* RAD) + i * sin(( t-90 ) * RAD)) + distanza;
                    accumulatore.at<int>(rho, t)++;
                }
            }
        }


    /*
        Per ogni valore di rho (i) della matrice accumulatore, e per ogni theta (t)
        controlliamo se il valore contenuto nell'accumulatore alla posizione (rho, theta)
        è maggiore o uguale della soglia che viene passata in input:
        troviamo il valore esatto di rho sottraendo il valore di "distanza" al valore indice considerato;
        trasformiamo l'angolo t in radianti;
        definiamo una coppia di punti (dato che per due punti passa una sola retta);
        Invochiamo la funzione PolarToCartesian in modo tale da riposrtare le coordinate dei due punti in coordiante cartesiane
        Inseriamo i due punti ottenuti nel vettore di input/output che contiene le coppie di punti che definiscono le rette presenti sull'immagine.
    */
    for(int i=0; i<accumulatore.rows; i++)
        for(int t=0; t<180; t++)
        {
            if( accumulatore.at<int>(i,t) >= threshold)
            {
                rho = i-distanza;
                theta = (t-90) * RAD;

                Point p1,p2;
                PolarToCartesian(rho, theta, p1, p2);

                punti.push_back( make_pair( p1, p2 ) );                
            }
        }
}



void PolarToCartesian(double rho, double theta, Point& P1, Point& P2)
{
    /*
        Trasformazione da coordinate polari a cartesiane.
        Dati un rho ed un theta, le corrispettive coordinate cartesiane sono date 
        dalla formula:
        x = rho * cos(theta)
        y = rho * sin(theta)
        Poichè dobbiamo disegnare le rette per tutta l'immagine, a questi valori verrà 
        sommato e sottratto successivmanete un fattore di scala
    */
    int x0 = cvRound(rho * cos(theta));
	int y0 = cvRound(rho * sin(theta));


    /*
        Determinazione delle coordinate dei punti che definiscono la retta.
        Il fattore di scala viene aggiunto e sottratto per permettere ala linea di variare
        tra il valore positivo e negativo del fattore di scala.
        Le x e le y vengono moltiplicate, rispettivamente, per -sin(theta) e cos(theta) in modo tale 
        da far variare le coordinate cartesiane nei valori sia positivi che negativi.
    */
	P1.x = cvRound(x0 + 1000 * (-sin(theta)));
	P1.y = cvRound(y0 + 1000 * (cos(theta)));
	P2.x = cvRound(x0 - 1000 * (-sin(theta)));
	P2.y = cvRound(y0 - 1000 * (cos(theta)));
}