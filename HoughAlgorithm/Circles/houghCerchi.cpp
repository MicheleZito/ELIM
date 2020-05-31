#include<iostream>
#include<opencv2/opencv.hpp>
#include<cmath>
#include<vector>
#define GRADI 180
#define RAD CV_PI/180


using namespace std;
using namespace cv;

void HoughTransform(Mat output, Mat img, int threshold, int minRad, int maxRad);

int main(int argc, char* argv[])
{
    if(argc != 5)
    {
        cerr << "Utilizza: <nome eseguibile> <path immagine> <threshold hough> <minRad> <maxRad>" << endl;
        exit(1);
    }

    Mat image = imread( argv[1], IMREAD_GRAYSCALE ); /// imread_color
    Mat canny, gauss, result;
    int threshold = atoi( argv[2] );
    int minRad = atoi(argv[3]);
    int maxRad = atoi(argv[4]);

    if( image.data == nullptr)
    {
        cerr << "Errore apertura immagine" << endl;
        exit(1);
    }

    namedWindow("Immagine Originale", WINDOW_NORMAL);
    imshow("Immagine Originale", image);

/////////////////////////////////////////// da utilizzare se l'immagine di input viene ottenuta a colori
    //Mat gray_temp = image.clone();
    //cvtColor(image, gray_temp, COLOR_BGR2GRAY);
    /////////////////////////////////////////////////////

    GaussianBlur( image, gauss, Size(5,5), 1.4 ); ///gray_temp = image
    Canny( gauss, canny, 100, 150, 3 );

    HoughTransform(image, canny, threshold, minRad, maxRad);

    
    waitKey();

    return 0;
}

void HoughTransform( Mat output, Mat img, int threshold, int minRad, int maxRad)
{
    int i,j,r;   

    //In Hough per cerchi l'accumulatore è tridimensionale
    //Prima dimensione = (maxRad-minRad)+1 Seconda Dim= img.rows Terza Dim= img.cols
    vector<vector<vector<int> > > accumulatore; 
    accumulatore.resize( (maxRad-minRad) + 1 );


    // inizializzazione accumulatore
    for(r = 0; r < (maxRad-minRad)+1; r++)
        accumulatore.at(r).resize(img.rows);
    
    for(r = 0; r < (maxRad-minRad)+1; r++)
        for(i = 0; i < img.rows; i++)
            accumulatore.at(r).at(i).resize(img.cols);
    
    for(r = 0; r < (maxRad-minRad)+1; r++)
        for(i = 0; i < img.rows; i++)
            for(j = 0; j < img.cols; j++)
                accumulatore.at(r).at(i).at(j) = 0;



    /*
        Per ogni pixel valido dell'immagine di input, che è stata sottoposta al Canny edge detector, e che quindi ha
        solo i pixel di bordo con valore > 250, eseguiamo un ciclo per tutti i raggi possibili dei cerchi da trovare,
        range di raggi he viene  specificato dall'utente. Poichè nello spazio dei parametri abbiamo che a e b sono le coordinate
        del centro del cerchio nello spazio immagine. Poichè più punti nello SI appartenenti ad uno stesso cerchio producono
        gli stessi risultati nello SP, alla fine i centri più votati in SP sono più probabilmente cerchi reali nell'immagine.
        poichè gli assi cartesiani in una immagine sono invertiti, e poichè le formule per trasformare le coordinate di un punto
        in polari sono a = x - R*cos(theta), b = y - R*sin(theta) a queste due formule per x ed y sostituiamo gli indice che, 
        rispettivamente, scorrono le colonne e le righe della matrice che rappresena l'immagine.
        Dopo aver effettuato un controllo sul valore di questi due risultati, viene incrementato il corrispettivo elemento
        nella matrice dell'accumulatore. In questo modo, più punti fanno parte dello stesso cerchio, maggiore sarà l'elemento contenuto in
        una posizione dell'accumulatore.
    */
    for(int i = 0; i < img.rows; i++)
        for(int j = 0; j < img.cols; j++)
            if(img.at<uchar>(i,j) > 250 ) 
                for(r = minRad; r < maxRad; r++)
                    for(int t = 0; t < 360; t++)
                    {
                        int a = ( j -  r * cos(t * CV_PI/180) );
                        int b = ( i -  r * sin(t * CV_PI/180) );
                        if( (a > 0) && (b > 0) && (a < img.cols) && (b < img.rows))
                            accumulatore.at(r-minRad).at(b).at(a)++;
                    }

    //Dichiariamo un vector di Vec3f che rappresenta i cerchi da mostrare sull'immagine
    vector<Vec3f> cerchi;

    // per ogni elemento contenuto nell'accumulatore viene controllato se il suo valore è maggiore o ugulae della soglia
    // stabilita. in caso affermativo viene aggiunto un nuovo elemento al vector, contenente le tre componenti del cerchio.
    for(r=0; r< (maxRad-minRad)+1; r++)
        for(i=0; i<img.rows; i++)
            for(j=0; j<img.cols; j++)
                if( accumulatore.at(r).at(i).at(j) >= threshold)
                {
                    Vec3f c(r+minRad, i, j);
                    cerchi.push_back( c );                
                }
    

    //Mat mask = Mat::zeros(output.size(), output.type());/////////////
    //cvtColor(mask, mask, COLOR_GRAY2BGR); // da usare se l'output è in scala di grigio
    cvtColor(output, output, COLOR_GRAY2BGR);
    for(i=0; i<cerchi.size(); i++)
    {
        Point centro(cvRound(cerchi[i][2]), cvRound(cerchi[i][1]));
        int raggio = cvRound(cerchi[i][0]);

        circle(output, centro, raggio, Scalar(0,0,255), 1, 8, 0);
        //circle(mask, centro, raggio, Scalar(0,0,255), -1, 8, 0); cerchio pieno da disegnare sulla maschera
    }


    //////////////////////////////////////////////////// Segmentazione di un cerchio
    //for(int i = 0; i < output.rows; i++)
    //    for(int j = 0; j < output.cols; j++)
    //        if(mask.at<Vec3b>(i,j) == Vec3b(0,0,0))
    //            output.at<Vec3b>(i,j) = Vec3b(0, 0, 0);   // se l'immagine di output è in scala di grigio allora dovrà essere: output.at<uchar>(i,j) = uchar(0);
    ////////////////////////////////////////////////////


    namedWindow("Hough Cerchi", WINDOW_NORMAL);
    imshow("Hough Cerchi", output);
}
