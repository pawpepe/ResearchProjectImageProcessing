#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include "raspi.h"
#include <time.h>
#include <fstream>



float tempo(clock_t t){

	float temp;
	t = clock() - t;
	temp = ((float)t) / CLOCKS_PER_SEC;
	cout << temp << endl;

	return temp;
}

float Angle(Mat img, Point point, int d){
	int ang;
	int x, y, z;
	float l1,l2;

	x = point.x - d;

	for (y = img.size().height; y > 0; y--){
		if (img.at<uchar>(y, x) == 0) {
			z = y;
			break;
		}
	}
		
		
			 l1 = sqrt(point.x*point.x + point.y*point.y);
			 l2 = sqrt(x*x + z*z);

			float dot = point.x * x + point.y * z;

			float a = dot / (l1 * l2);

			if (a >= 1.0)
				return 0.0;
			else if (a <= -1.0)
				return acos(a);
			else
				return acos(a); // 0..PI
	
}

void MyLine(Mat img, Point start, Point end, string image_name, float angle)
{
	int lineEnd;
	int thickness = 2;
	int lineType = 8;
	// create 8bit color image. IMPORTANT: initialize image otherwise it will result in 32F
	cv::Mat img_rgb(img.size(), CV_8UC3);

	// convert grayscale to color image
	cv::cvtColor(img, img_rgb, CV_GRAY2RGB);

	lineEnd = end.x;
	end.x = end.x + lineEnd * angle;

	line(img_rgb,
		start,
		end,
		Scalar(0, 0, 255),
		thickness,
		lineType);

	end.x = lineEnd - lineEnd*angle;

	line(img_rgb,
		start,
		end,
		Scalar(0, 0, 255),
		thickness,
		lineType);

	String name = "Line" + image_name;
	imwrite(name, img_rgb);

	
	ShowImage(img_rgb, img);

	


}

/*void save(Mat img, string name){
	string filename;

	time_t t = time(0);   // get time now
	struct tm * now = localtime(&t);

	char buffer[80];
	strftime(buffer, 80, "%Y-%m-%d_%H-%M-%s.jpg", now);

	string date = (buffer);

	filename = name + date;

	imwrite(filename, img);

}*/

void savePictures(Mat img, string name, string filter){

	string filename = filter + "_" + name ;
	imwrite(filename, img);
}

Point maxPoint(Mat aux){
	
	int hight = 0; 
	int width = 0;
	int x, y;

	for (x = aux.size().width * 0.5; x < aux.size().width - (aux.size().width*0); x++){
		for ( y = 0; y < aux.size().height; y++){
			if (aux.at<uchar>(y, x)==0) {
				if (y > hight){
					hight = y;
					width = x; 
					printf(" x Y : %d %d ", width, hight);
				}
				break;
			}
		}
	}

	printf(" CIMA P BAIXO : x Y : %d %d", width, hight);
	

	Point max = Point(width, hight);

		return max;
	}

	Point minPoint(Mat aux){ 

		int hight = aux.size().height;
		int width = 0;
		int x, y;


		for (x = 50; x < aux.size().width - 50; x++){
			for (y = aux.size().height; y > 0; y--){
				if (aux.at<uchar>(y, x) == 0) {
					if (hight > y){
						hight = y;
						width = x;
						printf(" x Y : %d %d ", width, hight);
					}
					break;
				}
			}
		}

		printf(" Baixo P CIMA : x Y : %d %d", width, hight);


		Point min = Point(width, hight);
		return min;
	}


	int intermadiatePoint(Point point, Mat img){
		int minPointY; 

	
			for (int y = point.y; y < img.size().height; y++){
				if (img.at<uchar>(y, point.x) == 255) {
					minPointY = y;
					break;
				}
					
			}
		

		return minPointY;
	}




int main(int argc, char ** argv)
{

	string gauss = "Gaussino";
	string canny = "Canny";
	string hough = "Hough";
	string binarizar = "Binarizar";
	string Otsu = "Otsu";
	string image_name = "";
	int number;
	Point min, max, start;

	ofstream myfile;

	myfile.open("data.txt");

	myfile << "ESCREVE QUALQUER COISA\n";
	

	clock_t t1, t2, t3, t4;
	double threshold1, threshold2, thres, minLength, maxGap;
	bool f1, f2, f3, f4, f5, f6, f7, f8, f9;
	string Result;
	ostringstream convert;
	//int i;
	float temp;

	//for (i = 1;  i <= 6; i++){

		//number = i;
		//convert << number;
		//Result = convert.str();
		//image_name = "a" + Result + ".JPG";
		image_name = "a2.JPG";
		//number++;
		//cout << number << endl;
		cout << image_name;


		myfile << image_name;
		myfile << "\n";

		t1 = clock();
		f1 = false;
		f2 = true;
		f3 = false;
		f4 = false;
		f5 = false;
		f6 = true;
		f7 = true;
		if (f7 == true){
			threshold1 = 10;
			threshold2 = 19;
		}
		f8 = false;
		f9 = true;
		if (f9 == true){
			thres = 10;// 40
			minLength = 20; //50
			maxGap = 30; //80

			/*
			CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );

			if ( !capture ) {
			fprintf( stderr, "ERROR: capture is NULL \n" );
			getchar();
			return -1;
			}
			string original = "original.jpg";
			string foto ="img";

			IplImage* frame = cvQueryFrame( capture );
			Mat img(frame);
			Mat I, I1, imge;
			cvtColor(img,imge,CV_RGB2GRAY);
			imge.convertTo(I, CV_8U);
			equalizeHist(I,I1);
			Mat aux = I1;
			savePictures(I1, original, foto);

			*/

			//realiza a leitura e carrega a imagem para a matriz I1
			// a imagem tem apenas 1 canal de cor e por isso foi usado o parametro CV_LOAD_IMAGE_GRAYSCALE
			Mat lara = imread("lara.JPG", CV_LOAD_IMAGE_GRAYSCALE);
			Mat I = imread(image_name, CV_LOAD_IMAGE_GRAYSCALE);
			if (I.empty())
				return -1;
			resize(I, I, lara.size(), 1.0, 1.0, INTER_LINEAR);
			Mat I1;
			//Mat aux = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE); 
			equalizeHist(I, I1);


			Mat aux, original;

			aux = I1;

			//ShowImage(I, I1);
			// verifica se carregou e alocou a imagem com sucesso
			if (I1.empty())
				return -1;

			// tipo Size contem largura e altura da imagem, recebe o retorno do metodo .size()
			//imSize = I1.size();

			// Cria uma matriz do tamanho imSize, de 8 bits e 1 canal

			Mat I2 = Mat::zeros(I1.size(), CV_8UC1);


			if (f2 == true) {
				t2 = clock();
				for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
					GaussianBlur(I1, I1, Size(i, i), 0, 0, BORDER_DEFAULT);
				//ShowImage(aux, I1);
				cout << "Guassiano tempo : ";
				temp = tempo(t2);
				savePictures(I1, image_name, gauss);
				myfile << "Gauss: ";
				myfile << temp;
				myfile << "\n";

			}

			if (f1 == true){
				t2 = clock();
				binarizacao(I1, 125);
				//ShowImage(aux, I1);
				cout << "binarizacao : ";
				temp = tempo(t2);
				savePictures(I1, image_name, binarizar);
				myfile << "Binarizacao: ";
				myfile << temp;
				myfile << "\n";


			}




			if (f3 == true){
				t2 = clock();
				inversao(I1);
				cout << "inversao : ";
				tempo(t2);

			}


			if (f4 == true){
				adaptiveThreshold(I1, I1, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 7, 0);
			}


			if (f5 == true)
				Laplacian(I1, I1, 125, 1, 1, 0, BORDER_DEFAULT);



			if (f7 == true){
				t2 = clock();
				Canny(I1, I2, threshold1, threshold2, 3, false);
				cout << "canny : ";
				temp = tempo(t2);
				savePictures(I2, image_name, canny);
				myfile << "Canny: " + (int)(temp * 1000);
				myfile << "\n";
			}



			if (f9 == true){
				t2 = clock();
				Hough(I2, aux, thres, minLength, maxGap);
				cout << "hough : ";
				temp = tempo(t2);
				savePictures(aux, image_name, hough);
				myfile << "Hough: ";
				myfile << temp;
				myfile << "\n";
			}

			if (f6 == true){
				t2 = clock();
				threshold_type = THRESH_BINARY;

				threshold(aux, I1, 9, max_BINARY_value, threshold_type);
				cout << "Threshold : ";
				//savePictures(aux, image_name, Otsu);
				temp = tempo(t2);
				myfile << "Threshold/OTSU: ";
				myfile << temp;
				myfile << "\n";
			}


			string name = Otsu + image_name;
			imwrite(name, aux);
			ShowImage(I1, aux);

			t2 = clock();
			max = maxPoint(aux);
			min = minPoint(aux);

			/*start.y = (max.y + min.y) / 2;
			start.x = (max.x + min.x) /2;*/

			start.x = max.x;
			start.y = max.y;

			Point end;

			end.x = start.x;
			end.y = aux.size().height;

			
			MyLine(I, start, end, image_name, 0.3);
			temp = tempo(t2);
			ShowImage(I, aux);

			myfile << "Rota: ";
			myfile << temp;
			myfile << "\n";

			temp = tempo(t1);
			cout << "Final time : ";
			myfile << "Final Time: ";
			myfile << temp;
			myfile << "\n";




			//float angle = Angle(aux, min, 5);

			//cout << angle; 

			

		}

	//}

		
		
		
		myfile.close();
		//ShowImage(aux, I1);

		//imwrite(argv[2], I2); // salva imagem I2 no arquivo definido pelo usuario em argv[2]
	//}
		return 0;
}