
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <math.h>
#include <string.h>

#define MAX_KERNEL_LENGTH  31
using namespace cv;
using namespace std;

Size imSize;



/// Global variables for Threshold and Canny tests 

int threshold_value = 0;
int threshold_type = 3;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;


Mat detected_edges;

int edgeThresh = 1;
int lowThreshold = 100;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;

//Mat src, dst;
char* window_name = "Threshold Demo";

char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";

/// Function headers Threshold and Canny tests
void Threshold_Demo(int, void*);

void TresholdCannyTest();



//HIGHLIGH 
// Methods that poprably will enhance the image 
void adaptiveBilateralFilter(InputArray src, OutputArray dst, Size ksize, double sigmaSpace, double maxSigmaColor = 20.0, Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT);




/**
src – input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
dst – output image of the same size and type as src.
ksize – Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero’s and then they are computed from sigma* .
sigmaX – Gaussian kernel standard deviation in X direction.
sigmaY – Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height , respectively (see getGaussianKernel() for details); to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
borderType – pixel extrapolation method (see borderInterpolate() for details)
*/
void cv::GaussianBlur(InputArray, OutputArray, Size, double, double, int);

/**
src – input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
dst – destination array of the same size and type as src.
ksize – aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ..
*/
void cv::medianBlur(InputArray, OutputArray, int);





/** This functions displays two images
Param 1 : Image 1
Param 2 : Image 2
*/
void ShowImage(Mat I1, Mat I2){

	namedWindow("I1", CV_WINDOW_AUTOSIZE);// Create a window for display.
	namedWindow("I2", CV_WINDOW_AUTOSIZE);// Create a window for display.

	imshow("I1", I1);                   // Show our image inside it.
	imshow("I2", I2);                   // Show our image inside it.

	waitKey(0); // aguarda indefinidamente por um pressionamento de tecla
}

/**
This functions finds the maximum value of a pixel into the image
return : max pixel value for the input image
*/
int maxPixel(Mat Image){

	int maxPix = 0;

	for (int x = 0; x < imSize.height; x++) {
		for (int y = 0; y < imSize.width; y++){// o metodo at<tipo>(i,j) eh usado para acessar um pixel da imagem
			if (Image.at<uchar>(x, y) > maxPix)
				maxPix = Image.at<uchar>(x, y);
		}
	}

	return maxPix;
}

/**
T(r) = c*log(1 + |r|)
c = (255) / log(1+R)

Esta funcao comprime o intervalo dinamico(razao entre o menor e o maior intensidade)
Desta forma, os pixels com menores valores sao realcados.

Param 1:  Imagem a ter os pixels realcados
Param 2:  Valor do maior pixel da imagem do Param 1

*/
void operador_logatimo(Mat Image, int maxPix){
	float c;

	c = (255) / log(1 + maxPix);

	for (int x = 0; x < Image.size().height; x++) {
		for (int y = 0; y < Image.size().width; y++)// o metodo at<tipo>(i,j) eh usado para acessar um pixel da imagem
			Image.at<uchar>(x, y) = c * log(1 + abs(Image.at<uchar>(x, y)));
	}

}
/**
Inversao(Negativo)
T(r) = 255 - r
*/
void inversao(Mat Image){
	for (int x = 0; x < Image.size().height; x++) {
		for (int y = 0; y < Image.size().width; y++)// o metodo at<tipo>(i,j) eh usado para acessar um pixel da imagem
			Image.at<uchar>(x, y) = 255 - Image.at<uchar>(x, y);
	}
}


/**
Tecnica de binarizacao serve para encontrar componentes conexos. Uma dos tipos de limiarizacao(Threshold).

g(x,y) = { 1, if f(x,y) => max value
{ else 2555
*/
void binarizacao(Mat Image, double value){

	for (int x = 0; x < Image.size().height; x++) {
		for (int y = 0; y < Image.size().width; y++) {
			// o metodo at<tipo>(i,j) eh usado para acessar um pixel da imagem
			if (Image.at<uchar>(x, y) <= value)
				Image.at<uchar>(x, y) = 0;

			else Image.at<uchar>(x, y) = 255;
		}
	}
}


/**
* @function Threshold_Demo
*/
void Threshold_Demo(int, void*, Mat src, Mat dst)
{
	/* 0: Binary
	1: Binary Inverted
	2: Threshold Truncated
	3: Threshold to Zero
	4: Threshold to Zero Inverted
	*/

	threshold(src, dst, threshold_value, max_BINARY_value, threshold_type);

	imshow(window_name, dst);
}



//Restoration :
// Functions that reduce image noise based on how the image was deteriored

/** Arithimetic average to reduce the noise, bt using the average of neighbors pixels 
	Param 1 : Image to be restored 
*/
void averageFilter(Mat I1){
	Mat I2;
	imSize = I1.size();
	for (int x = 0; x < imSize.height; x++) {
		for (int y = 0; y < imSize.width; y++){// o metodo at<tipo>(i,j) eh usado para acessar um pixel da imagem
			if (x == 0 && y == 0){
				I2.at<uchar>(x, y) = (I1.at<uchar>(x, y) + I1.at<uchar>(x + 1, y) + I1.at<uchar>(x, y + 1)) / 3;
			}
			else if (x == 0){
				I2.at<uchar>(x, y) = (I1.at<uchar>(x, y) + I1.at<uchar>(x + 1, y) + I1.at<uchar>(x, y + 1) + I1.at<uchar>(x, y - 1)) / 4;
			}
			else if (y == 0){
				I2.at<uchar>(x, y) = (I1.at<uchar>(x, y) + I1.at<uchar>(x, y + 1) + I1.at<uchar>(x + 1, y) + I1.at<uchar>(x - 1, y)) / 4;
			}
			else
				I2.at<uchar>(x, y) = (I1.at<uchar>(x, y) + I1.at<uchar>(x + 1, y) + I1.at<uchar>(x - 1, y) + I1.at<uchar>(x, y + 1) + I1.at<uchar>(x, y - 1)) / 5;	
		}
	}

	I1 = I2;
}
/**
    It performs a image denoise in a greyscale image.

	Param 1 : src – Input 8 - bit 1 - channel, 2 - channel or 3 - channel image.
	Param 2 : dst – Output image with the same size and type as src .
	Param 3 : templateWindowSize – Size in pixels of the template patch that is used to compute weights.Should be odd.Recommended value 7 pixels
	Param 4 : searchWindowSize – Size in pixels of the window that is used to compute weighted average for given pixel.Should be odd.Affect performance linearly : greater searchWindowsSize - greater denoising time.Recommended value 21 pixels
	Param 5 : h = 3 – Parameter regulating filter strength.Big h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noisesrc – Input 8 - bit 1 - channel, 2 - channel or 3 - channel image.
	Param 6 : templateWindownSize = 7 
	Param 7 : searchWindowSize = 21
	*/
void fastNlMeansDenoising(InputArray, OutputArray, float, int, int); 

//SEGMENTATION :  Process of partitioning an image into mutiple segments. The goals is simplify the analysis 
//                of the image, and give some meaning for it. 
// Methods : 
//			Threshold 
//			Border detector 
//		    Laplacian of Gauss ( LoG )
//			Sobel
//			Canny

/**
src – Source 8-bit single-channel image.
dst – Destination image of the same size and the same type as src .
maxValue – Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.
adaptiveMethod – Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C . See the details below.
thresholdType – Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
C – Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
*/
void cv::adaptiveThreshold(InputArray, OutputArray, double, int, int, int, double);


/** agucamento da imagem
src – Source image.
dst – Destination image of the same size and the same number of channels as src .
ddepth – Desired depth of the destination image.
ksize – Aperture size used to compute the second-derivative filters. See getDerivKernels() for details. The size must be positive and odd.
scale – Optional scale factor for the computed Laplacian values. By default, no scaling is applied. See getDerivKernels() for details.
delta – Optional delta value that is added to the results prior to storing them in dst .
borderType – Pixel extrapolation method. See borderInterpolate() for details.
*/
void cv::Laplacian(InputArray, OutputArray, int, int, double, double, int);

/**
image – single-channel 8-bit input image.
edges – output edge map; it has the same size and type as image .
threshold1 – first threshold for the hysteresis procedure.
threshold2 – second threshold for the hysteresis procedure.
apertureSize – aperture size for the Sobel() operator.
L2gradient – a flag, indicating whether a more accurate
L_2 norm  =\sqrt{(dI/dx)^2 + (dI/dy)^2} should be used to calculate the image
gradient magnitude ( L2gradient=true ), or whether the default  L_1 norm  =|dI/dx|+|dI/dy| is enough ( L2gradient=false ).
*/
void cv::Canny(InputArray, OutputArray, double, double, int, bool);

/**void CannyThreshold( Mat src, Mat dst)
{

	/// Reduce noise with a kernel 3x3
	//blur(src, detected_edges, Size(3, 3));

	//for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	/// Canny detector
	Canny(src, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	src.copyTo(dst, detected_edges);
	imshow(window_name, dst);
}
*/





/**
* @function main
*/
/*void TresholdCannyTest(){


	/// Create a window
	namedWindow("Canny", CV_WINDOW_AUTOSIZE);
	//namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Create a Trackbar for user to enter threshold
	createTrackbar("Min Threshold:", "Canny", &lowThreshold, max_lowThreshold, CannyThreshold);
	//createTrackbar(trackbar_type,window_name, &threshold_type, max_type, Threshold_Demo);

	//Threshold_Demo(0, 0);

	/// Show the image
	///CannyThreshold(0, 0);

	while (true)
	{
		int c;
		c = waitKey(20);
		if ((char)c == 27)
		{
			break;
		}
	}

	/// Wait until user exit program by pressing a key
	waitKey(0);

}
*/

/**
src: imagem 
dst: Output of the edge detector.It should be a grayscale image(although in fact it is a binary one)
lines : A vector that will store the parameters(x_{ start }, y_{ start }, x_{ end }, y_{ end }) of the detected lines
rho : The resolution of the parameter r in pixels.We use 1 pixel.
theta : The resolution of the parameter \theta in radians.We use 1 degree(CV_PI / 180)
threshold: The minimum number of intersections to “detect” a line
minLinLength : The minimum number of points that can form a line.Lines with less than this number of points are disregarded.
maxLineGap : The maximum gap between two points to be considered in the same line.
*/
void cv::HoughLinesP(InputArray, OutputArray, double, double, int, double, double );


int fourier(char filename[36]){
	Mat I = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (I.empty())
		return -1;



	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix

	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
	// viewable image form (float between values 0 and 1).

	imshow("Input Image", I);    // Show the result
	imshow("spectrum magnitude", magI);
	waitKey();

	return 0;
}

/**transformada de Hough */

void Hough(Mat dst, Mat color_dst, int threshold, int minLength, int maxGap){
	vector<Vec4i> lines;
	HoughLinesP(dst, lines, 1, CV_PI / 180,threshold, minLength, maxGap);
	for (size_t i = 0; i < lines.size(); i++)
	{
		line(color_dst, Point(lines[i][0], lines[i][1]),
			Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8);
	}
}

/// CLASSIFICACAO :
