#include "descriptor.h"
 
#define eps 0.0001

#define EPSILON 0.01
#define BOUNDARY(X, MIN, MAX) ( (X) < (MIN) ? (MIN) : (X) >= MAX ? MAX-1 : (X) )

static inline double minimum(double x, double y) { return (x <= y ? x : y); }
static inline double maximum(double x, double y) { return (x <= y ? y : x); }

static inline int minimum(int x, int y) { return (x <= y ? x : y); }
static inline int maximum(int x, int y) { return (x <= y ? y : x); }


/***************************
Function to get the image data as a double in row-major order
Currently, only for grayscale images.
*********************/
void getImgData_asDouble_inRowMajorOrder(IplImage* im, double **data)
{
    Mat imgMat(im);
    int cnt = 0;
    if(data)
    {
        if(!*data)
            (*data) = (double*)calloc(imgMat.rows*imgMat.cols,sizeof(double));

        for(int i = 0; i < imgMat.rows; i++)
            for(int j = 0; j < imgMat.cols; j++)
                (*data)[cnt++] = (double)imgMat.at<uchar>(i,j);
    }
    else
    {
        fprintf(stderr,"getImgData_asDouble_inRowMajorOrder :: No pointer to the image in array format given!\n");
        exit(-1);
    }
}

/*****************************************************
Function to sort the intensity values using quicksort
******************************************************/

void quickSort(double arr[], int left, int right)
{
	int i = left, j = right;
	double tmp;
	double pivot = arr[(left + right) / 2];

	/* partition */
	while (i <= j)
	{
		while (arr[i] < pivot)
			i++;
		while (arr[j] > pivot)
			j--;
		if (i <= j)
		{
			tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;
			i++;
			j--;
		}
	}

	/* recursion */
	if (left < j)
		quickSort(arr, left, j);
	if (i < right)
		quickSort(arr, i, right);
}

bool isSorted(float *array, int size)
{
    bool isSorted = true;
    for(int i = 0; i < size-1; i++)
        if(array[i] > array[i+1])
        {
            isSorted = false;
            break;
        }
    return isSorted;
}

/***************************************************************
Function for smoothing the image - a simple (3x3) box filter [w w w; w (1-8w) w; w w w]
****************************************************************/
void apply_filter(double *img_rank, double *img_rank_s, int rows, int cols)
{
	double w=0.05, wt;
    int x,y;
    int i,j;
	for(x=0;x<rows;x++)
	{
		for(y=0;y<cols;y++)
		{
			img_rank_s[rows*y+x]=0;
            wt = 0;
            for(i=-1;i<=1;i++)
			{
				for(j=-1;j<=1;j++)
                {
					if(x+i>0 && x+i < rows)
					{
						if(y+j>0 && y+j < cols)
						{
							img_rank_s[rows*(y)+(x)] += w * img_rank[rows*(y+j)+(x+i)];
                            wt += w;
                        }
                    }
				}
             }
             //printf("\n%f %f",wt,img_rank_s[rows*(y)+(x)]);
             img_rank_s[rows*(y)+(x)] += (1 - wt)*img_rank[rows*(y)+(x)];
             //printf("\n%f %f",wt,img_rank_s[rows*(y)+(x)]);
         }
     }
}
/***************************************************************
Function for smoothing the image - a simple (3x3) box filter [w w w; w (1-8w) w; w w w]
****************************************************************/
void apply_filter(float **img_rank, double *img_rank_s, int rows, int cols)
{
	double w=0.05, wt;
    int x,y;
    int i,j;
	for(x=0;x<rows;x++)
	{
		for(y=0;y<cols;y++)
		{
			img_rank_s[x*cols + y]=0; //changed from img_rank_s[x+y*rows]
            wt = 0;
            for(i=-1;i<=1;i++)
			{
				for(j=-1;j<=1;j++)
                {
					if(x+i>0 && x+i < rows)
					{
						if(y+j>0 && y+j < cols)
						{
							img_rank_s[x*cols + y] += *(*(img_rank+x+i)+y+j) * w;
                            wt += w;
                        }
                    }
				}
             }
             img_rank_s[cols*x + y] += *(*(img_rank+x) + y) * (1 - wt);
         }
     }
}

/* replicates the find() function of Matlab */
void find(const cv::Mat& binary, std::vector<cv::Point>& idx)
{
    assert(binary.cols > 0 && binary.rows > 0 && binary.channels() == 1 && binary.depth() == CV_8U);
    const int M = binary.rows;
    const int N = binary.cols;
    for (int m = 0; m < M; ++m)
    {
        const unsigned char* bin_ptr = binary.ptr<unsigned char>(m);
        for (int n = 0; n < N; ++n)
        {
            if (bin_ptr[n] > 0)
				idx.push_back(cv::Point(n,m));
        }
    }
}


/********************************************************************************************************************
Function to compute the gradient of an image
Input - inputImg: image whose gradient will be calculated
Outparameter - imgGrad: pointer to Gradient structure
Algorithm:
	Compute the gradient in X and Y directions by convolving with [ -1 0 1 ] and [ -1 0 1 ]' respectively,
	for each channel i.e. RGB.  The gradient channel with the highest magnitude will be the gradient for that
	pixel.
	The orientation of the pixel will be the orientation of the dominant gradient channel.
*********************************************************************************************************************/

int imageGradient(const IplImage *inputImg, Grad_s *imgGrad)
{

	int height_i = inputImg->height;
	int width_i	= inputImg->width;

	int h, w, ch;
	int nChannels_i = inputImg->nChannels;
	register long widthStep_i = inputImg->widthStep;
	float gradMax_f, gradMag_f;
	float deltaX_f,deltaY_f,theta_f;

	for (h=0; h<height_i; h++){
		for (w=0; w<width_i; w++)
		 {
			//initialize max of gradient to -1 (gradient magnitude is always > 0)
			gradMax_f = -1;
			for (ch=0; ch< nChannels_i; ch++)
			 {
				// convolve current pixel with [-1 0 1]' to get change in Y direction
				deltaY_f = (float)((uchar*)(inputImg->imageData + (BOUNDARY(h+1, 0, height_i) * widthStep_i)))[w*nChannels_i  + ch] - \
						((uchar*)(inputImg->imageData + (BOUNDARY(h-1, 0, height_i) * widthStep_i)))[w*nChannels_i  + ch];

				// convolve current pixel with [-1 0 1] to get change in X direction
				deltaX_f = (float)((uchar*)(inputImg->imageData + (h  * widthStep_i)))[nChannels_i*BOUNDARY(w+1, 0, width_i) + ch] - \
						((uchar*)(inputImg->imageData + (h  * widthStep_i)))[ nChannels_i*BOUNDARY(w-1, 0, width_i) + ch];

				// gradient magnitude for the current colour channel
				gradMag_f = (float)(fabs(deltaY_f)+fabs(deltaX_f));

				// check if the current channel has max gradient
				if(gradMax_f < gradMag_f)
				{
					gradMax_f = gradMag_f;
					// calculate orientation at the current pixel, atan2 gives result between -PI and PI
					theta_f = (float)atan2((float) deltaY_f, (float) deltaX_f);
					// the orientations should be between 0 and PI, hence add PI if angle < 0
					if(theta_f < 0)
						theta_f = theta_f + (float)CV_PI;
				}
			}
		// store the dominant gradient magnitude and the corresponding orientation after thresholding
		imgGrad->magnitude_ppf[h][w] = gradMax_f;
		imgGrad->orientation_ppf[h][w] = theta_f;
		}
	}

	return 0;
}




/***************************************************************
Function to compute histogram of a cell
****************************************************************/
void computeCellHistogram(int dims[2], double *img_rank_s, cell_info c_inf, int row_index, int col_index, Grad_s *imgGrad, imgHist_s *imgHist, dtct_s dtctInfo, int method_no, int n_channels)
{
	int *binCenter=0; //stores the bins to which current pixel contributes its vote
	int binCnt; //stores the number of bins to which a pixel contributes its vote

	int chan;
	//two for loops iterate over the cell
	for(int m=c_inf.cell_x; m < c_inf.cell_x+c_inf.hCell; m++)
	{
		for(int n=c_inf.cell_y; n < c_inf.cell_y+c_inf.wCell; n++)
		{
			binCenter = 0;

            //for HOG
                    binCnt = 1;
                    binCenter = (int*)calloc(binCnt,sizeof(int));
                    *binCenter = (int)((floor)(imgGrad->orientation_ppf[m][n] / (CV_PI / imgHist->nBins_i)));
                    if(binCenter[0] == imgHist->nBins_i)
                        binCenter[0]--; //happens when orientation is CV_PI
                    imgHist->hist_ppf[row_index][binCenter[0] + (imgHist->nBins_i*col_index)] += (imgGrad->magnitude_ppf[m][n]);

			if(binCenter)
				free(binCenter);
		}
	}
}


/*****************************************************************************************************************
Function to divide the gradient image into cells and calculate histogram for each cell
Input - imgGrad: Gradient image
Outparameter - imgHist : pointer to the cell wise image histogram

******************************************************************************************************************/
int cellwiseImgHistogram(double *img_rank_s, Grad_s *imgGrad,imgHist_s *imgHist,dtct_s dtctInfo,int method_no,int n_channels,int width,int height)
{
	int i,j,k,l;
	int widthImg_i,heightImg_i;
	int wCell_i,hCell_i;
	widthImg_i = width;
	heightImg_i = height;
	wCell_i = dtctInfo.wCell_i;
	hCell_i = dtctInfo.hCell_i;
	int dims[2];
	dims[0] = widthImg_i;
	dims[1] = heightImg_i;

	/**********************************************************************************
		 histogram indexes- k points to row, l points to column
		 k is incremented when we go to the next cell row wise and
	 	 l is incremented when we go to the next cell columnwise
		 array traversal for histogram calculation - left to right,top to bottom
	***********************************************************************************/

	k=0,l=0;
	for(i=0; i < heightImg_i; i+=hCell_i)
	{
		for(j=0; j < widthImg_i; j+=wCell_i)
		{
			cell_info c_inf;
			c_inf.cell_x = i;	c_inf.cell_y = j;
			c_inf.hCell = hCell_i; c_inf.wCell = wCell_i;
			computeCellHistogram(dims,img_rank_s, c_inf,k,l,imgGrad,imgHist,dtctInfo,method_no,n_channels);
			l++; // increment column index for histogram
		}
		k++; // increment row index for histogram and reset column index
		l=0;
	}
	return 0;
}

/*************************************************************************************************************************
Used for HOG to make them invariant to illumination
Function to compute the normalized respones of each cell by pooling them spatially in blocks
Input- (rowIndex_i,colIndex_i): starting index for the detection window
imgHist: cell wise image histogram of the entire image
Output - descriptor
**************************************************************************************************************************/

descriptor_s* normalizeWindow(int rowIndex_i,int colIndex_i,const dtct_s *dtctInfo,imgHist_s *imgHist,int normalize_reqd)
{

	int i,j,k,l,m,n;

	int stride_i = dtctInfo->strideBlck_i/dtctInfo->hCell_i;
	int nCellinBlk_i = dtctInfo->wBlck_i * dtctInfo->hBlck_i;
	int nBlkinWin_i = ((dtctInfo->wDtct_i/dtctInfo->wCell_i)-stride_i) * ((dtctInfo->hDtct_i/dtctInfo->hCell_i)-stride_i);

	int blockFeatureSize_i = dtctInfo->nBins_i * nCellinBlk_i;
	int featureVectorSize_i = blockFeatureSize_i * nBlkinWin_i;

	descriptor_s *windowDescriptor;
	windowDescriptor = (descriptor_s*) malloc(sizeof(descriptor_s));
	windowDescriptor->descriptorInit(featureVectorSize_i);

	float vectorMag_f=0.0,sqrtSum_f,normalizeValue_f;
	int maxRowIndex_i = rowIndex_i+ (dtctInfo->hDtct_i/dtctInfo->hCell_i) - dtctInfo->hBlck_i;
	//int maxColIndex_i = colIndex_i+ (dtctInfo->wDtct_i/dtctInfo->wCell_i) - dtctInfo->wBlck_i;

	k=0;l=0;
	for(i=rowIndex_i; i <= maxRowIndex_i; i=i+stride_i) //window
	{
		for(j=colIndex_i; j <= (imgHist->nBins_i*(imgHist->width_i-dtctInfo->wBlck_i)); j=j+dtctInfo->nBins_i)
		{

			for(m=i; m < (i+dtctInfo->hBlck_i); m++) //block
			{
				for(n=j; n < j+(dtctInfo->nBins_i * dtctInfo->wBlck_i); n++)
				{
					vectorMag_f += imgHist->hist_ppf[m][n] * imgHist->hist_ppf[m][n];l++;
				}
			}


			sqrtSum_f = (float)sqrt(vectorMag_f + EPSILON);

			for(m=i; m < (i+dtctInfo->hBlck_i); m++) //block
			{
				for(n=j; n < j+(dtctInfo->nBins_i * dtctInfo->wBlck_i); n++)
				{
					if(normalize_reqd == 1)
					{
						normalizeValue_f = imgHist->hist_ppf[m][n]/sqrtSum_f;
						//if(normalizeValue_f != normalizeValue_f) normalizeValue_f=0.0; // check for nan value
						windowDescriptor->featureVector_pf[k++] = normalizeValue_f;
                    }
					else
						windowDescriptor->featureVector_pf[k++] = imgHist->hist_ppf[m][n]; //normalizeValue_f;
				}
			}
			vectorMag_f=0.0;
		}
	}
	return windowDescriptor;
}

/***************************************************************************
Function to merge the feature vector values
Input :- 2 feature vectors
Output :- Single feature vector
Used when using combination of descriptors (like HOG+HOG)
****************************************************************************/

descriptor_s* mergeDescriptors(descriptor_s* descriptorData1,descriptor_s* descriptorData2)
{

	int i;
	descriptor_s *descriptorData;
	descriptorData = (descriptor_s*)malloc(sizeof(descriptor_s));
	descriptorData->descriptorInit(descriptorData1->descriptorSize_i+descriptorData2->descriptorSize_i);
	for(i=0;i<descriptorData1->descriptorSize_i;i++)
		descriptorData->featureVector_pf[i] = descriptorData1->featureVector_pf[i];
	for(i=descriptorData1->descriptorSize_i;i<descriptorData->descriptorSize_i;i++)
		descriptorData->featureVector_pf[i] = descriptorData2->featureVector_pf[i-descriptorData1->descriptorSize_i];

	descriptorData1->descriptorRelease();
	free(descriptorData1);

	descriptorData2->descriptorRelease();
	free(descriptorData2);

	return descriptorData;
}

