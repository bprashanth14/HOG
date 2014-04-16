#ifndef HOG_TYPES_H
#define HOG_TYPES_H

#define STR_SIZE 500

#include <stdio.h>
#include <stdlib.h>
//#include "stdafx.h"
#include <ctype.h>
#include <cv.h>
#include <highgui.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include "math.h"
#include <omp.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>

using namespace cv;
using namespace std;

/*********
structure to hold the frequency of occurency of a value
*********/
typedef struct sFreq
{
    float val;
    long freq;
}freq;

/***********************************************************
structure to hold the feature descriptor
************************************************************/

typedef struct descriptor
{
	int descriptorSize_i;  		// length of feature vector
	float* featureVector_pf;
	void descriptorInit(int descpSize_i)
	{
		descriptorSize_i = descpSize_i;
		featureVector_pf = (float*) malloc(sizeof(float) * descpSize_i);
		if(featureVector_pf == NULL)
		{
			free(featureVector_pf);
			printf("Memory allocation failed while allocating for featureVector_pf.\n");
			exit(-1);
		}
	}
	void descriptorRelease()
	{
		free(featureVector_pf);
	}

} descriptor_s;

/**************************************************************
structure to hold information required to create the descriptor
***************************************************************/

typedef struct dtct
 {
    int	    	nBins_i, descriptorSize_i;
    int	    	wDtct_i, hDtct_i;
    int	    	wBlck_i, hBlck_i;
    int	    	wCell_i, hCell_i;
    int	    	strideBlck_i, strideWin_i;

    dtct()
	{
		nBins_i = 9;
		wDtct_i = 64;  hDtct_i = 128;
		wBlck_i = 2;   hBlck_i = 2;
		wCell_i = 8;   hCell_i = 8;
		strideBlck_i = 8;  strideWin_i = 8;

	}
} dtct_s;

/***************************************************************************************************
					structure to hold the gradient of an image
***************************************************************************************************/
typedef struct Gradient
{
	float **magnitude_ppf;
	float **orientation_ppf;
	int widthImg_i,heightImg_i;

	void Gradientinit(int wImg_i,int htImg_i)
	{
		int i,j;
		widthImg_i = wImg_i; heightImg_i = htImg_i;

		/***************************************************************************************************
			Allocate memory for the gradient magnitude
		***************************************************************************************************/
		magnitude_ppf = (float **) calloc(heightImg_i , sizeof(float *));

		if(magnitude_ppf == NULL)
		{
			//free(magnitude_ppf);
			printf("Memory allocation failed while allocating for magnitude_ppf.\n");
			exit(-1);
		}
		for(i = 0; i < heightImg_i; i++)
		{
    		magnitude_ppf[i] = (float *) calloc(widthImg_i , sizeof(float));
    		if(magnitude_ppf[i] == NULL)
			{
				//free(magnitude_ppf[i]);
				printf("Memory allocation failed while allocating for magnitude_ppf[i][].\n");
				exit(-1);
			}
		}

		// initialize gradient magnitude to 0
		for(i = 0; i < widthImg_i; i++)
    		for(j = 0; j < heightImg_i; j++)
        		magnitude_ppf[j][i] = 0;


		/***************************************************************************************************
			Allocate memory for the gradient orientation
		***************************************************************************************************/
		orientation_ppf = (float **) calloc(heightImg_i , sizeof(float *));
		if(orientation_ppf == NULL)
		{
			//free(orientation_ppf);
			printf("Memory allocation failed while allocating for orientation_ppf.\n");
			exit(-1);
		}
		for(i = 0; i < heightImg_i; i++)
		{
    		orientation_ppf[i] = (float *) calloc(widthImg_i , sizeof(float));
    		if(orientation_ppf[i] == NULL)
			{
				//free(orientation_ppf[i]);
				printf("Memory allocation failed while allocating for orientation_ppf[i][].\n");
				exit(-1);
			}
		}

		// initialize gradient orientation to 0
		for(i = 0; i < widthImg_i; i++)
    		for(j = 0; j < heightImg_i; j++)
        		orientation_ppf[j][i] = 0;
	}

	void GradientRelease()
	{
		int i;

		// release the memory allocated to gradient magnitude and direction
		for(i = 0; i < heightImg_i; i++)
    		free(magnitude_ppf[i]);
		free(magnitude_ppf);

		for(i = 0; i < heightImg_i; i++)
    		free(orientation_ppf[i]);
		free(orientation_ppf);
	}
} Grad_s;


/***************************************************************************************************
					structure to hold the rank of an image
***************************************************************************************************/
typedef struct rank
{
	double *img_rank,*img_rank_t,*img_rank_s,*rank_range;
	double range_rank;
	int rank_bin;

	void rankInit(int dims[],int no_bins,int nChannels, long pxlCnt)
	{
		rank_bin = no_bins;

		img_rank   = (double *) calloc(1,dims[0]*dims[1]*nChannels*sizeof(double));
		img_rank_s = (double *) calloc(1,dims[0]*dims[1]*nChannels*sizeof(double));
		img_rank_t = (double *) calloc(1,pxlCnt*nChannels*sizeof(double)); //changed from calloc(1,dims[0]*dims[1]*nChannels*sizeof(double)) // BP - Wed, Feb-5
		rank_range = (double *) calloc(1,rank_bin * nChannels * sizeof(double)); //changed from calloc(1,rank_bin * 3 * sizeof(double)) //BP - Wed,Feb-5

		range_rank = floor(pxlCnt / ((double)rank_bin)); //average #pxl per bin
	}

	void rankRelease()
	{
		free(img_rank);
		free(img_rank_t);
		free(rank_range);
		free(img_rank_s);
	}
}rank;


/******************************************************
structure to store cell-wise histogram of the image
*******************************************************/
typedef struct imgHistogram
{
	int nBins_i;
	int height_i,width_i;
	float **hist_ppf;

	void imgHistogramInit(dtct_s dtctInfo,int width,int height)
	{
		int i;
		int widthImg_i,heightImg_i;
		int wCell_i,hCell_i;
		widthImg_i = width;
		heightImg_i = height;
		wCell_i = dtctInfo.wCell_i;
		hCell_i = dtctInfo.hCell_i;

		//number of cells along height and width
		width_i = widthImg_i / wCell_i;
		height_i = heightImg_i /hCell_i;

		//no.of.orientation bins
		nBins_i = dtctInfo.nBins_i;

		hist_ppf = (float**) calloc(height_i, sizeof(float*));
		if(hist_ppf == NULL)
		{
			free(hist_ppf);
			printf("Memory allocation failed while allocating for image hist_ppf.\n");
			exit(-1);
		}
		for(i=0;i< height_i;i++)
		{
			hist_ppf[i] = (float*) calloc(width_i * nBins_i , sizeof(float));
			if(hist_ppf[i] == NULL)
			{
				free(hist_ppf[i]);
				printf("Memory allocation failed while allocating for Image hist_ppf[i][].\n");
				exit(-1);
			}
		}

		//initialization
/*		for(i=0;i<height_i;i++)
			for(int j=0;j<width_i*nBins_i;j++)
				hist_ppf[i][j]=0;*/

	}

	void imgHistogramRelease()
	{
		int i;
		for(i=0;i < height_i; i++)
			if(hist_ppf[i])
				free(hist_ppf[i]);

		if(hist_ppf)
			free(hist_ppf);
	}

} imgHist_s;

/*****************************************************
structure to hold various parameters required for
applying descriptors on images
*****************************************************/

typedef struct imgProcesParams
{
	int no_bins,cell_xy,block_xy,hop;
	float mix1,mix2;
}imgProcessParams;

/*******************************************
structure to hold the cell information
********************************************/

typedef struct cellInfo
{
	int cell_x,cell_y;
	int hCell,wCell;
}cell_info;

/****************************************************
structure to hold the location of neighbouring pixels
*****************************************************/

struct neighborpts
{
   int x;
   int y;
   double neighborI[8];
};


#endif
