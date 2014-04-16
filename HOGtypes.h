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

//~ #undef _CRT_SECURE_CPP_OVERLOAD_SECURE_NAMES
//~ #undef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES
//~ #define _CRT_SECURE_CPP_OVERLOAD_SECURE_NAMES 1
//~ #define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1

#define POS 1
#define NEG 0

#define DEBUG 1

#define INRIA data[0]
#define	CALTECH	data[1]
#define TUD	data[2]

#define SVM_TRAINFEAT_FILE(x) ((x)->fileNames[0])
#define SVM_MODEL_FILE(x) ((x)->fileNames[1])
#define SVM_TESTFEAT_FILE(x) ((x)->fileNames[2])
#define SVM_OUTPUT_FILE(x) ((x)->fileNames[3])
#define FINAL_OUTPUT_FILE(x) ((x)->fileNames[4])
#define THSHLDED_POS_ACCUR_FILE(x) ((x)->fileNames[5])
#define THSHLDED_NEG_ACCUR_FILE(x) ((x)->fileNames[6])
#define MR_FPPW_FILE(x) ((x)->fileNames[7])
#define MR_FILE(x) ((x)->fileNames[8])
#define FPPW_FILE(x) ((x)->fileNames[9])
#define DESC_COLLECTION_FILE(x) ((x)->fileNames[10])
#define TEMP_FILE(x) ((x)->fileNames[11])
#define SVM_TEST_MESSAGE_FILE(x) ((x)->fileNames[12])
#define SVM_TRAIN_MESSAGE_FILE(x) ((x)->fileNames[13])

#define POS_TRAIN_SET_LOC (datasetPath[0])
#define NEG_TRAIN_SET_LOC (datasetPath[1])
#define POS_TEST_SET_LOC (datasetPath[2])
#define NEG_TEST_SET_LOC (datasetPath[3])


/*
 * Parameters for LBP variants
 */
#define THRES_CS_LBP 3
#define THRES_CS_LTP 10
#define NBD_SZ 3
#define NBD_RAD 3
#define NBR_CNT 16 //multiple of 8
#define UNIF_LBP_THSHLD 2 //number of transitions beyond which a pattern fails to be a uniform lbp

typedef float descpData_f;
extern char method_name[], method_id[];

typedef struct structFilesUsed
{
	int fileCount;
	char **fileNames;

	void allocate(int count=14)
	{
		this->fileCount = count;

		this->fileNames = (char**)calloc(count,sizeof(char*));
		for(int i = 0; i < count; i++)
			this->fileNames[i] = (char*)calloc(STR_SIZE,sizeof(char));
	}

	void print()
	{
		for(int i = 0; i < this->fileCount; i++)
			printf("fileNames[%d] : %s\n",i,this->fileNames[i]);
	}

	void setFileNameParameters(char **ipFileContents, char mName[STR_SIZE], int fileCount = 14)
	{
		this->release();
		this->allocate(fileCount);

		//svm_train_method_name.txt
		strcat(strcpy(SVM_TRAINFEAT_FILE(this),ipFileContents[5]),"_");
		strcat(strcat(SVM_TRAINFEAT_FILE(this),mName),".txt");

		//svm_train_method_name.model
		strcat(strcpy(SVM_MODEL_FILE(this),ipFileContents[5]),"_");
		strcat(strcat(SVM_MODEL_FILE(this),mName),".model");

		//svm_train_message_methodName.txt
		strcat(strcpy(SVM_TRAIN_MESSAGE_FILE(this),ipFileContents[5]),"_message_");
		strcat(strcat(SVM_TRAIN_MESSAGE_FILE(this),mName),".txt");

		//svm_testing_method_name.txt
		strcat(strcpy(SVM_TESTFEAT_FILE(this),ipFileContents[6]),"_");
		strcat(strcat(SVM_TESTFEAT_FILE(this),mName),".txt");

		//svm_out_method_name.txt or svm_out_method_name_pos.txt or svm_out_method_name_neg.txt
		//final_output.txt or final_output_pos.txt or final_output_neg.txt
		int category = atoi(ipFileContents[atoi(ipFileContents[0])]);
		if(category == 1)
		{
			strcat(strcat(strcpy(SVM_OUTPUT_FILE(this),ipFileContents[7]),mName),"_pos.txt");
			strcat(strcpy(FINAL_OUTPUT_FILE(this),ipFileContents[8]),"_pos.txt");
		}
		else if(category == 2)
		{
			strcat(strcat(strcpy(SVM_OUTPUT_FILE(this),ipFileContents[7]),mName),"_neg.txt");
			strcat(strcpy(FINAL_OUTPUT_FILE(this),ipFileContents[8]),"_neg.txt");
		}
		else
		{
			strcat(strcat(strcat(strcpy(SVM_OUTPUT_FILE(this),ipFileContents[7]),"_"),mName),".txt");
			strcat(strcpy(FINAL_OUTPUT_FILE(this),ipFileContents[8]),".txt");
		}

		//methodName_thshlded_pos_accuracies.txt
		strcat(strcat(strcpy(THSHLDED_POS_ACCUR_FILE(this),"thshlded_pos_accuracies_"),mName),".txt");

		//methodName_thshlded_neg_accuracies.txt
		strcat(strcat(strcpy(THSHLDED_NEG_ACCUR_FILE(this),"thshlded_neg_accuracies_"),mName),".txt");

		//methodName_MR_FPPW.txt
		strcat(strcat(strcpy(MR_FPPW_FILE(this),"MR_FPPW_"),mName),".txt");
		//methodName_MR.txt
		strcat(strcat(strcpy(MR_FILE(this),"MR_"),mName),".txt");
		//methodName_FPPW.txt
		strcat(strcat(strcpy(FPPW_FILE(this),"FPPW_"),mName),".txt");

		//descriptorsUsedSoFar.txt
		strcpy(DESC_COLLECTION_FILE(this),"descriptorsUsedSoFar.txt"); //currently, looks like we're not using this file

		//temp.txt
		strcpy(TEMP_FILE(this),"temp.txt"); //currently, looks like we're not using this file

		//eff.txt
		strcat(strcpy(SVM_TEST_MESSAGE_FILE(this),ipFileContents[6]),"_message_");
		strcat(strcat(SVM_TEST_MESSAGE_FILE(this),mName),".txt");
	}

	void release()
	{
		if(fileNames)
		{
			for(int i = 0; i < this->fileCount; i++)
				if(fileNames[i])
					free(fileNames[i]);
			free(fileNames);
		}
	}

}filesUsed;

//structure to hold the thresholded accuracies of the test-set and some meta-info about these accuracies
typedef struct struct_ThresholdedAccuracyInfo
{
	unsigned thsholdCnt; //total number of thresholds
	unsigned homoImgCnt; //number of homogeneous images (used in getting the accuracies)
	float  start, step, end;
	float ** posThresholdedAcc, **negThresholdedAcc;
	char opFileNames[2][STR_SIZE]; //corresponding to the positive and the negative thresholded accuracies
	char ipFileName[STR_SIZE]; //file containing the svm classifications along with their scores (i.e.)svm_out_methodId.txt

	void release()
	{
		if(posThresholdedAcc)
		{
			for(unsigned i = 0; i < this->thsholdCnt; i++)
				if(posThresholdedAcc[i])
					free(posThresholdedAcc[i]);
			free(posThresholdedAcc);
		}
		if(negThresholdedAcc)
		{
			for(unsigned i = 0; i < this->thsholdCnt; i++)
				if(negThresholdedAcc[i])
					free(negThresholdedAcc[i]);
			free(negThresholdedAcc);
		}
	}
	//void initialize(int thsholdCnt, char posAccFileName[], char negAccFileName[], char svmOPFileName[], float start, float step, float end)
	void initialize(filesUsed* fu, char** ipArgFileContents)
	{

		this->release();

		//set file names
		strcpy(ipFileName, SVM_OUTPUT_FILE(fu));
		strcpy(opFileNames[0],THSHLDED_POS_ACCUR_FILE(fu));
		strcpy(opFileNames[1],THSHLDED_NEG_ACCUR_FILE(fu));

		//printf("\nstruct_ThresholdedAccuracyInfo :: ipFileName = %s, opFileName[0] = %s, opFileName[1] = %s\n", ipFileName, opFileNames[0], opFileNames[1]);

		//set the threshold values
		start = atof(ipArgFileContents[10]);
		step = atof(ipArgFileContents[11]);
		end = atof(ipArgFileContents[12]);
		this->thsholdCnt = getNumElements(start, step, end);

		//printf("\nstruct_ThresholdedAccuracyInfo :: start = %f, step = %f, end = %f, cnt = %d\n",start, step, end,thsholdCnt);

		//allocate the memory for the accuracies
		this->posThresholdedAcc = (float**)calloc(this->thsholdCnt,sizeof(float*));
		this->negThresholdedAcc = (float**)calloc(this->thsholdCnt,sizeof(float*));

		for(unsigned i = 0; i < this->thsholdCnt; i++)
		{
			this->posThresholdedAcc[i] = (float*)calloc(2,sizeof(float)); //threshold, positive thresholded accuracy
			this->negThresholdedAcc[i] = (float*)calloc(2,sizeof(float)); //threshold, negative threshold accuracy
		}

		//homogeneous image count
		homoImgCnt = 0;//default
	}

	void print()
	{
		printf("start = %f\tstep = %f\tend =%f\n",start,step,end);
		for(unsigned i = 0; i < this->thsholdCnt; i++)
			printf("%f\t%f\t%f\n", this->posThresholdedAcc[i][0], this->posThresholdedAcc[i][1], this->negThresholdedAcc[i][1]);

		printf("Files containing these values are: %s & %s\n",opFileNames[0], opFileNames[1]);
	}

	unsigned getNumElements(float start, float step, float end)
	{
		return (unsigned)(floor((end-start)/step)+1);
	}

}thresholdedAccuracyInfo;


typedef struct Struct_FPPW_MR_INFO
{
	thresholdedAccuracyInfo tAccInfo;
	float **MR, **FPPW;

	void release()
	{
		if(MR)
		{
		for(unsigned i = 0; i < this->tAccInfo.thsholdCnt; i++)
			if(MR[i])
					free(MR[i]);
			free(MR);
		}
		if(FPPW)
		{
			for(unsigned i = 0; i < this->tAccInfo.thsholdCnt; i++)
				if(FPPW[i])
					free(FPPW[i]);
			free(FPPW);
		}
	}


	char opFileName[3][STR_SIZE]; //corresponding to MR, FPPW and MR_FPPW files
	char ipFileName[2][STR_SIZE]; //positive and negative thresholded accuracies

	char* thshlded_pos_accur_file()
	{
		return ipFileName[0];
	}

	char* thshlded_neg_accur_file()
	{
		return ipFileName[1];
	}

	char* fppw_file()
	{
		return opFileName[0];
	}

	char* mr_file()
	{
		return opFileName[1];
	}

	char* fppw_mr_file()
	{
		return opFileName[2];
	}
	void initialize(filesUsed* fu, char** ipArgFileContents)
	{

		this->release();

		tAccInfo.initialize(fu,ipArgFileContents);

		//set file names
		strcpy(ipFileName[0], THSHLDED_POS_ACCUR_FILE(fu));
		strcpy(ipFileName[1], THSHLDED_NEG_ACCUR_FILE(fu));
		strcpy(opFileName[0], FPPW_FILE(fu));
		strcpy(opFileName[1], MR_FILE(fu));
		strcpy(opFileName[2], MR_FPPW_FILE(fu));

		//allocate the memory for the accuracies
		MR = (float**)calloc(tAccInfo.thsholdCnt,sizeof(float*));
		FPPW = (float**)calloc(tAccInfo.thsholdCnt,sizeof(float*));

		for(unsigned i = 0; i < tAccInfo.thsholdCnt; i++)
		{
			MR[i] = (float*)calloc(2,sizeof(float)); //threshold, FPPW
			FPPW[i] = (float*)calloc(2,sizeof(float));//threshold, FPPW
		}
	}
}FalsePositivePerWindow_MissRateInfo;

/*typedef struct
{
	char **ipFileContents, **datasetPath, methodName[STR_SIZE], methodId[STR_SIZE/10];
	filesUsed fUused;
	thresholdedAccuracyInfo tAccInfo;

	void initialize()
	{

	}

}ipArgument;*/

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
