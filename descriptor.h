#ifndef HOG_PHASES_H
#define HOG_PHASES_H
 
#include "HOGtypes.h"

int imageGradient(const IplImage *inputImg, Grad_s *imgGrad);

int cellwiseImgHistogram(double *img_rank_s, Grad_s *imgGrad,imgHist_s *imgHist,dtct_s dtctInfo,int method_no,int n_channels,int width,int height);

descriptor_s* normalizeWindow(int rowIndex_i,int colIndex_i,const dtct_s *dtctInfo,imgHist_s *imgHist,int normalize_reqd);

descriptor_s* getFV(imgHist_s *imgHist,const dtct_s *dtctInfo);

descriptor_s* mergeDescriptors(descriptor_s* descriptorData1,descriptor_s* descriptorData2);

void getImgData_asDouble_inRowMajorOrder(IplImage* im, double **data);

void quickSort(double arr[], int left, int right);

void apply_filter(double *img_rank, double *img_rank_t, int rows, int cols);

void apply_filter(float **img_rank, double *img_rank_t, int rows, int cols);

void print_range(double *rank_range, int rank_bin);

int check_border_pt(int x_pos, int x_limit, int y_pos, int y_limit);

long getUnique(double *temp,double **uniqueInTemp,long tpxlCnt);

#endif
