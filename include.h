/* This file was automatically generated.  Do not edit! */
#ifndef _HOGTRAIN_HPP
#define _HOGTRAIN_HPP
 
#include "HOGtypes.h"
#include "descriptor.h"

extern descriptor_s *(*getDescp)(IplImage *,int,dtct_s);
extern int train_neg_images;
extern int train_pos_images;
extern int neg_total,neg_fp;
extern int no_of_neg_images;
extern int fp_homogenous;
extern int test_pos_images;
extern int test_neg_images;
extern float thresh;

#define DISPLAY 0
#define SVMTHRESH -1.0
#define eps 0.0001
#define SKIP_TRAINING 0
#define SKIP_TESTING 0
#define SKIP_DESC_COMPUTATION 0
#define SKIP_POS_DESC_COMPUTATION 0
#define SKIP_NEG_DESC_COMPUTATION 0

int rankCnt;
char orientBinCnt[] = "16";

int main(int argc,char *argv[]);
char *getStr(int num);
int getDigitCount(int num);
extern char method_name[STR_SIZE],method_id[STR_SIZE];
bool checkFile(char opFileName[]);
//void getTestFeatureVector(char *testFeatFile,imgProcesParams iPP,char datasetPath[][STR_SIZE]);
//void getTrainFeatureVector(char *trainFeatFile,imgProcesParams iPP,char datasetPath[][STR_SIZE]);
void readArgumentsFromIPFile(char *filename,char ***ipFileContents);
void operateMethod(FILE *fp,char **histParams);
void applyMethods(IplImage *inputImg,char **histParams);
descriptor_s* (*getDescp)(IplImage *, int, dtct_s);
int isHomogeneous(IplImage *img) ;
void calcHistogram(int hist[],int bins,IplImage *img);
descriptor_s *computeDescriptor(IplImage *inputImg,char **histParams);
descriptor_s *getHOG_Descp(IplImage *inputImg,int method_no,dtct_s dtctInfo);
descriptor_s *operateRankMethod(IplImage *inputImg,dtct_s dtctInfo,imgHist_s **imgHist,int method_num);
void modifyFV(descriptor_s **descriptorData);
int isHomogeneous(IplImage *img);
void calcHistogram(int hist[],int bins,IplImage *img);
IplImage *extractImage(FILE **fp,char *trainImgDir,char *pos_or_neg,int h);
imgHist_s *computeHistogram(double *img_rank_s,Grad_s *imgGrad,dtct_s dtctInfo,int method_no,int n_channels,int width,int height);
Grad_s *computeGradient(IplImage *inputImg);
dtct_s setDescriptorParams(char *hCell,char *wCell,char *hBlock,char *wBlock,char *n_bins);
void resizeImage(IplImage **inputImg,int n_width,int n_height);
void trainSVM(char *trainFeatFile,char *trainMsgFile);
void setImageProcessParams(char **histParams,imgProcesParams iPP);
void writeOutTrainParameters(char **argv, int argc);
bool fileExists(char *fileName);
float getNthPercentile(vector<float> scor, float n);

#endif
