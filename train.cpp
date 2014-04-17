
/*********************************************************************************************************
HOGtrain.c is used to train the pedestrian detection system.
*********************************************************************************************************/
float* model;
int modelflag=0;

#include "include.h"


using namespace cv;
using namespace std;


char method_id[STR_SIZE];
static inline float min(float x, float y) { return (x <= y ? x : y); }
static inline float max(float x, float y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

/*****************************************************************
Helper function to set the descriptor parameters
******************************************************************/

dtct_s setDescriptorParams(char *hCell,char *wCell,char *hBlock,char *wBlock,char *n_bins)
{
	dtct_s dtctInfo;
	dtctInfo.hCell_i = atoi(hCell);
	dtctInfo.wCell_i = atoi(wCell);
	dtctInfo.hBlck_i = atoi(hBlock);
	dtctInfo.wBlck_i = atoi(wBlock);
	dtctInfo.nBins_i = atoi(n_bins);
	return dtctInfo;
}

/*****************************************************************
Helper function to compute the image gradient
******************************************************************/

Grad_s *computeGradient(IplImage *inputImg)
{
	Grad_s *imgGrad;
	imgGrad = (Grad_s*) malloc(sizeof(Grad_s));
	imgGrad->Gradientinit(inputImg->width,inputImg->height);
	imageGradient(inputImg,imgGrad);
	return imgGrad;
}

/**********************************************************************
Helper function to compute the image histogram
**********************************************************************/

imgHist_s *computeHistogram(double *img_rank_s, Grad_s *imgGrad, dtct_s dtctInfo, int method_no, int n_channels, int width, int height)
{
	imgHist_s *imgHist;
	imgHist = (imgHist_s*) malloc(sizeof(imgHist_s));
	imgHist->imgHistogramInit(dtctInfo,width,height);
	cellwiseImgHistogram(img_rank_s, imgGrad,imgHist,dtctInfo,method_no,n_channels,width,height);
	return imgHist;
}

/**********************************************************************
Returns the pixel intensity values in a single dimensional array
This might be buggy.  It doesn't take into account the widthStep param
and the depth-major order of opencv's iplimage.
**********************************************************************/

rank *getIntensityValues(IplImage *inputImg,int no_bins)
{
	char *imagechar;
	int dims[2];
	int index;
	imagechar = inputImg->imageData;
	dims[0] = inputImg->width;
	dims[1] = inputImg->height;
    long pxlCnt = dims[0]*dims[1];

    rank *rk = (rank *)malloc(sizeof(rank));
	rk->rankInit(dims,no_bins,inputImg->nChannels,pxlCnt);

	for(index = 0; index < dims[0]*dims[1]*inputImg->nChannels; index++)
	{
		rk->img_rank[index] = imagechar[index]; //store the intensity values
	}

	return rk;
}

/**********************************************************
computes the intensity histogram of an image
Used to detect a homogeneous patch in an image
***********************************************************/

void calcHistogram(int hist[],int bins,IplImage *img)
{
	int div_fac = 255/bins; div_fac++;
	int i;
	for(i=0;i<bins;i++)hist[i]=0;
	for(i=0;i<img->height;i++)
	   {
			uchar *ptr = (uchar *)(img->imageData + i*img->widthStep);
			for(int j=0;j<img->widthStep;j++)
			{
				double intensity = ptr[j];
				int bin = (int)intensity/div_fac;
				hist[bin]++;
			}

	   }
}

/**********************************************************
checks if the image is homogeneous
***********************************************************/
int isHomogeneous(IplImage *img)
{
	int hist[25],bins = 25;
	calcHistogram(hist,bins,img);
	int no_pixels = img->width*img->height;
	int thresh = 0.7*no_pixels;
	for(int i=0;i<bins;i++) {
		//printf("%d\t",hist[i]);
		if(hist[i]>=thresh)
		{
			return 1;
		}
	}
	return 0;
}

/*****************************************************
Modify the descriptor data for homogeneous patches
******************************************************/

void modifyFV(descriptor_s **descriptorData)
{
	int fv_size = (*descriptorData)->descriptorSize_i;
	for(int i=0;i<fv_size;i++)
		(*descriptorData)->featureVector_pf[i] = 0.0;
}

/**********************************************************
Returns the feature vector formed by HOG
***********************************************************/

descriptor_s *getHOG_Descp(IplImage *inputImg, int method_no, dtct_s dtctInfo)
{
	Grad_s *imgGrad;
	imgHist_s *imgHist;
	descriptor_s *descriptorData;
	imgGrad = computeGradient(inputImg); //compute the gradient
	imgHist = computeHistogram(NULL,imgGrad,dtctInfo,method_no,inputImg->nChannels,inputImg->width,inputImg->height);
	descriptorData = normalizeWindow(0,0,&dtctInfo,imgHist,1);

	imgGrad->GradientRelease();
	free(imgGrad);

	imgHist->imgHistogramRelease();
    free(imgHist);

	return descriptorData;
}


/*********************************************************************
Function to apply different methods
**********************************************************************/
descriptor_s *computeDescriptor(IplImage *inputImg,char **histParams)
{
	dtct_s dtctInfo;

	if(atoi(histParams[0]) == 2 && atoi(histParams[4])<16)
		strcpy(histParams[4],"16");

	//int no_of_bins = atoi(histParams[9]);
	int method_no = atoi(histParams[0]);

	dtctInfo = setDescriptorParams(histParams[0],histParams[1],histParams[2],histParams[3],histParams[4]);
	dtctInfo.wDtct_i = inputImg->width;
	dtctInfo.hDtct_i = inputImg->height;
    //compute other descriptors
    descriptor_s * descriptorData = getHOG_Descp(inputImg, method_no, dtctInfo);
	return descriptorData;
}

bool checkFile(char opFileName[])
{
	return (bool)fopen(opFileName,"r");
}

string int2String(unsigned long long num)
{
    char str[64+1];//can handle upto 2^64 digits for unsigned long long
    sprintf(str,"%lld",num);
    return(string(str));
}

//get the descriptor libsvm
string float2String(double num)
{
    char strNum[308+1]; //10^308
    sprintf(strNum,"%.7f",num);

    //discard trailing zeroes
    int i=strlen(strNum);

    while(strNum[--i] == '0');
    if(strNum[i] != '.')
        ++i;
    //i = (strNum[i] == '.') ? i : ++i;
    strNum[i] = 0;

    return(string(strNum));
}

//convert the given descriptor into the libsvm format and return as a string
string getFV_LIBSVMFormat(float* desc, int size, int classlabel)
{
    string imgDesc(int2String(classlabel));
    imgDesc += " ";

    for(int i = 0; i < size; i++)
    {
        imgDesc += int2String(i+1);
        imgDesc += ":";
        imgDesc += float2String(desc[i]);
        imgDesc += " ";
    }
    imgDesc += "\n";

    return imgDesc;
}

bool fileExists(char *fileName)
{
	ifstream ifile(fileName);
	return (bool) ifile;
}

/***************************************************
Function to train the SVM with descriptor data
****************************************************/
void trainSVM(char* trainFeatFile, char* trainModelFile)
{
    if(fileExists(trainModelFile))
    {
        fprintf(stderr,"%s exists.  Skipping training.\n",trainModelFile);
    }
    else
    {
        //building model file
        char train_cmd[2*STR_SIZE];
        char trainMsgFile[1000];
        strcpy(trainMsgFile,trainModelFile);
        //get training info file from the model file
        char* modelPtr = strstr(trainMsgFile,".model");
        *modelPtr = '_';
        strcat(trainMsgFile,".txt");

        fprintf(stderr, "Training SVM....\n");
        sprintf(train_cmd, "./svmTrain -B 1 %s %s", trainFeatFile, trainModelFile);
        strcat(strcat(train_cmd," > "), trainMsgFile);

        //sprintf(train_cmd,"svm-train -s 0 -b 1 %s",trainFile);
        //printf("%s\n", train_cmd);
        system(train_cmd);
    }
}

/********************************************
 Main function
********************************************/

const char* helpMsg = "\ntrain.cpp:\nCreates a HOG descriptor for each of the images in the given direcotory and trains a linear-svm based on them. \nExpects a txt file containing the names of the images (pos.txt or neg.txt, as the case may be.)\nUsage: ./a.out posImgDir negImgDir descriptorFileName svmModelFileName.\n\n";

int main(int argc,char *argv[])
{
	if(argc != 5)
	{
		fprintf(stderr,"%s", helpMsg);
		exit(-1);
	}

	//argv[1] contains the path of the positive images & pos.txt
	//argv[2] contains the path of the negative images & neg.txt
	//argv[3] contains the filename to which descriptors will be written
	//argv[4] contains the model file containing the SVM-parameters

    char **histParams = 0;
    string class_Label = "0";


	histParams = new char*[6];
    histParams[0]=(char*)"8"; //horizontal width of a cell in terms of pixels
    histParams[1]=(char*)"8"; //vertical width of a cell in terms of pixels
    histParams[2]=(char*)"2"; //horizontal width of a block in terms of cells
    histParams[3]=(char*)"2"; //vertical width of a block in terms of cells
    histParams[4]=(char*)orientBinCnt; //no.of orientation bins

    ofstream des;

    //open descriptor file dir
    if(fileExists(argv[3]))
    {
        fprintf(stdout,"%s already exists! Skipping description computation\n",argv[3]);
    }
    else
    {
        Mat trainImg;
        Mat img3 = Mat(128,64,CV_8UC1);

        //get the number of images
        FILE *fp = NULL;
        char op[50];
        long int numPosImg,numNegImg;

        //positive images
        string txtFileName = string(argv[1]) + string("pos.txt");
        if(access(txtFileName.c_str(), F_OK) == -1)//check for existence
        {
            //pos.txt doesn't exist
            fprintf(stderr,"%s doesn't exist!\n",txtFileName.c_str());
            exit(-1);
        }
        //number of positive images
        fp = popen(string(string("cat ") + txtFileName + string(" | wc")).c_str(),"r");
        fscanf(fp,"%s",op);
        numPosImg = atol(op);
        pclose(fp);

        //negative images
        txtFileName.clear();
        txtFileName = string(argv[2]) + string("neg.txt");
        if(access(txtFileName.c_str(), F_OK) == -1)
        {
            //neg.txt doesn't exist
            fprintf(stderr,"%s doesn't exist!\n",txtFileName.c_str());
            exit(-1);
        }
        //number of negative images
        fp = popen(string(string("cat ") + txtFileName + string(" | wc")).c_str(),"r");
        fscanf(fp,"%s",op);
        numNegImg = atol(op);
        pclose(fp);

        int label;
        string desc_libSVMFormat[numPosImg+numNegImg];


        //create the descriptors file
        FILE* descFile = fopen(argv[3],"w");
        if(!descFile)
        {
            fprintf(stderr,"%s cannot be opened :(\n",argv[3]) ;
            exit(-1);
        }

        //Descriptors for pos images
        string ymlFileName = string(argv[1]) + string("pos.yml");
        if(access(ymlFileName.c_str(),F_OK) == -1)
        {
            //pos.yml doesn't exist!
            fprintf(stderr, "%s doesn't exist!\n", ymlFileName.c_str());
            exit(-1);
        }
        fprintf(stderr,"Opening the positive image storage\n");
        FileStorage posStorage(ymlFileName.c_str(),FileStorage::READ);
        label = 1;

        //compute the descriptors
        fprintf(stderr,"Collecing the positive descriptors\n");
        //omp_set_num_threads(NUM_THREADS);
        //#pragma omp parallel for
        for(int cnt=0; cnt < numPosImg; cnt++)
        {
            //private variables
            string imgId="img";
            descriptor_s *descriptorData_curr;
            float *a;
            int ln;

            //get the image id that is used to hash into the storage
            imgId += int2String(cnt);
            fprintf(stderr,"%s: train positives: %s/%ld\n",argv[4], imgId.c_str(),numPosImg);

            //store the image in Mat
            posStorage[imgId]>>trainImg;

            //resize
            resize(trainImg,img3,img3.size(),0.0,0.0,1);

            //display
            if(DISPLAY)
            {
                imshow("pos",img3);
                waitKey(2);
            }

            //compute the descriptor
            IplImage image=img3.operator IplImage();
            descriptorData_curr = computeDescriptor(&image,histParams);
            ln=descriptorData_curr->descriptorSize_i;
            a = descriptorData_curr->featureVector_pf;

            //write it to a vector of string descriptor
            desc_libSVMFormat[cnt] = getFV_LIBSVMFormat(a, ln, label);

            //reset
            descriptorData_curr->descriptorRelease();
            free(descriptorData_curr);

            imgId = "img";
        }

        //reset
        posStorage.release();

//***********************************************************************************************************************

        //Descriptors for neg images
        ymlFileName.clear();
        ymlFileName = string(argv[2]) + string("neg.yml");
        if(access(ymlFileName.c_str(), F_OK) == -1)
        {
            //neg.yml doesn't exist!
            fprintf(stderr,"%s doesn't exist!\n",ymlFileName.c_str());
            exit(-1);
        }
        fprintf(stderr,"Opening the negative image storage\n");
        FileStorage negStorage(ymlFileName.c_str(),FileStorage::READ);

        //set
        label = 0;

        //compute the descriptors
        fprintf(stderr,"Collecing the negative descriptors\n");
        //omp_set_num_threads(NUM_THREADS);
        //#pragma omp parallel for
        for(int cnt=numPosImg; cnt < numNegImg+numPosImg; cnt++)
        {
            //private variables
            string imgId="img";
            //Mat img3=Mat(128,64,CV_8UC1);
            descriptor_s *descriptorData_curr;
            float *a;
            int ln;

            //get the image id that is used to hash into the storage
            imgId += int2String(cnt-numPosImg);
            fprintf(stderr,"%s: train negatives: %s/%ld\n",argv[4],imgId.c_str(),numNegImg);

            //store the image in Mat
            negStorage[imgId]>>trainImg;

            //resize
            resize(trainImg,img3,img3.size(),0.0,0.0,1);

            //display
            if(DISPLAY)
            {
                imshow("neg",img3);
                waitKey(2);
            }

            //compute the descriptor
            IplImage image=img3.operator IplImage();
            descriptorData_curr = computeDescriptor(&image,histParams);
            ln=descriptorData_curr->descriptorSize_i;
            a = descriptorData_curr->featureVector_pf;

            //write it to a vector of string descriptor
            desc_libSVMFormat[cnt] = getFV_LIBSVMFormat(a, ln, label);

            //reset
            descriptorData_curr->descriptorRelease();
            free(descriptorData_curr);

            imgId = "img";
        }
        //reset
        negStorage.release();

        fprintf(stderr,"Saving the descriptors...\n");

        //write the descriptors to the feature vectors file in one-shot
        for(int i = 0; i < numNegImg+numPosImg; i++)
            fprintf(descFile,"%s",desc_libSVMFormat[i].c_str());
    }

    if(fileExists(argv[4]))
        fprintf(stderr,"%s exists!  Skipping training.\n",argv[4]);
    else
        trainSVM(argv[3],argv[4]);

    delete [] histParams;
}
