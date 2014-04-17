/*********************************************************************************************************
HOGtrain.c is used to train the pedestrian detection system.
*********************************************************************************************************/
float* model;
int modelflag=0;
float svmthresh;


#define display 0

#include "include.h"

#define SKIP_IMAGES_BOOL 1
#if SKIP_IMAGES_BOOL
    #define SKIP_IMAGES_CNT 5
#else
    #define SKIP_IMAGES_CNT 0
#endif

#define SVMTHRESH -1.0

using namespace cv;
using namespace std;

float thresh;
char method_name[STR_SIZE], method_id[STR_SIZE];

int det_h = 128;
int det_w = 64;

//descriptor_s* (*getDescp)(IplImage *, int, dtct_s);
int isHomogeneous(IplImage *img) ;
void calcHistogram(int hist[],int bins,IplImage *img);


/**************************************************************************
 setImageProcessParams()
 sets the required parameters like cell size, block size, no. of bins, etc.
***************************************************************************/

void setImageProcessParams(char **histParams,imgProcesParams iPP)
{
	char num[50];

	printf("\nbins=%d cell=%d block=%d\n",iPP.no_bins,iPP.cell_xy,iPP.block_xy);

	//printf("\nmethod no :: %s",ipFileContents[9]);

	strcpy(histParams[0],method_name);

	//cell dimensions
	//itoa(iPP.cell_xy,num,10);

	sprintf(num,"%d",iPP.cell_xy);

	strcpy(histParams[0], num);
	strcpy(histParams[1], num);

	//block dimensions
	//itoa(iPP.block_xy,num,10);

	sprintf(num,"%d",iPP.block_xy);
	strcpy(histParams[2], num);
	strcpy(histParams[3], num);

	//no. of bins
	//itoa(iPP.no_bins,num,10);

	sprintf(num,"%d",iPP.no_bins);
	strcpy(histParams[4], num);

	//mix proprtion
	sprintf(num,"%f",iPP.mix1);
	strcpy(histParams[10], num);
	sprintf(num,"%f",iPP.mix2);
	strcpy(histParams[12], num);

	//hop
	//itoa(iPP.hop,num,10);

	sprintf(num,"%d",iPP.hop);
	strcpy(histParams[11],num);

	//stride
	//itoa(1,num,10);
	sprintf(num,"%d",1);
	strcpy(histParams[13],num);
}

/**************************************************
Function to resize the image
***************************************************/

void resizeImage(IplImage **inputImg,int n_width,int n_height)
{
	IplImage *inputImgOld;
	CvSize sizeImg;
	if((*inputImg)->width!=n_width || (*inputImg)->height!=n_height)
	{
		sizeImg =  cvSize(n_width,n_height);
		cvResetImageROI( *inputImg );
		inputImgOld = *inputImg;
		*inputImg = cvCreateImage( sizeImg,(*inputImg)->depth, (*inputImg)->nChannels);
		cvResize( inputImgOld,*inputImg,CV_INTER_CUBIC );
		cvReleaseImage( &inputImgOld );
	}

}

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

imgHist_s *computeHistogram(double *img_rank_s, Grad_s *imgGrad,dtct_s dtctInfo,int method_no,int n_channels,int width,int height)
{
	imgHist_s *imgHist;
	imgHist = (imgHist_s*) malloc(sizeof(imgHist_s));
	//imgHist->nBins_i = dtctInfo.nBins_i;
	imgHist->imgHistogramInit(dtctInfo,width,height);
	cellwiseImgHistogram(img_rank_s, imgGrad,imgHist,dtctInfo,method_no,n_channels,width,height);
	return imgHist;
}

/**********************************************************************
 extractImage()
 loads image from given directory, resizes it to 64 x 128(if necessary)
 and then returns the image
***********************************************************************/

IplImage *extractImage(FILE **fp, char *trainImgDir, char *pos_or_neg, int h)
{
	IplImage *inputImg;
	char imgFile[50],readFile[300];
	char fileType[2][9] = {"neg.txt","pos.txt"};
	fscanf(*fp,"%s",imgFile);
	if(!strcmp(imgFile,fileType[atoi(pos_or_neg)]))
		return NULL;

	printf("loading image : %s\\%s\n",trainImgDir,imgFile);
	sprintf(readFile,"%s\\%s",trainImgDir,imgFile);

	if ( (inputImg = cvLoadImage( readFile, 2)) == 0 )
			{
				printf("\nError loading image :: %d\n",h);
				return NULL;
			}
	resizeImage(&inputImg,64,128);
	return inputImg;
}

/**********************************************************************
Returns the pixel intensity values in a single dimensional array
**********************************************************************/

rank *getIntensityValues(IplImage *inputImg,int no_bins)
{
	char *imagechar;
	int dims[2];
	int index;
	imagechar = inputImg->imageData;
	dims[0] = inputImg->width;
	dims[1] = inputImg->height;
    long pxlCnt = dims[0] * dims[1];
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

int isHomogeneous(IplImage *img)
{
	int hist[25],bins = 25;
	calcHistogram(hist,bins,img);
	int no_pixels = img->width*img->height;
	int thresh = 0.5*no_pixels;
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

	int no_of_bins = atoi(histParams[4]);
	int method_no = atoi(histParams[0]);

	dtctInfo = setDescriptorParams(histParams[0],histParams[1],histParams[2],histParams[3],histParams[4]);
	dtctInfo.wDtct_i = inputImg->width;
	dtctInfo.hDtct_i = inputImg->height;

    descriptor_s *descriptorData = getHOG_Descp(inputImg, method_no, dtctInfo);

	return descriptorData;
}



//read the model file
void readModel(char *modelFileName)
{

    FILE *fp = fopen(modelFileName,"r");
    int metaDataLen_modelFile = 6;
    char temp1[STR_SIZE],temp2[STR_SIZE];

    for(int x = 0; x < metaDataLen_modelFile; x++)
        if(x != 3)
            fscanf(fp, "%*[^\n]\n", NULL);
        else
            fscanf(fp,"%s %s",temp1,temp2);
	rewind(fp);

    int ln = (int)atol(temp2)+1;
    model = (float*) malloc((ln)*sizeof(float));

    for(int x=0;x < ln + metaDataLen_modelFile; x++)
        {
            if(x >= metaDataLen_modelFile)
                fscanf(fp,"%f",&model[x-metaDataLen_modelFile]);
            else
                fscanf(fp, "%*[^\n]\n", NULL);
      }

    fclose(fp);

    fp = fopen("checkModel_FPPI.txt","w");
    for(int x = 0; x < ln; x++)
        fprintf(fp,"%f\n",model[x]);
    fclose(fp);
}

//compute the svm score
float score(IplImage *inputImg, char **histParams)
{
    float svm_score=0.0;
    descriptor_s *descriptorData_curr = computeDescriptor(inputImg,histParams);

    if(descriptorData_curr)
    {
        int ln=descriptorData_curr->descriptorSize_i;
        float *a = descriptorData_curr->featureVector_pf;
        
        //+1 for bias
        // compute the score or predict
        for(int x=0; x < ln; x++)
        {
            svm_score+=a[x]*model[x];
        }
        svm_score+=model[ln];

        //release
        descriptorData_curr->descriptorRelease();
        free(descriptorData_curr);
}
return svm_score;
}

bool checkFile(char opFileName[])
{
	return (bool)fopen(opFileName,"r");
}

//gets the number of digits in an integer
int getDigitCount(int num)
{
	float windowCnt = 1;
	while(num >= powf(10,windowCnt))
		windowCnt++;

	return (int)windowCnt;
}

//gets the equivalent string of the given natural number
char* getStr(int num)
{
	int dgtCnt = getDigitCount(num);
	char* numStr = (char*) malloc(sizeof(char)*(dgtCnt+1));
	int windowCnt = 1, rem;

	while(windowCnt <= dgtCnt)
	{
		rem = num % 10;
		num /= 10;
		numStr[dgtCnt-windowCnt] = (rem + '0') ;
		windowCnt++;
	}
	numStr[dgtCnt] = '\0';

	return numStr;
}

void getDetectionBoxes(Mat frame, vector<Rect>* boxes, vector<float>* scores, char **histParams, char* modelFileName)
{

    long long windowCnt = 0;
    vector <float> scale;
    vector<Point> pts;
    vector<float> scor;
    vector<float> maxCandidateScor;
    vector<Point> maxCandidatePts;
    vector<float> maxCandidateScale;
    Mat draw(frame);


    //sliding window
    for(int si=15;si>2;si=si-1)
    {
        float s=si/10.0;

        Mat frame_scaled = Mat(frame.rows*s, frame.cols*s, CV_8UC1);

        Mat winWithBdry = Mat(det_h+6,det_w+6,CV_8UC1);
        Mat win = Mat(det_h,det_w,CV_8UC1);

        //resize the image to the current scale
        resize(frame, frame_scaled, frame_scaled.size(),0.0,0.0,1);

        //extract each sliding window and examine
        for(int y=0;y<frame_scaled.rows-det_h; y=y+det_w)  //for(int y=0;y<frame_scaled.rows-det_w*2; y=y+det_w)
        {
            for(int x=0;x<frame_scaled.cols-det_w; x=x+(det_w/2)) //for(int x=0;x<frame_scaled.cols-(det_w/2)*2; x=x+(det_w/2))
            {
                Mat image2,cimg;

                //extract a window with a context information of 3 pixels on all 4 sides
                frame_scaled(Rect(x,y,det_w+6,det_h+6)).copyTo(image2,noArray());
                resize(image2, winWithBdry, winWithBdry.size(),0.0,0.0,1); //actually a copy!

                //extract the central (det_h x det_w) window from the window with context
                winWithBdry(Rect(3,3,det_w,det_h)).copyTo(win,noArray());
                IplImage winIplImg = win.operator IplImage();
                rectangle(draw, Point(x/s,y/s), Point((x+det_w)/s,(y+det_h)/s), Scalar(255,255,255), 1, 8, 0);

                //compute the svm score
                float svm = score(&winIplImg,histParams);

                if(display)
                {
                    imshow("draw",draw);
                    waitKey(1);
                }

                if(isHomogeneous(&winIplImg)==1)
                    svm=-100.0;

                scale.push_back(s);
                pts.push_back(Point(x,y));
                scor.push_back(svm);

            }
        }
    }

    //collect scores above the given threshold
    for(int i = 0; i < scor.size(); i++)
        if(scor.at(i) >= svmthresh)
        {
            maxCandidateScor.push_back(scor.at(i));
            maxCandidatePts.push_back(pts.at(i));
            maxCandidateScale.push_back(scale.at(i));
            int x=pts.at(i).x;
            int y=pts.at(i).y;
            float s=scale.at(i);
            float score=scor.at(i);

            if(display)
            {
                rectangle(draw, Point((int)x/s,(int)y/s), Point((int)((x+det_w)/s),(int)((y+det_h)/s)),Scalar(0,0,255),1,8,0);
                stringstream ob1;
                ob1.str(string());
                ob1<<score;
                putText(draw, ob1.str(), Point((int)x/s+15,(int)y/s+15),FONT_HERSHEY_COMPLEX_SMALL, 0.5, cvScalar(255,255,255), 1, CV_AA);
            }
        }

    // nms begin
    for(int i=0;i<maxCandidateScale.size();i++)
    {
        //loop to look at a particular max-cadidate
        vector<Point> ngb;
        vector<float> ngbscale;
        vector<float> ngbscor;

        int ct=0;

        //transforming coordinates in the frame scale
        float s1=maxCandidateScale.at(i);
        int x1=(maxCandidatePts.at(i).x)/s1;
        int y1=(maxCandidatePts.at(i).y)/s1;

        for(int j=0;j<maxCandidateScale.size();j++)
        {

            //transforming coordinates in the frame scale
            float s2=maxCandidateScale.at(j);
            int x2=(maxCandidatePts.at(j).x)/s2;
            int y2=(maxCandidatePts.at(j).y)/s2;

            float iarea=0.0;
            float uarea=1.0;

            //coordinates of intersection area
            int xx1=max(x1,x2);
            int yy1=max(y1,y2);
            int xx2=min(x1+det_w/s1,x2+det_w/s2);
            int yy2=min(y1+det_h/s1,y2+det_h/s2);

            //width and height of intersection area
            int iw=xx2-xx1+1;
            int ih=yy2-yy1+1;

            //intersection and union area
            if(iw < 0 || ih < 0)
                iarea = 0;
            else
                iarea=iw*ih;

            uarea=(det_w*det_h)/(s1*s1)+(det_w*det_h)/(s2*s2)-iarea;

            //love thy neighbour and put him in a vector
            if((iw>0) && (ih>0) && (iarea/uarea>0.5))
            {
                ngb.push_back(maxCandidatePts.at(j));
                ngbscor.push_back(maxCandidateScor.at(j));
                ngbscale.push_back(maxCandidateScale.at(j));
            }
        }
        //end of loop to calculate neighbourhoods


        float curr=maxCandidateScor.at(i);
        Point currpt=maxCandidatePts.at(i);
        float currsc=maxCandidateScale.at(i);

        //find the maximum in the neighbourhood
        for(int k=0;k<ngb.size();k++)
        {
            if(ngbscor.at(k)>curr)
            {
                curr=ngbscor.at(k);
                currpt=ngb.at(k);
                currsc=ngbscale.at(k);
            }
        }

        //collect the maxima
        int flag = 0;
        if(ngb.size()>0)
        {
            for(int l=0; l<(*scores).size(); l++)
            {
                if((*scores).at(l)==curr)
                    flag=1;
            }
            //avoid scores that are duplicates
            if(flag==0)
            {
                (*boxes).push_back(Rect((int)currpt.x/currsc, (int)currpt.y/currsc, (int)det_w/currsc, (int)det_h/currsc));
                (*scores).push_back(currsc);
            }
        }

        //clear neighbourhood vectors for next point
        ct=0;
        ngb.clear();
        ngbscor.clear();
        ngbscale.clear();
    }

    printf("no. of detections - %ld\n", (*scores).size());
}


/********************************************
 Main function
********************************************/
const char* helpMsg = "\ndetect.cpp:\nPerforms the sliding-window detection of the object for which the model was created on each of the images given in the test directory. \nWrites out the detection information to text files in the given destination directory in the [x,y,w,h,score] format. \nExpects a txt file containing the names of the images (pos.txt or neg.txt, as the case may be.)\nUsage: ./a.out testImgDir detectionDir svmModelFileName svm_threshold.\n\n";

int main(int argc,char *argv[])
{
	if(argc != 5)
	{
		fprintf(stderr,"%s", helpMsg);
		exit(-1);
	}

	//argv[1] contains the src directory which contains the test images
	//argv[2] contains the dest directory where the detection results will be saved
	//argv[3] has model file name
	//argv[4] has svm threshold
	
	
    char **histParams = 0;

    histParams = new char*[5];

	histParams[0]=(char*)"8"; //x-cellsz in pixels
	histParams[1]=(char*)"8"; //y-cellsz in pixels
	histParams[2]=(char*)"2"; //x-blockSz in cells
	histParams[3]=(char*)"2"; //y-blockSz in cells
	histParams[4]=(char*)"16"; //stride factor for blocks

	string line, filename;
	std::ifstream posFile;
	std::ofstream detectionResultFile;
    
	svmthresh=atof(argv[4]);

	long long windowCnt = 0;

    //open input pos.txt file
	filename = string(argv[1]);
	filename += "pos.txt";
	printf("%s\n",filename.c_str());
	posFile.open(filename.c_str());
    if(!posFile.is_open())
	{
		fprintf(stderr,"%s\n",string(filename +" does not exist!!").c_str());
		exit(-1);
	}

	//read the model file
	readModel(argv[3]);

    while (getline(posFile,line))
	{
		string data;

		filename.clear();
		filename = string(argv[2]);
		filename+=line;
		filename+=".txt";
		detectionResultFile.open(filename.c_str());
		if(!detectionResultFile.is_open())
		{
			fprintf(stderr,"Detection output file %s\n",string(filename +" does not exist!!@@").c_str());
			exit(-1);
		}

		filename.clear();
		filename = string(argv[1]);
		filename+=line;
		Mat draw=imread(filename,1);
		if(draw.data == NULL)
		{
			fprintf(stderr,"%s\n",string(filename +" does not exist!!~").c_str());
			exit(-1);
		}

        //grayscale
		Mat frame = Mat(draw.rows,draw.cols,CV_8UC1);
		cv::cvtColor(draw,frame,CV_RGB2GRAY,0);
		Mat findraw(draw);

		//get the detection boxes
        vector<float> scores;
        vector<Rect> boxes;
        getDetectionBoxes(frame, &boxes, &scores, histParams, argv[3]);

        //highlight and write-out the detections
		for(int i=0; i<boxes.size(); i++)
		{
			std::ostringstream ob;
			ob<<scores.at(i);
			putText(findraw, ob.str(), Point(boxes.at(i).x, boxes.at(i).y), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
			rectangle(findraw, boxes.at(i), Scalar(0,255,0), 1, 8, 0);
			circle(findraw,Point(boxes.at(i).x + boxes.at(i).width/2, boxes.at(i).y + boxes.at(i).height/2),1,Scalar(255,0,255),1,8,0);
			detectionResultFile<<boxes.at(i).x<<" "<<boxes.at(i).y<<" "<<boxes.at(i).width<<" "<<boxes.at(i).height<<" "<<scores.at(i)<<std::endl;
		}
		filename = string(argv[2]);
		filename+=line;
		filename+=".jpg";
		imwrite(string(filename),findraw);

		detectionResultFile.close();
	}
	posFile.close();
	free(model);
	delete[] histParams;

	return 0;
}

