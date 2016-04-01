/****************************
*							*
*	FreeTouch				*
*	Val(ZhenWenJin)			*
*	zhenwenjin@gmail.com	*
*	ver 1.0 2016.03.11		*
*	In HuaRuan				*
*							*
****************************/

#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <ctime>

#define FreeTouch_Cr_Max 173 
#define FreeTouch_Cr_Min 133
#define FreeTouch_Cb_Max 127
#define FreeTouch_Cb_Min 77

#define U8_White 255
#define U8_Black 0

using namespace std;
using namespace cv;

struct FreeTouch_Convexity_Defect{
    Point start;
    Point end;
    Point depth_point;
    float depth;
};

string FreeTouch_Cascade_File_Path = "./FreeTouchCascade/FreeTouchCascade1.0.xml";
CascadeClassifier FreeTouch_Cascade;
//To check the hand first.

vector<Rect> FreeTouch_Hand_Rect;
//Get the hand's place.

int FreeToucn_Tmp_Int;
//Use to loop times.

Point FreeTouch_Haar_Hand_Center;

int FreeTouch_Hand_In_Contour;
//Save hand is in whitch contor.
vector<vector<Point> > FreeTouch_Contours;
//Save some Contours.
vector<Vec4i> FreeTouch_Hierarchy;
//Save next Contours.

void FreeTouch_Fill_Hole(const Mat srcBw, Mat &dstBw);
//Fill some hole (This code is refer from internet.)
void FreeTouch_Find_Convexity_Defects(vector<Point>& contour, vector<int>& hull, vector<FreeTouch_Convexity_Defect>& convexDefects);
//(This code is refer from internet.)

double FreeTouch_Start_Time , FreeTouch_End_Time;
//Count the loop time.


int main( int argc, const char** argv ){

	VideoCapture FreeTouch_Capture(0);
	//Select the camera.
	Mat FreeTouch_Frame;
	//To Save camera frame.

	if(!FreeTouch_Cascade.load(FreeTouch_Cascade_File_Path)){
		printf("FreeTouch load cascade file fail.\n"); 
		return -1;
	}

	for(;;){
		FreeTouch_Capture >> FreeTouch_Frame;
		//Get frame frome camera.
		
		FreeTouch_Start_Time = clock(); 		

		Mat FreeTouch_Frame_YCrCb;
		Mat FreeTouch_Frame_YCrCb_Cr;
		Mat FreeTouch_Frame_YCrCb_Cb;
		vector<Mat> FreeTouch_YCrCb_Channels;
		//To save YCrCb channels.
		
		cvtColor(FreeTouch_Frame,FreeTouch_Frame_YCrCb,CV_BGR2YCrCb);
		split(FreeTouch_Frame_YCrCb,FreeTouch_YCrCb_Channels);
		FreeTouch_Frame_YCrCb_Cr = FreeTouch_YCrCb_Channels.at(1);
		FreeTouch_Frame_YCrCb_Cb = FreeTouch_YCrCb_Channels.at(2);
		//BGR to YCrBr and split channel.
		
		int FreeTouch_Blur_Times = 6;
		while(FreeTouch_Blur_Times--){
			GaussianBlur(FreeTouch_Frame_YCrCb_Cr,FreeTouch_Frame_YCrCb_Cr,Size(5,5),0,0);
			GaussianBlur(FreeTouch_Frame_YCrCb_Cb,FreeTouch_Frame_YCrCb_Cb,Size(5,5),0,0);
		}

		for(int y = 0;y < FreeTouch_Frame.rows;y++){
			uchar* FreeTouch_Frame_YCrCb_Cr_Data = FreeTouch_Frame_YCrCb_Cr.ptr<uchar>(y); 
			uchar* FreeTouch_Frame_YCrCb_Cb_Data = FreeTouch_Frame_YCrCb_Cb.ptr<uchar>(y);
			//Get row first data place.
			for(int x = 0 ; x < FreeTouch_Frame.cols ; x++){
				if((FreeTouch_Frame_YCrCb_Cr_Data[x]>=FreeTouch_Cr_Min&&FreeTouch_Frame_YCrCb_Cr_Data[x]<= FreeTouch_Cr_Max)
					&&(FreeTouch_Frame_YCrCb_Cb_Data[x]>=FreeTouch_Cb_Min&&FreeTouch_Frame_YCrCb_Cb_Data[x]<=FreeTouch_Cb_Max)										
				  ){
					FreeTouch_Frame_YCrCb_Cr_Data[x]=U8_White;
				}
				else{
					FreeTouch_Frame_YCrCb_Cr_Data[x]=U8_Black;	
				}				
			}
		}
		Mat FreeTouch_Frame_1B = FreeTouch_Frame_YCrCb_Cr.clone();
		//To 2 value.
	
		Mat FreeTouch_Element = getStructuringElement(MORPH_ELLIPSE,Size(3,3));
		erode(FreeTouch_Frame_1B,FreeTouch_Frame_1B,FreeTouch_Element);
		erode(FreeTouch_Frame_1B,FreeTouch_Frame_1B,FreeTouch_Element);
		FreeTouch_Fill_Hole(FreeTouch_Frame_1B,FreeTouch_Frame_1B);
		medianBlur(FreeTouch_Frame_1B,FreeTouch_Frame_1B, 5);

		FreeTouch_Cascade.detectMultiScale(FreeTouch_Frame_1B,FreeTouch_Hand_Rect,1.1,2,0|CV_HAAR_SCALE_IMAGE,cvSize(20,20));
		if(FreeTouch_Hand_Rect.size()>0){
			FreeTouch_Haar_Hand_Center = Point( FreeTouch_Hand_Rect[0].x + FreeTouch_Hand_Rect[0].width*0.4, FreeTouch_Hand_Rect[0].y + FreeTouch_Hand_Rect[0].height*0.6 );
			Point FreeTouch_Haar_Hand_Center_1( FreeTouch_Hand_Rect[0].x + FreeTouch_Hand_Rect[0].width*0.4+20, FreeTouch_Hand_Rect[0].y + FreeTouch_Hand_Rect[0].height*0.6 );
			Point FreeTouch_Haar_Hand_Center_2( FreeTouch_Hand_Rect[0].x + FreeTouch_Hand_Rect[0].width*0.4-20, FreeTouch_Hand_Rect[0].y + FreeTouch_Hand_Rect[0].height*0.6 );
			Point FreeTouch_Haar_Hand_Center_3( FreeTouch_Hand_Rect[0].x + FreeTouch_Hand_Rect[0].width*0.4, FreeTouch_Hand_Rect[0].y + FreeTouch_Hand_Rect[0].height*0.6+20 );
			Point FreeTouch_Haar_Hand_Center_4( FreeTouch_Hand_Rect[0].x + FreeTouch_Hand_Rect[0].width*0.4, FreeTouch_Hand_Rect[0].y + FreeTouch_Hand_Rect[0].height*0.6-20 );

			findContours(FreeTouch_Frame_1B,FreeTouch_Contours,FreeTouch_Hierarchy,RETR_EXTERNAL,CHAIN_APPROX_NONE);
			FreeTouch_Hand_In_Contour = -1;
			int FreeTouch_Contours_Index = 0;
			for(;FreeTouch_Contours_Index>=0;FreeTouch_Contours_Index = FreeTouch_Hierarchy[FreeTouch_Contours_Index][0]){
				if( pointPolygonTest(FreeTouch_Contours[FreeTouch_Contours_Index],FreeTouch_Haar_Hand_Center  ,true)>=0 ||
					pointPolygonTest(FreeTouch_Contours[FreeTouch_Contours_Index],FreeTouch_Haar_Hand_Center_1,true)>=0 ||
					pointPolygonTest(FreeTouch_Contours[FreeTouch_Contours_Index],FreeTouch_Haar_Hand_Center_2,true)>=0 ||
					pointPolygonTest(FreeTouch_Contours[FreeTouch_Contours_Index],FreeTouch_Haar_Hand_Center_3,true)>=0 ||
					pointPolygonTest(FreeTouch_Contours[FreeTouch_Contours_Index],FreeTouch_Haar_Hand_Center_4,true)>=0 
				){
					FreeTouch_Hand_In_Contour = FreeTouch_Contours_Index;
                    vector<Point> approxCurve;
                    approxPolyDP(FreeTouch_Contours[FreeTouch_Hand_In_Contour], FreeTouch_Contours[FreeTouch_Hand_In_Contour], 30, true);
					//Get this aim contours.
					break;
				}
			}
		}

		FreeTouch_Frame_1B.setTo(Scalar(U8_Black));
		vector<vector<Point> > FreeTouch_Hull(FreeTouch_Contours.size());
		vector<vector<int> > FreeTouch_Hull_Int(FreeTouch_Contours.size());
		if(FreeTouch_Hand_In_Contour>0){
			convexHull(FreeTouch_Contours[FreeTouch_Hand_In_Contour],FreeTouch_Hull[FreeTouch_Hand_In_Contour]); 
			convexHull(FreeTouch_Contours[FreeTouch_Hand_In_Contour],FreeTouch_Hull_Int[FreeTouch_Hand_In_Contour]); 
			
			
			Moments FreeTouch_Hull_Mu 	= moments(FreeTouch_Hull[FreeTouch_Hand_In_Contour],false);
			Moments FreeTouch_Con_Mu 	= moments(FreeTouch_Contours[FreeTouch_Hand_In_Contour],false);
			Point2f FreeTouch_Hull_Mc	= Point2f( FreeTouch_Hull_Mu.m10/FreeTouch_Hull_Mu.m00 , FreeTouch_Hull_Mu.m01/FreeTouch_Hull_Mu.m00 );
			Point2f FreeTouch_Con_Mc	= Point2f( FreeTouch_Con_Mu.m10/FreeTouch_Con_Mu.m00 , FreeTouch_Con_Mu.m01/FreeTouch_Con_Mu.m00 );

			
			vector<FreeTouch_Convexity_Defect> FreeTouch_Convex_Defects;
			FreeTouch_Find_Convexity_Defects(FreeTouch_Contours[FreeTouch_Hand_In_Contour],FreeTouch_Hull_Int[FreeTouch_Hand_In_Contour],FreeTouch_Convex_Defects);

			//printf("%ld\n",FreeTouch_Hull_Int[FreeTouch_Hand_In_Contour].size());

			
				
				ellipse( FreeTouch_Frame, FreeTouch_Haar_Hand_Center, Size( FreeTouch_Hand_Rect[0].width*0.5, FreeTouch_Hand_Rect[0].height*0.5), 0, 0, 360, Scalar( 0, 255, 0 ), 4, 8, 0 );
				
				
				ellipse( FreeTouch_Frame, FreeTouch_Con_Mc, Size(4,4), 0, 0, 360, Scalar( 0, 0 ,255 ), 5, 0, 0 );
				ellipse( FreeTouch_Frame_1B, FreeTouch_Hull_Mc, Size(4,4), 0, 0, 360,U8_White, 5, 0, 0 );
				
			
				drawContours(FreeTouch_Frame,FreeTouch_Contours,FreeTouch_Hand_In_Contour,Scalar(255,0,0),1,8,FreeTouch_Hierarchy);
				drawContours(FreeTouch_Frame_1B,FreeTouch_Contours,FreeTouch_Hand_In_Contour,U8_White,1,8,FreeTouch_Hierarchy);

				
				drawContours(FreeTouch_Frame_1B,FreeTouch_Hull,FreeTouch_Hand_In_Contour,U8_White,1,8,vector<Vec4i>());
			
			
				for(FreeToucn_Tmp_Int = 0; FreeToucn_Tmp_Int < FreeTouch_Hull_Int[FreeTouch_Hand_In_Contour].size(); FreeToucn_Tmp_Int++){
					int FreeTouch_Hull_Index = FreeTouch_Hull_Int[FreeTouch_Hand_In_Contour][FreeToucn_Tmp_Int];
					circle(FreeTouch_Frame, FreeTouch_Contours[FreeTouch_Hand_In_Contour][FreeTouch_Hull_Index], 3, Scalar(0,0,255), 2);
				}
			

				for(FreeToucn_Tmp_Int = 0; FreeToucn_Tmp_Int < FreeTouch_Convex_Defects.size(); FreeToucn_Tmp_Int++){
					circle(FreeTouch_Frame, FreeTouch_Convex_Defects[FreeToucn_Tmp_Int].depth_point, 3, Scalar(255,0,0), 2);
				}


		}

		FreeTouch_End_Time = clock(); 
		stringstream ss;
		ss<<(1.0/((FreeTouch_End_Time-FreeTouch_Start_Time)/CLOCKS_PER_SEC));
		string s1 = ss.str()+"FPS";
		putText(FreeTouch_Frame,s1,Point(20,40),CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 0) ); 
		putText(FreeTouch_Frame_1B,s1,Point(20,40),CV_FONT_HERSHEY_COMPLEX, 1, U8_White ); 

		imshow("FreeTouch_Frame",FreeTouch_Frame);
		imshow("FreeTouch_Frame_1B",FreeTouch_Frame_1B);

		char InputKey = waitKey(1);
		if(InputKey == 'c' || InputKey == 'C'){
			break;
		}
	}

	return 0;
	//Should never get hever.
}


void FreeTouch_Fill_Hole(const Mat srcBw, Mat &dstBw){
    Size m_Size = srcBw.size();
    Mat Temp = Mat::zeros(m_Size.height+2,m_Size.width+2,srcBw.type());//延展图像
    srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
    floodFill(Temp, Point(0, 0), Scalar(255));
    Mat cutImg;//裁剪延展的图像
    Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
    dstBw = srcBw | (~cutImg);
}

void FreeTouch_Find_Convexity_Defects(vector<Point>& contour, vector<int>& hull, vector<FreeTouch_Convexity_Defect>& convexDefects){
    if(hull.size() > 0 && contour.size() > 0){
        CvSeq* contourPoints;
        CvSeq* defects;
        CvMemStorage* storage;
        CvMemStorage* strDefects;
        CvMemStorage* contourStr;
        CvConvexityDefect *defectArray = 0;

        strDefects = cvCreateMemStorage();
        defects = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvSeq),sizeof(CvPoint), strDefects );

        //We transform our vector<Point> into a CvSeq* object of CvPoint.
        contourStr = cvCreateMemStorage();
        contourPoints = cvCreateSeq(CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), contourStr);
        for(int i = 0; i < (int)contour.size(); i++) {
            CvPoint cp = Point(contour[i].x,  contour[i].y);
            cvSeqPush(contourPoints, &cp);
        }

        //Now, we do the same thing with the hull index
        int count = (int) hull.size();
        //int hullK[count];
        int* hullK = (int*) malloc(count*sizeof(int));
        for(int i = 0; i < count; i++) { hullK[i] = hull.at(i); }
        CvMat hullMat = cvMat(1, count, CV_32SC1, hullK);

        // calculate convexity defects
        storage = cvCreateMemStorage(0);
        defects = cvConvexityDefects(contourPoints, &hullMat, storage);
        defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*defects->total);
        cvCvtSeqToArray(defects, defectArray, CV_WHOLE_SEQ);

        for(int i = 0; i<defects->total; i++){
            FreeTouch_Convexity_Defect def;
            def.start       = Point(defectArray[i].start->x, defectArray[i].start->y);
            def.end         = Point(defectArray[i].end->x, defectArray[i].end->y);
            def.depth_point = Point(defectArray[i].depth_point->x, defectArray[i].depth_point->y);
            def.depth       = defectArray[i].depth;
            convexDefects.push_back(def);
        }

        // release memory
        cvReleaseMemStorage(&contourStr);
        cvReleaseMemStorage(&strDefects);
        cvReleaseMemStorage(&storage);

    }
}
