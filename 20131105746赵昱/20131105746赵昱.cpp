#include "stdafx.h"
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

int main(int argc, char *argv[]) {
	CvHaarClassifierCascade *pCascadeFrontal = 0, *pCascadeProfile = 0;
	CvMemStorage *pStorage = cvCreateMemStorage(0);
	CvSeq *pFaceRectSeq;
	int i;
	IplImage *pInpImg = cvLoadImage("D:/软件工程与计算/人脸识别/20131105746赵昱/6.jpg",CV_LOAD_IMAGE_COLOR);
	pCascadeFrontal = (CvHaarClassifierCascade *) cvLoad ("D:/软件工程与计算/人脸识别/20131105746赵昱/haarcascade/haarcascade_frontalface_default.xml");
	pCascadeProfile = (CvHaarClassifierCascade *) cvLoad ("D:/软件工程与计算/人脸识别/20131105746赵昱/haarcascade/haarcascade_profileface.xml");
	if (!pInpImg || !pCascadeFrontal || !pCascadeProfile) {
		printf("缺失文件\n");
		exit(0);
	}
	cvNamedWindow("照片", CV_WINDOW_NORMAL);
	cvShowImage("照片", pInpImg);
	cvWaitKey(50);
	pFaceRectSeq = cvHaarDetectObjects(pInpImg, pCascadeFrontal, pStorage,1.1,3,CV_HAAR_DO_CANNY_PRUNING,cvSize(40, 40));	
	for (i=0 ; i < (pFaceRectSeq ? pFaceRectSeq->total : 0) ; i++) {
		CvRect* r = (CvRect*)cvGetSeqElem(pFaceRectSeq, i);
		CvPoint pt1 = { r->x, r->y };
		CvPoint pt2 = { r->x + r->width, r->y + r->height };
		printf("Frontal face坐标(%d, %d)\n", pt2.x, pt2.y);
		cvRectangle(pInpImg, pt1, pt2, CV_RGB(0,255,0), 3, 4, 0);
		cvSetImageROI(pInpImg, *r);
		cvSmooth(pInpImg, pInpImg, CV_GAUSSIAN, 5, 3);
		cvResetImageROI(pInpImg);
	}
	cvShowImage("照片", pInpImg);
	cvWaitKey(1);
	pFaceRectSeq = cvHaarDetectObjects(pInpImg, pCascadeProfile, pStorage,1.1,3,CV_HAAR_DO_CANNY_PRUNING,cvSize(40,40));
	for (i=0 ; i < (pFaceRectSeq ? pFaceRectSeq->total : 0) ; i++) {
		CvRect* r = (CvRect*)cvGetSeqElem(pFaceRectSeq, i);
		CvPoint pt1 = { r->x, r->y };
		CvPoint pt2 = { r->x + r->width, r->y + r->height };
		printf("Profile face坐标(%d, %d)\n", pt2.x, pt2.y);
		cvRectangle(pInpImg, pt1, pt2, CV_RGB(255,255,0), 3, 4, 0);
		cvSetImageROI(pInpImg, *r);
		cvSmooth(pInpImg, pInpImg, CV_GAUSSIAN, 5, 3);
		cvResetImageROI(pInpImg);
	}
	cvShowImage("照片", pInpImg);
	cvWaitKey(0);
	cvDestroyWindow("照片");
	cvReleaseImage(&pInpImg);
	if (pCascadeFrontal) cvReleaseHaarClassifierCascade(&pCascadeFrontal);
	if (pCascadeProfile) cvReleaseHaarClassifierCascade(&pCascadeProfile);
	if (pStorage) cvReleaseMemStorage(&pStorage);
}