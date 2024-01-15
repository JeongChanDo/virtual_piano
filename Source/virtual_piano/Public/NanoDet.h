// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "PreOpenCVHeaders.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "PostOpenCVHeaders.h"
//https://github.com/hpc203/nanodet-opncv-dnn-cpp-python/tree/main

#include "CoreMinimal.h"

using namespace cv;
using namespace dnn;
using namespace std;

/**
 * 
 */
class VIRTUAL_PIANO_API NanoDet
{
public:
	typedef struct Bbox{
		float score;
		cv::Rect rect;
	};

	NanoDet();
	~NanoDet();

	NanoDet(int input_shape, float confThreshold, float nmsThreshold);
	void detect(Mat& srcimg);
	void detect(Mat& srcimg, std::vector<Bbox>& bboxes);

	std::vector<cv::Rect> get_color_filtered_boxes(cv::Mat image, cv::Mat& skin_image);

private:
	const int stride[3] = { 8, 16, 32 };
	int input_shape[2];   //// height, width
	const float mean[3] = { 103.53, 116.28, 123.675 };
	const float std[3] = { 57.375, 57.12, 58.395 };

	const int reg_max = 7;
	float prob_threshold;
	float iou_threshold;

	Scalar lower_skin = Scalar(0, 20, 70);
	Scalar upper_skin = Scalar(20, 255, 255);

	//const string classesFile = "coco.names";
	//std::vector<string> classes;

	int num_class;
	Net net;

	Mat resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left);
	void normalize(Mat& srcimg);
	void softmax(float* x, int length);
	void post_process(std::vector<Mat> outs, Mat& frame, int newh, int neww, int top, int left);
	void post_process(std::vector<Mat> outs, Mat& frame, int newh, int neww, int top, int left, std::vector<Bbox>& bboxes);
	void generate_proposal(std::vector<int>& classIds, std::vector<float>& confidences, std::vector<cv::Rect>& boxes, const int stride_, Mat out_score, Mat out_box);
	const bool keep_ratio = false;

};
