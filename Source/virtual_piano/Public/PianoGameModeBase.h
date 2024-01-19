// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include "PreOpenCVHeaders.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "PostOpenCVHeaders.h"
#include "NanoDet.h"
#include <map>
#include <iostream>
#include <fstream>

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "PianoGameModeBase.generated.h"

/**
 * 
 */
UCLASS()
class VIRTUAL_PIANO_API APianoGameModeBase : public AGameModeBase
{
	GENERATED_BODY()
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:
	static bool compareRectByX(cv::Rect& rect1, cv::Rect& rect2) {
		return rect1.x < rect2.x;
	}
	static bool compareRectByArea(cv::Rect& rect1, cv::Rect& rect2) {
		return rect1.width * rect1.height > rect2.width * rect2.height;
	}


	typedef struct TrackerItem
	{
		cv::Rect bbox;
		std::vector<cv::Point2f> pts;
		std::vector<cv::Point2f> train_pt;
		std::vector<int> preds;
		bool isPtOutofBox;
		int lifespan = 30;
	};
	int max_lifespan = 30;
	int max_pres_len = 3;

	cv::VideoCapture capture;
	cv::Mat bgraImage;
	cv::Mat image, skin_image;
	NanoDet nanodet = NanoDet(320, 0.4, 0.3);

	UFUNCTION(BlueprintCallable)
	void ReadFrame();
	void Inference();

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	UTexture2D* imageTexture;

	UTexture2D* MatToTexture2D(const cv::Mat InMat);
	




	cv::Mat prevFrame, gray, prevGray;
	cv::TermCriteria criteria;

	// vars for optical flow
	std::vector<cv::Point2f> p0, p1;
	std::vector<uchar> status;
	std::vector<float> err;

	//tracker
	bool is_tracker_init = false;

	std::map<int, TrackerItem> tracker;
	int tracker_pts_maxlen = 10;
	float tracker_dist_limit = 10;
	int next_id = 0;
	
	void InitTracker();
	std::vector<NanoDet::Bbox> getFullyContainedRects(const std::vector<cv::Rect>& big_boxes, const std::vector<NanoDet::Bbox>& small_boxes);
	void UpdateTracker();



	void UpdatePts(std::vector<int>& indexes, std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& pts_next);
	void GetLatestPtsFromTracker(std::vector<int>& indexes, std::vector<cv::Point2f>& pts);
	std::vector<cv::Point2f> GetLatestPtsFromTracker();

	void UpdateNextPtsToTracker(std::vector<int> indexes, std::vector<cv::Point2f>& pts);
	void DrawTrackerPtsAndRects(cv::Mat& draw_image);

	void UpdatePtsByBboxes();
	void FindClosestPoint(NanoDet::Bbox bbox, int* id, float* dist);
	float euclideanDist(cv::Point2f& a, cv::Point2f& b);
	void TrackerLifespanUpdate();


	std::vector<cv::Rect> color_boxes;
	std::vector<NanoDet::Bbox> bboxes;
	std::vector<int> correct_tracker_id;
	std::vector<int> correct_bbox_id;

	std::map<int, int> correct_index_map;

	bool isProblem();
	bool isPtsTooClose(std::vector<cv::Point2f>& pts);
	bool isPtsTooFar(std::vector<cv::Point2f>& pts);
	bool isOutsideOfCbox();
	bool isOutsideOfBbox();

	std::vector<cv::Point2f> getTrackerPtsInColorBox(cv::Rect color_box);
	std::vector<cv::Rect> getRectsInColorBox(cv::Rect color_box);

	void clear_tracker();


	//for ml data
	bool isTrain = false;
	bool isSaved = false;
	void saveCsvForTrain();
	void savePointsToCsv(const std::vector<cv::Point2f>& pts_train, int subVectorSize, const std::string& filename);
	float getVariance(const std::vector<float>& dists);
	float getSum(const std::vector<float>& dists);
	float getMidPointsDistance(const std::vector<cv::Point2f>& pts_train);
	// ����� �� �ε�
	cv::Ptr<cv::ml::RTrees> MLModel = cv::Algorithm::load<cv::ml::RTrees>("c:/240119_02_pts_finger_rftree.yml");
	void PredictTracker();
	int findMostFrequent(std::vector<int>& dists);
	
	cv::Mat GetMLDataFromTracker(int index);
	bool containsInt(std::vector<int>& vals, int a);

};