// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include "PreOpenCVHeaders.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "PostOpenCVHeaders.h"
#include "NanoDet.h"
#include <map>

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
	};

	cv::VideoCapture capture;
	cv::Mat bgraImage;
	cv::Mat image;
	NanoDet nanodet = NanoDet(320, 0.45, 0.3);

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

	void InitTracker();
	std::vector<NanoDet::Bbox> getFullyContainedRects(const std::vector<cv::Rect>& big_boxes, const std::vector<NanoDet::Bbox>& small_boxes);
	void UpdateTracker();
	void GetLatestPtsFromTracker(std::vector<int>& indexes, std::vector<cv::Point2f>& pts);
	void UpdateNextPts(std::vector<int> indexes, std::vector<cv::Point2f>& pts);
	void DrawTrackerPts(cv::Mat& draw_image);

	std::vector<cv::Rect> color_boxes;
	std::vector<NanoDet::Bbox> bboxes;
};
