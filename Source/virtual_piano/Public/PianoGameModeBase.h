// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include "PreOpenCVHeaders.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "PostOpenCVHeaders.h"
#include "NanoDet.h"

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


};
