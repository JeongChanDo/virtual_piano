// Fill out your copyright notice in the Description page of Project Settings.

#include "PianoGameModeBase.h"

void APianoGameModeBase::BeginPlay()
{
	Super::BeginPlay();
	//capture = cv::VideoCapture(0);
	capture = cv::VideoCapture("rtsp://192.168.0.181:8080/video/h264");

	if (!capture.isOpened())
	{
		UE_LOG(LogTemp, Log, TEXT("Open Webcam failed"));
		return;
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("Open Webcam Success"));
	}

}


void APianoGameModeBase::ReadFrame()
{
	UE_LOG(LogTemp, Log, TEXT("ReadFrame is called"));

	if (!capture.isOpened())
	{
		return;
	}
	capture.read(image);

	Inference();

	imageTexture = MatToTexture2D(image);
}




void APianoGameModeBase::Inference()
{

    auto time_start = std::chrono::steady_clock::now();
    //cv::flip(image, image, 1);


	cv::Mat skin_image;
	std::vector<cv::Rect> color_boxes = nanodet.get_color_filtered_boxes(image, skin_image);
	nanodet.detect(skin_image);

	image = skin_image;
    auto time_end = std::chrono::steady_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    std::string time_spent = "Time spent: " + std::to_string(time_diff) + "ms";
    cv::putText(image, time_spent, cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(125, 125, 125), 2);
}




UTexture2D* APianoGameModeBase::MatToTexture2D(const cv::Mat InMat)
{
	//create new texture, set its values
	UTexture2D* Texture = UTexture2D::CreateTransient(InMat.cols, InMat.rows, PF_B8G8R8A8);

	if (InMat.type() == CV_8UC3)//example for pre-conversion of Mat
	{
		//if the Mat is in BGR space, convert it to BGRA. There is no three channel texture in UE (at least with eight bit)
		cv::cvtColor(InMat, bgraImage, cv::COLOR_BGR2BGRA);



		// 검은색 영역 투명화
		cv::Mat mask;
		cv::cvtColor(InMat, mask, cv::COLOR_BGR2GRAY);
		cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);
		std::vector<cv::Mat> channels;
		cv::split(bgraImage, channels);
		channels[3] = mask;
		cv::merge(channels, bgraImage);





		//Texture->SRGB = 0;//set to 0 if Mat is not in srgb (which is likely when coming from a webcam)
		//other settings of the texture can also be changed here
		//Texture->UpdateResource();

		//actually copy the data to the new texture
		FTexture2DMipMap& Mip = Texture->GetPlatformData()->Mips[0];
		void* Data = Mip.BulkData.Lock(LOCK_READ_WRITE);//lock the texture data
		FMemory::Memcpy(Data, bgraImage.data, bgraImage.total() * bgraImage.elemSize());//copy the data
		Mip.BulkData.Unlock();
		Texture->PostEditChange();
		Texture->UpdateResource();
		return Texture;
	}
	//UE_LOG(LogTemp, Log, TEXT("CV_8UC3"));
	//if the texture hasnt the right pixel format, abort.
	Texture->PostEditChange();
	Texture->UpdateResource();
	return Texture;
}
