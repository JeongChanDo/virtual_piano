// Fill out your copyright notice in the Description page of Project Settings.

#include "PianoGameModeBase.h"

void APianoGameModeBase::BeginPlay()
{
	Super::BeginPlay();
	//capture = cv::VideoCapture(0);
	capture = cv::VideoCapture("rtsp://192.168.0.181:8080/video/h264");
	//capture = cv::VideoCapture("c:/nanodet_nail_test_crop.mp4");

	if (!capture.isOpened())
	{
		UE_LOG(LogTemp, Log, TEXT("Open Webcam failed"));
		return;
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("Open Webcam Success"));
	}



	criteria = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 0.03);


	
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
	color_boxes = nanodet.get_color_filtered_boxes(image, skin_image);
	bboxes.clear();
	nanodet.detect(image, bboxes);

	for (const auto& color_rect : color_boxes) {
		cv::rectangle(skin_image, Point(color_rect.x, color_rect.y), Point(color_rect.x + color_rect.width, color_rect.y + color_rect.height), Scalar(0, 0, 255), 2);
	}

	//remove boxes out side of color boxes
	bboxes = getFullyContainedRects(color_boxes, bboxes);


	// color_boxes가 2개면 x기준 정렬
	if (color_boxes.size() == 2)
		sort(color_boxes.begin(), color_boxes.end(), compareRectByX);;


	// color_boxes와 bboxes 수에 따라 tracker 초기화
	if (is_tracker_init == false)
		InitTracker();


	if (is_tracker_init)
	{
		//기존과 동일한 id에 새 pt들을 할당하기 위한 변수들
		std::vector<int> tracker_indexes;
		std::vector<cv::Point2f> tracker_pt_prev;
		std::vector<cv::Point2f> tracker_pt_next;

		//트래커 최근 pt와 id 가져옴
		GetLatestPtsFromTracker(tracker_indexes, tracker_pt_prev);

		// 기존 pt에 매칭된 새 pt 가져옴
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
		cv::calcOpticalFlowPyrLK(prevGray, gray, tracker_pt_prev, tracker_pt_next, status, err, cv::Size(15, 15), 2, criteria);

		// 코너 표시
		for (size_t i = 0; i < tracker_pt_prev.size(); ++i)
		{
			if (status[i])
			{
				cv::Point2f prevPt = tracker_pt_prev[i];
				cv::Point2f nextPt = tracker_pt_next[i];

				cv::line(skin_image, prevPt, nextPt, cv::Scalar(0, 255, 0), 2);
				cv::circle(skin_image, prevPt, 4, cv::Scalar(0, 0, 255), 1);
				cv::circle(skin_image, nextPt, 2, cv::Scalar(255, 0, 0), -1);
			}
		}
		// 다음 프레임 준비를 위한 변수 업데이트
		gray.copyTo(prevGray);


		// pt_next를 트래커 pts에 추가
		UpdateNextPts(tracker_indexes, tracker_pt_next);

		/*
		std::string tmp_str = "tracker[0].pts.size() : " + std::to_string(tracker[0].pts.size());
		cv::putText(skin_image, tmp_str, cv::Point(0, 150), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(125, 125, 125), 2);
		*/


		// tracker_item의 pts drawing
		DrawTrackerPts(skin_image);
	}





	//drawing detection result
	for (auto const bbox : bboxes)
	{
		cv::rectangle(skin_image, Point(bbox.rect.x, bbox.rect.y), Point(bbox.rect.x + bbox.rect.width, bbox.rect.y + bbox.rect.height), Scalar(255, 0, 0), 1);
		string label = cv::format("%.2f", bbox.score);
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		int ymin = max(bbox.rect.y, labelSize.height);
		putText(skin_image, label, Point(bbox.rect.x, ymin), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);

	}





    auto time_end = std::chrono::steady_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    std::string time_spent = "Time spent: " + std::to_string(time_diff) + "ms";
    cv::putText(skin_image, time_spent, cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(125, 125, 125), 2);
	image = skin_image;
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

void APianoGameModeBase::InitTracker()
{
	if (color_boxes.size() == 0 || color_boxes.size() > 2)
		return;

	if (!((bboxes.size() == 5) || (bboxes.size() == 10)))
		return;


	cv::cvtColor(image, prevGray, cv::COLOR_BGR2GRAY);
	capture.read(image);


	int bbox_idx = 0;

	for (const auto color_box : color_boxes)
	{
		for (const auto bbox : bboxes) {
			cv::Point2f pt(bbox.rect.x + bbox.rect.width / 2, bbox.rect.y + bbox.rect.height / 2);
			//p0.push_back(pt);

			TrackerItem item;
			item.bbox = bbox.rect;
			item.pts.push_back(pt);
			tracker[bbox_idx++] = item;
		}
	}
	is_tracker_init = true;
}


std::vector<NanoDet::Bbox> APianoGameModeBase::getFullyContainedRects(const std::vector<cv::Rect>& big_boxes, const std::vector<NanoDet::Bbox>& small_boxes) {
	std::vector<NanoDet::Bbox> result;

	for (const auto& big_rect : big_boxes) {
		for (const auto& bbox : small_boxes) {
			if (big_rect.contains(bbox.rect.tl()) && big_rect.contains(bbox.rect.br())) {
				result.push_back(bbox);
			}
		}
	}

	return result;
}


void APianoGameModeBase::UpdateTracker()
{
	std::vector<int> foundIndex;


	for (const auto& bbox : bboxes)
	{
		bool foundMatch = false;
		for (auto& tracked_item : tracker)
		{
			/*
			// Calculate IOU between detected rectangle and tracked rectangle
			float iou = calculateIOU(skinRegion, trackedRect.second.rect);

			// If IOU is above a threshold, update the tracked rectangle
			if (iou > 0.4) {
				trackedRect.second.rect = skinRegion;
				foundMatch = true;
				foundIndex.push_back(trackedRect.first);
				break;
			}
			*/
		}

		// If no match found, add new tracked rectangle
		/*
		if (!foundMatch) {
			trackedRects[nextId].rect = skinRegion;
			trackedRects[nextId].lifespan = 5;
			trackedRects[nextId].isHandDetected = false;
			nextId++;
		}
		*/
	}
	/*
	//if trackedRect not found, lifespan -1
	for (auto& trackedRect : trackedRects)
	{
		bool foundMatch = false;
		for (auto& id : foundIndex)
		{
			if (id == trackedRect.first)
			{
				foundMatch = true;
				trackedRect.second.lifespan = 5; //찾으면 5로 다시 설정
				break;
			}
		}
		if (foundMatch == false)
		{
			trackedRect.second.lifespan -= 1;
		}
	}

	// lifespan == 0인 trackedRect 삭제
	for (auto& trackedRect : trackedRects)
	{
		if (trackedRect.second.lifespan == 0)
		{
			trackedRects.erase(trackedRect.first);
		}
	}
	return trackedRects;
	*/

}

void APianoGameModeBase::GetLatestPtsFromTracker(std::vector<int>& indexes, std::vector<cv::Point2f>& pts)
{
	for (auto& tracked_item : tracker)
	{
		int id = tracked_item.first;
		TrackerItem item = tracked_item.second;
		pts.push_back(item.pts.back());
		indexes.push_back(id);
	}
}

void APianoGameModeBase::UpdateNextPts(std::vector<int> indexes, std::vector<cv::Point2f>& pts)
{
	for (auto& index : indexes)
	{
		tracker[index].pts.push_back(pts[index]);
		if (tracker[index].pts.size() > tracker_pts_maxlen)
			tracker[index].pts.erase(tracker[index].pts.begin());
	}
}

void APianoGameModeBase::DrawTrackerPts(cv::Mat& draw_image)
{
	for (auto& tracker_item : tracker)
	{
		std::vector<cv::Point2f> tracker_item_pts = tracker_item.second.pts;
		for (size_t i = 0; i < tracker_item_pts.size() - 1; i++)
		{
			cv::Point2f prevPt = tracker_item_pts[i];
			cv::Point2f nextPt = tracker_item_pts[i + 1];

			cv::line(draw_image, prevPt, nextPt, cv::Scalar(255, 255, 0), 2);
		}
	}
}