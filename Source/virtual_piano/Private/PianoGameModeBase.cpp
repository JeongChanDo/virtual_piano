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

	imageTexture = MatToTexture2D(skin_image);
}




void APianoGameModeBase::Inference()
{

    auto time_start = std::chrono::steady_clock::now();
    //cv::flip(image, image, 1);

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
	//if ((bboxes.size() == 5) || (bboxes.size() == 10))
	//	ReInitTracker();

	if (is_tracker_init)
	{
		//기존과 동일한 id에 새 pt들을 할당하기 위한 변수들
		std::vector<int> tracker_indexes;
		std::vector<cv::Point2f> tracker_pt_prev;
		std::vector<cv::Point2f> tracker_pt_next;

		UpdatePts(tracker_indexes, tracker_pt_prev, tracker_pt_next);

		if (isProblem())
		{
			clear_tracker();
			is_tracker_init = false;
			return;
		}



		// pt_next를 트래커 pts에 추가
		UpdateNextPtsToTracker(tracker_indexes, tracker_pt_next);

		//UpdateTracker();


		correct_bboxes.clear();
		UpdatePtsByBboxes();








		// tracker_item의 pts drawing
		DrawTrackerPtsAndRects(skin_image);




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

	std::string tracker_num = "track num: " + std::to_string(tracker.size()) ;
	cv::putText(skin_image, tracker_num, cv::Point(300, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(125, 125, 125), 2);

}




UTexture2D* APianoGameModeBase::MatToTexture2D(const cv::Mat InMat)
{
	//create new texture, set its values
	UTexture2D* Texture = UTexture2D::CreateTransient(InMat.cols, InMat.rows, PF_B8G8R8A8);

	if (InMat.type() == CV_8UC3)//example for pre-conversion of Mat
	{

		// 검은색 영역 투명화
		cv::Mat mask;
		cv::cvtColor(InMat, mask, cv::COLOR_BGR2GRAY);
		cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);

		//mask.setTo(100, mask == 0);
		/*
		std::ostringstream os;
		os << mask.row(100);                         // Put to the stream
		std::string asStr = os.str();             // Get string 

		cv::putText(mask, asStr, Point2d(10, 100), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0), 1);
		*/


		//if the Mat is in BGR space, convert it to BGRA. There is no three channel texture in UE (at least with eight bit)
		cv::cvtColor(InMat, bgraImage, cv::COLOR_BGR2BGRA);



		//mask.setTo(125, mask == 0);
		//mask.setTo(255, mask > 0);
		
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

	next_id = 0;

	for (const auto color_box : color_boxes)
	{

		std::vector<cv::Rect> rects = getRectsInColorBox(color_box);

		for (const auto rect : rects) {
			cv::Point2f pt(rect.x + rect.width / 2, rect.y + rect.height / 2);
			//p0.push_back(pt);

			TrackerItem item;
			item.bbox = rect;
			//item.bbox = bbox.rect;

			item.pts.push_back(pt);
			tracker[next_id++] = item;
		}
	}




	std::string time_spent = "tracker inited ";
	cv::putText(skin_image, time_spent, cv::Point(0, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(125, 125, 125), 2);


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
		cv::Point2f bbox_pt(bbox.rect.x + bbox.rect.width / 2, bbox.rect.y + bbox.rect.height / 2);

		bool foundMatch = false;
		for (auto& tracker_item : tracker)
		{
			//cv::Rect tracker_item_rect = tracker_item.second.bbox;
			cv::Point2f tracker_item_pt = tracker_item.second.pts.back();

			float dist = euclideanDist(bbox_pt, tracker_item_pt);
			if (dist < tracker_dist_limit)
			{
				tracker_item.second.bbox = bbox.rect;
				foundIndex.push_back(tracker_item.first);
				foundMatch = true;
			}
		}

		if (!foundMatch)
		{
			tracker[next_id].bbox = bbox.rect;
			tracker[next_id].pts = std::vector<cv::Point2f>();
			tracker[next_id].pts.push_back(bbox_pt);
			next_id++;
		}

	}



	//if tracker_item not found, lifespan -1
	for (auto& tracker_item : tracker)
	{
		bool foundMatch = false;
		for (auto& id : foundIndex)
		{
			if (id == tracker_item.first)
			{
				foundMatch = true;
				tracker_item.second.lifespan = max_lifespan; //찾으면 로 다시 설정
				break;
			}
		}
		if (foundMatch == false)
		{
			tracker_item.second.lifespan -= 1;
		}
	}

	// lifespan == 0인 trackedRect 삭제
	for (auto& tracker_item : tracker)
	{
		if (tracker_item.second.lifespan == 0)
		{
			tracker.erase(tracker_item.first);
		}
	}
}

void APianoGameModeBase::UpdatePts(std::vector<int>& tracker_indexes, std::vector<cv::Point2f>& tracker_pt_prev, std::vector<cv::Point2f>& tracker_pt_next)
{
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

std::vector<cv::Point2f> APianoGameModeBase::GetLatestPtsFromTracker()
{
	std::vector<cv::Point2f> pts;

	for (auto& tracked_item : tracker)
	{
		int id = tracked_item.first;
		TrackerItem item = tracked_item.second;
		pts.push_back(item.pts.back());
	}
	return pts;
}




void APianoGameModeBase::UpdateNextPtsToTracker(std::vector<int> indexes, std::vector<cv::Point2f>& pts)
{
	for (auto& index : indexes)
	{
		tracker[index].pts.push_back(pts[index]);
		if (tracker[index].pts.size() > tracker_pts_maxlen)
			tracker[index].pts.erase(tracker[index].pts.begin());
	}
}

void APianoGameModeBase::DrawTrackerPtsAndRects(cv::Mat& draw_image)
{
	for (auto& tracker_item : tracker)
	{
		cv::Rect tracker_item_rect = tracker_item.second.bbox;
		std::vector<cv::Point2f> tracker_item_pts = tracker_item.second.pts;
		for (size_t i = 0; i < tracker_item_pts.size() - 1; i++)
		{
			cv::Point2f prevPt = tracker_item_pts[i];
			cv::Point2f nextPt = tracker_item_pts[i + 1];

			cv::line(draw_image, prevPt, nextPt, cv::Scalar(255, 255, 0), 2);
		}

		string label = cv::format("id(%d) : %d", tracker_item.first, tracker_item.second.lifespan);
		cv::Point2f last_pt = cv::Point2f(tracker_item.second.pts.back().x - 20, tracker_item.second.pts.back().y - 20);
		putText(draw_image, label, last_pt, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 255), 1.5);

		/*
		cv::rectangle(draw_image, 
			Point(tracker_item_rect.x, tracker_item_rect.y),
			Point(tracker_item_rect.x + tracker_item_rect.width,
				tracker_item_rect.y + tracker_item_rect.height),
			cv::Scalar(255, 255, 0), 2
		);
		*/
	}
}


void APianoGameModeBase::UpdatePtsByBboxes()
{
	for (auto bbox : bboxes)
	{
		int tracker_id;
		float dist;
		cv::Point2f pt(bbox.rect.x + bbox.rect.width / 2, bbox.rect.y + bbox.rect.height / 2);

		FindClosestPoint(bbox, &tracker_id, &dist);

		if (dist < tracker_dist_limit)
			correct_bboxes.push_back(tracker_id);


		// if pt far replace with bbox centroid
		if (dist >= tracker_dist_limit && dist < tracker_dist_limit * 3)
		{
			tracker[tracker_id].pts.pop_back();
			tracker[tracker_id].pts.push_back(pt);
			correct_bboxes.push_back(tracker_id);
		}
	}
}


void APianoGameModeBase::FindClosestPoint(NanoDet::Bbox bbox, int* id, float* dist)
{
	cv::Point2f pt(bbox.rect.x + bbox.rect.width / 2, bbox.rect.y + bbox.rect.height / 2);
	
	int closet_id;
	float closet_dist = 9999;

	for (auto& tracker_item : tracker)
	{
		int current_id = tracker_item.first;
		cv::Point2f last_pt = tracker_item.second.pts.back();
		float current_dist = euclideanDist(pt, last_pt);
		if (current_dist < closet_dist)
		{
			closet_dist = current_dist;
			closet_id = current_id;
		}
	}
	*id = closet_id;
	*dist = closet_dist;
}


float APianoGameModeBase::euclideanDist(cv::Point2f& a, cv::Point2f& b)
{
	cv::Point2f diff = a - b;
	return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}


/*
void APianoGameModeBase::TrackerLifespanUpdate()
{
	for (auto& tracker_item : tracker)
	{
		tracker_item.second.lifespan -= 1;
		if (tracker_item.second.lifespan == 0)
			tracker.erase(tracker_item.first);



		if (std::find(correct_bboxes.begin(), correct_bboxes.end(), tracker_item.first) != correct_bboxes.end()) {
			tracker_item.second.lifespan = 40;
		}
	}

}
*/








bool APianoGameModeBase::isProblem()
{
	bool isProblem = false;

	//is num tracker item and bbox difference
	if ((bboxes.size() == 5) || (bboxes.size() == 10))
	{
		if (bboxes.size() == 5 && tracker.size() == 10)
			return true;
		else if (bboxes.size() == 10 && tracker.size() == 5)
			return true;
	}


	//detect all fingers
	if ((bboxes.size() == 5) || (bboxes.size() == 10))
	{
		//check any pts outside of bbox
		if (isOutsideOfCbox())
		{
			std::string time_spent = "pt outside of bbox ";
			cv::putText(skin_image, time_spent, cv::Point(0, 150), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(125, 125, 125), 2);

			return true;
		}
	}
	//detect fingers but not all
	else
	{
		//check any pts outside of color_boxes
		if (isOutsideOfCbox())
		{
			std::string time_spent = "pt outside of cbox ";
			cv::putText(skin_image, time_spent, cv::Point(0, 200), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(125, 125, 125), 2);

			return true;
		}
	}


	//check pts seperatly in cbox
	for (auto& color_box : color_boxes)
	{
		std::vector<cv::Point2f> tracker_pts_in_cbox = getTrackerPtsInColorBox(color_box);

		//check pts in cbox is too close
		if (!isPtsTooClose(tracker_pts_in_cbox))
		{
			//check pts in cbox is too far
			if (!isPtsTooFar(tracker_pts_in_cbox))
				isProblem = false;
		}
	}

	return isProblem;
}


bool APianoGameModeBase::isPtsTooClose(std::vector<cv::Point2f>& pts)
{
	for (int i = 0; i < 5; i++) {
		for (int j = i + 1; j < 5; j++) {
			float distance = cv::norm(pts[i] - pts[j]);
			if (distance <= tracker_dist_limit * 2) {
				return true;
			}
		}
	}
	return false;
}


bool APianoGameModeBase::isPtsTooFar(std::vector<cv::Point2f>& pts)
{

	return false;
}

bool APianoGameModeBase::isOutsideOfCbox()
{
	std::vector<cv::Point2f> tracker_all_pts = GetLatestPtsFromTracker();


	for (auto& pt : tracker_all_pts)
	{
		bool isInside = false;
		for (auto& cbox : color_boxes)
		{
			if (cbox.contains(pt))
				isInside = true;
		}

		//if any pt is not in cbox, return true
		if (isInside == false)
			return true;
	}
	// no pt is not in cbox, return false. there is no pt outside of cbox.
	return false;
}


bool APianoGameModeBase::isOutsideOfBbox()
{
	std::vector<cv::Point2f> tracker_all_pts = GetLatestPtsFromTracker();


	for (auto& pt : tracker_all_pts)
	{
		bool isInside = false;
		for (auto& bbox : bboxes)
		{
			if (bbox.rect.contains(pt))
				isInside = true;
		}

		//if any pt is not in cbox, return true
		if (isInside == false)
			return true;
	}
	// no pt is not in cbox, return false. there is no pt outside of cbox.
	return false;
}


std::vector<cv::Point2f> APianoGameModeBase::getTrackerPtsInColorBox(cv::Rect color_box)
{
	std::vector<cv::Point2f> ptsInColorbox;

	for (auto& tracked_item : tracker)
	{
		int id = tracked_item.first;
		TrackerItem item = tracked_item.second;
		cv::Point2f pt = item.pts.back();
		if (color_box.contains(pt))
			ptsInColorbox.push_back(pt);
	}

	return ptsInColorbox;
}


std::vector<cv::Rect> APianoGameModeBase::getRectsInColorBox(cv::Rect color_box)
{
	std::vector<cv::Rect> rectsInColorbox;

	for (auto& bbox : bboxes)
	{
		if (color_box.contains(bbox.rect.tl()) && color_box.contains(bbox.rect.br()))
			rectsInColorbox.push_back(bbox.rect);
	}

	return rectsInColorbox;
}


void APianoGameModeBase::clear_tracker()
{
	for (auto& tracker_item : tracker)
		tracker_item.second.pts.clear();
	tracker.clear();
}