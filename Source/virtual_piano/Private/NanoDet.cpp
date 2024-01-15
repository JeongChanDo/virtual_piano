// Fill out your copyright notice in the Description page of Project Settings.


#include "NanoDet.h"
NanoDet::NanoDet()
{
}

NanoDet::~NanoDet()
{
}

NanoDet::NanoDet(int input_shape, float confThreshold, float nmsThreshold)
{
	assert(input_shape == 320 || input_shape == 416);
	this->input_shape[0] = input_shape;
	this->input_shape[1] = input_shape;
	this->prob_threshold = confThreshold;
	this->iou_threshold = nmsThreshold;

	/*
	ifstream ifs(this->classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->classes.push_back(line);
	this->num_class = this->classes.size();
	*/
	this->num_class = 1;
	if (input_shape == 320)
	{
		this->net = readNet("c:/nanodet_finger_v4_048_sim.onnx");
		//this->net = readNet("nanodet_finger_v3_sim_fp16.onnx");

	}
	else
	{
		this->net = readNet("c:/nanodet_finger_v3_sim.onnx");
	}
}

Mat NanoDet::resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->input_shape[0];
	*neww = this->input_shape[1];
	Mat dstimg;
	if (this->keep_ratio && srch != srcw)
	{
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1)
		{
			*newh = this->input_shape[0];
			*neww = int(this->input_shape[1] / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->input_shape[1] - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->input_shape[1] - *neww - *left, BORDER_CONSTANT, 0);
		}
		else
		{
			*newh = (int)this->input_shape[0] * hw_scale;
			*neww = this->input_shape[1];
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->input_shape[0] - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->input_shape[0] - *newh - *top, 0, 0, BORDER_CONSTANT, 0);
		}
	}
	else
	{
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void NanoDet::normalize(Mat& img)
{
	img.convertTo(img, CV_32F);
	int i = 0, j = 0;
	for (i = 0; i < img.rows; i++)
	{
		float* pdata = (float*)(img.data + i * img.step);
		for (j = 0; j < img.cols; j++)
		{
			pdata[0] = (pdata[0] - this->mean[0]) / this->std[0];
			pdata[1] = (pdata[1] - this->mean[1]) / this->std[1];
			pdata[2] = (pdata[2] - this->mean[2]) / this->std[2];
			pdata += 3;
		}

		//        float* pdata = img.ptr<float>(i);
		//        for(j = 0; j < img.cols; j++)
		//        {
		//            pdata[3 * j] = (pdata[3 * j] - this->mean[0]) / this->std[0];
		//            pdata[3 * j + 1] = (pdata[3 * j + 1] - this->mean[1]) / this->std[1];
		//            pdata[3 * j + 2] = (pdata[3 * j + 2] - this->mean[2]) / this->std[2];
		//        }
	}
}

//Mat NanoDet::normalize(Mat src)
//{
//	std::vector<Mat> bgrChannels(3);
//	split(src, bgrChannels);
//	for (auto i = 0; i < bgrChannels.size(); i++)
//	{
//		bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / this->std[i], (0.0 - this->mean[i]) / this->std[i]);
//	}
//	Mat dst;
//	merge(bgrChannels, dst);
//	return dst;
//}

void NanoDet::detect(Mat& srcimg)
{
	int newh = 0, neww = 0, top = 0, left = 0;
	Mat dstimg = this->resize_image(srcimg, &newh, &neww, &top, &left);
	this->normalize(dstimg);
	Mat blob = blobFromImage(dstimg);

	this->net.setInput(blob);
	std::vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	/*
	cout << outs[0].reshape(1, outs[0].size[1]).rowRange(0, 10) << endl << endl;

	for (int i = 0; i < 6; i++)
	{
		cout << this->net.getUnconnectedOutLayersNames()[i] << endl;
		cout << outs[i].size() << outs[i].size[0] << " x " << outs[i].size[1] << " x " << outs[i].size[2] << endl;

		cout << outs[i].reshape(1, outs[i].size[1]).size() << endl;
		cout << outs[i].reshape(1, outs[i].size[1]).rowRange(0, 1) << endl << endl;
	}
	*/

	this->post_process(outs, srcimg, newh, neww, top, left);
}

void NanoDet::detect(Mat& srcimg, std::vector<Bbox>& bboxes)
{
	int newh = 0, neww = 0, top = 0, left = 0;
	Mat dstimg = this->resize_image(srcimg, &newh, &neww, &top, &left);
	this->normalize(dstimg);
	Mat blob = blobFromImage(dstimg);

	this->net.setInput(blob);
	std::vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	/*
	cout << outs[0].reshape(1, outs[0].size[1]).rowRange(0, 10) << endl << endl;

	for (int i = 0; i < 6; i++)
	{
		cout << this->net.getUnconnectedOutLayersNames()[i] << endl;
		cout << outs[i].size() << outs[i].size[0] << " x " << outs[i].size[1] << " x " << outs[i].size[2] << endl;

		cout << outs[i].reshape(1, outs[i].size[1]).size() << endl;
		cout << outs[i].reshape(1, outs[i].size[1]).rowRange(0, 1) << endl << endl;
	}
	*/

	this->post_process(outs, srcimg, newh, neww, top, left, bboxes);
}


void NanoDet::softmax(float* x, int length)
{
	float sum = 0;
	int i = 0;
	for (i = 0; i < length; i++)
	{
		x[i] = exp(x[i]);
		sum += x[i];
	}
	for (i = 0; i < length; i++)
	{
		x[i] /= sum;
	}
}

void NanoDet::generate_proposal(std::vector<int>& classIds, std::vector<float>& confidences, std::vector<cv::Rect>& boxes, const int stride_, Mat out_score, Mat out_box)
{
	const int num_grid_y = (int)this->input_shape[0] / stride_;
	const int num_grid_x = (int)this->input_shape[1] / stride_;
	const int reg_1max = this->reg_max + 1;

	if (out_score.dims == 3)
	{
		out_score = out_score.reshape(0, num_grid_x * num_grid_y);
	}
	if (out_box.dims == 3)
	{
		out_box = out_box.reshape(0, num_grid_x * num_grid_y);
	}
	for (int i = 0; i < num_grid_y; i++)
	{
		for (int j = 0; j < num_grid_x; j++)
		{
			const int idx = i * num_grid_x + j;
			Mat scores = out_score.row(idx).colRange(0, num_class);
			Point classIdPoint;
			double score;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &score, 0, &classIdPoint);

			if (score >= this->prob_threshold)
			{
				float* pbox = (float*)out_box.data + idx * reg_1max * 4;
				float dis_pred[4];
				for (int k = 0; k < 4; k++)
				{
					this->softmax(pbox, reg_1max);
					float dis = 0.f;
					for (int l = 0; l < reg_1max; l++)
					{
						dis += l * pbox[l];
					}
					dis_pred[k] = dis * stride_;
					pbox += reg_1max;
				}

				float pb_cx = (j + 0.5f) * stride_ - 0.5;
				float pb_cy = (i + 0.5f) * stride_ - 0.5;
				float x0 = pb_cx - dis_pred[0];
				float y0 = pb_cy - dis_pred[1];
				float x1 = pb_cx + dis_pred[2];
				float y1 = pb_cy + dis_pred[3];

				classIds.push_back(classIdPoint.x);
				confidences.push_back(score);
				boxes.push_back(cv::Rect((int)x0, (int)y0, (int)(x1 - x0), (int)(y1 - y0)));
			}
		}
	}
}


void NanoDet::post_process(std::vector<Mat> outs, Mat& frame, int newh, int neww, int top, int left)
{
	/////generate proposals
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	this->generate_proposal(classIds, confidences, boxes, this->stride[0], outs[0], outs[1]);
	this->generate_proposal(classIds, confidences, boxes, this->stride[1], outs[2], outs[3]);
	this->generate_proposal(classIds, confidences, boxes, this->stride[2], outs[4], outs[5]);



	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	NMSBoxes(boxes, confidences, this->prob_threshold, this->iou_threshold, indices);

	float ratioh = (float)frame.rows / newh;
	float ratiow = (float)frame.cols / neww;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		int xmin = (int)max((box.x - left) * ratiow, 0.f);
		int ymin = (int)max((box.y - top) * ratioh, 0.f);
		int xmax = (int)min((box.x - left + box.width) * ratiow, (float)frame.cols);
		int ymax = (int)min((box.y - top + box.height) * ratioh, (float)frame.rows);
		rectangle(frame, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 3);


		//Get the label for the class name and its confidence
		string label = cv::format("%.2f", confidences[idx]);
		//Display the label at the top of the bounding box
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		ymin = max(ymin, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(frame, label, Point(xmin, ymin), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}

void NanoDet::post_process(std::vector<Mat> outs, Mat& frame, int newh, int neww, int top, int left, std::vector<Bbox>& bboxes)
{
	/////generate proposals
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	this->generate_proposal(classIds, confidences, boxes, this->stride[0], outs[0], outs[1]);
	this->generate_proposal(classIds, confidences, boxes, this->stride[1], outs[2], outs[3]);
	this->generate_proposal(classIds, confidences, boxes, this->stride[2], outs[4], outs[5]);



	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	NMSBoxes(boxes, confidences, this->prob_threshold, this->iou_threshold, indices);

	float ratioh = (float)frame.rows / newh;
	float ratiow = (float)frame.cols / neww;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		int xmin = (int)max((box.x - left) * ratiow, 0.f);
		int ymin = (int)max((box.y - top) * ratioh, 0.f);
		int xmax = (int)min((box.x - left + box.width) * ratiow, (float)frame.cols);
		int ymax = (int)min((box.y - top + box.height) * ratioh, (float)frame.rows);
		cv::Rect rect = cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin);
		//rectangle(frame, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 3);
		Bbox bbox;
		bbox.score = confidences[idx];
		bbox.rect = rect;
		bboxes.emplace_back(bbox);

		/*
		//Get the label for the class name and its confidence
		string label = cv::format("%.2f", confidences[idx]);
		//Display the label at the top of the bounding box
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		ymin = max(ymin, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(frame, label, Point(xmin, ymin), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
		*/
	}
}


std::vector<cv::Rect> NanoDet::get_color_filtered_boxes(cv::Mat image, cv::Mat& skin_image) {
	cv::Mat hsv_image;
	cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

	cv::Mat skin_mask;
	cv::inRange(hsv_image, lower_skin, upper_skin, skin_mask);

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
	cv::morphologyEx(skin_mask, skin_mask, cv::MORPH_OPEN, kernel);

	cv::bitwise_and(image, image, skin_image, skin_mask);
	std::vector<cv::Rect> bounding_boxes;

	/*
	std::std::vector<std::std::vector<cv::Point>> contours;
	cv::findContours(skin_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cout << "bounding_boxes" << endl;

	for (const auto& contour : contours) {
		cv::Rect bounding_box = cv::boundingRect(contour);
		if (bounding_box.width * bounding_box.height > 100 * 100) {
			bounding_box.x -= 10;
			bounding_box.y -= 10;
			if (bounding_box.x < 0) bounding_box.x = 0;
			if (bounding_box.y < 0) bounding_box.y = 0;

			bounding_box.width += 20;
			bounding_box.height += 20;
			if (bounding_box.x + bounding_box.width > skin_image.size[0]) bounding_box.width = skin_image.size[0] - bounding_box.x - 22;
			if (bounding_box.y + bounding_box.height > skin_image.size[1]) bounding_box.height = skin_image.size[1] - bounding_box.y - 22;


			bounding_boxes.push_back(bounding_box);
			cout << bounding_box << endl;
		}
	}
	*/
	return bounding_boxes;
}
