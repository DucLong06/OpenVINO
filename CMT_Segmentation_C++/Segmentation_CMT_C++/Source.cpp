// Khai báo thư viện cần thiết
# include <iostream>
//# include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/dnn.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

// Nguyên mẫu hàm decoder()
void decode(const Mat& scores, const Mat& geometry, float confThreshold, vector<RotatedRect>& boxes, vector<float>& confidences);


int main(int argc, char** argv)
{
	// 1. Các giá trị cần thiết
	// Ngưỡng xác suất Text detection 
	float confThreshold = 0.1f;
	// Ngưỡng triệt tiêu các bounding box chồng chéo nhau
	float nmsThreshold = 0.4f;
	// Kích thước chiều rộng và chiều cao của ảnh đầu vào dnn (nên là bội số của 32)
	int inpWidth = 320;
	int inpHeight = 320;

	// Pre-trained model cho text detection
	String model = "frozen_east_text_detection.pb";

	// 2. Xử lý
	// Kiểm tra file model
	CV_Assert(!model.empty());

	// Load network.
	Net net = cv::dnn::readNet(model);

	// Open a image file
	Mat frame = imread("21901161_408427682.jpg");
	
	// Các đại lượng đầu ra của mạng
	std::vector<Mat> outs;
	std::vector<String> outNames(2);
	outNames[0] = "feature_fusion/Conv_7/Sigmoid";
	outNames[1] = "feature_fusion/concat_3";

	// Các đại lượng đầu vào của mạng
	Mat blob;

	// Biểu diễn ảnh đầu vào sử dụng phép trừ ảnh, các giá trị trung bình dựa trên mạng ImageNet
	blobFromImage(frame, blob, 1.0, Size(320, 320), Scalar(123.68, 116.78, 103.94), true, false);
	
	// Gọi mạng dnn xử lý đầu vào và đầu ra, outs được gán giá trị bằng 2 lớp đầu ra của mạng
	net.setInput(blob);
	net.forward(outs, outNames);

	// Lấy giá trị xác suất và hình học đầu ra để xử lý
	Mat scores = outs[0];
	Mat geometry = outs[1];

	// Decode predicted bounding boxes.
	vector<RotatedRect> boxes;
	vector<float> confidences;

	// Gọi hàm decode() để xử lý
	decode(scores, geometry, confThreshold, boxes, confidences);

	// Apply non-maximum suppression procedure
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	// Render detections.
	Point2f ratio((float)frame.cols / inpWidth, (float)frame.rows / inpHeight);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		RotatedRect& box = boxes[indices[i]];

		Point2f vertices[4];
		box.points(vertices);
		for (int j = 0; j < 4; ++j)
		{
			vertices[j].x *= ratio.x;
			vertices[j].y *= ratio.y;
		}

		// Vẽ 4 đường thẳng để tạo thành tứ giác bao quanh khối text
		for (int j = 0; j < 4; ++j)
		{
			line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(255, 0, 0), 2);
		}

		

		// Xem kết quả
		//imshow("Ket qua", frame);
	}

	// Xem kết quả
	imshow("Ket qua", frame);

	waitKey(0);
	destroyAllWindows();
	//cout << "Success!" << endl;
	return 0;
}




// Định nghĩa hàm decode()
void decode(const Mat& scores, const Mat& geometry, float confThreshold,
	std::vector<RotatedRect>& boxes, std::vector<float>& confidences)
{
	boxes.clear();
	CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
	CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
	CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);

	const int height = scores.size[2];
	const int width = scores.size[3];
	for (int y = 0; y < height; ++y)
	{
		const float* scoresData = scores.ptr<float>(0, 0, y);
		const float* x0_data = geometry.ptr<float>(0, 0, y);
		const float* x1_data = geometry.ptr<float>(0, 1, y);
		const float* x2_data = geometry.ptr<float>(0, 2, y);
		const float* x3_data = geometry.ptr<float>(0, 3, y);
		const float* anglesData = geometry.ptr<float>(0, 4, y);

		for (int x = 0; x < width; ++x)
		{
			float score = scoresData[x];
			if (score < confThreshold)
				continue;

			// Decode a prediction.
			// Multiple by 4 because feature maps are 4 time less than input image.
			float offsetX = x * 4.0f, offsetY = y * 4.0f;
			float angle = anglesData[x];
			float cosA = std::cos(angle);
			float sinA = std::sin(angle);
			float h = x0_data[x] + x2_data[x];
			float w = x1_data[x] + x3_data[x];

			Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
					offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
			Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
			Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
			RotatedRect r(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);
			boxes.push_back(r);
			confidences.push_back(score);

		}
	}
}
