#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <time.h>

#include "utils.h"

using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;
using namespace cv;
using namespace std::chrono;

int main(int argc, char* argv[]) {

    // Set dirs variables
    string ROOTDIR = "../";
    string LABELS = "demo/ssd_inception_v2/classes.pbtxt";
    string GRAPH = "demo/ssd_inception_v2/frozen_inference_graph.pb";
    string VIDEO_FILE = "demo/ssd_inception_v2/ride_2.mp4";
    
    // Set input & output nodes names
    string inputLayer = "image_tensor:0";
    vector<string> outputLayer = {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"};

    // Load and initialize the model from .pb file
    std::unique_ptr<tensorflow::Session> session;
    string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
    
    LOG(INFO) << "graphPath:" << graphPath;
    Status loadGraphStatus = loadGraph(graphPath, &session);
    if (!loadGraphStatus.ok()) {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return -1;
    } else
        LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;


    // Load labels map from .pbtxt file
    std::map<int, std::string> labelsMap = std::map<int,std::string>();
    Status readLabelsMapStatus = readLabelsMapFile(tensorflow::io::JoinPath(ROOTDIR, LABELS), labelsMap);
    if (!readLabelsMapStatus.ok()) {
        LOG(ERROR) << "readLabelsMapFile(): ERROR" << loadGraphStatus;
        return -1;
    } else
        LOG(INFO) << "readLabelsMapFile(): labels map loaded with " << labelsMap.size() << " label(s)" << endl;

    Mat frame;
    Tensor tensor;
    std::vector<Tensor> outputs;
    double thresholdScore = 0.5;
    double thresholdIOU = 0.8;

    // FPS count
    int nFrames = 30;
    int iFrame = 0;
    double fps = 0.;
    double duration = 0.;

    high_resolution_clock::time_point start = high_resolution_clock::now();
    high_resolution_clock::time_point end, infer_end;

    string videoPath = tensorflow::io::JoinPath(ROOTDIR, VIDEO_FILE);
    // Start streaming frames from camera
    VideoCapture cap(videoPath);

    tensorflow::TensorShape shape = tensorflow::TensorShape();
    shape.AddDim(1);
    shape.AddDim((int64)cap.get(CAP_PROP_FRAME_HEIGHT));
    shape.AddDim((int64)cap.get(CAP_PROP_FRAME_WIDTH));
    shape.AddDim(3);

    tensor = Tensor(tensorflow::DT_UINT8, shape);
    bool ret = false;

    while (cap.isOpened()) {
        start = high_resolution_clock::now();
        
        ret = cap.read(frame);
        if(!ret)
        {
            cap.release();
            continue;
        }

        cvtColor(frame, frame, COLOR_BGR2RGB);

        if (++iFrame % nFrames == 0) {
            fps = 1. * nFrames / duration * 1000.;
            duration = 0.;
        }

        // Convert mat to tensor
        Status readTensorStatus = readTensorFromMat(frame, tensor);
        if (!readTensorStatus.ok()) {
            LOG(ERROR) << "Mat->Tensor conversion failed: " << readTensorStatus;
            return -1;
        }

        // Run the graph on tensor
        outputs.clear();
        Status runStatus = session->Run({{inputLayer, tensor}}, outputLayer, {}, &outputs);
        if (!runStatus.ok()) {
            LOG(ERROR) << "Running model failed: " << runStatus;
            return -1;
        }
        end = high_resolution_clock::now();
        duration += (double) duration_cast<milliseconds>(end - start).count();

        // Extract results from the outputs vector
        tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
        tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
        tensorflow::TTypes<float>::Flat numDetections = outputs[3].flat<float>();
        tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float,3>();

        vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore);
        
        // Draw bboxes and captions
        cvtColor(frame, frame, COLOR_BGR2RGB);
        drawBoundingBoxesOnImage(frame, scores, classes, boxes, labelsMap, goodIdxs);

        putText(frame, "TensorFlow: CPU feed", Point(0, frame.rows - 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 255), 2);
        putText(frame, to_string(fps).substr(0, 5), Point(0, frame.rows - 5), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        imshow("stream", frame);
        waitKey(1);
        

        if (iFrame % 100 == 0)
        {
            LOG(INFO) << "Speed: " << to_string(fps).substr(0, 5);
        }

    }
    destroyAllWindows();

    return 0;
}