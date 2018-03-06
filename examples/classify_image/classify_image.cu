/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * Full license terms provided in LICENSE.md file.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "examples/classify_image/utils.h"


using namespace std;
using namespace nvinfer1;


class Logger : public ILogger
{
  void log(Severity severity, const char * msg) override
  {
    if (severity != Severity::kINFO)
      cout << msg << endl;
  }
} gLogger;


/**
 * image_file: path to image
 * plan_file: path of the serialized engine file
 * label_file: file with <class_name> per line
 * input_name: name of the input tensor
 * output_name: name of the output tensor
 * preprocessing_fn: 'vgg' or 'inception'
 */
int main(int argc, char *argv[])
{
  if (argc != 7)
  {
    cout << "Usage: classify_image <image_file> <plan_file> <label_file> <input_name> <output_name> <preprocessing_fn>\n";
    return 0;
  }

  string imageFilename = argv[1];
  string planFilename = argv[2];
  string labelFilename = argv[3];
  string inputName = argv[4];
  string outputName = argv[5];
  string preprocessingFn = argv[6];

  /* load the engine */
  cout << "Loading TensorRT engine from plan file..." << endl;
  ifstream planFile(planFilename); 

  if (!planFile.is_open())
  {
    cout << "Could not open plan file." << endl;
    return 1;
  }

  stringstream planBuffer;
  planBuffer << planFile.rdbuf();
  string plan = planBuffer.str();
  IRuntime *runtime = createInferRuntime(gLogger);
  ICudaEngine *engine = runtime->deserializeCudaEngine((void*)plan.data(), plan.size(), nullptr);
  IExecutionContext *context = engine->createExecutionContext();
  
  /* get the input / output dimensions */
  int inputBindingIndex, outputBindingIndex;
  inputBindingIndex = engine->getBindingIndex(inputName.c_str());
  outputBindingIndex = engine->getBindingIndex(outputName.c_str());

  if (inputBindingIndex < 0)
  {
    cout << "Invalid input name." << endl;
    return 1;
  }

  if (outputBindingIndex < 0)
  {
    cout << "Invalid output name." << endl;
    return 1;
  }

  Dims inputDims, outputDims;
  inputDims = engine->getBindingDimensions(inputBindingIndex);
  outputDims = engine->getBindingDimensions(outputBindingIndex);
  int inputWidth, inputHeight;
  inputHeight = inputDims.d[1];
  inputWidth = inputDims.d[2];

  /* read image, convert color, and resize */
  cout << "Preprocessing input..." << endl;
  cv::Mat image = cv::imread(imageFilename, CV_LOAD_IMAGE_COLOR);

  if (image.data == NULL)
  {
    cout << "Could not read image from file." << endl;
    return 1;
  }

  cv::cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
  cv::resize(image, image, cv::Size(inputWidth, inputHeight));

  /* convert from uint8+NHWC to float+NCHW */
  float *inputDataHost, *outputDataHost;
  size_t numInput, numOutput;
  numInput = numTensorElements(inputDims);
  numOutput = numTensorElements(outputDims);
  inputDataHost = (float*) malloc(numInput * sizeof(float));
  outputDataHost = (float*) malloc(numOutput * sizeof(float));
  cvImageToTensor(image, inputDataHost, inputDims);
  if (preprocessingFn == "vgg")
    preprocessVgg(inputDataHost, inputDims);
  else if (preprocessingFn == "inception")
    preprocessInception(inputDataHost, inputDims);
  else
  {
    cout << "Invalid preprocessing function argument, must be vgg or inception. \n" << endl;
    return 1;
  }

  /* transfer to device */
  float *inputDataDevice, *outputDataDevice;
  cudaMalloc(&inputDataDevice, numInput * sizeof(float));
  cudaMalloc(&outputDataDevice, numOutput * sizeof(float));
  cudaMemcpy(inputDataDevice, inputDataHost, numInput * sizeof(float), cudaMemcpyHostToDevice);
  void *bindings[2];
  bindings[inputBindingIndex] = (void*) inputDataDevice;
  bindings[outputBindingIndex] = (void*) outputDataDevice;

  /* execute engine */
  cout << "Executing inference engine..." << endl;
  const int kBatchSize = 1;
  context->execute(kBatchSize, bindings);

  /* transfer output back to host */
  cudaMemcpy(outputDataHost, outputDataDevice, numOutput * sizeof(float), cudaMemcpyDeviceToHost);

  /* parse output */
  vector<size_t> sortedIndices = argsort(outputDataHost, outputDims);

  cout << "\nThe top-5 indices are: ";
  for (int i = 0; i < 5; i++)
    cout << sortedIndices[i] << " ";

  ifstream labelsFile(labelFilename);

  if (!labelsFile.is_open())
  {
    cout << "\nCould not open label file." << endl;
    return 1;
  }

  vector<string> labelMap;
  string label;
  while(getline(labelsFile, label))
  {
    labelMap.push_back(label);
  }

  cout << "\nWhich corresponds to class labels: ";
  for (int i = 0; i < 5; i++)
    cout << endl << i << ". " << labelMap[sortedIndices[i]];
  cout << endl;

  /* clean up */
  runtime->destroy();
  engine->destroy();
  context->destroy();
  free(inputDataHost);
  free(outputDataHost);
  cudaFree(inputDataDevice);
  cudaFree(outputDataDevice);

  return 0;
}
