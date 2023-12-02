#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace chrono;

__global__ void gaussianBlur(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = (y * width + x) * 3;

        int halfSize = 1;
        float sumR = 0.0f;
        float sumG = 0.0f;
        float sumB = 0.0f;
        float sumWeight = 0.0f;


        for (int dy = -halfSize; dy <= halfSize; dy++) {
            for (int dx = -halfSize; dx <= halfSize; dx++) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int nIndex = (ny * width + nx) * 3;


                    unsigned char r = inputImage[nIndex];
                    unsigned char g = inputImage[nIndex + 1];
                    unsigned char b = inputImage[nIndex + 2];

                    float weight = exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));


                    sumR += weight * r;
                    sumG += weight * g;
                    sumB += weight * b;
                    sumWeight += weight;
                }
            }
        }

        // Normalizacija
        unsigned char resultR = static_cast<unsigned char>(sumR / sumWeight);
        unsigned char resultG = static_cast<unsigned char>(sumG / sumWeight);
        unsigned char resultB = static_cast<unsigned char>(sumB / sumWeight);

        outputImage[index] = resultR;
        outputImage[index + 1] = resultG;
        outputImage[index + 2] = resultB;
    }
}
int main() {
    auto start = high_resolution_clock::now();

    Mat inputImage = imread("C:\\Users\\zeljk\\Downloads\\hrvatska.jpg", IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "Failed to open image file." << std::endl;
        return -1;
    }


    int width = inputImage.cols;
    int height = inputImage.rows;
    int numPixels = width * height;


    unsigned char* inputImageHost = inputImage.data;
    unsigned char* outputImageHost = new unsigned char[numPixels * 3];


    unsigned char* inputImageDevice;
    unsigned char* outputImageDevice;
    cudaMalloc((void**)&inputImageDevice, numPixels * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&outputImageDevice, numPixels * 3 * sizeof(unsigned char));


    cudaMemcpy(inputImageDevice, inputImageHost, numPixels * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);


    gaussianBlur << <gridSize, blockSize >> > (inputImageDevice, outputImageDevice, width, height, 1.0f);

    cudaMemcpy(outputImageHost, outputImageDevice, numPixels * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    Mat outputImage(height, width, CV_8UC3, outputImageHost);
    imwrite("C:\\Users\\zeljk\\Downloads\\output_image2.jpg", outputImage);


    cudaFree(inputImageDevice);
    cudaFree(outputImageDevice);


    delete[] outputImageHost;

    auto end = high_resolution_clock::now();

    duration<double, milli> duration = end - start;
    double executionTime = duration.count();

    cout << "Vrijeme izvršavanja: " << executionTime << " ms" << endl;

    return 0;
}
