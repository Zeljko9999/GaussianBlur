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

#define BLOCK_SIZE 32

__global__ void gaussianBlur(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory for inputImage
    __shared__ unsigned char sharedImage[BLOCK_SIZE + 2][BLOCK_SIZE + 2][3];

    // Populate shared memory
    int sharedX = threadIdx.x + 1;
    int sharedY = threadIdx.y + 1;

    if (x < width && y < height) {
        int index = (y * width + x) * 3;
        sharedImage[sharedY][sharedX][0] = inputImage[index];
        sharedImage[sharedY][sharedX][1] = inputImage[index + 1];
        sharedImage[sharedY][sharedX][2] = inputImage[index + 2];

        // Border regions of shared memory
        if (threadIdx.x == 0) {
            // Left border
            sharedImage[sharedY][0][0] = inputImage[(y * width + max(x - 1, 0)) * 3];
            sharedImage[sharedY][0][1] = inputImage[(y * width + max(x - 1, 0)) * 3 + 1];
            sharedImage[sharedY][0][2] = inputImage[(y * width + max(x - 1, 0)) * 3 + 2];
        }
        else if (threadIdx.x == blockDim.x - 1) {
            // Right border
            sharedImage[sharedY][BLOCK_SIZE + 1][0] = inputImage[(y * width + min(x + 1, width - 1)) * 3];
            sharedImage[sharedY][BLOCK_SIZE + 1][1] = inputImage[(y * width + min(x + 1, width - 1)) * 3 + 1];
            sharedImage[sharedY][BLOCK_SIZE + 1][2] = inputImage[(y * width + min(x + 1, width - 1)) * 3 + 2];
        }

        if (threadIdx.y == 0) {
            // Top border
            sharedImage[0][sharedX][0] = inputImage[(max(y - 1, 0) * width + x) * 3];
            sharedImage[0][sharedX][1] = inputImage[(max(y - 1, 0) * width + x) * 3 + 1];
            sharedImage[0][sharedX][2] = inputImage[(max(y - 1, 0) * width + x) * 3 + 2];
        }
        else if (threadIdx.y == blockDim.y - 1) {
            // Bottom border
            sharedImage[BLOCK_SIZE + 1][sharedX][0] = inputImage[(min(y + 1, height - 1) * width + x) * 3];
            sharedImage[BLOCK_SIZE + 1][sharedX][1] = inputImage[(min(y + 1, height - 1) * width + x) * 3];
            sharedImage[BLOCK_SIZE + 1][sharedX][2] = inputImage[(min(y + 1, height - 1) * width + x) * 3 + 2];
        }
    }

    // Synchronize threads to ensure all shared memory is populated
    __syncthreads();

    if (x < width && y < height) {
        int index = (y * width + x) * 3;

        int halfSize = 1;
        float sumR = 0.0f;
        float sumG = 0.0f;
        float sumB = 0.0f;
        float sumWeight = 0.0f;

        for (int dy = -halfSize; dy <= halfSize; dy++) {
            for (int dx = -halfSize; dx <= halfSize; dx++) {
                int nx = sharedX + dx;
                int ny = sharedY + dy;

                unsigned char r = sharedImage[ny][nx][0];
                unsigned char g = sharedImage[ny][nx][1];
                unsigned char b = sharedImage[ny][nx][2];

                float weight = exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));

                sumR += weight * r;
                sumG += weight * g;
                sumB += weight * b;
                sumWeight += weight;
            }
        }

        // Synchronize threads to ensure all shared memory is populated
        __syncthreads();

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

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    gaussianBlur << <gridSize, blockSize >> > (inputImageDevice, outputImageDevice, width, height, 1.0f);

    cudaMemcpy(outputImageHost, outputImageDevice, numPixels * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    Mat outputImage(height, width, CV_8UC3, outputImageHost);
    imwrite("C:\\Users\\zeljk\\Downloads\\output_image3.jpg", outputImage);

    cudaFree(inputImageDevice);
    cudaFree(outputImageDevice);

    delete[] outputImageHost;

    auto end = high_resolution_clock::now();
    duration<double, milli> duration = end - start;
    double executionTime = duration.count();

    cout << "Vrijeme izvršavanja: " << executionTime << " ms" << endl;

    return 0;
}
