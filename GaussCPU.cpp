#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace chrono;

// funkcija za spremanje RGB slike u jpg formatu
bool saveImage(const string& filename, const vector<vector<Vec3b>>& image) {
    int width = static_cast<int>(image[0].size());
    int height = static_cast<int>(image.size());

    Mat outputImage(height, width, CV_8UC3);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            outputImage.at<Vec3b>(i, j) = image[i][j];
        }
    }

    return imwrite(filename, outputImage);
}

// funkcija koja obavlja cjelokupno Gaussovo filtriranje
void applyGaussianBlur(vector<vector<Vec3b>>& image, int size, double sigma) {
    int height = static_cast<int>(image.size());
    int width = static_cast<int>(image[0].size());
    vector<vector<Vec3b>> blurredImage(height, vector<Vec3b>(width));

    // stvara Gaussian kernel koristeci zadane parametre i Gaussovu funkciju
    vector<vector<double>> kernel(size, vector<double>(size));
    double sum = 0.0;
    int halfSize = size / 2;
    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            double exponent = -(i * i + j * j) / (2.0 * sigma * sigma);
            double value = (1.0 / (2.0 * 3.14159 * sigma * sigma)) * exp(exponent);
            kernel[i + halfSize][j + halfSize] = value;
            sum += value;
        }
    }

    // normalizacija kernela
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel[i][j] /= sum;
        }
    }

    // primjena Gaussian blura na slici
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double sumR = 0.0;
            double sumG = 0.0;
            double sumB = 0.0;

            // konvolucija matrice kernela (filtra) i matrice slike
            for (int k = -halfSize; k <= halfSize; k++) {
                for (int l = -halfSize; l <= halfSize; l++) {
                    int rowIndex = min(max(i + k, 0), height - 1);
                    int colIndex = min(max(j + l, 0), width - 1);

                    double weight = kernel[k + halfSize][l + halfSize];
                    sumR += weight * image[rowIndex][colIndex][0];
                    sumG += weight * image[rowIndex][colIndex][1];
                    sumB += weight * image[rowIndex][colIndex][2];
                }
            }

            blurredImage[i][j][0] = static_cast<uint8_t>(sumR);
            blurredImage[i][j][1] = static_cast<uint8_t>(sumG);
            blurredImage[i][j][2] = static_cast<uint8_t>(sumB);
        }
    }

    image = blurredImage;
}
int main() {
    auto start = high_resolution_clock::now();

    // ucitaj ulaznu sliku
    Mat cvImage = imread("C:\\Users\\zeljk\\Downloads\\hrvatska.jpg");
    if (cvImage.empty()) {
        cout << "Failed to load the image." << endl;
        return -1;
    }

    // funkcija vezana uz koristenje OpenCV librarya - pretvorba u Vec3b vektor iz OpenCV slike
    vector<vector<Vec3b>> image(cvImage.rows, vector<Vec3b>(cvImage.cols));
    for (int i = 0; i < cvImage.rows; i++) {
        for (int j = 0; j < cvImage.cols; j++) {
            image[i][j] = cvImage.at<Vec3b>(i, j);
        }
    }

    // parametri kernela i primjena filtra na sliku
    int size = 3;
    double sigma = 5.0;
    applyGaussianBlur(image, size, sigma);

    // funkcija vezana uz koristenje OpenCV librarya - pretvorba nazad u OpenCV sliku iz Vec3b vektora
    Mat outputImage(cvImage.rows, cvImage.cols, CV_8UC3);
    for (int i = 0; i < cvImage.rows; i++) {
        for (int j = 0; j < cvImage.cols; j++) {
            outputImage.at<Vec3b>(i, j) = image[i][j];
        }
    }

    // sprema izlaznu sliku
    if (!imwrite("C:\\Users\\zeljk\\Downloads\\output_hrvatska.jpg", outputImage)) {
        cout << "Failed to save the image." << endl;
        return -1;
    }

    auto end = high_resolution_clock::now();

    // Izra?unavanje ukupnog vremena izvršavanja u milisekundama
    duration<double, milli> duration = end - start;
    double executionTime = duration.count();

    cout << "Vrijeme izvršavanja: " << executionTime << " ms" << endl;

    return 0;
