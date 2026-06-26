#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <functional>

using namespace cv;
using namespace std;

// 定义遗传编程个体（污点检测公式）
struct GPProgram {
    function<Mat(const Mat&)> process;
    double fitness;
    GPProgram(function<Mat(const Mat&)> func) : process(func), fitness(0.0) {}
};

// 计算适应度（污点检测效果）
double evaluate(const GPProgram& program, const Mat& input, const Mat& groundTruth) {
    Mat result = program.process(input);
    Mat diff;
    absdiff(result, groundTruth, diff);
    return 1.0 / (sum(diff)[0] + 1); // 误差越小适应度越高
}

// 生成随机 GP 公式（变异操作）
function<Mat(const Mat&)> randomGPFunction() {
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_int_distribution<int> opDist(0, 2);

    return [](const Mat& img) {
        Mat result;
        int op = opDist(gen);
        if (op == 0) {
            GaussianBlur(img, result, Size(5, 5), 1.5);
        }
        else if (op == 1) {
            Canny(img, result, 50, 150);
        }
        else {
            threshold(img, result, 128, 255, THRESH_BINARY);
        }
        return result;
        };
}

// 交叉操作（交换两个程序的一部分）// what is the relation between p1 and p2
GPProgram crossover(const GPProgram& p1, const GPProgram& p2) {
    return GPProgram(randomGPFunction()); // 简单随机组合
}

int main() {
    // 读取输入图像（灰度）
    Mat input = imread("input.jpg", IMREAD_GRAYSCALE);
    Mat groundTruth = imread("groundtruth.jpg", IMREAD_GRAYSCALE);

    if (input.empty() || groundTruth.empty()) {
        cerr << "Error loading images!" << endl;
        return -1;
    }

    // 初始化种群
    vector<GPProgram> population;
    for (int i = 0; i < 10; i++) {
        population.emplace_back(randomGPFunction());
    }

    // 进化循环
    for (int gen = 0; gen < 20; gen++) {
        // 评估适应度
        for (auto& individual : population) {
            individual.fitness = evaluate(individual, input, groundTruth);
        }

        // 选择前 50% 最佳个体
        sort(population.begin(), population.end(), [](const GPProgram& a, const GPProgram& b) {
            return a.fitness > b.fitness;
            });

        // 交叉和变异生成新个体
        for (size_t i = 5; i < population.size(); i++) {
            population[i] = crossover(population[i - 5], population[i - 4]);
        }
    }

    // 输出最优个体的检测结果
    Mat bestResult = population[0].process(input);
    imwrite("output.jpg", bestResult);
    cout << "Best fitness: " << population[0].fitness << endl;

    return 0;
}
