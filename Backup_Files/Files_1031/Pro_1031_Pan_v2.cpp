#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

#define ROWS 100
#define COLS 4

// 代表一代的统计数据结构
struct GenerationData {
    double eliteFValue;
    double genMinFValue;
    double genAveFValue;
    double genDevFValue;
};

// 定义生成数据的结构体向量
std::vector<GenerationData> generateData(double oriData[][COLS]) {
    std::vector<GenerationData> resVec = {};
    for (int i = 0; i < ROWS; i++) {
        GenerationData rowData = { oriData[i][0], oriData[i][1], oriData[i][2], oriData[i][3] };
        resVec.push_back(rowData);
    }
    return resVec;
}

// 画图函数
void plotGraph(const std::vector<GenerationData>& data) {
    int width = 800;
    int height = 600;
    int margin = 50;

    // 创建画布
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    // 计算坐标转换比例
    double maxY = 0.15;  // 设置Y轴最大值
    int numGenerations = ROWS;
    double xScale = (width - 2 * margin) / (double)(numGenerations - 1);
    double yScale = (height - 2 * margin) / maxY;

    // 画坐标轴
    cv::line(image, cv::Point(margin, height - margin), cv::Point(width - margin, height - margin), cv::Scalar(124, 99, 95), 2);
    cv::line(image, cv::Point(margin, margin), cv::Point(margin, height - margin), cv::Scalar(124, 99, 95), 2); // 95, 99, 124

    // 绘制数据折线
    auto plotLine = [&](const std::vector<double>& values, const cv::Scalar& color) {
        for (size_t i = 0; i < values.size() - 1; ++i) {
            cv::Point p1(margin + i * xScale, height - margin - values[i] * yScale);
            cv::Point p2(margin + (i + 1) * xScale, height - margin - values[i + 1] * yScale);
            cv::line(image, p1, p2, color, 2);
        }
        };

    // 提取数据并画线
    std::vector<double> eliteFValues, genMinFValues, genAveFValues, genDevFValues;
    for (const auto& d : data) {
        eliteFValues.push_back(d.eliteFValue);
        genMinFValues.push_back(d.genMinFValue);
        genAveFValues.push_back(d.genAveFValue);
        genDevFValues.push_back(d.genDevFValue);
    }

    plotLine(eliteFValues, cv::Scalar(255, 0, 0));    // blue -> eliteFValue
    plotLine(genMinFValues, cv::Scalar(0, 0, 255));   // red -> genMinFValue
    plotLine(genAveFValues, cv::Scalar(77, 77, 77));  // gray -> genAveFValue
    plotLine(genDevFValues, cv::Scalar(0, 255, 255)); // yellow -> genDevFValue

    // 显示结果
    cv::imshow("Generation Plot", image);
    cv::waitKey(0);
    cv::imwrite("./imgs_Pro_GA/linearDiagram.png", image);
}

int main() {
    FILE* file;
    double oriData[ROWS][COLS];
    int i = 0, j = 0;
    if (fopen_s(&file, "./imgs_Pro_GA/output/params.txt", "r") != 0) {
        printf("Cannot open the file\n");
        return 1;
    }
    while (fscanf_s(file, "%lf %lf %lf %lf", &oriData[i][0], &oriData[i][1], &oriData[i][2], &oriData[i][3]) == 4) {
        i += 1;
    }
    fclose(file);

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%.4f ", oriData[i][j]);
        }
        printf("\n");
    }

    // 获取数据
    std::vector<GenerationData> data = generateData(oriData);

    // 绘制折线图
    plotGraph(data);

    return 0;
}
