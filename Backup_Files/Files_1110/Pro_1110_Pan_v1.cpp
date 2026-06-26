#include <iostream>
#include <fstream>
#include <string>

void replaceCharacter(const std::string& inputFile, const std::string& outputFile) {
    std::ifstream inFile(inputFile);
    if (!inFile) {
        std::cerr << "无法打开输入文件: " << inputFile << std::endl;
        return;
    }

    std::ofstream outFile(outputFile);
    if (!outFile) {
        std::cerr << "无法打开输出文件: " << outputFile << std::endl;
        return;
    }

    std::string line;
    while (std::getline(inFile, line)) {
        for (char& ch : line) {
            if (ch == '、') {
                ch = '，';
            }
            if (ch == '。') {
                ch = '．';
            }
        }
        outFile << line << std::endl;
    }

    inFile.close();
    outFile.close();
    std::cout << "文件处理完成。" << std::endl;
}

int main() {

    std::string inputFile = "./imgs_1110_v1/inputText.txt";
    std::string outputFile = "./imgs_1110_v1/outputText.txt";

    replaceCharacter(inputFile, outputFile);

    return 0;
}
