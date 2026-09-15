#include <cureco/inference.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
int main(int argc, char **argv) {
    try {
        if (argc != 5) throw std::runtime_error("Usage: infer model.onnx image.rgb width height");
        auto parse = [](const char *s) { size_t n; int v = std::stoi(s, &n);
            if (n != std::string(s).size() || v < 1 || v > 16384) throw std::runtime_error("Invalid dimension"); return v; };
        int w = parse(argv[3]), h = parse(argv[4]);
        if (static_cast<size_t>(w) * h > 16777216) throw std::runtime_error("Image too large");
        std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 3);
        std::ifstream input(argv[2], std::ios::binary);
        if (!input.read(reinterpret_cast<char *>(pixels.data()), pixels.size()) || input.peek() != EOF)
            throw std::runtime_error("Expected exactly width * height * 3 RGB bytes");
        cureco::Engine engine(argv[1]);
        auto result = engine.infer(pixels.data(), pixels.size(), w, h, w * 3);
        std::cout << ci_result_json(result.get()) << '\n';
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
