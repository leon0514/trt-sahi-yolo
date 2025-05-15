#ifndef FORMAT_HPP__
#define FORMAT_HPP__
#include <cstdio>   // 用于 std::snprintf
#include <iomanip>  // 用于格式化输出 (std::fixed, std::setprecision)
#include <iostream> // 用于打印输出
#include <memory>
#include <stdexcept> // 用于异常处理 (std::runtime_error)
#include <string>
#include <tuple> // 用于 std::tuple
#include <vector>

namespace fmt
{

template <typename... Args>
static std::string str_format(const std::string &format, Args &&...args) // 接收参数时保留 Args&&
{
    // 第一次调用获取所需长度
    // 直接传递 args...，不使用 std::forward
    // 编译器会按 C 调用约定处理参数
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...); // 注意：这里去掉了 std::forward
    if (size_s < 0)
    {
        // 根据 cppreference，snprintf 在编码错误时返回负值
        throw std::runtime_error("str_format: 计算大小时 snprintf 返回错误");
    }

    // 分配缓冲区（+1 for null terminator）
    // 使用 size_t 更安全
    size_t buffer_size = static_cast<size_t>(size_s) + 1;
    std::vector<char> buf(buffer_size);

    // 实际格式化
    // 再次直接传递 args...，不使用 std::forward
    int result = std::snprintf(buf.data(), buf.size(), format.c_str(), args...); // 注意：这里也去掉了 std::forward

    // 检查第二次调用的结果
    if (result < 0)
    {
        // 实际格式化时也可能出错
        throw std::runtime_error("str_format: 格式化时 snprintf 返回错误");
    }
    // 理论上 result 应该等于 size_s，如果大于等于 buf.size() 说明缓冲区不够
    // （虽然我们计算过，但多一层检查总是好的，以防万一参数行为在两次调用间有诡异变化）
    if (static_cast<size_t>(result) >= buf.size())
    {
        // 这不应该发生，如果发生了说明第一次长度计算可能有问题或参数有意外行为
        throw std::runtime_error("str_format: snprintf 格式化时发生截断");
    }

    // 从缓冲区创建 string，长度为实际写入的字符数 (result)
    return std::string(buf.data(), static_cast<size_t>(result));
}

} // namespace fmt

#endif