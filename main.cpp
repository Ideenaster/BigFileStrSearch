#define NOMINMAX  // 防止 Windows.h 定义 min/max 宏
//#define DEBUG
#include <windows.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <immintrin.h>  // SIMD 指令集头文件

const int MAX_CHAR = 256;  // 使用单字节范围

// Boyer-Moore算法的坏字符规则
void buildBadCharTable(const std::string& pattern, std::vector<int>& badChar) {
    int m = pattern.length();
    
    // 初始化所有字符的位置为-1
    for (int i = 0; i < MAX_CHAR; i++) {
        badChar[i] = -1;
    }
    
    // 记录模式串中每个字符最右出现的位置
    for (int i = 0; i < m; i++) {
        // 使用 unsigned char 确保值在 0-255 范围内
        unsigned char c = static_cast<unsigned char>(pattern[i]);
        badChar[c] = i;
    }
}

// 添加好后缀规则的预处理函数
void buildGoodSuffixTable(const std::string& pattern, std::vector<int>& goodSuffix, std::vector<int>& borderPos) {
    int m = pattern.length();
    borderPos.resize(m + 1);
    goodSuffix.resize(m + 1);
    
    int i = m, j = m + 1;
    borderPos[i] = j;
    
    while (i > 0) {
        while (j <= m && pattern[i - 1] != pattern[j - 1]) {
            if (goodSuffix[j] == 0) {
                goodSuffix[j] = j - i;
            }
            j = borderPos[j];
        }
        i--; j--;
        borderPos[i] = j;
    }
    
    j = borderPos[0];
    for (i = 0; i <= m; i++) {
        if (goodSuffix[i] == 0) {
            goodSuffix[i] = j;
        }
        if (i == j) {
            j = borderPos[j];
        }
    }
}

// 添加内存映射文件的包装类
class MemoryMappedFile {
private:
    HANDLE hFile;
    HANDLE hMapping;
    void* pData;
    size_t fileSize;

public:
    MemoryMappedFile() : hFile(INVALID_HANDLE_VALUE), hMapping(NULL), pData(nullptr), fileSize(0) {}
    
    bool open(const std::string& filename) {
        // 打开文件
        hFile = CreateFileA(
            filename.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL
        );

        if (hFile == INVALID_HANDLE_VALUE) {
            std::cerr << "无法打开文件: " << GetLastError() << std::endl;
            return false;
        }

        // 获取文件大小
        LARGE_INTEGER size;
        if (!GetFileSizeEx(hFile, &size)) {
            std::cerr << "无法获取文件大小: " << GetLastError() << std::endl;
            CloseHandle(hFile);
            return false;
        }
        fileSize = size.QuadPart;

        // 创建文件映射
        hMapping = CreateFileMapping(
            hFile,
            NULL,
            PAGE_READONLY,
            0,
            0,
            NULL
        );

        if (hMapping == NULL) {
            std::cerr << "无法创建文件映射: " << GetLastError() << std::endl;
            CloseHandle(hFile);
            return false;
        }

        // 映射视图
        pData = MapViewOfFile(
            hMapping,
            FILE_MAP_READ,
            0,
            0,
            0
        );

        if (pData == nullptr) {
            std::cerr << "无法映射视图: " << GetLastError() << std::endl;
            CloseHandle(hMapping);
            CloseHandle(hFile);
            return false;
        }

        return true;
    }

    const char* getData() const { return static_cast<const char*>(pData); }
    size_t getSize() const { return fileSize; }

    ~MemoryMappedFile() {
        if (pData) {
            UnmapViewOfFile(pData);
        }
        if (hMapping) {
            CloseHandle(hMapping);
        }
        if (hFile != INVALID_HANDLE_VALUE) {
            CloseHandle(hFile);
        }
    }
};

// 修?? Boyer-Moore 搜索函数
std::vector<int> boyerMooreSearch(const char* text, size_t textLen, const std::string& pattern) {
    std::vector<int> positions;
    int n = textLen;
    int m = pattern.length();
    
    std::vector<int> badChar(MAX_CHAR);
    std::vector<int> goodSuffix(m + 1);
    std::vector<int> borderPos(m + 1);
    
    buildBadCharTable(pattern, badChar);
    buildGoodSuffixTable(pattern, goodSuffix, borderPos);
    
    int s = 0;
    while (s <= (n - m)) {
        int j = m - 1;
        
        while (j >= 0 && pattern[j] == text[s + j]) {
            j--;
        }
        
        if (j < 0) {
            positions.push_back(s);
            s += goodSuffix[0];
        } else {
            int badCharShift = j - badChar[(unsigned char)text[s + j]];
            int goodSuffixShift = goodSuffix[j + 1];
            s += std::max(badCharShift, goodSuffixShift);
        }
    }
    
    return positions;
}


// Horspool 算法的预处理函数
void buildHorspoolTable(const std::string& pattern, std::vector<int>& shift) {
    int m = pattern.length();
    
    // 初始化所有字符的移动距离为模式串长度
    for (int i = 0; i < MAX_CHAR; i++) {
        shift[i] = m;
    }
    
    // 记录模式串中除最后一个字符外的移动距离
    for (int i = 0; i < m - 1; i++) {
        unsigned char c = static_cast<unsigned char>(pattern[i]);
        shift[c] = m - 1 - i;
    }
}

std::vector<int> horspoolSearch(const char* text, size_t textLen, const std::string& pattern) {
    std::vector<int> positions;
    int n = textLen;
    int m = pattern.length();
    
    std::vector<int> shift(MAX_CHAR);
    buildHorspoolTable(pattern, shift);
    
    int s = 0;
    while (s <= n - m) {
        int j = m - 1;
        
        while (j >= 0 && pattern[j] == text[s + j]) {
            j--;
        }
        
        if (j < 0) {
            positions.push_back(s);
            s += m;
        } else {
            s += shift[(unsigned char)text[s + m - 1]];
        }
    }
    
    return positions;
}

// Sunday 算法的预处理函数
void buildShiftTable(const std::string& pattern, std::vector<int>& shift) {
    int m = pattern.length();
    
    // 初始化所有字符的移动距离为模式串长度 + 1
    for (int i = 0; i < MAX_CHAR; i++) {
        shift[i] = m + 1;
    }
    
    // 记录模式串中每个字符到末尾的距离
    for (int i = 0; i < m; i++) {
        unsigned char c = static_cast<unsigned char>(pattern[i]);
        shift[c] = m - i;
    }
}

std::vector<int> sundaySearch(const char* text, size_t textLen, const std::string& pattern) {
    std::vector<int> positions;
    int n = textLen;
    int m = pattern.length();
    
    std::vector<int> shift(MAX_CHAR);
    buildShiftTable(pattern, shift);
    
    int s = 0;
    while (s <= n - m) {
        int j = 0;
        
        // 尝试匹配
        while (j < m && pattern[j] == text[s + j]) {
            j++;
        }
        
        if (j == m) {
            positions.push_back(s);
        }
        
        // 计算移动距离
        if (s + m < n) {
            s += shift[(unsigned char)text[s + m]];
        } else {
            break;
        }
    }
    
    return positions;
}

// BMHBNFS Pattern 类
class BMHBNFSPattern {
private:
    std::string pattern;
    std::vector<bool> alphabet;
    std::vector<size_t> bm_bc;
    size_t k;

    // 计算k值（最长的重复前缀）
    static size_t computeK(const std::string& p) {
        size_t k = 0;
        size_t len = p.length();
        for (size_t i = 1; i < len; ++i) {
            bool is_period = true;
            for (size_t j = 0; j < len - i; ++j) {
                if (p[j] != p[j + i]) {
                    is_period = false;
                    break;
                }
            }
            if (is_period) {
                k = i;
                break;
            }
        }
        return k == 0 ? len : k;
    }

public:
    BMHBNFSPattern(const std::string& pat) : pattern(pat), alphabet(256, false), bm_bc(256, pat.length()) {
        if (pat.empty()) {
            throw std::invalid_argument("Pattern cannot be empty");
        }

        size_t lastpos = pat.length() - 1;

        // 构???字母表和坏字符表
        for (size_t i = 0; i < lastpos; ++i) {
            alphabet[static_cast<unsigned char>(pat[i])] = true;
            bm_bc[static_cast<unsigned char>(pat[i])] = lastpos - i;
        }
        alphabet[static_cast<unsigned char>(pat[lastpos])] = true;

        k = computeK(pat);
    }

    std::vector<int> findAll(const char* text, size_t textLen) {
        std::vector<int> result;
        const size_t pat_last_pos = pattern.length() - 1;
        const size_t patlen = pattern.length();
        size_t string_index = pat_last_pos;
        size_t offset = pat_last_pos;
        const size_t offset0 = k - 1;

        while (string_index < textLen) {
            if (text[string_index] == pattern[pat_last_pos]) {
                bool match = true;
                for (size_t i = 0; i < pat_last_pos; ++i) {
                    if (text[string_index - pat_last_pos + i] != pattern[i]) {
                        match = false;
                        break;
                    }
                }
                
                if (match) {
                    result.push_back(string_index - pat_last_pos);
                    offset = offset0;
                    string_index += k;  // Galil规则
                    continue;
                }
            }

            if (string_index + 1 >= textLen) {
                break;
            }

            offset = pat_last_pos;

            // 结合Sunday和Horspool的跳转规则
            if (!alphabet[static_cast<unsigned char>(text[string_index + 1])]) {
                string_index += patlen + 1;  // Sunday跳转
            } else {
                string_index += bm_bc[static_cast<unsigned char>(text[string_index])];  // Horspool跳转
            }
        }

        return result;
    }
};

// // 添加朴素算法实现
// std::vector<int> bruteForceSearch(const char* text, size_t textLen, const std::string& pattern) {
//     std::vector<int> positions;
//     int n = textLen;
//     int m = pattern.length();
    
//     for (int i = 0; i <= n - m; i++) {
//         int j;
//         for (j = 0; j < m; j++) {
//             if (text[i + j] != pattern[j]) {
//                 break;
//             }
//         }
//         if (j == m) {
//             positions.push_back(i);
//         }
//     }
    
//     return positions;
// }


#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

class ThreadPool {
public:
    ThreadPool(size_t);
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();

    // 添加获取工作线程数量的方法
    size_t getThreadCount() const { 
        return workers.size(); 
    }

private:
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
    // the task queue
    std::queue< std::function<void()> > tasks;
    
    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    #ifdef DEBUG
    std::atomic<int> activeThreads{0};  // 添加活动线程计数器
    #endif
};
 
// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
    :   stop(false)
{
    #ifdef DEBUG
    std::cout << "创建线程池，线程数: " << threads << std::endl;
    #endif

    for(size_t i = 0; i < threads; ++i)
        workers.emplace_back(
            [this]
            {
                for(;;)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                        
                        #ifdef DEBUG
                        activeThreads++;
                        std::cout << "线程 " << std::this_thread::get_id() << " 开始执行任务, 当前活动线程数: " 
                                 << activeThreads << std::endl;
                        #endif
                    }

                    task();
                    
                    #ifdef DEBUG
                    activeThreads--;
                    std::cout << "线程 " << std::this_thread::get_id() << " 完成任务, 当前活动线程数: " 
                             << activeThreads << std::endl;
                    #endif
                }
            }
        );
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

// 添加分块搜索结构体
struct SearchBlock {
    const char* start;
    size_t length;
    size_t offset;
};

class SIMDSearch {
private:
    std::string pattern;
    __m128i first_char_mask;
    int pattern_length;

    // 添加跨平台的 CTZ (Count Trailing Zeros) 实现
    static inline int countTrailingZeros(unsigned int x) {
        #ifdef _MSC_VER  // Microsoft Visual C++
            unsigned long index;
            _BitScanForward(&index, x);
            return static_cast<int>(index);
        #else  // GCC, Clang 等
            return __builtin_ctz(x);
        #endif
    }

public:
    SIMDSearch(const std::string& pat) : pattern(pat) {
        pattern_length = pat.length();
        if (pattern_length > 16) {
            throw std::invalid_argument("Pattern too long for SSE implementation");
        }
        first_char_mask = _mm_set1_epi8(pattern[0]);
    }

    std::vector<int> search(const char* text, size_t text_length) {
        std::vector<int> positions;
        
        if (text == nullptr || text_length < pattern_length) {
            return positions;
        }

        size_t safe_length = text_length - pattern_length + 1;
        size_t i = 0;

        while (i + 16 <= safe_length) {
            char temp_buffer[16];
            memcpy(temp_buffer, text + i, 16);
            __m128i text_chunk = _mm_loadu_si128((__m128i*)temp_buffer);
            
            __m128i cmp = _mm_cmpeq_epi8(text_chunk, first_char_mask);
            unsigned int mask = _mm_movemask_epi8(cmp);
            
            while (mask != 0) {
                // 使用跨平台的 CTZ 实现
                int pos = countTrailingZeros(mask);
                
                if (i + pos + pattern_length <= text_length) {
                    bool match = true;
                    for (int j = 1; j < pattern_length && match; j++) {
                        if (text[i + pos + j] != pattern[j]) {
                            match = false;
                        }
                    }
                    
                    if (match) {
                        positions.push_back(i + pos);
                    }
                }
                
                mask &= (mask - 1);
            }
            
            i += 16;
        }

        // 处理剩余部分
        while (i < safe_length) {
            bool match = true;
            for (int j = 0; j < pattern_length; j++) {
                if (text[i + j] != pattern[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                positions.push_back(i);
            }
            i++;
        }

        return positions;
    }
};

// 修改 smartSearchBlock 函数
std::vector<int> smartSearchBlock(const SearchBlock& block, const std::string& pattern) {
    std::vector<int> positions;
    
    // 添加安全检查
    if (block.start == nullptr || block.length == 0) {
        return positions;
    }
    bool is_alt = true;
    // try {
        if (pattern.length() <= 16 && is_alt) {  // 对于短模式串使用SIMD搜索
            SIMDSearch simd(pattern);
            positions = simd.search(block.start, block.length);
        } else {
            BMHBNFSPattern bmhbnfs(pattern);
            positions = bmhbnfs.findAll(block.start, block.length);
        }
        
        // 调整位置偏移
        for(auto& pos : positions) {
            pos += block.offset;
        }
    // } catch (const std::exception& e) {
    //     #ifdef DEBUG
    //     std::cerr << "搜索块处理出错: " << e.what() << std::endl;
    //     #endif
    //     // 发生错误时回退到 BMHBNFS
    //     BMHBNFSPattern bmhbnfs(pattern);
    //     positions = bmhbnfs.findAll(block.start, block.length);
    //     for(auto& pos : positions) {
    //         pos += block.offset;
    //     }
    // }
    
    return positions;
}

// 修改并行搜索函数
std::vector<int> parallelSearch(const char* text, size_t textLen, const std::string& pattern) {
    const size_t min_block_size = 1024 * 1024; // 1MB
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t pattern_len = pattern.length();
    
    #ifdef DEBUG
    std::cout << "\n开始并行搜索模式: " << pattern << std::endl;
    std::cout << "硬件支持的线程数: " << num_threads << std::endl;
    #endif
    
    static ThreadPool pool(num_threads);
    
    // 计算块大小和分块
    size_t block_size = std::max(min_block_size, textLen / num_threads);
    std::vector<SearchBlock> blocks;
    
    for(size_t offset = 0; offset < textLen; offset += block_size) {
        SearchBlock block;
        block.start = text + offset;
        block.offset = offset;
        block.length = std::min(block_size + pattern_len - 1, textLen - offset);
        blocks.push_back(block);
    }
    
    #ifdef DEBUG
    std::cout << "文本总长度: " << textLen << " 字节" << std::endl;
    std::cout << "分块数量: " << blocks.size() << std::endl;
    std::cout << "每块大小: " << block_size << " 字节" << std::endl;
    #endif

    std::vector<std::future<std::vector<int>>> futures;
    
    // 直接使用线程池的 enqueue 方法提交任务
    for(const auto& block : blocks) {
        futures.push_back(
            pool.enqueue(&smartSearchBlock, block, pattern)
        );
    }
    
    // 收集结果
    std::vector<int> all_positions;
    for(auto& future : futures) {
        auto positions = future.get();
        all_positions.insert(all_positions.end(), positions.begin(), positions.end());
    }
    
    // 排序并去重
    std::sort(all_positions.begin(), all_positions.end());
    all_positions.erase(
        std::unique(all_positions.begin(), all_positions.end()),
        all_positions.end()
    );
    
    return all_positions;
}


int main() {
    // 读取关键词文件
    std::ifstream keywordFile("../file/keyword.txt");
    std::vector<std::string> keywords;
    std::string keyword;
    while (std::getline(keywordFile, keyword)) {
        if (keyword.length() <= 300) {
            keywords.push_back(keyword);
        }
    }
    keywordFile.close();
    
    // 打开输出文件
    std::ofstream outputFile("output.txt");
    
    // 使用内存映射方式读取XML文件
    MemoryMappedFile mmFile;
    if (!mmFile.open("../file/enwiki-20231120-abstract1.xml")) {
        std::cerr << "无法打开XML文件" << std::endl;
        return 1;
    }
    
    // 对每个关键词进行搜索并计时
    for (const auto& kw : keywords) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<int> matches = parallelSearch(mmFile.getData(), mmFile.getSize(), kw);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        outputFile << "Keyword: " << kw << "\n";
        outputFile << "Occurrences: " << matches.size() << "\n";
        outputFile << "Search time: " << duration.count() << " ms\n\n";
    }
    
    outputFile.close();
    return 0;
}