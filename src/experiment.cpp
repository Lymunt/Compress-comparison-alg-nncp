#include "arithmetic.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <random>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

extern char** environ;

namespace {

namespace fs = std::filesystem;
using arithmetic::Bytes;
using Clock = std::chrono::steady_clock;

constexpr std::size_t dataset_size = 32 * 1024;
constexpr std::size_t period = 128;
constexpr std::uint32_t shuffle_seed = 42;

struct Config {
    fs::path input;
    fs::path output;
    fs::path nncp;
    std::string profile = "default";
    unsigned runs = 5;
    unsigned threads = 1;
};

unsigned positive_integer(const std::string& value) {
    unsigned number = 0;
    const auto end = value.data() + value.size();
    const auto [next, error] = std::from_chars(value.data(), end, number);
    if (error != std::errc{} || next != end || number == 0) {
        throw std::runtime_error("Expected a positive integer, got: " + value);
    }
    return number;
}

void usage(std::ostream& out) {
    out << "Usage: hw1_ex9 INPUT OUTPUT [--runs N] [--nncp PATH] "
           "[--profile NAME] [--threads N]\n"
           "INPUT is the Tiny Shakespeare text file or a directory containing\n"
           "original.txt, shuffled.txt and repeated.txt.\n";
}

Config parse_arguments(int argc, char* argv[]) {
    Config config{};
    config.input = fs::absolute(argv[1]);
    config.output = fs::absolute(argv[2]);
    for (int i = 3; i < argc; i += 2) {
        const std::string option = argv[i];
        if (i + 1 == argc) {
            throw std::runtime_error("Missing value for " + option);
        }
        const std::string value = argv[i + 1];
        if (option == "--runs") {
            config.runs = positive_integer(value);
        } else if (option == "--threads") {
            config.threads = positive_integer(value);
        } else if (option == "--profile") {
            config.profile = value;
        } else if (option == "--nncp") {
            config.nncp = fs::absolute(value);
            if (!fs::is_regular_file(config.nncp) || access(config.nncp.c_str(), X_OK) != 0) {
                throw std::runtime_error("NNCP is not executable: " + config.nncp.string());
            }
        } else {
            throw std::runtime_error("Unknown option: " + option);
        }
    }
    return config;
}

Bytes read_file(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    const auto size = stream.tellg();
    if (size < 0) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    Bytes bytes(static_cast<std::size_t>(size));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    return bytes;
}

void write_file(const fs::path& path, const Bytes& bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    stream.close();
    if (!stream) {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

std::array<std::size_t, 256> histogram(const Bytes& bytes) {
    std::array<std::size_t, 256> counts{};
    for (auto byte : bytes) {
        ++counts[byte];
    }
    return counts;
}

Bytes repeat_prefix(const Bytes& original) {
    Bytes repeated;
    repeated.reserve(dataset_size);
    for (std::size_t i = 0; i < dataset_size / period; ++i) {
        repeated.insert(repeated.end(), original.begin(), original.begin() + period);
    }
    return repeated;
}

void shuffle_bytes(Bytes& bytes) {
    std::mt19937 engine(shuffle_seed);
    constexpr std::uint64_t range = std::uint64_t{1} << 32;
    // Fisher–Yates with rejection sampling, reproducible across standard libraries.
    for (std::size_t remaining = bytes.size(); remaining > 1; --remaining) {
        const auto limit = range - range % remaining;
        auto draw = engine();
        while (draw >= limit) {
            draw = engine();
        }
        std::swap(bytes[remaining - 1], bytes[draw % remaining]);
    }
}

struct Dataset {
    const char* name;
    const char* stem;
    Bytes bytes;
};

std::array<Dataset, 3> prepare(const fs::path& input, bool supplied) {
    auto original = read_file(supplied ? input / "original.txt" : input);
    if (original.size() < dataset_size || (supplied && original.size() != dataset_size)) {
        throw std::runtime_error("A prepared original must have 32768 bytes. "
                                 "The Tiny Shakespeare source must contain at least that many.");
    }
    original.resize(dataset_size);
    auto shuffled = supplied ? read_file(input / "shuffled.txt") : original;
    auto repeated = repeat_prefix(original);
    if (supplied) {
        if (histogram(original) != histogram(shuffled) ||
                read_file(input / "repeated.txt") != repeated) {
            throw std::runtime_error("The supplied datasets do not match the task rules");
        }
    } else {
        shuffle_bytes(shuffled);
    }
    return {{{"Original", "original", std::move(original)},
             {"Shuffled", "shuffled", std::move(shuffled)},
             {"Repeated", "repeated", std::move(repeated)}}};
}

void run_nncp(const Config& config, bool compress, const fs::path& input, const fs::path& output) {
    std::vector<std::string> args{config.nncp.string(), "-T", std::to_string(config.threads), "-q"};
    if (compress) {
        args.insert(args.end(), {"--profile", config.profile});
    }
    args.insert(args.end(), {compress ? "c" : "d", input.string(), output.string()});
    std::vector<char*> argv;
    for (auto& arg : args) {
        argv.push_back(arg.data());
    }
    argv.push_back(nullptr);

    pid_t child;
    const int error = posix_spawn(&child, argv.front(), nullptr, nullptr, argv.data(), environ);
    if (error != 0) {
        throw std::system_error(error, std::generic_category(), "Start NNCP");
    }
    int status;
    while (waitpid(child, &status, 0) == -1) {
        if (errno != EINTR) {
            throw std::system_error(errno, std::generic_category(), "Wait for NNCP");
        }
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        const auto reason = WIFEXITED(status) ? "exit code " + std::to_string(WEXITSTATUS(status))
                                              : "signal " + std::to_string(WTERMSIG(status));
        throw std::runtime_error("NNCP failed on " + input.string() + " (" + reason + ")");
    }
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const auto middle = samples.size() / 2;
    return samples.size() % 2 ? samples[middle] : (samples[middle - 1] + samples[middle]) / 2.0;
}

struct Measurement {
    const char* dataset;
    const char* method;
    std::uintmax_t compressed;
    double entropy;
    unsigned distinct_bytes;
    std::vector<double> times;
};

Measurement measure(const Config& config, const Dataset& dataset, bool nncp,
                    const fs::path& input, const fs::path& archive) {
    Measurement row{dataset.name, nncp ? "NNCP" : "Arithmetic", 0, 0, 0, {}};
    auto counts = histogram(dataset.bytes);
    std::sort(counts.begin(), counts.end());
    for (auto count : counts) {
        if (count != 0) {
            const auto probability = static_cast<double>(count) / static_cast<double>(dataset.bytes.size());
            row.entropy -= probability * std::log2(probability);
            ++row.distinct_bytes;
        }
    }

    for (unsigned i = 0; i < config.runs; ++i) {
        const auto start = Clock::now();
        if (nncp) {
            run_nncp(config, true, input, archive);
        } else {
            write_file(archive, arithmetic::encode(read_file(input)));
        }
        row.times.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    }

    Bytes decoded;
    if (nncp) {
        const auto directory = config.output / "decoded";
        fs::create_directories(directory);
        const auto path = directory / input.filename();
        run_nncp(config, false, archive, path);
        decoded = read_file(path);
    } else {
        decoded = arithmetic::decode(read_file(archive));
    }
    if (decoded != dataset.bytes) {
        throw std::runtime_error("Round trip mismatch for " + std::string(dataset.name) +
                                 " / " + row.method);
    }
    row.compressed = fs::file_size(archive);
    if (row.compressed == 0) {
        throw std::runtime_error("Empty archive: " + archive.string());
    }
    return row;
}

void json_string(std::ostream& out, const std::string& value) {
    out << '"';
    for (char character : value) {
        const auto ch = static_cast<unsigned char>(character);
        if (ch == '\\' || ch == '"') {
            out << '\\' << static_cast<char>(ch);
        } else if (ch < 32) {
            const char* hex = "0123456789abcdef";
            out << "\\u00" << hex[ch >> 4] << hex[ch & 15];
        } else {
            out << static_cast<char>(ch);
        }
    }
    out << '"';
}

void save_results(const Config& config, bool supplied, const std::vector<Measurement>& rows) {
    const auto path = config.output / "results.json";
    std::ofstream out(path);
    out.imbue(std::locale::classic());
    out << std::setprecision(15);
    utsname system{};
    out << "{\n  \"platform\": ";
    json_string(out, uname(&system) == 0 ? std::string(system.sysname) + " " + system.release +
                                               " " + system.machine : "unknown");
    out << ",\n  \"datasets_supplied\": " << (supplied ? "true" : "false")
        << ",\n  \"shuffle_seed\": " << (supplied ? "null" : std::to_string(shuffle_seed))
        << ",\n  \"compression_runs\": " << config.runs << ",\n  \"nncp\": ";
    if (config.nncp.empty()) {
        out << "null";
    } else {
        out << "{\"profile\": ";
        json_string(out, config.profile);
        out << ", \"threads\": " << config.threads << ", \"device\": \"cpu\"}";
    }
    out << ",\n  \"results\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        const auto size = static_cast<double>(row.compressed);
        out << (i ? ",\n" : "") << "    {\n      \"dataset\": \"" << row.dataset
            << "\",\n      \"method\": \"" << row.method
            << "\",\n      \"input_bytes\": " << dataset_size
            << ",\n      \"compressed_bytes\": " << row.compressed
            << ",\n      \"compression_ratio\": " << dataset_size / size
            << ",\n      \"compressed_bits_per_byte\": " << 8.0 * size / dataset_size
            << ",\n      \"empirical_entropy_bits_per_byte\": " << row.entropy
            << ",\n      \"distinct_bytes\": " << row.distinct_bytes
            << ",\n      \"compression_seconds_median\": " << median(row.times)
            << ",\n      \"compression_seconds_each_run\": [";
        for (std::size_t j = 0; j < row.times.size(); ++j) {
            out << (j ? ", " : "") << row.times[j];
        }
        out << "],\n      \"round_trip_equal\": true\n    }";
    }
    out << "\n  ]\n}\n";
    out.close();
    if (!out) {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        usage(std::cout);
        return 0;
    }
    if (argc < 3) {
        usage(std::cerr);
        return 1;
    }
    try {
        const auto config = parse_arguments(argc, argv);
        const auto supplied = fs::is_directory(config.input);
        const auto datasets = prepare(config.input, supplied);
        const auto inputs = config.output / "datasets";
        const auto archives = config.output / "compressed";
        fs::create_directories(inputs);
        fs::create_directories(archives);
        std::vector<Measurement> rows;
        for (const auto& dataset : datasets) {
            const std::string stem = dataset.stem;
            const auto input = inputs / (stem + ".txt");
            write_file(input, dataset.bytes);
            rows.push_back(measure(config, dataset, false, input, archives / (stem + ".ac")));
            if (!config.nncp.empty()) {
                std::cout << "Benchmarking NNCP on " << dataset.name << std::endl;
                rows.push_back(measure(config, dataset, true, input, archives / (stem + ".nncp")));
            }
        }
        save_results(config, supplied, rows);
        for (const auto& row : rows) {
            std::cout << row.dataset << ' ' << row.method << ": " << row.compressed
                      << " bytes, " << std::fixed << std::setprecision(3)
                      << 1000 * median(row.times) << " ms, H0=" << std::setprecision(6)
                      << row.entropy << " bits/byte\n";
        }
        std::cout << "Results: " << config.output / "results.json" << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
