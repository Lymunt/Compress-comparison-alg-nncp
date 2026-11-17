#include "arithmetic.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace arithmetic {
namespace {

// A 32-bit coding interval uses 64-bit intermediates for width * frequency.
constexpr std::uint64_t full = std::uint64_t{1} << 32;
constexpr std::uint64_t half = full / 2;
constexpr std::uint64_t quarter = full / 4;
constexpr std::uint64_t three_quarters = 3 * quarter;

void append_integer(Bytes& bytes, std::uint32_t value, unsigned length) {
    for (unsigned remaining = length; remaining > 0; --remaining) {
        bytes.push_back(static_cast<std::uint8_t>(value >> (8 * (remaining - 1))));
    }
}

std::uint32_t read_integer(const Bytes& bytes, std::size_t& offset, unsigned length) {
    if (length > bytes.size() - offset) {
        throw std::runtime_error("Truncated AC01 header");
    }
    std::uint32_t value = 0;
    for (unsigned i = 0; i < length; ++i) {
        value = (value << 8) | bytes[offset++];
    }
    return value;
}

class BitWriter {
public:
    void write(unsigned bit) {
        current_ = static_cast<std::uint8_t>((static_cast<unsigned>(current_) << 1) | bit);
        if (++used_ == 8) {
            bytes_.push_back(current_);
            current_ = 0;
            used_ = 0;
        }
    }

    void finish(Bytes& archive) {
        if (used_ > 0) {
            bytes_.push_back(static_cast<std::uint8_t>(current_ << (8 - used_)));
        }
        archive.insert(archive.end(), bytes_.begin(), bytes_.end());
    }

private:
    Bytes bytes_;
    std::uint8_t current_ = 0;
    unsigned used_ = 0;
};

class BitReader {
public:
    BitReader(const Bytes& bytes, std::size_t offset) : bytes_(bytes), byte_(offset) {}

    unsigned read() {
        // The final interval is completed with zero bits after the payload.
        if (byte_ == bytes_.size()) {
            return 0;
        }
        const unsigned bit = (bytes_[byte_] >> (7 - bit_)) & 1;
        if (++bit_ == 8) {
            bit_ = 0;
            ++byte_;
        }
        return bit;
    }

private:
    const Bytes& bytes_;
    std::size_t byte_;
    unsigned bit_ = 0;
};

std::array<std::uint32_t, 257> cumulative_counts(
        const std::array<std::uint32_t, 256>& counts) {
    std::array<std::uint32_t, 257> cumulative{};
    for (std::size_t i = 0; i < counts.size(); ++i) {
        cumulative[i + 1] = cumulative[i] + counts[i];
    }
    return cumulative;
}

}  // namespace

Bytes encode(const Bytes& data) {
    if (data.empty() || data.size() > 65535) {
        throw std::runtime_error("AC01 supports files of 1–65535 bytes");
    }
    const auto total = static_cast<std::uint32_t>(data.size());
    std::array<std::uint32_t, 256> counts{};
    for (auto symbol : data) {
        ++counts[symbol];
    }
    const auto cumulative = cumulative_counts(counts);
    const auto distinct = std::count_if(counts.begin(), counts.end(),
                                      [](auto count) { return count > 0; });
    Bytes archive{'A', 'C', '0', '1'};
    append_integer(archive, total, 4);
    append_integer(archive, static_cast<std::uint32_t>(distinct), 2);
    for (std::size_t symbol = 0; symbol < counts.size(); ++symbol) {
        if (counts[symbol] > 0) {
            archive.push_back(static_cast<std::uint8_t>(symbol));
            append_integer(archive, counts[symbol], 2);
        }
    }

    BitWriter writer;
    std::uint64_t low = 0;
    std::uint64_t high = full - 1;
    std::size_t pending = 0;
    auto emit = [&](unsigned bit) {
        writer.write(bit);
        for (std::size_t i = 0; i < pending; ++i) {
            writer.write(1 - bit);
        }
        pending = 0;
    };

    for (auto symbol : data) {
        const auto width = high - low + 1;
        high = low + width * cumulative[symbol + 1] / total - 1;
        low += width * cumulative[symbol] / total;

        while (true) {
            if (high < half) {
                emit(0);
            } else if (low >= half) {
                emit(1);
                low -= half;
                high -= half;
            } else if (low >= quarter && high < three_quarters) {
                // Defer the output while the interval straddles the midpoint.
                ++pending;
                low -= quarter;
                high -= quarter;
            } else {
                break;
            }
            low *= 2;
            high = 2 * high + 1;
        }
    }
    ++pending;
    emit(low >= quarter);
    writer.finish(archive);
    return archive;
}

Bytes decode(const Bytes& archive) {
    if (archive.size() < 10 || archive[0] != 'A' || archive[1] != 'C'
            || archive[2] != '0' || archive[3] != '1') {
        throw std::runtime_error("Invalid AC01 header");
    }
    std::size_t offset = 4;
    const auto total = read_integer(archive, offset, 4);
    const auto distinct = read_integer(archive, offset, 2);
    if (total == 0 || total > 65535 || distinct == 0 || distinct > 256) {
        throw std::runtime_error("Invalid AC01 size or alphabet");
    }
    std::array<std::uint32_t, 256> counts{};
    for (std::uint32_t i = 0; i < distinct; ++i) {
        const auto symbol = read_integer(archive, offset, 1);
        const auto count = read_integer(archive, offset, 2);
        if (count == 0 || counts[symbol] != 0) {
            throw std::runtime_error("Invalid AC01 frequency table");
        }
        counts[symbol] = count;
    }
    const auto cumulative = cumulative_counts(counts);
    if (cumulative.back() != total || offset == archive.size()) {
        throw std::runtime_error("Invalid AC01 frequency sum or missing payload");
    }

    BitReader reader(archive, offset);
    std::uint64_t low = 0;
    std::uint64_t high = full - 1;
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 32; ++i) {
        value = 2 * value + reader.read();
    }
    Bytes data;
    data.reserve(total);
    for (std::uint32_t i = 0; i < total; ++i) {
        if (value < low || value > high) {
            throw std::runtime_error("Invalid AC01 coding interval");
        }
        const auto width = high - low + 1;
        const auto position = ((value - low + 1) * total - 1) / width;
        const auto upper = std::upper_bound(cumulative.begin(), cumulative.end(), position);
        const auto symbol = static_cast<std::size_t>(upper - cumulative.begin() - 1);
        data.push_back(static_cast<std::uint8_t>(symbol));
        high = low + width * cumulative[symbol + 1] / total - 1;
        low += width * cumulative[symbol] / total;

        while (true) {
            std::uint64_t adjustment;
            if (high < half) {
                adjustment = 0;
            } else if (low >= half) {
                adjustment = half;
            } else if (low >= quarter && high < three_quarters) {
                adjustment = quarter;
            } else {
                break;
            }
            low = 2 * (low - adjustment);
            high = 2 * (high - adjustment) + 1;
            value = 2 * (value - adjustment) + reader.read();
        }
    }
    return data;
}

}  // namespace arithmetic
