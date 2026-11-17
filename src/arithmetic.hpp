#pragma once

#include <cstdint>
#include <vector>

namespace arithmetic {

using Bytes = std::vector<std::uint8_t>;

// AC01 stores the byte histogram in the archive. Frequencies stay fixed while
// coding the entire file. The existing format supports 1–65535 input bytes.
Bytes encode(const Bytes& data);
Bytes decode(const Bytes& archive);

}  // namespace arithmetic
