#include "google/protobuf/message.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "io.h"
#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <fcntl.h>
#include <glog/logging.h>

using namespace google::protobuf;

namespace ssdnative {
	bool ReadProtoFromTextFile(const char* filename, Message* proto);

	inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
		return ReadProtoFromTextFile(filename.c_str(), proto);
	}

	inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
		CHECK(ReadProtoFromTextFile(filename, proto));
	}

	inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
		ReadProtoFromTextFileOrDie(filename.c_str(), proto);
	}

	void WriteProtoToTextFile(const Message& proto, const char* filename);
	inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
		WriteProtoToTextFile(proto, filename.c_str());
	}

	bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

	inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
		return ReadProtoFromBinaryFile(filename.c_str(), proto);
	}

	inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
		CHECK(ReadProtoFromBinaryFile(filename, proto));
	}

	inline void ReadProtoFromBinaryFileOrDie(const string& filename,
		Message* proto) {
		ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
	}

	void WriteProtoToBinaryFile(const Message& proto, const char* filename);
	inline void WriteProtoToBinaryFile(
		const Message& proto, const string& filename) {
		WriteProtoToBinaryFile(proto, filename.c_str());
	}
}