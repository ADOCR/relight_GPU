#include "jpeg_decoder.h"
#include <cstring>
#include <limits>

JpegDecoder::JpegDecoder() {
	decInfo.err = jpeg_std_error(&errMgr);
	jpeg_create_decompress(&decInfo);
}

JpegDecoder::~JpegDecoder() {
	jpeg_destroy_decompress(&decInfo);
#ifdef _WIN32
	if (mapped_data)
		mapped_file.unmap(mapped_data);
#endif
	if(file)
		fclose(file);
}

void JpegDecoder::setColorSpace(J_COLOR_SPACE colorSpace) {
	requested_out_color_space = colorSpace;
}


J_COLOR_SPACE JpegDecoder::getColorSpace() const {
	return decInfo.out_color_space;
}

J_COLOR_SPACE JpegDecoder::getJpegColorSpace() const {
	return decInfo.jpeg_color_space;
}

bool JpegDecoder::decode(uint8_t* buffer, size_t len, uint8_t*& img, int& width, int& height) {
	if (buffer == nullptr)
		return false;

	jpeg_mem_src(&decInfo, buffer, len);
	return decode(img, width, height);
}

bool JpegDecoder::decode(const char* path, uint8_t*& img, int& width, int& height) {
	FILE* file = fopen(path, "rb");
	if(!file) return false;
	jpeg_stdio_src(&decInfo, file);
	bool rv = decode(img, width, height);
	fclose(file);
	file = nullptr;
	return rv;
}


bool JpegDecoder::decode(FILE *file, uint8_t*& img, int& width, int& height) {
	jpeg_stdio_src(&decInfo, file);
	return decode(img, width, height);
}


bool JpegDecoder::decode(uint8_t*& img, int& width, int& height) {
	init(width, height);

	img = new uint8_t[decInfo.image_height * rowSize()];

	int readed = readRows(height, img);
	if(readed != height)
		return false;
/*	JSAMPROW rows[1];
	size_t offset = 0;
	while (decInfo.output_scanline < decInfo.image_height) {
		rows[0] = img + offset;
		jpeg_read_scanlines(&decInfo, rows, 1);
		offset += rowSize;
	} */

//	jpeg_finish_decompress(&decInfo);
	return true;
}

bool JpegDecoder::init(const char* path, int &width, int &height) {
#ifdef _WIN32
	jpeg_abort_decompress(&decInfo);
	if (mapped_data) {
		mapped_file.unmap(mapped_data);
		mapped_data = nullptr;
	}
	mapped_file.close();
	mapped_file.setFileName(QString::fromUtf8(path));
	if (!mapped_file.open(QIODevice::ReadOnly)) return false;
	mapped_size = mapped_file.size();
	if (mapped_size <= 0 || quint64(mapped_size) > std::numeric_limits<unsigned long>::max()) {
		mapped_file.close();
		return false;
	}
	mapped_data = mapped_file.map(0, mapped_size);
	if (!mapped_data) {
		mapped_file.close();
		return false;
	}
	jpeg_mem_src(&decInfo, mapped_data, static_cast<unsigned long>(mapped_size));
#else
	file = fopen(path, "rb");
	if(!file) return false;
	jpeg_stdio_src(&decInfo, file);
#endif
	return init(width, height);
}

bool JpegDecoder::init(int &width, int &height) {
	// Save APP2 markers to capture ICC profile
	jpeg_save_markers(&decInfo, JPEG_APP0 + 2, 0xFFFF);
	
	jpeg_read_header(&decInfo, (boolean)true);
	
	// Extract ICC profile if present
	extractICCProfile();

	// Apply requested output color space (must happen after jpeg_read_header
	// which resets out_color_space to the source's color space, and before
	// jpeg_start_decompress which derives output_components from it).
	if(requested_out_color_space != JCS_UNKNOWN)
		decInfo.out_color_space = requested_out_color_space;

	decInfo.raw_data_out = (boolean)false;
	
	if(decInfo.num_components > 1) 
		subsampled =  decInfo.comp_info[1].h_samp_factor != 1;

	jpeg_start_decompress(&decInfo);

	width = decInfo.image_width;
	height = decInfo.image_height;
	return true;
}

void JpegDecoder::extractICCProfile() {
	icc_profile.clear();
	
	// ICC profiles can span multiple APP2 segments
	// Format: "ICC_PROFILE\0" + seq_no (1 byte) + num_markers (1 byte) + data
	const int MAX_SEGMENTS = 255;
	std::vector<std::vector<uint8_t>> segments(MAX_SEGMENTS);
	int num_markers = 0;
	
	for (jpeg_saved_marker_ptr marker = decInfo.marker_list; marker != nullptr; marker = marker->next) {
		if (marker->marker == JPEG_APP0 + 2 && marker->data_length >= 14) {
			// Check for ICC_PROFILE signature
			if (std::memcmp(marker->data, "ICC_PROFILE\0", 12) == 0) {
				int seq_no = marker->data[12];
				num_markers = marker->data[13];
				
				if (seq_no > 0 && seq_no <= MAX_SEGMENTS) {
					// Store segment data (skip 14-byte header)
					segments[seq_no - 1].assign(
						marker->data + 14,
						marker->data + marker->data_length
					);
				}
			}
		}
	}
	
	// Reassemble segments in order
	if (num_markers > 0) {
		for (int i = 0; i < num_markers; i++) {
			if (!segments[i].empty()) {
				icc_profile.insert(icc_profile.end(), segments[i].begin(), segments[i].end());
			}
		}
	}
}

size_t JpegDecoder::readRows(int nrows, uint8_t *buffer) { //return false on end.
	if(decInfo.output_scanline == decInfo.image_height)
		restart();

	size_t rowSize = decInfo.image_width * decInfo.output_components;
	JSAMPROW rows[1];
	size_t offset = 0;
	int readed = 0;
	while (decInfo.output_scanline < decInfo.image_height && readed < nrows) {
		readed++;
		rows[0] = buffer + offset;
		jpeg_read_scanlines(&decInfo, rows, 1);
		offset += rowSize;
	}

	if(decInfo.output_scanline == decInfo.image_height)
		jpeg_finish_decompress(&decInfo);
	return readed;
}

bool JpegDecoder::finish() {
	// readRows already finishes complete images; abort also safely ends partial reads.
	jpeg_abort_decompress(&decInfo);
	if(file) {
		fclose(file);
		file = nullptr;
	}
#ifdef _WIN32
	if (mapped_data) {
		mapped_file.unmap(mapped_data);
		mapped_data = nullptr;
	}
	mapped_file.close();
	mapped_size = 0;
#endif
	return true;
}

bool JpegDecoder::restart() {
	//jpeg_finish_decompress(&decInfo);
	jpeg_abort_decompress(&decInfo);
#ifdef _WIN32
	if (mapped_data) {
		jpeg_mem_src(&decInfo, mapped_data, static_cast<unsigned long>(mapped_size));
	} else
#endif
	{
	if (!file) return false;
	rewind(file);
	jpeg_stdio_src(&decInfo, file);
	}
	int w, h;
	return init(w, h);
}


