//File EEGStream.h
//Written by Timo Duchrow, Marc Tabie, Johannes Teiwes
//Date: Feb. 2009
//Version 1.0

#ifndef _EEGSTREAM
#define _EEGSTREAM


// All numbers are sent in little endian format.
struct Marker {
	uint32_t position;	// relative position in samples from beginning of block
	uint32_t type; 		// marker type
};

// Header wich comes with all Messages
struct MessageHeader {
	uint32_t message_size; // = Size of Message in Bytes
	uint32_t message_type; // = Type of Message: 1=StartMessage, 2=DataMessage, 3=StopMessage, 4=MuxDataMessage
};

// StartMessage, Type = 1
struct EEGStartMessage : MessageHeader {
	uint32_t n_variables; // = number of channels
	uint32_t n_observations; // = number of samples
	uint32_t frequency;
	uint32_t sample_size; 	// size of one sample (1, 2, 4, maybe even 8 Bytes)
	uint32_t protocol_version;
    uint32_t abs_start_time[2];
	uint8_t resolutions[256]; // as defined in BrainAmpIoCtl.h
	uint32_t variable_names_size; // size in byte
	char variable_names[1];
	uint32_t marker_names_size; // size in byte
	char marker_names[1];
};

// DataMessage, Type = 2
struct EEGDataMessage : MessageHeader {

	uint32_t time_code; // number of ms since start of recording
	uint32_t n_marker;  // number of Markers in this Block
	Marker markers[1];	// Array of Markers, length = n_marker*sizeof(Marker)
	// if not enough data left to fill complete block, rest of the block is
	// padded with 0 (should only happen in case of offline processing)
	int16_t data[1];		// Array of Data, length = n_variables*n_observations*sizeof(short)
};

// StopMessage, Type = 3
struct EEGStopMessage : MessageHeader {
};

// RawData Message, Type = 4
struct EEGMuxDataMarkerMessage : MessageHeader {
	uint32_t time_code;		// number of ms since start of recording
	uint32_t sample_size; 	// size of one sample (1, 2, 4, maybe even 8 Bytes)
	char data[1];			// this field will contain n_variables x n_observations samples
							// .. much like the raw data coming out of the EEG-Hardware
};


#endif // _EEGSTREAM
