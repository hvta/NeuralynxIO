from __future__ import division
import os
import warnings
import mmap
import numpy as np
import ntpath

HEADER_LENGTH = 16 * 1024  # 16 kilobytes of header

NCS_SAMPLES_PER_RECORD = 512
NCS_RECORD = np.dtype([('TimeStamp',       np.uint64),       # Cheetah timestamp for this record. This corresponds to
                                                             # the sample time for the first data point in the Samples
                                                             # array. This value is in microseconds.
                       ('ChannelNumber',   np.uint32),       # The channel number for this record. This is NOT the A/D
                                                             # channel number
                       ('SampleFreq',      np.uint32),       # The sampling frequency (Hz) for the data stored in the
                                                             # Samples Field in this record
                       ('NumValidSamples', np.uint32),       # Number of values in Samples containing valid data
                       ('Samples',         np.int16, NCS_SAMPLES_PER_RECORD)])  # Data points for this record. Cheetah
                                                                                # currently supports 512 data points per
                                                                                # record. At this time, the Samples
                                                                                # array is a [512] array.

NEV_RECORD = np.dtype([('stx',           np.int16),      # Reserved
                       ('pkt_id',        np.int16),      # ID for the originating system of this packet
                       ('pkt_data_size', np.int16),      # This value should always be two (2)
                       ('TimeStamp',     np.uint64),     # Cheetah timestamp for this record. This value is in
                                                         # microseconds.
                       ('event_id',      np.int16),      # ID value for this event
                       ('ttl',           np.int16),      # Decimal TTL value read from the TTL input port
                       ('crc',           np.int16),      # Record CRC check from Cheetah. Not used in consumer
                                                         # applications.
                       ('dummy1',        np.int16),      # Reserved
                       ('dummy2',        np.int16),      # Reserved
                       ('Extra',         np.int32, 8),   # Extra bit values for this event. This array has a fixed
                                                         # length of eight (8)
                       ('EventString',   'S', 128)])  # Event string associated with this event record. This string
                                                         # consists of 127 characters plus the required null termination
                                                         # character. If the string is less than 127 characters, the
                                                         # remainder of the characters will be null.

VOLT_SCALING = (1, u'V')
MILLIVOLT_SCALING = (1000, u'mV')
MICROVOLT_SCALING = (1000000, u'µV')


logging_on = False


def log(msg):
    global logging_on
    if logging_on:
        print(msg)


def read_header(fid):
    # Read the raw header data (16 kb) from the file object fid. Restores the position in the file object after reading.
    pos = fid.tell()
    fid.seek(0)
    raw_hdr = fid.read(HEADER_LENGTH).strip(b'\0')
    fid.seek(pos)

    return raw_hdr


def parse_header(raw_hdr):
    # Parse the header string into a dictionary of name value pairs
    hdr = dict()

    # Decode the header as iso-8859-1 (the spec says ASCII, but there is at least one case of 0xB5 in some headers)
    raw_hdr = raw_hdr.decode('iso-8859-1')

    # Neuralynx headers seem to start with a line identifying the file, so
    # let's check for it
    hdr_lines = [line.strip() for line in raw_hdr.split('\r\n') if line != '']
    if hdr_lines[0] != '######## Neuralynx Data File Header':
        warnings.warn('Unexpected start to header: ' + hdr_lines[0])

    """
    # Try to read the original file path
    try:
        assert hdr_lines[1].split()[1:3] == ['File', 'Name']
        hdr[u'FileName']  = ' '.join(hdr_lines[1].split()[3:])
        # hdr['save_path'] = hdr['FileName']
    except:
        warnings.warn('Unable to parse original file path from Neuralynx header: ' + hdr_lines[1])

    # Process lines with file opening and closing times
    hdr[u'TimeOpened'] = hdr_lines[2][3:]
    hdr[u'TimeOpened_dt'] = parse_neuralynx_time_string(hdr_lines[2])
    hdr[u'TimeClosed'] = hdr_lines[3][3:]
    hdr[u'TimeClosed_dt'] = parse_neuralynx_time_string(hdr_lines[3])
    """

    # Read the parameters, assuming "-PARAM_NAME PARAM_VALUE" format
    #for line in hdr_lines[4:]:
    for line in hdr_lines[1:]:
        try:
            name, value = line[1:].split(' ', 1)  # Ignore the dash and split PARAM_NAME and PARAM_VALUE
            hdr[name] = value
        except Exception as e:
            warnings.warn('Unable to parse parameter line from Neuralynx header: ' + line)

    return hdr


"""
def read_records_with_numpy_memmap(fid, record_dtype, record_skip=0):
    rec_count = 0
    while True:
        rec_chunk = fid.read(record_dtype.itemsize)
        if not rec_chunk:
            break
        rec_count += 1

    # filename = path.join(mkdtemp(), 'memmap_test.dat')
    filename = 'memmap_test.dat'
    fp = np.memmap(filename, dtype=record_dtype, mode='w+', shape=(rec_count))

    fid.seek(HEADER_LENGTH, 0)
    fid.seek(record_skip * record_dtype.itemsize, 1)

    chunk_size = 32768  # 2^15 - number of records to read at once
    for i in range(0, rec_count, chunk_size):
        rec_chunk = fid.read(record_dtype.itemsize * chunk_size)
        record = np.frombuffer(rec_chunk, dtype=record_dtype)
        fp[i:i + chunk_size] = record
        fp.flush()

    return fp
"""


def read_records(fid, record_dtype, record_skip=0, memmap=True, count=None, record_count=None):
    pos = fid.tell()
    fid.seek(HEADER_LENGTH, 0)
    fid.seek(record_skip * record_dtype.itemsize, 1)

    if memmap:
        """
        record_count = 0
        while True:
            rec_chunk = fid.read(record_dtype.itemsize)
            if not rec_chunk:
                break
            record_count += 1
        """
        # Source: https://stackoverflow.com/questions/60493766/read-binary-flatfile-and-skip-bytes
        mm = mmap.mmap(fid.fileno(), length=0, access=mmap.ACCESS_READ)
        rec = np.ndarray(buffer=mm, dtype=record_dtype, offset=HEADER_LENGTH, shape=record_count)
        #rec = np.ndarray(buffer=mm, dtype=record_dtype, offset=HEADER_LENGTH, shape=record_count).copy()
        # mm.close()  # TODO needed?
    else:
        count = -1 if count is None else count
        rec = np.fromfile(fid, record_dtype, count=count)  # Original read

    fid.seek(pos)

    return rec


def estimate_record_count(file_path, record_dtype):
    # Estimate the number of records from the file size
    file_size = os.path.getsize(file_path)
    file_size -= HEADER_LENGTH

    if file_size % record_dtype.itemsize != 0:
        warnings.warn('File size is not divisible by record size (some bytes unaccounted for)')
        raise Exception('File size is not divisible by record size (some bytes unaccounted for)')

    return int(file_size / record_dtype.itemsize)


def check_ncs_records(records):
    # Check that all the records in the array are "similar" (have the same sampling frequency etc.)
    dt = np.diff(records['TimeStamp'])
    dt = dt.astype('int64')
    dt = np.abs(dt - dt[0])

    if not np.all(records['ChannelNumber'] == records[0]['ChannelNumber']):
        warnings.warn('Channel number changed during record sequence')
        return False
    elif not np.all(records['SampleFreq'] == records[0]['SampleFreq']):
        warnings.warn('Sampling frequency changed during record sequence')
        return False
    elif not np.all(records['NumValidSamples'] == 512):
        warnings.warn('Invalid samples in one or more records')
        return False
    elif not np.all(dt <= 1):
        warnings.warn('Time stamp difference tolerance exceeded')
        return False
    else:
        return True


def load_ncs(file_path, rescale_data=False, signal_scaling=MICROVOLT_SCALING, verbose=False):
    global logging_on
    logging_on = verbose

    log('Loading NCS...')
    file_path = os.path.abspath(file_path)
    record_count = estimate_record_count(file_path, NCS_RECORD)

    with open(file_path, 'rb') as fid:
        log('Reading header')
        raw_header = read_header(fid)
        log('Reading records')
        records = read_records(fid, NCS_RECORD, record_count=record_count)
        fid.close()

    header = parse_header(raw_header)
    check_ncs_records(records)

    # Reshape (and rescale, if requested) the data into a 1D array
    data = records['Samples'].ravel()

    if rescale_data:
        log('Rescaling data')
        try:
            # ADBitVolts specifies the conversion factor between the ADC counts and volts
            data = data.astype(np.float64) * (np.float64(header['ADBitVolts']) * signal_scaling[0])
        except KeyError:
            warnings.warn('Unable to rescale data, no ADBitVolts value specified in header')
            rescale_data = False

    ncs = dict()
    ncs['file_path'] = file_path
    ncs['file_name'] = ntpath.basename(file_path)
    ncs['raw_header'] = raw_header
    ncs['header'] = header
    ncs['data'] = data
    ncs['data_units'] = signal_scaling[1] if rescale_data else 'ADC counts'
    ncs['sampling_rate'] = records['SampleFreq'][0]
    ncs['channel_number'] = records['ChannelNumber'][0]
    ncs['timestamp'] = records['TimeStamp']
    ncs['scaling_factor'] = signal_scaling[0]

    log('Loading NCS finished')

    return ncs


def load_nev(file_path):
    file_path = os.path.abspath(file_path)

    with open(file_path, 'rb') as fid:
        raw_header = read_header(fid)
        records = read_records(fid, NEV_RECORD, memmap=False)

    header = parse_header(raw_header)

    nev = dict()
    nev['file_path'] = file_path
    nev['raw_header'] = raw_header
    nev['header'] = header
    nev['records'] = records
    nev['events'] = records[['pkt_id', 'TimeStamp', 'event_id', 'ttl', 'Extra', 'EventString']]

    return nev
