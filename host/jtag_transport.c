#include "jtag_transport.h"

#include <errno.h>
#include <libftdi1/ftdi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* FT232H MPSSE commands. Data is shifted LSB-first, changed on TCK's falling
 * edge, and sampled on its rising edge, matching Xilinx JTAG timing. */
enum {
    MPSSE_SET_LOW = 0x80,
    MPSSE_SET_HIGH = 0x82,
    MPSSE_DIVISOR = 0x86,
    MPSSE_DISABLE_DIV5 = 0x8a,
    MPSSE_DISABLE_3PHASE = 0x8d,
    MPSSE_DISABLE_ADAPTIVE = 0x97,
    MPSSE_BYTES_IO_LSB_NEG = 0x39,
    MPSSE_BITS_OUT_LSB_NEG = 0x1b,
    MPSSE_TMS_OUT_NEG = 0x4b,
};

static struct ftdi_context *device;
static char last_error[256] = "transport is not open";

static int fail(int code, const char *message) {
    snprintf(last_error, sizeof(last_error), "%s", message);
    return code < 0 ? code : -code;
}

static int write_all(const uint8_t *data, size_t length) {
    size_t offset = 0;
    while (offset < length) {
        int written = ftdi_write_data(device, (unsigned char *)data + offset,
                                      (int)(length - offset));
        if (written < 0)
            return fail(EIO, ftdi_get_error_string(device));
        if (written == 0)
            return fail(EIO, "short FTDI write");
        offset += (size_t)written;
    }
    return 0;
}

static uint64_t milliseconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000u + (uint64_t)now.tv_nsec / 1000000u;
}

static int read_exact(uint8_t *data, size_t length, unsigned timeout_ms) {
    size_t offset = 0;
    uint64_t deadline = milliseconds() + timeout_ms;
    while (offset < length) {
        int count = ftdi_read_data(device, data + offset, (int)(length - offset));
        if (count < 0)
            return fail(EIO, ftdi_get_error_string(device));
        offset += (size_t)count;
        if (offset == length)
            return 0;
        if (milliseconds() >= deadline)
            return fail(ETIMEDOUT, "FTDI read timed out");
        usleep(1000);
    }
    return 0;
}

static int tms(unsigned count, uint8_t bits, int tdi) {
    if (count == 0 || count > 7)
        return fail(EINVAL, "invalid TMS clock count");
    uint8_t command[3] = {
        MPSSE_TMS_OUT_NEG, (uint8_t)(count - 1),
        (uint8_t)((bits & 0x7f) | (tdi ? 0x80 : 0)),
    };
    return write_all(command, sizeof(command));
}

static int select_user(unsigned chain) {
    int result;
    /* Test-Logic-Reset, Run-Test/Idle. */
    if ((result = tms(6, 0x1f, 0)) < 0)
        return result;
    /* Idle -> Select-DR -> Select-IR -> Capture-IR -> Shift-IR. */
    if ((result = tms(4, 0x03, 0)) < 0)
        return result;
    /* XC7 USER1/USER2 are six-bit instructions 000010/000011, LSB first. The
     * sixth bit is clocked together with TMS=1 to leave Shift-IR. */
    if (chain != 1 && chain != 2)
        return fail(EINVAL, "invalid USER JTAG chain");
    uint8_t first_five[3] = {
        MPSSE_BITS_OUT_LSB_NEG, 4, (uint8_t)(chain == 1 ? 0x02 : 0x03)
    };
    if ((result = write_all(first_five, sizeof(first_five))) < 0)
        return result;
    if ((result = tms(1, 0x01, 0)) < 0)
        return result;
    /* Exit1-IR -> Update-IR -> Idle. */
    return tms(2, 0x01, 0);
}

int kj_open(const char *serial) {
    if (device)
        return fail(EBUSY, "transport is already open");
    device = ftdi_new();
    if (!device)
        return fail(ENOMEM, "cannot allocate libftdi context");
    int result = ftdi_set_interface(device, INTERFACE_A);
    if (result >= 0)
        result = ftdi_usb_open_desc(device, 0x0403, 0x6014, NULL, serial);
    if (result < 0) {
        fail(EIO, ftdi_get_error_string(device));
        ftdi_free(device);
        device = NULL;
        return -EIO;
    }
    ftdi_usb_reset(device);
    ftdi_set_latency_timer(device, 1);
    ftdi_write_data_set_chunksize(device, 65536);
    ftdi_read_data_set_chunksize(device, 65536);
    ftdi_set_bitmode(device, 0, BITMODE_RESET);
    usleep(20000);
    if (ftdi_set_bitmode(device, 0, BITMODE_MPSSE) < 0) {
        kj_close();
        return fail(EIO, "cannot enable FTDI MPSSE mode");
    }
    usleep(20000);
    uint8_t setup[] = {
        MPSSE_DISABLE_DIV5, MPSSE_DISABLE_ADAPTIVE, MPSSE_DISABLE_3PHASE,
        MPSSE_DIVISOR, 4, 0, /* 60 MHz / (2 * (4 + 1)) = 6 MHz. */
        /* Digilent HS3 buffer enables and directions, matching the locked
         * openFPGALoader cable definition.  Bit 7 enables the low-port JTAG
         * drivers; high-port bits 4/5 control the remaining HS3 buffers. */
        MPSSE_SET_LOW, 0x88, 0x8b,
        MPSSE_SET_HIGH, 0x20, 0x30,
    };
    if (write_all(setup, sizeof(setup)) < 0) {
        kj_close();
        return -EIO;
    }
    ftdi_tcioflush(device);
    snprintf(last_error, sizeof(last_error), "ok");
    return 0;
}

static int user_exchange(unsigned chain, const uint8_t *tx, size_t tx_len,
                         uint8_t *rx, size_t rx_cap, unsigned timeout_ms) {
    if (!device)
        return fail(ENODEV, "transport is not open");
    if ((!tx && tx_len) || (!rx && rx_cap) || (tx_len == 0 && rx_cap == 0) ||
        tx_len > 65536 || rx_cap > 65536)
        return fail(EINVAL, "invalid exchange buffer or length");
    size_t length = tx_len > rx_cap ? tx_len : rx_cap;
    uint8_t *out = calloc(length, 1);
    uint8_t *incoming = calloc(length, 1);
    if (!out || !incoming) {
        free(out);free(incoming);
        return fail(ENOMEM, "cannot allocate exchange buffer");
    }
    if (tx_len)
        memcpy(out, tx, tx_len);
    ftdi_tciflush(device);
    int result = select_user(chain);
    /* Idle -> Select-DR -> Capture-DR -> Shift-DR. */
    if (result >= 0)
        result = tms(3, 0x01, 0);
    if (result >= 0) {
        uint8_t header[3] = {
            MPSSE_BYTES_IO_LSB_NEG,
            (uint8_t)((length - 1) & 0xff),
            (uint8_t)((length - 1) >> 8),
        };
        result = write_all(header, sizeof(header));
    }
    if (result >= 0)
        result = write_all(out, length);
    if (result >= 0)
        result = read_exact(incoming, length, timeout_ms ? timeout_ms : 1);
    /* One deliberately ignored partial bit leaves Shift-DR. UPDATE/CAPTURE
     * resets the FPGA byte phase before the next packet. */
    if (result >= 0)
        result = tms(1, 0x01, 0);
    if (result >= 0)
        result = tms(2, 0x01, 0); /* Update-DR, Idle. */
    if(result>=0&&rx_cap)memcpy(rx,incoming,rx_cap<length?rx_cap:length);
    free(out);free(incoming);
    if (result < 0)
        return result;
    snprintf(last_error, sizeof(last_error), "ok");
    return (int)(rx_cap < length ? rx_cap : length);
}

int kj_user1_exchange(const uint8_t *tx, size_t tx_len,
                      uint8_t *rx, size_t rx_cap, unsigned timeout_ms) {
    return user_exchange(1, tx, tx_len, rx, rx_cap, timeout_ms);
}

void kj_close(void) {
    if (!device)
        return;
    ftdi_set_bitmode(device, 0, BITMODE_RESET);
    ftdi_usb_close(device);
    ftdi_free(device);
    device = NULL;
}

const char *kj_last_error(void) {
    return last_error;
}
