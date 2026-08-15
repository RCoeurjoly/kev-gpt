#ifndef KEVIN_JTAG_TRANSPORT_H
#define KEVIN_JTAG_TRANSPORT_H

#include <stddef.h>
#include <stdint.h>

int kj_open(const char *serial);
int kj_user1_exchange(const uint8_t *tx, size_t tx_len,
                      uint8_t *rx, size_t rx_cap, unsigned timeout_ms);
void kj_close(void);
const char *kj_last_error(void);

#endif
