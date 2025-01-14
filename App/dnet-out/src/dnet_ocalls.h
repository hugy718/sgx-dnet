/*
 * Created on Fri Feb 14 2020
 *
 * Copyright (c) 2020 xxx xxx, xxxx
 */

#ifndef DNET_OCALLS_H
#define DNET_OCALLS_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "darknet.h"

//for open
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

//edgerator header file
#include "Enclave_u.h"

#if defined(__cplusplus)

extern "C"
{
#endif

    void ocall_free_sec(section *sec);
    void ocall_free_list(list *list);
    // void ocall_print_string(const char *str);
#ifndef NDEBUG
    void ocall_sgxdnet_print_float(float number);
    void ocall_sgxdnet_print_int(int number);
    void ocall_sgxdnet_print_size_t(size_t number);
#endif // NDEBUG
    void ocall_open_file(const char *filename, flag oflag);
    void ocall_close_file();
    void ocall_fread(void *ptr, size_t size, size_t nmemb);
    void ocall_fwrite(void *ptr, size_t size, size_t nmemb);

#if defined(__cplusplus)
}
#endif

#endif /* DNET_OCALLS_H */
