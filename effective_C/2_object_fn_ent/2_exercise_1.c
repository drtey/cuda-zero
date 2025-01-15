#include <stddef.h>    // For NULL definition
#include <stdio.h>     // For input/output functions if we need them

struct sigrecord {
    int signum;
    char signame[20];
    char sigdesc[100];
    int counter;    // Added counter as per exercise requirement
} sigline, *sigline_p;

// Function to retrieve the counter value
int retrieve_counter(const struct sigrecord *record) {
    if (record == NULL) {
        return -1;    // Error condition
    }
    return record->counter;
}

int main() {
    struct sigrecord my_record;
    
    // Initialize the record
    my_record.signum = 42;
    my_record.counter = 1;
    
    // Retrieve and use the counter value
    int retrieved_value = retrieve_counter(&my_record);
    
    // Print the value to avoid unused variable warning
    printf("Retrieved counter value: %d\n", retrieved_value);
    
    return 0;
}