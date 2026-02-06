#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

class TestRunner {
public:
    
    void run();

private:
    void run_histogram_tests();
    void run_scan_hillis_tests();
    void run_scan_blelloch_tests();
};

#endif