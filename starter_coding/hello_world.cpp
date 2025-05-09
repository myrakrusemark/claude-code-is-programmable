// Import required libraries
#include <iostream>          // For input/output operations
#include <curl/curl.h>      // For making HTTP requests (libcurl)
#include <string>           // For working with strings

// Callback function to write response data
size_t WriteCallback(char* ptr, size_t size, size_t nmemb, std::string* data) {
    // Calculate total size of the received data
    size_t totalSize = size * nmemb;
    
    // Append received data to the provided string
    data->append(ptr, totalSize);
    
    // Return the size of processed data
    return totalSize;
}

int main(int argc, char* argv[]) {
    // Check if URL was provided as command line argument
    if (argc < 2) {
        // Print usage message if no URL was provided
        std::cout << "Usage: " << argv[0] << " <url>" << std::endl;
        // Exit program with error code 1
        return 1;
    }
    
    // Get URL from first command line argument
    const char* url = argv[1];
    
    // Initialize CURL
    CURL* curl = curl_easy_init();
    
    // Check if CURL was initialized successfully
    if (curl) {
        // String to store the response
        std::string response;
        
        // Set URL to request
        curl_easy_setopt(curl, CURLOPT_URL, url);
        
        // Set callback function to handle response data
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        
        // Set the response string as the data to be passed to the callback
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        
        // Perform the request
        CURLcode res = curl_easy_perform(curl);
        
        // Check if request was successful
        if (res == CURLE_OK) {
            // Get HTTP status code
            long status_code;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);
            
            // Print status code
            std::cout << "Status Code: " << status_code << std::endl;
            
            // Print response content (first 100 characters)
            std::cout << "Response Content: " << 
                      response.substr(0, 100) << "..." << std::endl;
        } else {
            // Print error message if request failed
            std::cout << "Error making request: " << 
                      curl_easy_strerror(res) << std::endl;
        }
        
        // Clean up CURL resources
        curl_easy_cleanup(curl);
    } else {
        // Print error message if CURL initialization failed
        std::cout << "Error initializing CURL" << std::endl;
    }
    
    return 0;
}

// Compile with: g++ hello_world.cpp -o hello_world -lcurl
// Run with: ./hello_world <url>
