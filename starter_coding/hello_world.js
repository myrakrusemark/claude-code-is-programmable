// Import required modules
const https = require('https');  // For making HTTPS requests
const http = require('http');   // For making HTTP requests
const url = require('url');    // For parsing URLs

// Function to make an HTTP request
function makeRequest(urlString) {
    // Parse the provided URL
    const parsedUrl = url.parse(urlString);
    
    // Choose the appropriate protocol (http or https)
    const protocol = parsedUrl.protocol === 'https:' ? https : http;
    
    // Return a new Promise to handle the asynchronous request
    return new Promise((resolve, reject) => {
        // Log that request is being made
        console.log(`Making request to: ${urlString}`);
        
        // Make the HTTP/HTTPS request
        const req = protocol.get(urlString, (res) => {
            // Variable to store the response data
            let data = '';
            
            // Log the status code
            console.log(`Status Code: ${res.statusCode}`);
            
            // Event handler for receiving data chunks
            res.on('data', (chunk) => {
                // Append the chunk to our data variable
                data += chunk;
            });
            
            // Event handler for when the response is complete
            res.on('end', () => {
                // Resolve the promise with the collected data
                resolve(data);
            });
        });
        
        // Event handler for request errors
        req.on('error', (err) => {
            // Reject the promise with the error
            reject(`Error making request: ${err.message}`);
        });
        
        // End the request
        req.end();
    });
}

// Main function
async function main() {
    // Check if URL was provided as command line argument
    if (process.argv.length < 3) {
        // Print usage message if no URL was provided
        console.log('Usage: node hello_world.js <url>');
        // Exit program with error code 1
        process.exit(1);
    }
    
    // Get URL from command line argument
    const url = process.argv[2];
    
    try {
        // Make the request and await the response
        const response = await makeRequest(url);
        // Print the first 100 characters of the response
        console.log(`Response Content: ${response.substring(0, 100)}...`);
    } catch (error) {
        // Print any errors that occurred
        console.error(error);
    }
}

// Call the main function
main();
