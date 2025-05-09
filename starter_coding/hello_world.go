// Package declaration
package main

// Import required packages
import (
	"fmt"           // For formatted I/O operations
	"io"            // For I/O primitives
	"net/http"      // For making HTTP requests
	"os"            // For accessing command-line arguments
	"strings"       // For string manipulation
)

// Main function, entry point of the program
func main() {
	// Check if URL was provided as command line argument
	if len(os.Args) < 2 {
		// Print usage message if no URL was provided
		fmt.Println("Usage: go run hello_world.go <url>")
		// Exit program with error code 1
		os.Exit(1)
	}
	
	// Get URL from first command line argument
	url := os.Args[1]
	
	// Print message indicating request is being made
	fmt.Printf("Making request to: %s\n", url)
	
	// Make HTTP GET request to the provided URL
	resp, err := http.Get(url)
	
	// Check if an error occurred during the request
	if err != nil {
		// Print error message if request failed
		fmt.Printf("Error making request: %s\n", err)
		// Exit program with error code 1
		os.Exit(1)
	}
	
	// Ensure the response body is closed after function returns
	defer resp.Body.Close()
	
	// Print the HTTP status code
	fmt.Printf("Status Code: %d\n", resp.StatusCode)
	
	// Read the first 100 bytes of the response body
	body := make([]byte, 100)
	n, err := resp.Body.Read(body)
	
	// Check if an error occurred while reading the response body
	if err != nil && err != io.EOF {
		// Print error message if reading failed
		fmt.Printf("Error reading response: %s\n", err)
		// Exit program with error code 1
		os.Exit(1)
	}
	
	// Convert read bytes to string and trim any null bytes
	bodyString := strings.TrimRight(string(body[:n]), "\x00")
	
	// Print the response content
	fmt.Printf("Response Content: %s...\n", bodyString)
}
