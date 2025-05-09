# Import required libraries
import requests  # For making HTTP requests
import sys  # For accessing command line arguments

def main():
    # Check if URL was provided as command line argument
    if len(sys.argv) < 2:
        # Print usage message if no URL was provided
        print("Usage: python hello_world.py <url>")
        # Exit program with error code 1
        sys.exit(1)
    
    # Get URL from first command line argument
    url = sys.argv[1]
    
    try:
        # Make HTTP GET request to the provided URL
        response = requests.get(url)
        # Check if request was successful (status code 200)
        response.raise_for_status()
        # Print status code
        print(f"Status Code: {response.status_code}")
        # Print response content
        print(f"Response Content: {response.text[:100]}...")
    except requests.exceptions.RequestException as e:
        # Print error message if request failed
        print(f"Error making request: {e}")

# Check if this file is being run directly
if __name__ == "__main__":
    # Call the main function
    main()
