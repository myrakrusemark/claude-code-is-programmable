# Import required libraries
require 'net/http'  # For making HTTP requests
require 'uri'       # For parsing URIs

# Check if URL was provided as command line argument
if ARGV.length < 1
  # Print usage message if no URL was provided
  puts "Usage: ruby hello_world.rb <url>"
  # Exit program with error code 1
  exit 1
end

# Get URL from first command line argument
url_string = ARGV[0]

begin
  # Parse the URL string into a URI object
  uri = URI.parse(url_string)
  
  # Create a new HTTP object using the host and port from the URI
  http = Net::HTTP.new(uri.host, uri.port)
  
  # Use HTTPS if the scheme is https
  http.use_ssl = (uri.scheme == 'https')
  
  # Set timeouts to 5 seconds
  http.open_timeout = 5
  http.read_timeout = 5
  
  # Make a GET request to the URI path
  response = http.get(uri.request_uri)
  
  # Print the HTTP status code
  puts "Status Code: #{response.code}"
  
  # Print the first 100 characters of the response body
  puts "Response Content: #{response.body[0...100]}..."
  
rescue => e
  # Print error message if request failed
  puts "Error making request: #{e.message}"
  # Exit program with error code 1
  exit 1
end
