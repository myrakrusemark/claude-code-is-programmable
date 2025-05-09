// Import required libraries
import java.io.BufferedReader;          // For reading response data
import java.io.InputStreamReader;      // For converting bytes to characters
import java.net.HttpURLConnection;     // For making HTTP connections
import java.net.URL;                   // For working with URLs

public class HelloWorld {
    public static void main(String[] args) {
        // Check if URL was provided as command line argument
        if (args.length < 1) {
            // Print usage message if no URL was provided
            System.out.println("Usage: java HelloWorld <url>");
            // Exit program with error code 1
            System.exit(1);
        }
        
        // Get URL from command line argument
        String urlString = args[0];
        
        try {
            // Create URL object from the string
            URL url = new URL(urlString);
            
            // Open connection to the URL
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            
            // Set request method to GET
            connection.setRequestMethod("GET");
            
            // Set connection timeout to 5 seconds
            connection.setConnectTimeout(5000);
            
            // Set read timeout to 5 seconds
            connection.setReadTimeout(5000);
            
            // Get response code
            int statusCode = connection.getResponseCode();
            System.out.println("Status Code: " + statusCode);
            
            // Create reader for the input stream
            BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getInputStream()));
            
            // Variable to hold a line of the response
            String line;
            
            // StringBuilder to accumulate the response
            StringBuilder response = new StringBuilder();
            
            // Read the response line by line
            while ((line = reader.readLine()) != null) {
                response.append(line);
            }
            
            // Close the reader
            reader.close();
            
            // Disconnect the connection
            connection.disconnect();
            
            // Get the response as a string
            String responseContent = response.toString();
            
            // Print the first 100 characters of the response
            System.out.println("Response Content: " + 
                              responseContent.substring(0, Math.min(100, responseContent.length())) + 
                              "...");
            
        } catch (Exception e) {
            // Print error message if request failed
            System.out.println("Error making request: " + e.getMessage());
        }
    }
}
