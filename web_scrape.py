import requests
from bs4 import BeautifulSoup
from fpdf import FPDF

# Function to scrape web page content
def scrape_web_page(url):
    try:
        # Send a request to the website
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Function to extract main content and ignore menus
def extract_main_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove unnecessary sections like menus, footers, and navigation bars
    for element in soup(['header', 'nav', 'footer', 'aside', 'script', 'style']):
        element.decompose()  # Remove these elements completely

    # Attempt to extract the main content (assuming it's within <main> or <article>)
    main_content = soup.find('main') or soup.find('article')
    
    # If <main> or <article> is not found, fall back to <body> content
    if not main_content:
        main_content = soup.find('body')
    
    text = main_content.get_text(separator="\n") if main_content else "No main content found"
    return text

# Function to remove non-ASCII characters from text
def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

# Function to save text to a PDF
def save_to_pdf(text, pdf_filename):
    # Create PDF document
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add text to the PDF
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    
    # Save the PDF to a file
    pdf.output(pdf_filename)
    print(f"PDF saved as {pdf_filename}")

# Main function
def main():
    url = input("Enter the URL to scrape: ")
    
    # Step 1: Scrape the web page
    html_content = scrape_web_page(url)
    
    if html_content:
        # Step 2: Extract the main content
        main_content = extract_main_content(html_content)
        
        # Step 3: Remove non-ASCII characters
        cleaned_text = remove_non_ascii(main_content)
        
        # Step 4: Save the cleaned text to a PDF
        pdf_filename = "dbs_home_loan.pdf"
        save_to_pdf(cleaned_text, pdf_filename)

# Run the main function
if __name__ == "__main__":
    main()
