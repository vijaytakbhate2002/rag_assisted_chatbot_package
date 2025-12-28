import urllib3
import os
import requests
import markdown
from xhtml2pdf import pisa


class GithubScrapper:
    def __init__(self, output_dir=r"scrapped_data\github_pdfs"):
        """
        Initialize the GithubScrapper with an output directory.
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def get_readme_content(self, username):
        """
        Fetches the README content for a user's profile repository.
        Attempts to find the special repository named after the username.
        """
        # GitHub API to get the README content (raw)
        # https://docs.github.com/en/rest/repos/contents?apiVersion=2022-11-28#get-repository-content
        # The user profile README is in a repo with the same name as the user
        url = f"https://api.github.com/repos/{username}/{username}/readme"
        headers = {'Accept': 'application/vnd.github.v3.raw'}
        
        try:
            print(f"Fetching README for user: {username}...")
            # Using verify=False to bypass potential SSL certificate issues in some environments
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(url, headers=headers, verify=False)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching README for {username}: {e}")
            return None

    def convert_to_pdf(self, markdown_content, output_filename):
        """
        Converts markdown content to PDF and saves it to the output directory.
        """
        # Convert Markdown to HTML
        # enabling extensions for tables, fenced code blocks, etc.
        html_content = markdown.markdown(markdown_content, extensions=['extra', 'codehilite'])
        
        # Simple CSS styling to make the PDF look decent
        styled_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Helvetica, sans-serif; font-size: 12px; line-height: 1.5; color: #333; }}
                h1, h2, h3 {{ color: #24292e; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
                h1 {{ font-size: 2em; }}
                h2 {{ font-size: 1.5em; }}
                code {{ background-color: #f6f8fa; padding: 0.2em 0.4em; border-radius: 3px; font-family: monospace; }}
                pre {{ background-color: #f6f8fa; padding: 16px; overflow: auto; border-radius: 6px; }}
                blockquote {{ border-left: 4px solid #dfe2e5; color: #6a737d; padding: 0 1em; }}
                img {{ max-width: 100%; }}
                a {{ color: #0366d6; text-decoration: none; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            print(f"Converting to PDF: {output_path}...")
            with open(output_path, "wb") as pdf_file:
                pisa_status = pisa.CreatePDF(styled_html, dest=pdf_file)
            
            if pisa_status.err:
                print(f"Error generating PDF for {output_filename}")
                return False
            
            print(f"Successfully saved PDF to {output_path}")
            return True
        except Exception as e:
            print(f"Exception converting to PDF: {e}")
            return False

    def scrape_profile(self, username):
        """
        Main method to scrape a profile README and save as PDF.
        """
        content = self.get_readme_content(username)
        
        if content:
            filename = f"{username}_profile_readme.pdf"
            self.convert_to_pdf(content, filename)
        else:
            print(f"Could not retrieve README for {username}")

if __name__ == "__main__":
    # Example usage
    scrapper = GithubScrapper()
    # Test with a user who likely has a profile readme, e.g., 'octocat' or the user's name if known
    # You can add usernames here to test
    users_to_scrape = ['vijaytakbhate2002'] 
    for user in users_to_scrape:
        scrapper.scrape_profile(user)
