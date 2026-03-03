from importlib_metadata import metadata
from xhtml2pdf import pisa
import feedparser
from typing import Union
import json
import cloudscraper
import re


class NameFormatter:
    """A utility class for formatting names to be suitable for file names by replacing spaces with underscores and removing special characters.
    Methods:
        format_name(name: str) -> str: Formats the name and makes it suitable for saving as a file name.
    """

    def format_name(self, name: str) -> str:
        """
        Allow only A-Z, a-z and underscore.
        Remove everything else including numbers and emojis.
        """

        name = name.replace(" ", "_")
        
        name = re.sub(r'[^A-Za-z_]', '', name)
        
        name = re.sub(r'_+', '_', name)

        name = name.strip('_')

        name = name.strip()
        
        return name
    

class MediumDataCollector(NameFormatter):
    """Collects and formats data from a Medium user's RSS feed, and saves it in PDF format along with metadata in JSON format.
    Args:
        medium_username (str): The Medium username to collect data from.
    
    Methods:
        collect_raw_data() -> list: Collects raw data from the Medium RSS feed and returns
        a list of entries.
        style_html(html_content: str) -> str: Styles the HTML content for better PDF formatting
        format_pdf_html() -> Union[str, None]: Formats the collected data into a styled HTML string suitable for PDF generation.
        save_data(output_folder: str): Saves the formatted HTML data in PDF format, and saves
    """
    def __init__(self, medium_username):
        self.medium_username = medium_username
        self.feed_url = f"https://medium.com/feed/@{medium_username}"
        self.scraper = cloudscraper.create_scraper()
        self.response = self.scraper.get(self.feed_url)


    def collect_raw_data(self) -> list:
        """Collects raw data from the Medium RSS feed and returns a list of entries."""
        feed = feedparser.parse(self.response.text)
        return feed.entries
    
    
    def style_html(self, html_content: str) -> str:
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
        return styled_html
    
    
    def format_pdf_html(self) -> Union[str, None]:
        """Formats the collected data into a styled HTML string suitable for PDF generation."""
        entries = self.collect_raw_data()
        if entries:
            data = {"medium": []}
            for entry in entries:
                data['medium'].append({
                    "full_name": self.format_name(entry.title),
                    "repo_name": self.format_name(entry.title),
                    "created_at": entry.published,
                    "updated_at": entry.published,
                    "pushed_at": entry.published,
                    "download_url": entry.link,
                    "repository_url": entry.link,
                    "language": "English",
                    "private": False,
                    "description": None,
                    "full_html_content": self.style_html(entry.content[0].value),
                    "size": len(entry.content[0].value)
                })
            return data
        else:
            raise ValueError(f"No entries found in the Medium feed for {self.medium_username} Please check the username and try again.")
        

    def save_data(self, pdf_folder_path: str, metadata_file_path: str):
        """Saves the formatted html data in pdf format, plus save the metadata into a json format"""
        data = self.format_pdf_html()
        if data:
            for details in data['medium']:
                full_html_content = details["full_html_content"]
                styled_html = self.style_html(full_html_content)
                output_path = f"{pdf_folder_path}/{details['full_name']}.pdf"
                with open(output_path, "wb") as pdf_file:
                    pisa_status = pisa.CreatePDF(styled_html, dest=pdf_file)

            with open(metadata_file_path, "w") as f:
                metadata = {'medium': []}
           
                for item in data['medium']:
                    item.pop("full_html_content", None)
                    metadata['medium'].append(item)
                json.dump(metadata, f)
    


            
