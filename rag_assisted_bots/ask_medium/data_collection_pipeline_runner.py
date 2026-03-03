from rag_assisted_bots.ask_medium.src import MediumDataCollector

def collect_data(medium_username, pdf_folder_path:str, metadata_file_path:str):
    """Run the data collection pipeline for a given Medium username and save the data to the specified output folder."""
    collector = MediumDataCollector(medium_username)
    collector.save_data(pdf_folder_path, metadata_file_path)


if __name__ == "__main__":
    medium_username = "vijaytakbhate45" 
    output_folder = "medium_data"  
    collect_data(medium_username, output_folder)