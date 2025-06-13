#!/usr/bin/env python3
import pypdf
import os

def combine_pdfs(pdf_files, output_filename):
    """Combine multiple PDF files into a single PDF."""
    writer = pypdf.PdfWriter()
    
    for pdf_file in pdf_files:
        print(f"Adding: {os.path.basename(pdf_file)}")
        reader = pypdf.PdfReader(pdf_file)
        for page in reader.pages:
            writer.add_page(page)
    
    with open(output_filename, 'wb') as output_file:
        writer.write(output_file)
    print(f"Created: {output_filename}\n")

# Define the base directory
base_dir = "/home/albaraazain/Documents/PROJECTS/METU/DeepLearning/THE2_QUIZ_PREPERATION"

# PDFs from the code folder
code_pdfs = [
    os.path.join(base_dir, "code/Template_for_rapid_homework_typesetting-6.pdf"),
    os.path.join(base_dir, "code/Template_for_rapid_homework_typesetting-7.pdf"),
    os.path.join(base_dir, "code/Template_for_rapid_homework_typesetting-8.pdf")
]

# PDFs from the memorization_guide folder
memorization_pdfs = [
    os.path.join(base_dir, "code/memorization_guide/Template_for_rapid_homework_typesetting-12.pdf"),
    os.path.join(base_dir, "code/memorization_guide/Template_for_rapid_homework_typesetting-13.pdf"),
    os.path.join(base_dir, "code/memorization_guide/Template_for_rapid_homework_typesetting-14.pdf"),
    os.path.join(base_dir, "code/memorization_guide/Template_for_rapid_homework_typesetting__2_-1.pdf"),
    os.path.join(base_dir, "code/memorization_guide/Template_for_rapid_homework_typesetting__2_.pdf")
]

# Combine PDFs from code folder
print("Combining PDFs from code folder...")
combine_pdfs(code_pdfs, os.path.join(base_dir, "combined_code.pdf"))

# Combine PDFs from memorization_guide folder
print("Combining PDFs from memorization_guide folder...")
combine_pdfs(memorization_pdfs, os.path.join(base_dir, "combined_memorization_guide.pdf"))

print("Done! Created two combined PDFs:")
print(f"1. {os.path.join(base_dir, 'combined_code.pdf')}")
print(f"2. {os.path.join(base_dir, 'combined_memorization_guide.pdf')}")