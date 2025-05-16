from pdf2image import convert_from_path
import os

def pdf_to_png(pdf_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert PDF to a list of images
    images = convert_from_path(pdf_path)
    
    # Save each image as PNG
    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f"page_{i+1}.png")
        image.save(output_path, "PNG")
        print(f"Saved {output_path}")

if __name__ == "__main__":
    # Specify the path to your PDF file
    pdf_file = "/workspace/Paddle_OCR/testing_tribe/CALAX_Concession Agreement 10July2015.pdf"  # Replace with your PDF file path
    output_directory = "/workspace/Paddle_OCR/images_png"  # Directory where PNGs will be saved
    
    # Convert PDF to PNG
    pdf_to_png(pdf_file, output_directory)