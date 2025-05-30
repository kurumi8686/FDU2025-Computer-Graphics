from PIL import Image
import os


def combine_images_horizontally(image_files, output_filename="combined_gpr_plot.png"):
    """
    Combines a list of images horizontally.

    Args:
        image_files (list): A list of filenames for the images to combine,
                              in the order they should appear from left to right.
        output_filename (str): The filename for the combined output image.
    """
    images = []
    for file in image_files:
        try:
            img = Image.open(file)
            images.append(img)
        except FileNotFoundError:
            print(f"Error: Image file '{file}' not found. Skipping.")
            return
        except Exception as e:
            print(f"Error opening image '{file}': {e}. Skipping.")
            return

    if not images:
        print("No images were loaded. Exiting.")
        return

    # Assume all images have the same height for simplicity in this typical use case.
    # If heights vary, you might want to resize or align them.
    # For this script, we'll use the height of the first image.
    # More robustly, you could use max_height = max(img.height for img in images)
    # and then decide on an alignment strategy if widths/heights vary significantly.

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)  # Use the maximum height to ensure all images fit

    # Create a new image with the total width and max height
    combined_image = Image.new('RGB', (total_width, max_height), color='white')  # Added white background

    current_x_offset = 0
    for img in images:
        # If image height is less than max_height, you might want to center it.
        # For now, it will be pasted at the top (y=0)
        y_offset = (max_height - img.height) // 2  # Center vertically
        combined_image.paste(img, (current_x_offset, y_offset))
        current_x_offset += img.width

    try:
        combined_image.save(output_filename)
        print(f"Combined image saved as '{output_filename}'")
    except Exception as e:
        print(f"Error saving combined image: {e}")


if __name__ == "__main__":
    # Define the image filenames in the desired order (left to right)
    image_filenames = [
        "gpr_embedding_visualization_n3.png",
        "gpr_embedding_visualization_n30.png",
        "gpr_embedding_visualization_n300.png",
        "gpr_embedding_visualization_n3000.png"
    ]

    # Check if all specified image files exist before attempting to combine
    missing_files = [f for f in image_filenames if not os.path.exists(f)]
    if missing_files:
        print("Error: The following image files are missing:")
        for f in missing_files:
            print(f"- {f}")
        print("Please ensure all files are in the same directory as the script, or provide correct paths.")
    else:
        combine_images_horizontally(image_filenames, "combined_gpr_visualizations.png")