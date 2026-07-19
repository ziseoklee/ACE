import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_layout(
    image: Image.Image,
    phrases: list[str],
    boxes: list[tuple[float, float, float, float]],
) -> Image.Image:
    """
    Draws bounding boxes and annotates them with phrases on an image.

    Args:
        image (Image.Image): The input PIL image.
        phrases (list[str]): A list of phrases to annotate.
        boxes (list[tuple[float, float, float, float]]): A list of bounding boxes
            in [xmin, ymin, xmax, ymax] format, normalized to [0, 1].

    Returns:
        Image.Image: The image with bounding boxes and annotations.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Scale font size and box width based on image size
    base_width = 512  # Reference width for scaling
    scale_factor = width / base_width
    font_size = max(10, int(30 * scale_factor))
    box_width = max(1, int(3 * scale_factor))
    text_offset = font_size

    # Generate distinct colors for each box
    # Using a colormap to generate more distinct colors
    num_colors = len(boxes)
    colors = [tuple(int(c * 255) for c in plt.get_cmap("viridis", num_colors)(i)[:3]) for i in range(num_colors)]

    try:
        font = ImageFont.truetype("fonts/TimesNewRoman.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for i, (box, phrase) in enumerate(zip(boxes, phrases)):
        xmin, ymin, xmax, ymax = box

        # Denormalize coordinates
        abs_xmin = xmin * width
        abs_ymin = ymin * height
        abs_xmax = xmax * width
        abs_ymax = ymax * height

        # Get color for the box
        color = colors[i]

        # Draw bounding box
        draw.rectangle([(abs_xmin, abs_ymin), (abs_xmax, abs_ymax)], outline=color, width=box_width)

        # Draw text
        text_position = (abs_xmin, abs_ymin - text_offset) if abs_ymin - text_offset > 0 else (abs_xmin, abs_ymin + 5)

        # Draw a small background for the text for better readability
        text_bbox = draw.textbbox(text_position, phrase, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text(text_position, phrase, fill="white", font=font)

    return image


def draw_instance_with_masks(instance_img_rgb: np.ndarray, masks: np.ndarray, phrases: list[str]):
    """
    Draws segmentation masks on an image.

    Args:
        instance_img_rgb (np.ndarray): The input RGB instance image.
        masks (np.ndarray): The segmentation masks.
        phrases (list[str]): The phrases corresponding to each mask.

    Returns:
        np.ndarray: The image with segmentation masks applied.
    """
    # Ensure masks are converted to the right shape (N, 512, 512)
    masks = masks.squeeze(1)

    # Generate random colors for each mask (N, 3)
    N = masks.shape[0]
    # colors = np.random.rand(N, 3)
    color_dict = {
        "black": [0.0, 0.0, 0.0],
        "white": [255.0, 255.0, 255.0],
        "red": [255.0, 0.0, 0.0],
        "green": [0.0, 255.0, 0.0],
        "blue": [0.0, 0.0, 255.0],
        "yellow": [255.0, 255.0, 0.0],
        "purple": [255.0, 0.0, 255.0],
        "pink": [255.0, 192.0, 203.0],
        "brown": [165.0, 42.0, 42.0],
        "gray": [128.0, 128.0, 128.0],
        "grey": [128.0, 128.0, 128.0],
        "orange": [255.0, 165.0, 0.0],
    }
    colors = []
    for i in range(N):
        flag = False
        for color in color_dict:
            if color in phrases[i]:
                now_color = np.array(color_dict[color])
                now_color[0], now_color[2] = now_color[2], now_color[0]
                colors.append(now_color)
                flag = True
                break
        if not flag:
            colors.append(np.random.rand(3) * 255.0)

    # Create a copy of the instance image to overlay masks
    overlay_image = instance_img_rgb.copy()

    # Iterate over each mask and apply color
    for i in range(N):
        mask = masks[i]
        color = colors[i]

        # Expand mask to 3 channels
        mask_rgb = np.stack([mask] * 3, axis=-1)

        # Apply the color to the mask area
        overlay_image = np.where(mask_rgb, overlay_image * 0.5 + color * 0.5, overlay_image)

    return overlay_image


# It's good practice to have a small example of how to use the function
if __name__ == "__main__":
    # Load an example image
    input_image = Image.open("images/Gemini_001_hard.jpg")

    # Example inputs from demo_gligen.py
    phrases = ["cats", "cats", "dogs", "cars", "cars", "bicycle"]
    boxes = [
        (0.281, 0.169, 0.451, 0.417),
        (0.419, 0.187, 0.563, 0.440),
        (0.471, 0.461, 0.887, 0.807),
        (0.000, 0.245, 0.562, 0.584),
        (0.438, 0.259, 0.999, 0.617),
        (0.341, 0.319, 0.650, 0.936),
    ]

    # Draw the layout
    layout_image = draw_layout(input_image.copy(), phrases, boxes)

    # Display the image
    plt.imshow(layout_image)
    plt.axis("off")
    plt.show()

    # Save the image
    layout_image.save("layout_example.jpg")
    print("Saved example image to layout_example.jpg")
