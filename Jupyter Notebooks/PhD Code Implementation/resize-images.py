def resize_and_save_images(input_dir, output_dir, target_width=512, target_height=256):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of image filenames in the input directory
    image_filenames = [filename for filename in os.listdir(input_dir) if filename.endswith('.png')]
    print(image_filenames)

    for image_filename in image_filenames:
        try:
            # Load the image
            img = imread(os.path.join(input_dir, image_filename))

            # Resize the image to the target dimensions
            img_resized = resize(img, (target_height, target_width), mode='constant', preserve_range=True)

            # Save the resized image
            output_filename = os.path.splitext(image_filename)[0] + ".png"
            imsave(os.path.join(output_dir, output_filename), img_resized.astype(np.uint8))
        except Exception as e:
            print(f"Error processing {image_filename}: {e}")