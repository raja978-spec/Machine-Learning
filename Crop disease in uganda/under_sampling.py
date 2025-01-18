#     UNDER SAMPLING
'''
 Under-sampling is a technique used in data preprocessing, 
 particularly in machine learning, to address the issue of 
 imbalanced datasets.

 By undersampling, you reduce the number of samples in the majority class to match the minority class. 

 For example:

 Original dataset:
 
 Class A: 900 samples
 Class B: 100 samples

 After undersampling (target_count = 100):
 
 Class A: 100 samples (reduced from 900)
 Class B: 100 samples (unchanged)
 
 NEED OF THIS:

 Imagine training a model to classify medical images:

 Class A: Healthy (90% of samples)
 Class B: Disease (10% of samples)

 Without undersampling:

 The model learns to predict healthy most of the time because it minimizes overall error.

 And it will predict disease as healthy most of the time.

 This could result in life-threatening misdiagnoses for diseased cases.

 You can check if a dataset has under sampling by visuliation the
 data in bar chart, if all the bars in bar chart are not in same
 level then the dataset has under sampling.

 Following function is used to change the dataset to under sampled
 dataset, means it makes all bar in equal level.

 def undersample_dataset(dataset_dir, output_dir, target_count=None):
    """
    Undersample the dataset to have a uniform distribution across classes.

    Parameters:
    - dataset_dir: Path to the directory containing the class folders.
    - output_dir: Path to the directory where the undersampled dataset will be stored.
    - target_count: Number of instances to keep in each class. If None, the class with the least instances will set the target.
    """
    # Mapping each class to its files
    classes_files = {}
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            files = os.listdir(class_dir)
            classes_files[class_name] = files

    # Determine the minimum class size if target_count is not set
    if target_count is None:
        target_count = min(len(files) for files in classes_files.values())

      # Creating the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Perform undersampling
    for class_name, files in classes_files.items():
        print("Copying images for class", class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        # Randomly select target_count images
        selected_files = random.sample(files, min(len(files), target_count))

        # Copy selected files to the output directory
        for file_name in tqdm(selected_files):
            src_path = os.path.join(dataset_dir, class_name, file_name)
            dst_path = os.path.join(class_output_dir, file_name)
            copy2(src_path, dst_path)

    print(f"Undersampling completed. Each class has up to {target_count} instances.")
'''